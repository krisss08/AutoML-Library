import tempfile
import mlflow

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
import pandas as pd
import utils as automl_utils
from models.ensemble import EnsembleModel
from models.stacking import StackingModel
from preprocessor import AutoMLPreprocess
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import RFE

from pprint import pprint
from IPython.display import display

from constants import (
    ENV, 
    CLIENT
)
class AutoMLModelTrain(BaseEstimator):
    
    def __init__(self, configs):
        self.configs = configs
        self.task = configs['task']
        self.stacking = configs.get('stacking', None)
        self.ensemble = configs.get('ensemble', None)
        self.include_models = configs.get('include_models', [])
        self.train_models = None
        self.ensemble_model = None
        self.stacking_model = None
        self.focus = configs.get('focus', None) # set default for classification and regression 
        self.tune = configs.get('tune', False)
        self.val_size = configs.get('validation_set_size', 0.2)
        self.best_model = None
        self.run_name = None
        self.best_model_name = None
        self.model_description = None
        self.verbose = configs.get('verbose', True)
        self.ignore_columns = configs.get('ignore_columns', [])
        self.experiment_name = configs['experiment_name']

        # categorize columns
        self._fx = None

        # preproc object 
        self.pp_ = None
        self.fit_num_to_cat = configs.get('encode', {}).get('fit_numerical_to_categorical', [])
        self.encoder_method = configs.get('encode', {}).get('categorical_encoder_method', [])
        self.include_features = configs.get('include_features')
        self.features = []
        self.selected_features = None

        self.run_id = configs.get("run_id", None)

        if not len(self.include_models):
            if self.task == 'Classification':
                self.include_models = ['LogisticRegression', 'XGBoost', 'DecisionTree']
            else:
                self.include_models = ['LinearRegression', 'XGBoost']
        if not self.focus: 
            if self.task == 'Classification': 
                self.focus = 'accuracy'
            else: 
                self.focus = 'r2'

    def preprocess_data(self, X, y, train = True, artifact_uri = None):
        
        # categorize the columns
        self._fx = automl_utils.categorize_columns(X)

        # convert datetime to numerical feature
        current_date = pd.to_datetime('today')
        for col in self._fx['DateTime']:
            X[col] = pd.to_datetime(X[col], errors = 'coerce')
            X[col] = (current_date - X[col]).dt.days
        
        self._fx['Numeric'] += self._fx['DateTime']

        if self.run_id:
            self.pp_ = AutoMLPreprocess(self.configs, self._fx)
            pipeline_path = f"{artifact_uri}/pipeline"
            self.pp_.pipeline = mlflow.sklearn.load_model(pipeline_path)
            print("successfully loaded pipeline from ML Client.")
            X_preprocessed = self.pp_.transform(X)
            print("processed data:")
            display(pd.DataFrame(X_preprocessed, columns=self.features).head())
            print()

        else:
            if train:
                self.pp_ = AutoMLPreprocess(self.configs, self._fx)
                self.pp_.fit(X, y)
            X_preprocessed = self.pp_.transform(X)

        return X_preprocessed, y

    def feature_selection(self, X_train, y_train, X_val):
        xgb = XGBClassifier() if self.task == 'Classification' else XGBRegressor()
        X_train_df = pd.DataFrame(X_train, columns = self.features)
        y_train_df = pd.DataFrame(y_train, columns = [y_train.name])
        X_val_df = pd.DataFrame(X_val, columns = self.features)
        xgb.fit(X_train_df, y_train_df)

        rfe = RFE(xgb, n_features_to_select=self.include_features, step=30)
        rfe.fit(X_train_df, y_train_df)
        X_train = rfe.transform(X_train_df)
        X_val = rfe.transform(X_val_df)

        selected_features = [feature for feature, status in zip(self.features, rfe.support_) if status]
        return X_train, X_val, selected_features

    def fit(self, X, y):
        # ignoring some columns
        X = X[[col for col in X.columns if col not in self.ignore_columns]]

        # converting numerical columns to catergorical columns
        for col in self.fit_num_to_cat:
            X[col] = X[col].astype(str)

        # split the data into training and validation

        stratify = y if self.task == 'Classification' else None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, 
                                                          shuffle=True, stratify=stratify,
                                                          random_state=42)
        
        print("Sample pre processed data: ")
        display(X.head())
        print()
        
        # data preprocessing 
        print("Data Preprocessing in progress...")
        X_train, y_train = self.preprocess_data(X_train, y_train)
        X_val, _ = self.preprocess_data(X_val, None, train = False)

        # display(pd.DataFrame(X_val, columns=self.features).head())
        print("*"*20)

        if self.features == []:
            # keep track of the column names
            if self.encoder_method == 'one_hot':
                self.features = automl_utils.get_column_names(X, self._fx)
            else:
                self.features = X.columns

        # feature selection

        if self.include_features != 'all':
            if self.verbose: 
                print("Feature Selection in progress...")

            X_train, X_val, self.selected_features = self.feature_selection(X_train, y_train, X_val)
            
        else:
            self.selected_features = list(self.features)
            
        print("\n Selected features - Sample: ")
        pprint(list(self.selected_features)[:10])
        print()

        if self.ensemble:

            with mlflow.start_run(run_name=f"automl_ensemble-{self.experiment_name}-{model_identifier}", 
                                  description=f"Creating an Ensemble of {self.include_models}"): 
            
                if self.verbose:
                    print("\n Data preprocessing for the training data has been completed successfully \n")
                    print(f"Training of Ensemble model of {', '.join(self.include_models)}")
                    print("This might take a moment...")

                self.ensemble_model = EnsembleModel(self.configs)
                self.ensemble_model.fit(X_train, y_train)

                if self.verbose:
                    print("\n Training has been completed successfully \n")

                self.best_model = self.ensemble_model

                mlflow.sklearn.log_model(self.best_model, "model")
                mlflow.sklearn.log_model(self.pp_.pipeline, "pipeline")

                training_prediction = self.ensemble_model.predict(X_train)
                training_metrics = automl_utils.calculate_metrics(training_prediction, y_train, self.task)
                training_metrics = {f'training_{key}': value for key, value in training_metrics.items()}
                mlflow.log_metrics(training_metrics)
                
                validation_prediction = self.ensemble_model.predict(X_val)
                validation_metrics = automl_utils.calculate_metrics(validation_prediction, y_val, self.task)
                validation_metrics = {f'validation_{key}': value for key, value in validation_metrics.items()}
                mlflow.log_metrics(validation_metrics)
                
                # path for interpretation plots and csv
                features, scores = self.ensemble_model.interpret(X_train, X_val, y_val, self.features, 'Ensemble')
                img_buffer, table_buffer = automl_utils.save_interpretation_to_s3('Ensemble', features, scores)

                # Log image to MLflow
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img_file:
                    temp_img_file.write(img_buffer.getvalue())
                    mlflow.log_artifact(temp_img_file.name, artifact_path="feature_importance_plots")
                
                # Log table to MLflow
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_table_file:
                    temp_table_file.write(table_buffer.getvalue().encode())
                    mlflow.log_artifact(temp_table_file.name, artifact_path="feature_importance_tables")

                mlflow.log_dict(self.configs, "config.json")

        elif self.stacking:

            with mlflow.start_run(run_name=f"automl_stacking-{self.experiment_name}-{model_identifier}", 
                                  description=f"Creating an Stack of {self.include_models}"): 

                if self.verbose:
                    print("\n Data preprocessing for the training data has been completed successfully \n")

                self.stacking_model = StackingModel(self.configs)
                self.stacking_model.fit(X_train, y_train)

                if self.verbose:
                    print("Training has been completed successfully \n")

                self.best_model = self.stacking_model
                mlflow.sklearn.log_model(self.best_model, "model")
                mlflow.sklearn.log_model(self.pp_.pipeline, "pipeline")

                training_prediction = self.stacking_model.predict(X_train)
                training_metrics = automl_utils.calculate_metrics(training_prediction, y_train, self.task)
                training_metrics = {f'training_{key}': value for key, value in training_metrics.items()}
                mlflow.log_metrics(training_metrics)
                
                validation_prediction = self.stacking_model.predict(X_val)
                validation_metrics = automl_utils.calculate_metrics(validation_prediction, y_val, self.task)
                validation_metrics = {f'validation_{key}': value for key, value in validation_metrics.items()}
                mlflow.log_metrics(validation_metrics)

                # path for interpretation plots and csv
                features, scores = self.stacking_model.interpret(X_train, X_val, y_val, self.features, 'Ensemble')
                img_buffer, table_buffer = automl_utils.save_interpretation_to_s3('Ensemble', features, scores)

                # Log image to MLflow
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img_file:
                    temp_img_file.write(img_buffer.getvalue())
                    mlflow.log_artifact(temp_img_file.name, artifact_path="feature_importance_plots")
                
                # Log table to MLflow
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_table_file:
                    temp_table_file.write(table_buffer.getvalue().encode())
                    mlflow.log_artifact(temp_table_file.name, artifact_path="feature_importance_tables")

                mlflow.log_dict(self.configs, "config.json")
  
        else:   
            # Neither ensemble nor stacking - train all the models and find the best
            self.train_models = [automl_utils.models[self.task][model](self.configs) for model in self.include_models]
            model_run_id_mapping = {}

            all_model_metrics = []
            for model, name in zip(self.train_models, self.include_models):

                with mlflow.start_run(): 

                    if self.verbose:
                        print(f"{name} Training began successfully.")
                        print("This might take a moment...")

                    if self.tune:
                        model.tune_and_fit(X_train, y_train)
                    else:
                        model.fit(X_train, y_train)

                    if self.verbose:
                        print(f"Training completed, proceeding to register.")

                    model_identifier = automl_utils.generate_uuid()

                    self.run_name = f"automl-all_models-{name}-{model_identifier}"
                    self.model_description = f"Fitting all models, compare charts and pick the model."

                    mlflow.sklearn.log_model(model, "model")
                    mlflow.sklearn.log_model(self.pp_.pipeline, "pipeline")

                    training_prediction = model.predict(X_train)
                    training_metrics = automl_utils.calculate_metrics(training_prediction, y_train, self.task)
                    training_metrics = {f'training_{key}': value for key, value in training_metrics.items()}
                    mlflow.log_metrics(training_metrics)
                    
                    validation_prediction = model.predict(X_val)
                    validation_metrics = automl_utils.calculate_metrics(validation_prediction, y_val, self.task)
                    validation_metrics = {f'validation_{key}': value for key, value in validation_metrics.items()}
                    mlflow.log_metrics(validation_metrics)

                    all_model_metrics.append(validation_metrics)

                    # path for interpretation plots and csv
                    features, scores = model.interpret(X_train, X_val, y_val, self.features, 'Ensemble')
                    img_buffer, table_buffer = automl_utils.save_interpretation_to_s3('Ensemble', features, scores)

                    # Log image to MLflow
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img_file:
                        temp_img_file.write(img_buffer.getvalue())
                        mlflow.log_artifact(temp_img_file.name, artifact_path="feature_importance_plots")
                    
                    # Log table to MLflow
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_table_file:
                        temp_table_file.write(table_buffer.getvalue().encode())
                        mlflow.log_artifact(temp_table_file.name, artifact_path="feature_importance_tables")
                        
                if self.verbose:
                    print("\n Data preprocessing and training has been completed successfully")

                # get the best model based on focus
                self.best_model, self.best_model_name = automl_utils.select_best(self.include_models, self.train_models,
                                                                    all_model_metrics, self.task, self.focus)

                if self.verbose:
                    print("The best model is ", self.best_model_name)
                    print("Model ID: ", model_run_id_mapping[self.best_model_name]['model_id'])
                    print("ML Flow Run ID: ", model_run_id_mapping[self.best_model_name]['mlflow_run_id'])
                    all_model_metrics = pd.DataFrame(all_model_metrics).T
                    all_model_metrics.columns = self.include_models
                    display(all_model_metrics)

                mlflow.log_dict(self.configs, "config.json")

    def predict(self, X):
        
        artifact_uri = None

        if self.run_id:
            logged_model = f"runs:/{self.run_id}/model"
            self.best_model = mlflow.pyfunc.load_model(logged_model)

            model_config = automl_utils.get_model_config(self.run_id)
            artifact_uri = model_config['data']['ml_client_model_config']['artifact_uri']
            model_library = model_config['data']['model_parameters']['library']

            ml_client_run_id = model_config['data']['ml_client_model_config']['run_id']

            self.features = automl_utils.fetch_features_from_s3(self.configs, ml_client_run_id)

            configs = automl_utils.read_config(self.experiment_name, ml_client_run_id)
            self.ignore_columns = configs.get('ignore_columns', [])
            self.fit_num_to_cat = configs.get('encode', {}).get('fit_numerical_to_categorical', [])

            self.selected_features = model_config['data']['artifact_config']['data_preprocessing_pipeline'][0]['encoding_fields']

            logged_model = f"{artifact_uri}/model"

            mlflow_flavour = automl_utils.library_mapping.get(model_library)

            self.best_model =  mlflow_flavour.load_model(logged_model)

            if self.verbose:
                print("Loading the model from the given run id")
            

        # ignoring some columns
        X = X[[col for col in X.columns if col not in self.ignore_columns]]

        # converting numerical columns to categorical columns
        for col in self.fit_num_to_cat:
            X[col] = X[col].astype(str)

        X_test, _ = self.preprocess_data(X, None, train = False, artifact_uri = artifact_uri)
        if self.verbose:
            print("Preprocessing for test data has been completed")

        X_test_df = pd.DataFrame(X_test, columns = self.features)
        X_test_df = X_test_df[self.selected_features]
        return self.best_model.predict(X_test_df)
    
    def predict_proba(self, X):            
        
        artifact_uri = None

        if self.run_id:
            model_config = automl_utils.get_model_config(self.run_id)
            artifact_uri = model_config['data']['ml_client_model_config']['artifact_uri']
            model_library = model_config['data']['model_parameters']['library']

            ml_client_run_id = model_config['data']['ml_client_model_config']['run_id']

            self.features = automl_utils.fetch_features_from_s3(self.configs, ml_client_run_id)

            configs = automl_utils.read_config(self.experiment_name, ml_client_run_id)
            self.ignore_columns = configs['ignore_columns']
            self.fit_num_to_cat = configs['encode']['fit_numerical_to_categorical']

            self.selected_features = model_config['data']['artifact_config']['data_preprocessing_pipeline'][0]['encoding_fields']

            logged_model = f"{artifact_uri}/model"

            mlflow_flavour = automl_utils.library_mapping.get(model_library)

            self.best_model =  mlflow_flavour.load_model(logged_model)

            if self.verbose:
                print("Loading the model from the given run id")
            

        # ignoring some columns
        X = X[[col for col in X.columns if col not in self.ignore_columns]]

        # converting numerical columns to categorical columns
        for col in self.fit_num_to_cat:
            X[col] = X[col].astype(str)

        X_test, _ = self.preprocess_data(X, None, train = False, artifact_uri = artifact_uri)
        if self.verbose:
            print("Preprocessing for test data has been completed")

        X_test_df = pd.DataFrame(X_test, columns = self.features)
        X_test_df = X_test_df[self.selected_features]
        return self.best_model.predict_proba(X_test_df)