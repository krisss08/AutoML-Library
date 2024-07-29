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

    def construct_mlops_payload_for_model(self, payload_contruction_dict): 

        model_payload = {}

        model_payload['model_name'] = self.run_name
        model_payload['model_description'] = self.model_description
        model_payload['experiment_name'] = self.experiment_name
        model_payload['task'] = self.task
        model_payload['is_automl'] = True
        model_payload['model_parameters'] = {}
        model_payload['model_parameters']['model_architecture'] = payload_contruction_dict['model_architecture']
        model_payload['model_parameters']['library'] = 'scikit-learn'
        try: 
            model_args = payload_contruction_dict['model'].get_params()
            model_payload['model_parameters']['model_args'] = {key: str(value) for key, value in model_args.items()}
            
        except: 
            model_payload['model_parameters']['model_args'] = {}
            
        model_payload['metrics'] = {}
        model_payload['metrics']['training_metrics'] = payload_contruction_dict['metrics']['training_metrics']
        model_payload['metrics']['validation_metrics'] = payload_contruction_dict['metrics']['validation_metrics']

        model_payload['artifact_config'] = {}
        model_payload['artifact_config']['model'] = payload_contruction_dict['model_s3_path']

        model_payload['artifact_config']['data_preprocessing_pipeline'] = []
        model_payload['artifact_config']['data_preprocessing_pipeline'].append({'step_name': 'pipeline', 'object_path': payload_contruction_dict['preproc_pipeline_s3_path'], 'encoding_fields':list(self.selected_features)})

        model_payload['model_interpretability'] = {}
        model_payload['model_interpretability']['feature_scores'] = {}
        model_payload['model_interpretability']['feature_scores']['visual_representation'] = payload_contruction_dict['interpret_plot_path']
        model_payload['model_interpretability']['feature_scores']['tabular_representation'] = payload_contruction_dict['interpret_table_path']

        return model_payload

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
            
            if self.verbose:
                print("\n Data preprocessing for the training data has been completed successfully \n")
                print(f"Training of Ensemble model of {', '.join(self.include_models)}")
                print("This might take a moment...")

            self.ensemble_model = EnsembleModel(self.configs)
            self.ensemble_model.fit(X_train, y_train)

            if self.verbose:
                print("\n Training has been completed successfully \n")

            self.best_model = self.ensemble_model
            model_identifier = automl_utils.generate_uuid()
            
            self.best_model_name = f"automl_ensemble-{self.experiment_name}-{model_identifier}"
            self.run_name = f"automl_ensemble-{self.experiment_name}-{model_identifier}"
            self.model_description = f"Creating an Ensemble of {self.include_models}"

            training_prediction = self.ensemble_model.predict(X_train)
            training_metrics = automl_utils.calculate_metrics(training_prediction, y_train, self.task)
            
            validation_prediction = self.ensemble_model.predict(X_val)
            validation_metrics = automl_utils.calculate_metrics(validation_prediction, y_val, self.task)

            model_s3_path = automl_utils.upload_joblib_to_s3(self.best_model.model, f'AutoML_experiments/{ENV}/{CLIENT}/{self.experiment_name}/{self.run_name}/artifacts/model.joblib')
            preproc_pipeline_s3_path = automl_utils.upload_joblib_to_s3(self.pp_.pipeline, f'AutoML_experiments/{ENV}/{CLIENT}/{self.experiment_name}/{self.run_name}/artifacts/pipeline.joblib')

            # path for interpretation plots and csv
            common_path = f'AutoML_experiments/{ENV}/{CLIENT}/{self.experiment_name}/{self.run_name}/artifacts/'
            features, scores = self.ensemble_model.interpret(X_train, X_val, y_val, self.features, 'Ensemble')
            interpret_plot_path, interpret_table_path = automl_utils.save_interpretation_to_s3('Ensemble', features, scores, common_path+'interpretation_plot.png', common_path+'interpretation_table.csv')

            payload_contruction_dict = {
                'metrics' : {
                    'training_metrics': training_metrics,
                    'validation_metrics': validation_metrics
                },
                'model_s3_path': model_s3_path,
                'preproc_pipeline_s3_path': preproc_pipeline_s3_path,
                'interpret_plot_path': interpret_plot_path,
                'interpret_table_path': interpret_table_path,
                'model': self.ensemble_model.model,
                'model_architecture': 'ensemble'
            }

            model_payload = self.construct_mlops_payload_for_model(payload_contruction_dict)

            response = automl_utils.create_new_run(model_payload)
            if response.status_code==201: 
                print(f"Model {self.best_model_name} registered successfully in ML Client")
                print(f"Model ID: {response.json()['config']['model_id']}")
                print(f"ML Client Model ID: {response.json()['config']['ml_client_model_config']['run_id']}")
            else: 
                print(response.json())

            # dumping the train config 
            automl_utils.dump_config(self.configs, response.json()['config']['ml_client_model_config']['run_id'])
            
            # dump the feature list 
            automl_utils.dump_feature_list(self.configs, response.json()['config']['ml_client_model_config']['run_id'],list(self.features))

        elif self.stacking:

            if self.verbose:
                print("\n Data preprocessing for the training data has been completed successfully \n")

            self.stacking_model = StackingModel(self.configs)
            self.stacking_model.fit(X_train, y_train)

            if self.verbose:
                print("Training has been completed successfully \n")

            self.best_model = self.stacking_model
            model_identifier = automl_utils.generate_uuid()

            self.best_model_name = f"automl_stacking-{self.experiment_name}-{model_identifier}"
            self.run_name = f"automl_stacking-{self.experiment_name}-{model_identifier}"
            self.model_description = f"Creating a Stack of {self.include_models}"

            training_prediction = self.stacking_model.predict(X_train)
            training_metrics = automl_utils.calculate_metrics(training_prediction, y_train, self.task)
            
            validation_prediction = self.stacking_model.predict(X_val)
            validation_metrics = automl_utils.calculate_metrics(validation_prediction, y_val, self.task)

            model_s3_path = automl_utils.upload_joblib_to_s3(self.best_model.model, f'AutoML_experiments/{ENV}/{CLIENT}/{self.experiment_name}/{self.run_name}/artifacts/model.joblib')
            preproc_pipeline_s3_path = automl_utils.upload_joblib_to_s3(self.pp_.pipeline, f'AutoML_experiments/{ENV}/{CLIENT}/{self.experiment_name}/{self.run_name}/artifacts/pipeline.joblib')

            # path for interpretation plots and csv
            common_path = f'AutoML_experiments/{ENV}/{CLIENT}/{self.experiment_name}/{self.run_name}/artifacts/'
            features, scores = self.stacking_model.interpret(X_train, X_val, y_val, self.features, 'Stacking')
            interpret_plot_path, interpret_table_path = automl_utils.save_interpretation_to_s3('Stacking', features, scores, common_path+'interpretation_plot.png', common_path+'interpretation_table.csv')

            payload_contruction_dict = {
                'metrics' : {
                    'training_metrics': training_metrics,
                    'validation_metrics': validation_metrics
                },
                'model_s3_path': model_s3_path,
                'preproc_pipeline_s3_path': preproc_pipeline_s3_path,
                'interpret_plot_path': interpret_plot_path,
                'interpret_table_path': interpret_table_path,
                'model': self.stacking_model.model,
                'model_architecture': 'stacking'
            }

            model_payload = self.construct_mlops_payload_for_model(payload_contruction_dict)

            response = automl_utils.create_new_run(model_payload)
            print(response.text)
            if response.status_code==201: 
                print(f"Model {self.best_model_name} registered successfully in ML Client")
                print(f"Model ID: {response.json()['config']['model_id']}")
                print(f"ML Client Model ID: {response.json()['config']['ml_client_model_config']['run_id']}")
            else: 
                print(response.json())

            # dumping the train config 
            # automl_utils.dump_config(self.configs, self.best_model_name)
            automl_utils.dump_config(self.configs, response.json()['config']['ml_client_model_config']['run_id'])

            # dump the feature list 
            # automl_utils.dump_feature_list(self.configs, self.best_model_name, list(self.features))
            automl_utils.dump_feature_list(self.configs, response.json()['config']['ml_client_model_config']['run_id'], list(self.features))
  
        else:   
            # Neither ensemble nor stacking - train all the models and find the best
            self.train_models = [automl_utils.models[self.task][model](self.configs) for model in self.include_models]
            model_run_id_mapping = {}

            all_model_metrics = []
            for model, name in zip(self.train_models, self.include_models):

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

                training_prediction = model.predict(X_train)
                training_metrics = automl_utils.calculate_metrics(training_prediction, y_train, self.task)
                
                validation_prediction = model.predict(X_val)
                validation_metrics = automl_utils.calculate_metrics(validation_prediction, y_val, self.task)

                all_model_metrics.append(validation_metrics)

                model_s3_path = automl_utils.upload_joblib_to_s3(model.model, f'AutoML_experiments/{ENV}/{CLIENT}/{self.experiment_name}/{self.run_name}/artifacts/model.joblib')
                preproc_pipeline_s3_path = automl_utils.upload_joblib_to_s3(self.pp_.pipeline, f'AutoML_experiments/{ENV}/{CLIENT}/{self.experiment_name}/{self.run_name}/artifacts/pipeline.joblib')

                # path for interpretation plots and csv
                common_path = f'AutoML_experiments/{ENV}/{CLIENT}/{self.experiment_name}/{self.run_name}/artifacts/'
                print(model)
                features, scores = model.interpret(X_train, X_val, y_val, self.features, name)
                interpret_plot_path, interpret_table_path = automl_utils.save_interpretation_to_s3(name, features, scores, common_path+'interpretation_plot.png', common_path+'interpretation_table.csv')

                payload_contruction_dict = {
                    'metrics' : {
                        'training_metrics': training_metrics,
                        'validation_metrics': validation_metrics
                    },
                    'model_s3_path': model_s3_path,
                    'preproc_pipeline_s3_path': preproc_pipeline_s3_path,
                    'interpret_plot_path': interpret_plot_path,
                    'interpret_table_path': interpret_table_path,
                    'model': model.model,
                    'model_architecture': name
                }

                model_payload = self.construct_mlops_payload_for_model(payload_contruction_dict)

                response = automl_utils.create_new_run(model_payload)

                if response.status_code==201: 
                    print(f"Model {self.run_name} registered successfully in ML Client")
                    print(f"Model ID: {response.json()['config']['model_id']}")
                    print(f"ML Client Model ID: {response.json()['config']['ml_client_model_config']['run_id']}")
                    model_run_id_mapping[name] = {"model_id": response.json()['config']['model_id'], "mlflow_run_id": response.json()['config']['ml_client_model_config']['run_id']}

                else: 
                    print(response.json())

                # dumping the train config 
                # automl_utils.dump_config(self.configs, self.run_name)
                automl_utils.dump_config(self.configs, response.json()['config']['ml_client_model_config']['run_id'])

                # dump the feature list 
                # automl_utils.dump_feature_list(self.configs, self.run_name, list(self.features))
                automl_utils.dump_feature_list(self.configs, response.json()['config']['ml_client_model_config']['run_id'], list(self.features))
                
            
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

    def predict(self, X):
        
        artifact_uri = None

        if self.run_id:
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