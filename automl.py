from model import AutoMLModelTrain
import utils as automl_utils

import warnings
warnings.filterwarnings('ignore')

class AutoML:

    def __init__(self, configs):

        self.configs = configs
        self.experiment_name = configs['experiment_name']
        self.task = configs['task']
        self.model = AutoMLModelTrain(self.configs)
        self.verbose = configs.get('verbose', True)
        self.run_id = configs.get("run_id", None)

        # checking model config here - throws error if unknown tasks are provided 
        automl_utils.check_params(self.configs)

        if self.run_id is None:
            experiment_response = automl_utils.create_new_experiment(self.experiment_name)

            if experiment_response.status_code==201: 
                print("Experiment set up successful in ML Client")
                print(f"Expeiment ID: {experiment_response.json()['data']}")
                print("*"*20)
                print()

            elif experiment_response.status_code==409:
                print("Experiment with same name exists - runs will be logged under the existing experiment.")
                print("*"*20)
                print()

            else: 
                print("Failure in setting up Experiment in ML Client")
                print(experiment_response.text)
                print("*"*20)
                print()
        
        print(f"Experiment Name: {self.experiment_name}")
        print(f"Task: {self.task}")

        print("*"*20)
        print()


    def fit(self, X, y):
        # does all the preprocessing and the model training

        if self.run_id is not None:
            raise ValueError('Cannot set to train model when a run_id is specified, remove run_id to pregoress with training.')
        
        if self.verbose:
            print("SET TO TRAIN MODE")
            print("*"*20)
            print()
        
        self.model.fit(X, y)
        
    def predict(self, X):

        if self.verbose:
            print("SET TO INFERENCE MODE")
            print("*"*20)
            print()

        return self.model.predict(X)
    
    def predict_proba(self, X):

        if self.task == 'Regression':
            raise TypeError('Predict proba is not applicable for regression tasks')

        if self.verbose:
            print("SET TO INFERENCE MODE")
            print("*"*20)
            print()

        return self.model.predict_proba(X)
