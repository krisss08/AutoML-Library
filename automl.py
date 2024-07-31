import mlflow 
import os 
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
            
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            mlflow.set_experiment(self.experiment_name)
        
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
