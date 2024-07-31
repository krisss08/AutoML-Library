import pandas as pd
import numpy as np
import os
import io 
import mlflow
import requests
import boto3
import json
import os
import joblib
import tempfile
import uuid
import matplotlib.pyplot as plt

from models.linearRegressor import Linear
from models.lassoRegressor import LassoRegression

from models.logisticRegression import Logistic
from models.naivebayesClassifier import NaiveBayes

from models.knn import KNNRegressor, KNNClassifier
from models.decisiontree import DecisiontreeClassifier, DecisiontreeRegressor
from models.randomforest import RandomforestClassifier, RandomforestRegressor
from models.xgboost import XGBoostClassifier, XGBoostRegressor

from metrics import ClassificationMetrics, RegressionMetrics

from constants import (
    MLOPS_URL, 
    CLIENT, 
    CLIENT_ID, 
    ENV
)


models = {
    'Classification': {
        'LogisticRegression': Logistic,
        'KNN': KNNClassifier,
        'NaiveBayesClassifier': NaiveBayes,
        'DecisionTree': DecisiontreeClassifier,
        'RandomForest': RandomforestClassifier,
        'XGBoost': XGBoostClassifier
    },

    'Regression': {
        'LinearRegression': Linear,
        'KNN': KNNRegressor,
        'LassoRegression': LassoRegression,
        'DecisionTree': DecisiontreeRegressor,
        'RandomForest': RandomforestRegressor,
        'XGBoost': XGBoostRegressor
    }
}

metrics = { 
    'Classification': ['accuracy',
                    'precision', 
                    'recall', 
                    'f1'
                    ],
    'Regression': ['r2', 
                   'mean_absolute_error',
                   'mean_squared_error',
                   'root_mean_squared_error',
                   'explained_variance',
                   'mean_absolute_percentage_error'
                   ]
}

library_mapping = {
    "scikit-learn": mlflow.sklearn,
    "tensorflow": mlflow.tensorflow,
    "pytorch": mlflow.pytorch,
    "xgboost": mlflow.xgboost,
    "lightgbm": mlflow.lightgbm,
    # "keras": mlflow.keras # Commenting for now, i dont think this exists
}

headers = {
        "Content-Type": "application/json",
        "x-client-id": CLIENT_ID,
        "x-client-name": CLIENT,
        "x-request-id": "r1",
        "x-correlation-id": "c1",
    }

# HACK: check how to set 
# init s3 boto3 client
s3 = boto3.client('s3')

def identify_nonoverlapping_features(preproc_steps_columns, preproc_tuple_dict): 
    
    column_wise_preproc_steps = {}
    for key, columns in preproc_steps_columns.items():
        for column in columns:
            if column not in column_wise_preproc_steps:
                column_wise_preproc_steps[column] = []
            column_wise_preproc_steps[column].append(key)

            
    for column in column_wise_preproc_steps:
        column_wise_preproc_steps[column] = '_'.join(column_wise_preproc_steps[column])
    
    column_wise_preproc_group = {}
    for key, columns in column_wise_preproc_steps.items():
        if columns not in column_wise_preproc_group:
            column_wise_preproc_group[columns] = []
        column_wise_preproc_group[columns].append(key)
        
    function_steps = {}
    for key, value in column_wise_preproc_group.items():
        parts = key.split("_")

        function_list = []
        for part in parts:
            if part in preproc_tuple_dict:
                function_list.extend(preproc_tuple_dict[part])

        function_steps[key] = function_list
        
        
    return column_wise_preproc_group, function_steps

def upload_joblib_to_s3(data, s3_key):
    bucket_name = "msd-platform-models" ## can configure this later - bucket name / ENV / CLIENT / experiment / run name / artifacts 
    with tempfile.NamedTemporaryFile(delete = False) as temp_file:
        joblib.dump(data, temp_file.name)        
        s3.upload_file(temp_file.name, bucket_name, s3_key)
    return f's3://{bucket_name}/{s3_key}'

def dump_config(configs, model_name): 
    experiment_name = configs.get("experiment_name")
    configs_str = json.dumps(configs)

    bucket_name = 'msd-platform-models'
    object_name = f'{ENV}/{CLIENT}/{experiment_name}/{model_name}/config.json'

    s3.put_object(Body=configs_str, Bucket=bucket_name, Key=object_name)

    print("Config object uploaded to S3")

def read_config(experiment_name, run_id):
    bucket_name = 'msd-platform-models'
    object_name = f'{ENV}/{CLIENT}/{experiment_name}/{run_id}/config.json'

    obj = s3.get_object(Bucket=bucket_name, Key=object_name)
    data = obj['Body'].read().decode('utf-8')
    json_data = json.loads(data)

    return json_data

def dump_feature_list(configs, model_name, feature_list): 
    experiment_name = configs.get("experiment_name")
    configs_str = json.dumps(feature_list)


    bucket_name = 'msd-platform-models'
    object_name = f'{ENV}/{CLIENT}/{experiment_name}/{model_name}/features.json'
    
    s3.put_object(Body=configs_str, Bucket=bucket_name, Key=object_name)

    print("Feature list uploaded to S3")
    
def fetch_features_from_s3(configs, model_name):
    
    bucket_name = 'msd-platform-models'

    experiment_name = configs.get("experiment_name")

    config_path = f'{ENV}/{CLIENT}/{experiment_name}/{model_name}/features.json'

    response = s3.get_object(Bucket=bucket_name, Key=config_path)
    content = response['Body'].read().decode('utf-8')
    config = json.loads(content)

    return config
    
def categorize_columns(X):
    
    "Categorize columns as Numerical, Datetime or Categorical based on the dtypes."

    _fx = {
        "Numeric": [],
        "DateTime": [],
        "Categorical": [],
        "Others": []
    }

    for col in X.columns:

        if pd.api.types.is_numeric_dtype(X[col]):
            _fx["Numeric"].append(col)

        elif pd.api.types.is_datetime64_any_dtype(X[col]):
            _fx["DateTime"].append(col)
        
        elif pd.api.types.is_categorical_dtype(X[col]):
            _fx['Categorical'].append(col)
        
        elif pd.api.types.is_object_dtype(X[col]):
            try:
                pd.to_datetime(X[col], errors='raise')
                _fx['DateTime'].append(col)
            except (ValueError, TypeError):
                _fx['Categorical'].append(col)
        
        else:
            _fx["Others"].append(col)
        
    return _fx

def get_column_names(X, _fx):
    features = []
    for col in X.columns:
        if col in _fx['Categorical']:
            current_cols = X[col].unique()
            features += [f'{col}_{x}' for x in current_cols]
        else:
            features.append(col)
    return features

def calculate_metrics(prediction, y_test, task):
    metrics = ClassificationMetrics(y_test, prediction) if task == 'Classification' else RegressionMetrics(y_test, prediction)
    return metrics.evaluate_metrics()

def select_best(included_models, train_models, all_model_metrics, task, focus):
    best_model = None
    best_model_name = None
    best_metric_val = float('-inf') if task == 'Classification' or (task == 'Regression' and focus in ['r2', 'explained_variance']) else float('inf')

    for name, model, metric in zip(included_models, train_models, all_model_metrics):
        if (task == 'Classification' or (task == 'Regression' and focus in ['r2', 'explained_variance'])) and metric[focus] > best_metric_val:
            best_metric_val = metric[focus]
            best_model = model
            best_model_name = name
            
        if task == 'Regression' and focus not in ['r2', 'explained_variance'] and metric[focus] < best_metric_val:
            best_metric_val = metric[focus]
            best_model = model
            best_model_name = name
    return best_model, best_model_name

def check_params(configs):

    task = configs.get('task', None)
    experiment_name = configs.get('experiment_name', None)
    ensemble = configs.get('ensemble')
    stacking = configs.get('stacking')
    focus = configs.get('focus')

    if not task:
        raise NameError('Task must be specified')
    
    if not experiment_name:
        raise NameError('Experiment name must be specified')

    if task not in ['Classification', 'Regression']:
        raise NameError(f'The specified task {task} is invalid')
    
    if ensemble and stacking:
        raise ValueError('You cannot run both ensembling and stacking at the same time')

    if task == 'Classification' and focus and focus not in metrics['Classification']:
        raise NameError(f'The specified focus {focus} is invalid')
    
    if task == 'Regression' and focus and focus not in metrics['Regression']:
        raise NameError(f'The specified focus {focus} is invalid')

    for model in configs.get('include_models', []):
        if model not in models[task]:
            raise NameError(f'The given model {model} does not exist')

def mlflow_logging(model, model_name, metrics, configs):
    mlflow.log_metrics(metrics)
    mlflow.log_params(model.get_params())
    mlflow.sklearn.log_model(model, model_name)
    mlflow.set_tag("Model", model_name)
    mlflow.log_dict(configs, "config.json")

def get_tags_for_run(run_id):
    client = mlflow.tracking.MlflowClient()
    tags = client.get_run(run_id).data.tags
    return tags['Model']

def create_new_experiment(experiment_name):   
    URL = f"{MLOPS_URL}/api/v2/mlops/experiments"
    experiment_data={
        "experiment_name": experiment_name
    }
    response = requests.post(URL, headers=headers, json=experiment_data)
    return response

def create_new_run(model_payload):
    URL = f"{MLOPS_URL}/api/v2/mlops/models"
    response = requests.post(URL, headers=headers, json=model_payload)
    return response

def get_model_config(model_id):   
    URL = f"{MLOPS_URL}/api/v2/mlops/models/{model_id}"
    response = requests.get(URL, headers=headers)
    print('getting model config...')
    return response.json()

def generate_uuid():
    return str(uuid.uuid4())

def save_interpretation_to_s3(model_name, features, scores):
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name} - Feature Importance Scores")
    plt.bar(features, scores)
    
    # Save the plot to an in-memory bytes buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)  # Move the pointer to the start of the buffer
    plt.close()
    
    # Create the table
    table = pd.DataFrame({'Feature': features, 'Scores': scores})
    
    # Save the table to an in-memory bytes buffer
    table_buffer = io.StringIO()
    table.to_csv(table_buffer, index=False)
    table_buffer.seek(0)  # Move the pointer to the start of the buffer
    
    return img_buffer, table_buffer
