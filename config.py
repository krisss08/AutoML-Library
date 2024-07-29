## write the json to s3 

configurations = {
    "ignore_columns": ['Name'],

    "null_value_imputation": {
        "numerical_defaults": "mean",
        "categorical_defaults": "Unknown",
        "datetime_defaults": None,
        "drop_null": False
    },
    "encode": {
        "categorical_encoder_method": "one_hot",
        "fit_numerical_to_categorical": ['Pclass'],
        "numerical_encoder_method": 'min_max',
        "numerical_encoding_ignore_columns": ['Fare', 'Age']
    },
    "skew": {
        "skew_function": 'yeo-johnson',
        "skew_threshold": 0.5,
    },

    "outlier": {
        'handling_method': 'handle' # / 'drop' / 'handle' / None
    },

    'include_features': 4, # either all or some n(int) features
    'validation_set_size': 0.3,
    'cv_folds': 7,
    'task': 'Classification',
    'ensemble': True,
    'stacking': False,
    'tune': True,
    'include_models': ['DecisionTree'],
    'focus': 'recall',
    'verbose': True,

    'experiment_name': 'titanic exp186', ### do we have to change for every run 
    # 'run_id': 'fc2723d0-1f73-11ef-b4ad-42e8aff021e3'
}