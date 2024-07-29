from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import utils as automl_utils
from preprocessing.handle_missing import FillDefault
from preprocessing.handle_skew import SkewHandle
from preprocessing.encoding import Encoding
from preprocessing.handle_outliers import OutlierHandle

import warnings
warnings.filterwarnings(action='ignore')

class AutoMLPreprocess(BaseEstimator,TransformerMixin):

    '''
    This is a generalized pipeline for performing preprocessing tks. 

    >>> data = pd.read_csv('./Some_Data.csv')
    >>> target_var_name = "Target"
    >>> ml = AutoMLPreprocess(configs)
    >>> transformed_data  = ml.fit_transform(data, target_var_name)
    >>> transformed_data.head(4)

    '''

    def __init__(self, configs, _fx):

        self.configs = configs
        self._fx = _fx
        self.run_id = configs.get("run_id", None)
        self.pipeline = None

    def fit(self, X, y):
        self.pipeline, X, y = self._create_pipeline(X, y)
        self.pipeline.fit(X, y)
        return self

    def transform(self, X, y= None):
        return self.pipeline.transform(X)
        
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _create_pipeline(self, X, y):

        all_numeric_columns = self._fx['Numeric']
        all_categorical_columns = self._fx['Categorical']

        fill_defaults = FillDefault(self._fx, self.configs)
        skew_handle = SkewHandle(self._fx, self.configs, X)
        all_encoding = Encoding(self._fx, self.configs)
        outlier_handle = OutlierHandle(self._fx, self.configs, X, y)

        preproc_steps_columns = {
            "numfilldefaults": all_numeric_columns, 
            "numencoding": [col for col in all_numeric_columns if col not in self.configs.get('encode', {}).get('numerical_encoding_ignore_columns', [])],
            "catfilldefaults": all_categorical_columns, 
            "catencoding": all_categorical_columns, 
        }

        preproc_tuple_dict = {
            "numfilldefaults": fill_defaults.numeric_set, 
            "numencoding": all_encoding.numeric_set, 
            "catfilldefaults": fill_defaults.categorical_set,
            "catencoding": all_encoding.categorical_set,
        }

        # check if the null values needs to be dropped - for training nulls will be dropped and for inference nulls will be handled by some default method
        if self.configs.get('null_value_imputation', {}).get('drop_null', False):
            X = X.dropna()

        # check whether the skew needs to be handled
        if self.configs.get("skew", {}).get("skew_function", None): 
            preproc_steps_columns['skew'] = skew_handle.transform_cols
            preproc_tuple_dict['skew'] = skew_handle.numeric_set

        # check whether the outlier needs to be handled
        if self.configs.get("outlier", {}).get("handling_method", None): 
            if self.configs["outlier"]["handling_method"] == "handle":
                preproc_steps_columns['outlier'] = outlier_handle.transform_cols
                preproc_tuple_dict['outlier'] = outlier_handle.numeric_set

            else: # drop the outliers
                X, y = outlier_handle.drop_outliers()

        column_wise_preproc_group, function_steps = automl_utils.identify_nonoverlapping_features(preproc_steps_columns, preproc_tuple_dict)
        
        pipeline_mapping = {}
        for pipeline_name, pipeline_steps in function_steps.items(): 
            pipeline_mapping[pipeline_name] = Pipeline(steps=pipeline_steps)
        
        all_transformers = []
        for step_name, pipeline_obj in pipeline_mapping.items():
            transformer_step = (step_name, pipeline_obj, column_wise_preproc_group.get(step_name))
            all_transformers.append(transformer_step)

        # now construct the column transformer 
        preprocessor = ColumnTransformer(
            transformers=all_transformers)
        
        preprocessor_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

        print('preprocessor pipeline constructed\n')
        
        return preprocessor_pipeline, X, y
