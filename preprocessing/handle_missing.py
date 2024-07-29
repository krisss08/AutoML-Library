from sklearn.impute import SimpleImputer

class FillDefault:

    '''Fill with Standard/Custom Values
    Default Values :
        * numerical_defaults = 0
        * categorical_defaults = "Unknown"

    User defined: 
        * numerical_defaults : 
            1. Can be type of filling i.e 'mean','median','mode' 
            2. Value to be filled i.e any IntegerType

        * categorical_defaults:
            1. Can be any string user-defined

    Note: `mode` for numerical_defaults not implemented
    '''
    
    def __init__(self, _fx, configs):

        self._fx = _fx
        self.numerical_defaults = configs.get('null_value_imputation', {}).get('numerical_defaults', 'mean')
        self.categorical_defaults = configs.get('null_value_imputation', {}).get('categorical_defaults', 'Unknown')
        self.verbose = configs.get('verbose')

        self.numeric_set = []
        self.categorical_set = []
        
        self.construct_pipeline_tuples()

    def construct_pipeline_tuples(self): 

        # numercial defaults
        
        if isinstance(self.numerical_defaults, str):
            if self.numerical_defaults == "mean": 
                input_strategy = SimpleImputer(strategy='mean')
            
            elif self.numerical_defaults == 'median':
                input_strategy = SimpleImputer(strategy='median')

            elif self.numerical_defaults == 'mode': 
                input_strategy = SimpleImputer(strategy='mode')

        elif isinstance(self.numerical_defaults, (int, float)):
            input_strategy = SimpleImputer(strategy='constant', fill_value=self.numerical_defaults)
        
        else: 
            input_strategy = SimpleImputer(strategy='mean')

        numerical_imputation = ("fill_default_numerical", input_strategy)
        self.numeric_set.append(numerical_imputation)

        # categorical defaults

        if self.categorical_defaults == 'mode':
            input_strategy = SimpleImputer(strategy='most_frequent')

        elif isinstance(self.categorical_defaults, str): 
            input_strategy = SimpleImputer(strategy='constant', fill_value=self.categorical_defaults)
            
        else: 
            input_strategy = SimpleImputer(strategy='constant', fill_value='unknown')

        categorical_imputation = ("fill_default_categorical", input_strategy)
        self.categorical_set.append(categorical_imputation)
