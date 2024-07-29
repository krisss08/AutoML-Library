import numpy as np
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

# Define named functions
def reciprocal(x):
    return 1 / x

def inverse(x):
    return -x

def cube(x):
    return np.power(x, 3)

# Dictionary of transformations
transformations = {
    'log': np.log,
    'cube_root': np.cbrt,
    'square_root': np.sqrt,
    'reciprocal': reciprocal,
    'exponential': np.exp,
    'inverse': inverse,
    'absolute': np.abs,
    'square': np.square,
    'cube': cube,
}

class SkewHandle:

    '''
    Checks for a possible skew & applies log transformations. 

    The type of transformations can be given in the `function` parameter. 
    '''

    def __init__(self, _fx, configs, X):
        
        self.function = configs.get('skew', {}).get('skew_function', 'yeo-johnson')
        self.threshold = configs.get('skew', {}).get('skew_threshold', 0.5)
        self.verbose = configs.get('verbose')
        self.numerical_columns = _fx['Numeric']
        self.numeric_set = []
        self.X = X
        self.transform_cols = []

        self.construct_pipeline_tuples()

    def compute_skew(self): 

        for col in self.numerical_columns:
            skewness = self.X[col].skew()
            if (skewness < (-1 * self.threshold)) or (skewness > self.threshold):
                self.transform_cols.append(col)

    def construct_pipeline_tuples(self):

        self.compute_skew()

        if self.function in ['yeo-johnson', 'box-cox']:
            transformer = PowerTransformer(method=self.function)
        else:
            transformer = FunctionTransformer(func=transformations[self.function], validate=True)
        skew_handle = ('numeric_skew_handle', transformer)

        self.numeric_set.append(skew_handle)
