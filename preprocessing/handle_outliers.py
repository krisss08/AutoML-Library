import numpy as np
from sklearn.preprocessing import RobustScaler

class OutlierHandle:
    def __init__(self, _fx, configs, X, y):
        self.verbose = configs.get('verbose')

        self.numerical_columns = _fx['Numeric']
        self.numeric_set = []
        self.X = X
        self.y = y
        self.transform_cols = []

        self.construct_pipeline_tuples()

    def find_outlier_cols(self):
        for col in self.numerical_columns:
            Q1 = self.X[col].quantile(0.25)
            Q3 = self.X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if self.X[(self.X[col] < lower_bound) | (self.X[col] > upper_bound)].any(axis=None):
                self.transform_cols.append(col)

    def drop_outliers(self):
        for col in self.numerical_columns:
            Q1 = self.X[col].quantile(0.25)
            Q3 = self.X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            self.X[col] = np.where((self.X[col] < lower_bound) | (self.X[col] > upper_bound), np.nan, self.X[col])

        outlier_rows = self.X.isna().any(axis=1)
        self.X = self.X[~outlier_rows]
        self.y = self.y[~outlier_rows]
        return self.X, self.y

    def construct_pipeline_tuples(self):

        self.find_outlier_cols()

        transformer = RobustScaler()
        outlier_handle = ('outlier handle', transformer)
        self.numeric_set.append(outlier_handle)
