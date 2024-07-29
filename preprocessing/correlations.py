import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import pingouin as pg


### This class only has a fit functions, which computes the correlations of feats to target. 
### It does not yet remove columns based on the corr values. This has to be done in a `transform` function.

class Correlations(BaseEstimator, TransformerMixin):

    """Compute Correlation Values."""

    def __init__(
        self,
        _fx,
        cat_method = "annova",
        num_method = "pearson"
    ):
        
        self._corr = {}

        self._fx = _fx

        if cat_method in ["annova"]:
            self._cat = cat_method
        else:
            raise NameError(f"Specified scaling method {cat_method} is not valid")
        
        if num_method in ['pearson', 'spearman', 'kendall']:
            self._nume = num_method
        else:
            raise NameError(f"Specified scaling method {num_method} is not valid")
        
    def fit(self,X,y):

        focused_cols = [ele for lst in self._fx.values() for ele in lst]

        if any(X[col].isnull().any() for col in focused_cols):

            raise ValueError("DataFrame column(s) contain null value(s)")
        
        data = pd.concat([X, y], axis = 1)
        
        ### Categorical Correlations
        self._corr['Categorical'] = {
                                col_: pg.anova(data = data, dv = y.columns[0], between=col_).F.values[0]
                                for col_ in self._fx['Categorical']
                                }

        target = y[y.columns[0]]
        ### Numerical Correlations
        self._corr['Numeric'] = {
                                col_: X[col_].corr(target, method = self._nume)
                                for col_ in self._fx['Numeric']
                                }
        
        self.corr_values = pd.DataFrame(self._corr)

        return self
    
    
