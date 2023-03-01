from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ToNumber(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    def transform(self, X, y=None):    
        return X.astype(np.float64)
    def get_feature_names_out(self):
        pass