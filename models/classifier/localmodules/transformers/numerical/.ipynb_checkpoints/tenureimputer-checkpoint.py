from sklearn.base import BaseEstimator, TransformerMixin

class TenureImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    def transform(self, X, y=None):
        boolean_mask = X['tenure'].isna()
        X.loc[boolean_mask, 'tenure'] = round(X['totalcharges'] / X['monthlycharges'])
        return X
    def get_feature_names_out(self):
        pass