from sklearn.base import BaseEstimator, TransformerMixin

class DependentsImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    def transform(self, X, y=None):
        # se for idoso ou jovem solteiro
        mask_imputer    = (X['seniorcitizen'] == 'Yes') | ((X['seniorcitizen'] == 'No') & (X['partner'] == 'No'))
        mask_dependents = (X['dependents'].isna())
        mask_no_dependents  = (mask_imputer & mask_dependents)
        mask_yes_dependents = (~mask_imputer & mask_dependents)
        X.loc[mask_no_dependents, 'dependents']  = 'No'
        X.loc[mask_yes_dependents, 'dependents'] = 'Yes'
        return X
    def get_feature_names_out(self):
        pass