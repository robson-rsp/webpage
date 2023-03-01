from sklearn.base import BaseEstimator, TransformerMixin

class OutliersZScoreReplacer(BaseEstimator, TransformerMixin):
    """
    Substitui os outliers encontrados pelas medianas de cada atributo.
    """
    def fit(self, X, y=None):
        self.mean_std_median = list()
        for name in X.columns:
            mean   = X[name].mean()
            std    = X[name].std()
            median = X[name].median()
            self.mean_std_median.append((mean, std, median))
        return self
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    def transform(self, X, y=None):
        std_unit = 3
        for index, name in enumerate(X.columns):
            mean    = self.mean_std_median[index][0]
            std     = self.mean_std_median[index][1]
            median  = self.mean_std_median[index][2]
            scores  = ((X[name] - mean) / std)
            filter_mask = ((scores < -std_unit) | (scores > std_unit))
            X.loc[filter_mask, name] = -900_000_000
        return X
    def get_feature_names_out(self):
        pass