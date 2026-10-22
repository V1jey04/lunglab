"""A compact ensemble of original, wavelet and nonlinear CT radiomic views."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from .improve import CorrelationPruner


class RadiomicsViewEnsemble(ClassifierMixin, BaseEstimator):
    """Each view has fold-fitted redundancy pruning and supervised selection."""

    def __init__(self, C=0.1, k=20, original_weight=1.0, random_state=42):
        self.C = C
        self.k = k
        self.original_weight = original_weight
        self.random_state = random_state

    def fit(self, X, y):
        if not hasattr(X, "columns"):
            raise ValueError("Named radiomic feature columns are required.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.classes_ = np.unique(y)
        self.views_ = {}
        selectors = {
            "original": lambda n: n.startswith("original_"),
            "wavelet": lambda n: n.startswith("wavelet-"),
            "filtered": lambda n: not n.startswith(("original_", "wavelet-")),
        }
        for name, select in selectors.items():
            columns = [c for c in X if select(c)]
            if not columns:
                continue
            # The selector uses min(k, remaining columns), which is necessary
            # when correlated features remove most of a small original view.
            model = Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("prune", CorrelationPruner(0.95)),
                ("scale", QuantileTransformer(n_quantiles=40, output_distribution="normal",
                                               random_state=self.random_state)),
                ("select", CappedUnivariateSelector(self.k)),
                ("model", LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000)),
            ])
            model.fit(X[columns], y)
            self.views_[name] = (columns, model)
        if not self.views_:
            raise ValueError("No supported radiomic feature views.")
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "views_")
        probabilities, weights = [], []
        for name, (columns, model) in self.views_.items():
            probabilities.append(model.predict_proba(X[columns]))
            weights.append(self.original_weight if name == "original" else 1.0)
        return np.average(probabilities, axis=0, weights=weights)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class CappedUnivariateSelector(BaseEstimator):
    def __init__(self, k=20):
        self.k = k

    def fit(self, X, y):
        self.selector_ = SelectKBest(f_classif, k=min(self.k, np.asarray(X).shape[1])).fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)
