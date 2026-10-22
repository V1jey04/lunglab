"""The paper's best architecture and a regularized boosting alternative."""
import numpy as np
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


class LassoTopK(TransformerMixin, BaseEstimator):
    """Fit only within the training fold; break ties by original column order."""

    def __init__(self, k=10, alpha=0.001):
        self.k = k
        self.alpha = alpha

    def fit(self, X, y):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.lasso_ = Lasso(alpha=self.alpha, max_iter=100000).fit(X, y)
        self.scores_ = np.abs(self.lasso_.coef_)
        if not np.any(self.scores_):
            raise ValueError("LASSO discarded every feature; lower its alpha or inspect the data.")
        self.indices_ = np.sort(np.argsort(-self.scores_, kind="stable")[:min(self.k, X.shape[1])])
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count changed.")
        return X[:, self.indices_]


class AdaptiveSMOTE(BaseEstimator):
    """Reduce k on small cohorts; never resample validation or test patients."""

    def __init__(self, random_state=42, k_neighbors=5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        counts = np.bincount(y)
        if len(counts) != 2 or min(counts) < 2:
            raise ValueError("SMOTE needs at least two training examples in each class.")
        if counts[0] == counts[1]:
            return X, y
        return SMOTE(random_state=self.random_state,
                     k_neighbors=min(self.k_neighbors, int(min(counts)) - 1)).fit_resample(X, y)


def paper_mlp(seed=42):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("select", LassoTopK(k=10)),
        ("smote", AdaptiveSMOTE(random_state=seed)),
        ("model", MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                                learning_rate_init=0.001, solver="adam", max_iter=5000,
                                early_stopping=False, random_state=seed)),
    ])


def catboost_model(seed=42):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", CatBoostClassifier(iterations=300, depth=3, learning_rate=0.03,
                                     l2_leaf_reg=10, auto_class_weights="Balanced",
                                     loss_function="Logloss", random_seed=seed,
                                     thread_count=2, verbose=False, allow_writing_files=False)),
    ])


def candidates(seed=42, quick=False):
    return {
        "paper_mlp": (paper_mlp(seed), {"select__alpha": [0.001] if quick else [0.001, 0.01]}),
        "catboost": (catboost_model(seed), {
            "model__depth": [3] if quick else [3, 5],
            "model__l2_leaf_reg": [10] if quick else [3, 10],
        }),
    }
