"""Development-only model search. Previously inspected test patients are excluded."""
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from .data import load_dataset
from .evaluation import metrics, patient_predictions
from .models import catboost_model, paper_mlp
from .train import patient_folds


class CorrelationPruner(TransformerMixin, BaseEstimator):
    """Discard redundant measurements using training-fold statistics only."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        active = np.flatnonzero(np.nanstd(values, axis=0) > 1e-12)
        if not len(active):
            raise ValueError("All radiomic features are constant in this training fold.")
        correlation = np.abs(np.nan_to_num(np.corrcoef(values[:, active], rowvar=False)))
        retained = []
        for i in range(len(active)):
            if not retained or np.all(correlation[i, retained] < self.threshold):
                retained.append(i)
        self.indices_ = active[retained]
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        if np.asarray(X).shape[1] != self.n_features_in_:
            raise ValueError("Feature count changed.")
        return np.asarray(X)[:, self.indices_]


def search_candidates(seed=42):
    def pipeline(estimator, quantile=False):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("prune", CorrelationPruner()),
            ("scale", QuantileTransformer(n_quantiles=50, output_distribution="normal", random_state=seed)
             if quantile else RobustScaler()),
            ("select", SelectKBest(f_classif, k="all")),
            ("model", estimator),
        ])

    return {
        "paper_mlp": (paper_mlp(seed), {"select__alpha": [0.001, 0.01]}),
        "catboost_original": (catboost_model(seed), {"model__depth": [3, 5], "model__l2_leaf_reg": [3, 10]}),
        "robust_logistic": (pipeline(LogisticRegression(class_weight="balanced", max_iter=5000)), {
            "model__C": [0.01, 0.1, 1, 10], "select__k": [10, "all"]}),
        "quantile_svm": (pipeline(SVC(kernel="rbf", class_weight="balanced", probability=False), True), {
            "model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.1], "select__k": [10, "all"]}),
        "shrinkage_lda": (pipeline(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]), True), {
            "select__k": [10, "all"]}),
        "extra_trees": (pipeline(ExtraTreesClassifier(n_estimators=400, class_weight="balanced", n_jobs=2,
                                                       random_state=seed)), {
            "model__min_samples_leaf": [2, 5, 10], "model__max_features": [0.3, 0.7], "select__k": [10, "all"]}),
        "hist_gradient_boosting": (pipeline(HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=7,
                                                                            class_weight="balanced", random_state=seed)), {
            "model__learning_rate": [0.03, 0.1], "model__l2_regularization": [1, 10], "select__k": [10, "all"]}),
    }


def development_search(data="data/nsclc_subtype.csv", original_run="artifacts/nsclc_lung1_converged",
                       output="artifacts/improvement_development", seed=42, outer_cv=5):
    dataset = load_dataset(data)
    splits = pd.read_csv(Path(original_run) / "splits.csv", dtype={"patient_id": str})
    development_ids = set(splits.loc[splits.split == "development", "patient_id"])
    forbidden_ids = set(splits.loc[splits.split == "test", "patient_id"])
    if development_ids & forbidden_ids:
        raise ValueError("Invalid existing patient split.")
    use = np.isin(dataset.groups, list(development_ids))
    X = dataset.X.loc[use].reset_index(drop=True)
    y, groups = dataset.y[use], dataset.groups[use]
    if set(groups) != development_ids or set(groups) & forbidden_ids:
        raise ValueError("Dataset does not match the preserved development split.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "search_report.json").exists():
        raise ValueError("A completed search exists; choose a new output directory.")
    report = {
        "evaluation": "nested stratified patient CV on the original development patients only",
        "target_paper_accuracy": 0.83, "development_patients": len(set(groups)),
        "excluded_previously_inspected_test_ids": sorted(forbidden_ids),
        "dataset_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        "seed": seed, "outer_folds": outer_cv, "models": {},
        "selection_metric": "balanced_accuracy", "target_met": False,
    }
    folds = patient_folds(y, groups, outer_cv, seed)
    scorers = {"balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy", "roc_auc": "roc_auc",
               "macro_f1": "f1_macro"}
    warnings.filterwarnings("ignore", message="n_quantiles .* is greater")
    for name, (estimator, grid) in search_candidates(seed).items():
        print(f"Nested development search: {name}", flush=True)
        oof_predictions = np.empty(len(y), dtype=int)
        oof_scores = np.empty(len(y), dtype=float)
        parameters, fold_rows = [], []
        for fold_number, (tr, va) in enumerate(folds, 1):
            inner = patient_folds(y[tr], groups[tr], n_splits=3, seed=seed)
            search = GridSearchCV(clone(estimator), grid, scoring=scorers, refit="balanced_accuracy",
                                  cv=inner, n_jobs=1, error_score="raise")
            search.fit(X.iloc[tr], y[tr])
            fitted = search.best_estimator_
            oof_predictions[va] = fitted.predict(X.iloc[va]).ravel().astype(int)
            if hasattr(fitted, "decision_function"):
                oof_scores[va] = fitted.decision_function(X.iloc[va])
            else:
                oof_scores[va] = fitted.predict_proba(X.iloc[va])[:, 1]
            parameters.append(search.best_params_)
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
            fold_rows.append({"fold": fold_number,
                              "accuracy": float(accuracy_score(y[va], oof_predictions[va])),
                              "balanced_accuracy": float(balanced_accuracy_score(y[va], oof_predictions[va])),
                              "roc_auc": float(roc_auc_score(y[va], oof_scores[va]))})
            print(f"  fold {fold_number}/{outer_cv}: balanced accuracy {fold_rows[-1]['balanced_accuracy']:.3f}", flush=True)
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     f1_score, roc_auc_score)
        tn, fp, fn, tp = confusion_matrix(y, oof_predictions, labels=[0, 1]).ravel()
        result = {
            "nested_cv_accuracy": float(accuracy_score(y, oof_predictions)),
            "nested_cv_balanced_accuracy": float(balanced_accuracy_score(y, oof_predictions)),
            "nested_cv_roc_auc": float(roc_auc_score(y, oof_scores)),
            "nested_cv_macro_f1": float(f1_score(y, oof_predictions, average="macro")),
            "adenocarcinoma_recall": float(tn / (tn + fp)),
            "squamous_recall": float(tp / (tp + fn)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "folds": fold_rows, "parameters_by_fold": parameters,
        }
        report["models"][name] = result
        pd.DataFrame({"patient_id": groups, "y_true": y, "predicted": oof_predictions,
                      "score": oof_scores}).to_csv(output / f"{name}_development_oof.csv", index=False)
        (output / "search_report.json").write_text(json.dumps(report, indent=2))
    report["selected_model_by_nested_cv"] = max(report["models"], key=lambda n: (
        report["models"][n]["nested_cv_balanced_accuracy"], report["models"][n]["nested_cv_roc_auc"]))
    best = report["models"][report["selected_model_by_nested_cv"]]
    report["target_met"] = best["nested_cv_accuracy"] > 0.83
    report["interpretation"] = (
        "These are development results. Selecting a winner across model families still requires "
        "a new external holdout before claiming superiority. The original test cases were excluded.")
    (output / "search_report.json").write_text(json.dumps(report, indent=2))
    table = pd.DataFrame({n: {k: v for k, v in m.items() if k not in {"folds", "parameters_by_fold", "confusion_matrix"}}
                          for n, m in report["models"].items()}).T
    table.to_csv(output / "development_metrics.csv", index_label="model")
    print(table.to_string(), flush=True)
    return report


if __name__ == "__main__":
    development_search()
