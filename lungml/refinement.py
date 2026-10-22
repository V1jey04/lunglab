"""Prespecified accuracy refinement: calibration, bagging and stacking.

The existing 88.6% model remains unchanged. All base-model preprocessing is
inside the estimator evaluated by cross-validation; calibration and stacking
use another internal split, rather than fitting on outer validation cases.
"""
from pathlib import Path
import hashlib
import json
import warnings

import joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC

from .accuracy_experiment import candidate_specs, prepare_clinical, run_accuracy
from .improve import CorrelationPruner


def original_columns(X):
    return [column for column in X.columns if column.startswith("original_")]


def refinement_specs(output, seed=42):
    memory = joblib.Memory(Path(output) / "training_cache", verbose=0)
    baseline, baseline_grid, _, _ = candidate_specs(seed, memory)["rich_accuracy_logistic"]

    def pipeline(model, select=100, pca=None):
        steps = [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("prune", CorrelationPruner()),
            ("scale", QuantileTransformer(n_quantiles=30, output_distribution="normal", random_state=seed)),
        ]
        steps.append(("pca", PCA(n_components=pca, whiten=True, svd_solver="full")) if pca is not None
                     else ("select", SelectKBest(f_classif, k=select)))
        steps.append(("model", model))
        return Pipeline(steps, memory=memory)

    sigmoid = CalibratedClassifierCV(
        estimator=clone(baseline), method="sigmoid",
        cv=StratifiedKFold(3, shuffle=True, random_state=seed), ensemble=True, n_jobs=1)
    bagging = BaggingClassifier(
        estimator=clone(baseline), n_estimators=12, max_samples=0.8, bootstrap=True,
        random_state=seed, n_jobs=1)
    elastic = pipeline(LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.1,
        max_iter=15000, tol=0.001, random_state=seed), select="all")
    pca_logistic = pipeline(LogisticRegression(max_iter=4000, random_state=seed), pca=10)
    pca_svm = pipeline(SVC(probability=True, random_state=seed), pca=10)

    # Different feature views and model families provide candidate diversity.
    logistic_base = clone(baseline).set_params(select__k=100, model__C=0.1)
    svm_base = pipeline(SVC(C=3, gamma=0.01, probability=True, random_state=seed), select=40)
    tree_base = Pipeline([
        ("view", ColumnTransformer([("original", "passthrough", original_columns)], remainder="drop")),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5,
                                       max_features=0.7, random_state=seed, n_jobs=2)),
    ])
    stacked = StackingClassifier(
        estimators=[("logistic", logistic_base), ("svm", svm_base), ("trees", tree_base)],
        final_estimator=Pipeline([("scale", StandardScaler()),
                                  ("model", LogisticRegression(max_iter=4000, random_state=seed))]),
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        stack_method="predict_proba", passthrough=False, n_jobs=1)

    return {
        "baseline_rich_logistic": (baseline, baseline_grid, "rich", True),
        "sigmoid_rich_logistic": (sigmoid, {
            "estimator__select__k": [40, 100], "estimator__model__C": [0.03, 0.1, 0.3]}, "rich", True),
        "bagged_rich_logistic": (bagging, {
            "estimator__select__k": [40, 100], "estimator__model__C": [0.03, 0.1, 0.3]}, "rich", True),
        "elastic_net_radiomics": (elastic, {
            "model__C": [0.03, 0.1, 0.3], "model__l1_ratio": [0.1, 0.5, 0.9]}, "rich", True),
        "pca_radiomics_logistic": (pca_logistic, {
            "pca__n_components": [5, 15, 30], "model__C": [0.03, 0.1, 0.3, 1]}, "rich", True),
        "pca_radiomics_svm": (pca_svm, {
            "pca__n_components": [5, 15, 30], "model__C": [0.3, 3, 30],
            "model__gamma": ["scale", 0.03]}, "rich", True),
        "stacked_radiomics": (stacked, {
            "final_estimator__model__C": [0.03, 0.1, 0.3, 1]}, "rich", True),
    }


def run_refinement(output="artifacts/radiogenomics_refinement", evaluate=False):
    data = prepare_clinical()
    # Explicit penalty is retained for sklearn 1.5-1.7 compatibility. Only the
    # 1.8 API deprecation notice is hidden; convergence warnings remain visible.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'penalty' was deprecated", category=FutureWarning)
        result = run_accuracy(data, output, evaluate, specs=refinement_specs(output))
    if result.get("test_evaluated"):
        return summarize_refinement(output)
    return result


def summarize_refinement(output, previous="artifacts/radiogenomics_accuracy"):
    """Compare frozen development-selected models without any reselection."""
    output, previous = Path(output), Path(previous)
    report = json.loads((output / "external_report.json").read_text())
    baseline = json.loads((previous / "external_report.json").read_text())
    if (report["dataset_sha256"] != baseline["dataset_sha256"]
            or report["split_sha256"] != baseline["split_sha256"]):
        raise ValueError("Refinement and baseline must use identical data and patient splits.")
    candidate = report["selected_model_by_development"]
    reference = baseline["selected_model_by_development"]
    def rank(model):
        return (model["nested_cv_accuracy"], model["nested_cv_balanced_accuracy"], model["nested_cv_auc"])
    development_improved = rank(report["models"][candidate]) > rank(baseline["models"][reference])
    accuracy_improved = report["test_metrics"][candidate]["accuracy"] > baseline["test_metrics"][reference]["accuracy"]
    best_observed = max(report["test_metrics"], key=lambda name: report["test_metrics"][name]["accuracy"])
    report.update({
        "reference_run": str(previous), "reference_model": reference,
        "reference_model_sha256": hashlib.sha256((previous / f"{reference}.joblib").read_bytes()).hexdigest(),
        "reference_holdout_accuracy": baseline["test_metrics"][reference]["accuracy"],
        "development_selection_rank_improved": bool(development_improved),
        "observed_holdout_accuracy_improved": bool(accuracy_improved),
        "refinement_selected_as_default": bool(development_improved and accuracy_improved),
        "independently_confirmed_improvement": False,
        "best_observed_candidate": best_observed,
        "best_observed_holdout_accuracy": report["test_metrics"][best_observed]["accuracy"],
        "best_observed_candidate_is_development_selected": best_observed == candidate,
        "best_observed_candidate_interpretation": (
            "Highest observed test accuracy among these experiments; this descriptive comparison "
            "does not change the frozen development selection. The previously inspected holdout "
            "requires fresh independent patients to confirm any improvement."
        ),
    })
    (output / "external_report.json").write_text(json.dumps(report, indent=2))
    return report
