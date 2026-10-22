"""Development-only model selection for patient-level lymph-node involvement."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .improve import CorrelationPruner
from .multiview import CappedUnivariateSelector
from .nodal_data import DEFAULT_AUDIT, DEFAULT_COHORT, LABELS, load_nodal_cohort
from .recurrence_data import _sha256

CONFIG_PATH = Path(__file__).with_name("configs") / "nodal.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
FAMILIES = ("prevalence_reference", "volume_logistic", "radiomics_logistic")


def _folds(y, n_splits, seed):
    if np.bincount(np.asarray(y, dtype=int), minlength=2).min() < n_splits:
        raise ValueError(f"Nodal cross-validation requires at least {n_splits} patients of each class.")
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y))


def _inputs(rows, names, family):
    selected = ["original_shape_MeshVolume"] if family == "volume_logistic" else names
    return rows.loc[:, selected]


def _model(family, parameters):
    if family == "prevalence_reference":
        return DummyClassifier(strategy="prior")
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if family == "radiomics_logistic":
        steps += [("prune", CorrelationPruner(0.95)),
                  ("select", CappedUnivariateSelector(parameters["k"]))]
    steps += [("scale", StandardScaler()),
              ("model", LogisticRegression(C=parameters["C"], max_iter=5000, random_state=42))]
    return Pipeline(steps)


def _parameters(family):
    if family == "prevalence_reference":
        return [{}]
    if family == "volume_logistic":
        return [{"C": float(c)} for c in CONFIG["C_grid"]]
    return [{"C": float(c), "k": int(k)}
            for k in CONFIG["feature_count_grid"] for c in CONFIG["C_grid"]]


def _threshold(y, p):
    # Select the operating point on inner OOF predictions, never outer/test labels.
    candidates = np.linspace(0.05, 0.95, 19)
    return float(min(candidates, key=lambda t: (
        -balanced_accuracy_score(y, p >= t), abs(t - 0.5), -t)))


def _select(rows, names, family, seed):
    y = rows.nodal_target.to_numpy(dtype=int)
    X = _inputs(rows, names, family)
    folds = _folds(y, int(CONFIG["inner_folds"]), seed)
    results = []
    for parameters in _parameters(family):
        p = np.full(len(rows), np.nan)
        for train, valid in folds:
            estimator = _model(family, parameters).fit(X.iloc[train], y[train])
            p[valid] = estimator.predict_proba(X.iloc[valid])[:, 1]
        results.append({"parameters": parameters, "inner_brier_score": float(brier_score_loss(y, p)),
                        "threshold": _threshold(y, p)})
    best = min(results, key=lambda r: (r["inner_brier_score"], r["parameters"].get("k", 0),
                                      r["parameters"].get("C", 0)))
    return best, results


def _wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p, z2 = successes / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    half = z * np.sqrt(p * (1 - p) / total + z2 / (4 * total**2)) / (1 + z2 / total)
    return [float(center - half), float(center + half)]


def metrics(y, p, predictions):
    y, p, predictions = np.asarray(y, int), np.asarray(p, float), np.asarray(predictions, int)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    both = np.unique(y).size == 2
    return {
        "patients": len(y), "node_positive_patients": tp + fn, "node_negative_patients": tn + fp,
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "brier_score": float(brier_score_loss(y, p)),
        "node_positive_recall": tp / (tp + fn) if tp + fn else None,
        "node_negative_recall": tn / (tn + fp) if tn + fp else None,
        "node_positive_recall_ci95": _wilson(tp, tp + fn),
        "node_negative_recall_ci95": _wilson(tn, tn + fp),
        "confusion_matrix": matrix.tolist(),
    }


def verify_frozen_models(output, selection):
    output = Path(output)
    for family in FAMILIES:
        path = output / f"{family}.joblib"
        expected = selection.get("frozen_models", {}).get(family, {}).get("model_sha256")
        if not expected or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Frozen nodal model changed or is missing: {family}.")


def run_nodal(cohort_path=DEFAULT_COHORT, output="artifacts/nodal", *,
              audit_path=DEFAULT_AUDIT / "audit.json", evaluate=False):
    cohort_path, output, audit_path = Path(cohort_path), Path(output), Path(audit_path)
    if not audit_path.is_file():
        raise ValueError("A nodal audit is required before fitting.")
    audit = json.loads(audit_path.read_text())
    if not audit.get("training_allowed"):
        raise ValueError("Nodal training is blocked: verify the radiomic CT match and preoperative timing.")
    if (Path(audit.get("outputs", {}).get("cohort", "")).resolve() != cohort_path.resolve()
            or audit.get("cohort_sha256") != _sha256(cohort_path)):
        raise ValueError("Nodal audit does not match this cohort.")
    rows, provenance = load_nodal_cohort(cohort_path)
    development = rows.loc[rows.source_split.eq("development")].reset_index(drop=True)
    test_ids = rows.loc[rows.source_split.eq("test"), "patient_id"].tolist()
    y = development.nodal_target.to_numpy(dtype=int)
    if np.bincount(y, minlength=2).min() < int(CONFIG["minimum_development_patients_per_class"]):
        raise ValueError("Nodal development cohort has too few patients of one class for nested selection.")
    names = provenance["feature_names"]
    if "original_shape_MeshVolume" not in names:
        raise ValueError("Nodal volume benchmark requires original_shape_MeshVolume.")
    if (output / "selection.json").exists() or any(output.glob("*.joblib")):
        raise ValueError("Nodal experiment is already frozen; use a new output directory.")
    if cohort_path.resolve().is_relative_to(output.resolve()):
        raise ValueError("Keep the nodal source cohort outside the experiment output directory.")
    seed = int(CONFIG["seed"])
    outer = _folds(y, int(CONFIG["outer_folds"]), seed)
    comparisons, oof = {}, pd.DataFrame({"patient_id": development.patient_id, "nodal_target": y})
    for family in FAMILIES:
        p, decisions = np.full(len(development), np.nan), np.zeros(len(development), dtype=int)
        fold_details = []
        for number, (train, valid) in enumerate(outer, start=1):
            training, validation = development.iloc[train], development.iloc[valid]
            chosen, _ = _select(training, names, family, seed + number)
            estimator = _model(family, chosen["parameters"]).fit(
                _inputs(training, names, family), y[train])
            p[valid] = estimator.predict_proba(_inputs(validation, names, family))[:, 1]
            decisions[valid] = p[valid] >= chosen["threshold"]
            fold_details.append({"fold": number, **chosen,
                                 "metrics": metrics(y[valid], p[valid], decisions[valid])})
        if not np.isfinite(p).all():
            raise RuntimeError("Nested nodal CV did not predict every development patient.")
        comparisons[family] = {"nested_metrics": metrics(y, p, decisions), "outer_folds": fold_details}
        oof[f"{family}_p_node_positive"] = p
        oof[f"{family}_prediction"] = decisions
    selected = min(FAMILIES, key=lambda f: (
        comparisons[f]["nested_metrics"]["brier_score"], FAMILIES.index(f)))
    selection = {
        "schema_version": 1, "task": "lymph_node_involvement", "labels": list(LABELS),
        "dataset_path": str(cohort_path), "cohort_sha256": _sha256(cohort_path),
        "split_sha256": _sha256(cohort_path.with_suffix(".splits.csv")),
        "provenance_sha256": _sha256(cohort_path.with_suffix(".provenance.json")),
        "protocol": CONFIG, "feature_names": names,
        "development_patients": len(development), "test_patients": len(test_ids),
        "selected_model_by_development": selected, "development_metrics": comparisons,
        "test_patient_ids_excluded_from_selection": test_ids,
        "radiomics_incremental_value_in_development": (
            comparisons["radiomics_logistic"]["nested_metrics"]["brier_score"] < min(
                comparisons[f]["nested_metrics"]["brier_score"] for f in FAMILIES[:2])),
        "test_evaluated": False,
        "interpretation": "Exploratory retrospective nodal research; probabilities are not confirmed staging, node localization or treatment recommendations.",
        "evaluation": "Preserved patient split; test patients were previously inspected for subtype experiments. Independent confirmation is required.",
        "frozen_models": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        best, search = _select(development, names, family, seed)
        X = _inputs(development, names, family)
        model = _model(family, best["parameters"]).fit(X, y)
        bundle = {
            "schema_version": 1, "task": "lymph_node_involvement", "model_family": family,
            "model": model, "feature_names": X.columns.tolist(), "labels": list(LABELS),
            "threshold": best["threshold"], "training_patient_ids": development.patient_id.tolist(),
            "scan_timing_semantics_verified": True, "data_provenance": provenance,
            "prediction_context": selection["interpretation"],
        }
        path = output / f"{family}.joblib"
        joblib.dump(bundle, path)
        selection["frozen_models"][family] = {
            **best, "inner_search": search, "model_sha256": _sha256(path)}
    oof.to_csv(output / "development_oof_predictions.csv", index=False)
    rows[["patient_id", "source_split"]].rename(columns={"source_split": "split"}).to_csv(
        output / "splits.csv", index=False)
    (output / "selection.json").write_text(json.dumps(selection, indent=2))
    if evaluate:
        return evaluate_nodal(cohort_path, output)
    return selection


def evaluate_nodal(cohort_path=DEFAULT_COHORT, output="artifacts/nodal"):
    cohort_path, output = Path(cohort_path), Path(output)
    if (output / "external_report.json").exists() or (output / "test_predictions.csv").exists():
        raise ValueError("The nodal test cohort was already scored; retain the original report.")
    selection = json.loads((output / "selection.json").read_text())
    for path, key in [(cohort_path, "cohort_sha256"),
                      (cohort_path.with_suffix(".splits.csv"), "split_sha256"),
                      (cohort_path.with_suffix(".provenance.json"), "provenance_sha256")]:
        if _sha256(path) != selection[key]:
            raise ValueError("Nodal cohort, split or provenance changed after freezing.")
    verify_frozen_models(output, selection)
    rows, _ = load_nodal_cohort(cohort_path)
    test = rows.loc[rows.source_split.eq("test")].reset_index(drop=True)
    development_ids = set(rows.loc[rows.source_split.eq("development"), "patient_id"])
    if set(test.patient_id) != set(selection["test_patient_ids_excluded_from_selection"]):
        raise ValueError("Nodal test patients changed after freezing.")
    report, predictions = {**selection, "test_evaluated": True, "test_metrics": {}}, []
    from .nodal_predict import predict_frame
    for family in FAMILIES:
        bundle = joblib.load(output / f"{family}.joblib")
        if set(bundle["training_patient_ids"]) != development_ids or development_ids & set(test.patient_id):
            raise ValueError(f"Nodal training/test patient leakage found in {family}.")
        prediction = predict_frame(bundle, test)
        prediction["observed_nodal_target"] = test.nodal_target.to_numpy()
        report["test_metrics"][family] = metrics(
            test.nodal_target, prediction.p_node_positive, prediction.prediction.eq("node_positive"))
        predictions.append(prediction)
    pd.concat(predictions, ignore_index=True).to_csv(output / "test_predictions.csv", index=False)
    (output / "external_report.json").write_text(json.dumps(report, indent=2))
    return report
