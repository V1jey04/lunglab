"""Freeze a new cohort holdout, compare richer models using development only."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

from .data import TASK_LABELS, load_dataset
from .evaluation import bootstrap_intervals, metrics, patient_predictions, save_figures
from .improve import search_candidates
from .multiview import RadiomicsViewEnsemble
from .train import patient_folds


def candidates(seed=42):
    base = search_candidates(seed)
    # Prespecified comparison across model families; original baseline remains
    # restricted to original_* features, while candidates can use richer views.
    result = {k: base[k] for k in ["paper_mlp", "quantile_svm", "shrinkage_lda", "robust_logistic"]}
    result["multiview_ensemble"] = (RadiomicsViewEnsemble(random_state=seed), {
        "C": [0.01, 0.1, 1], "k": [10, 20], "original_weight": [1.0, 2.0]})
    return result


def run(data="data/radiogenomics_subtype.csv", output="artifacts/radiogenomics_rich", evaluate=False,
        feature_mode="all", candidate_specs=None, model_columns=None,
        evaluation_description="25% held-out patients within NSCLC Radiogenomics; not an external-hospital validation"):
    dataset = load_dataset(data, feature_mode=feature_mode)
    if len(np.unique(dataset.groups)) != len(dataset.groups):
        raise ValueError("This experiment requires exactly one primary tumour per patient.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selection.json"
    if selection_path.exists():
        if evaluate:
            return evaluate_frozen(data, output)
        raise ValueError("Model selection already exists; do not repeat selection using test outcomes.")
    splits_path = Path(data).with_suffix(".splits.csv")
    if not splits_path.exists():
        # Reserve a new cohort once, before any preprocessing/model selection.
        _, test_ids = train_test_split(dataset.frame.patient_id, test_size=0.25,
                                        stratify=dataset.frame.label, random_state=42)
        manifest = dataset.frame[["patient_id", "sample_id", "label"]].copy()
        manifest["split"] = np.where(manifest.patient_id.isin(test_ids), "test", "development")
        manifest.to_csv(splits_path, index=False)
    splits = pd.read_csv(splits_path, dtype={"patient_id": str})
    if (splits.patient_id.duplicated().any() or set(splits.patient_id) != set(dataset.groups)
            or set(splits.split) != {"development", "test"}):
        raise ValueError("Split manifest must assign every patient exactly once to development or test.")
    use = dataset.frame.patient_id.isin(splits.loc[splits.split == "development", "patient_id"]).to_numpy()
    X, y, groups = dataset.X.loc[use].reset_index(drop=True), dataset.y[use], dataset.groups[use]
    test_ids = set(splits.loc[splits.split == "test", "patient_id"])
    if test_ids & set(groups):
        raise ValueError("Patient leakage in new cohort.")
    selection = {"dataset_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(),
                 "split_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
                 "development_patients": len(groups), "test_patients": len(test_ids),
                 "target_accuracy": 0.83, "models": {}, "seed": 42,
                 "selection_metric": "nested CV balanced accuracy; AUC breaks ties",
                 "test_evaluated": False, "dataset_path": str(data),
                 "feature_mode": feature_mode, "evaluation": evaluation_description}
    folds = patient_folds(y, groups, 5, 42)
    source_provenance = json.loads(Path(data).with_suffix(".provenance.json").read_text())
    for name, (estimator, grid) in (candidates() if candidate_specs is None else candidate_specs).items():
        columns = [c for c in X if c.startswith("original_")] if name == "paper_mlp" else X.columns.tolist()
        if model_columns is not None and name in model_columns:
            columns = model_columns[name]
        view = X[columns]
        print(f"Richer cohort development: {name} ({len(columns)} features)", flush=True)
        predictions = np.empty(len(y), dtype=int)
        scores = np.empty(len(y))
        fold_parameters = []
        for number, (tr, va) in enumerate(folds, 1):
            inner = patient_folds(y[tr], groups[tr], 3, 42)
            search = GridSearchCV(clone(estimator), grid, cv=inner, n_jobs=1,
                                  scoring="balanced_accuracy", refit=True, error_score="raise")
            search.fit(view.iloc[tr], y[tr])
            fitted = search.best_estimator_
            predictions[va] = fitted.predict(view.iloc[va]).ravel().astype(int)
            if hasattr(fitted, "decision_function"):
                scores[va] = fitted.decision_function(view.iloc[va])
            else:
                scores[va] = fitted.predict_proba(view.iloc[va])[:, 1]
            fold_parameters.append(search.best_params_)
            print(f"  fold {number}/5: balanced accuracy {balanced_accuracy_score(y[va],predictions[va]):.3f}", flush=True)
        result = {"nested_cv_accuracy": float(accuracy_score(y, predictions)),
                  "input_features": len(columns), "prediction_rule": "native_predict",
                  "nested_cv_balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
                  "nested_cv_auc": float(roc_auc_score(y, scores)),
                  "confusion_matrix": confusion_matrix(y, predictions).tolist(),
                  "fold_parameters": fold_parameters}
        selection["models"][name] = result
        pd.DataFrame({"patient_id": groups, "y_true": y, "predicted": predictions, "score": scores}).to_csv(
            output / f"{name}_development_oof.csv", index=False)
        # Choose final parameters using development folds, then save the bundle
        # before any test predictions or labels are scored.
        search = GridSearchCV(clone(estimator), grid, cv=folds, n_jobs=1,
                              scoring="balanced_accuracy", refit=True, error_score="raise")
        search.fit(view, y)
        result["frozen_parameters"] = search.best_params_
        model = search.best_estimator_
        if name == "quantile_svm":
            # Native SVC probability calibration stays inside development data;
            # one row corresponds to one patient in this cohort.
            model.set_params(model__probability=True)
            model.fit(view, y)
        bundle = {"model": model, "feature_names": columns,
                  "labels": list(TASK_LABELS["subtype"]), "task": "subtype",
                  "threshold": 0.5, "data_provenance": "real", "name": name,
                  "training_patient_ids": groups.tolist(), "dataset_provenance": source_provenance,
                  "prediction_rule": "native_predict"}
        joblib.dump(bundle, output / f"{name}.joblib")
        (output / "selection_progress.json").write_text(json.dumps(selection, indent=2))
    selection["selected_model_by_development"] = max(selection["models"], key=lambda n: (
        selection["models"][n]["nested_cv_balanced_accuracy"], selection["models"][n]["nested_cv_auc"]))
    selection_path.write_text(json.dumps(selection, indent=2))
    splits.to_csv(output / "splits.csv", index=False)
    print("Froze all models and selected winner before evaluating the new holdout.", flush=True)
    if evaluate:
        return evaluate_frozen(data, output)
    return selection


def evaluate_frozen(data, output):
    output = Path(output)
    if (output / "external_report.json").exists():
        raise ValueError("The new holdout has already been evaluated. Retain its original result.")
    selection = json.loads((output / "selection.json").read_text())
    if hashlib.sha256(Path(data).read_bytes()).hexdigest() != selection["dataset_sha256"]:
        raise ValueError("Dataset changed after model selection.")
    split_path = output / "splits.csv"
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != selection["split_sha256"]:
        raise ValueError("Frozen split manifest changed after model selection.")
    dataset = load_dataset(data, feature_mode=selection.get("feature_mode", "all"))
    splits = pd.read_csv(output / "splits.csv")
    use = dataset.frame.patient_id.isin(splits.loc[splits.split == "test", "patient_id"]).to_numpy()
    y = dataset.y[use]
    if len(set(dataset.groups[use])) != selection["test_patients"]:
        raise ValueError("Held-out patient count changed.")
    majority = max(np.bincount(y)) / len(y)
    report = {**selection, "test_evaluated": True, "majority_baseline_accuracy": float(majority),
              "test_metrics": {}, "target_met": False,
              "evaluation": selection.get("evaluation", "25% held-out patients within NSCLC Radiogenomics; not an external-hospital validation"),
              "data_provenance": "real", "task": "subtype", "labels": list(TASK_LABELS["subtype"]),
              "positive_class": TASK_LABELS["subtype"][1], "feature_names": dataset.X.columns.tolist(),
              "test_intervals": {}}
    predictions = {}
    for name in selection["models"]:
        bundle = joblib.load(output / f"{name}.joblib")
        X = dataset.X.loc[use, bundle["feature_names"]]
        if set(dataset.groups[use]) & set(bundle["training_patient_ids"]):
            raise ValueError("Saved model contains test patient IDs.")
        p = bundle["model"].predict_proba(X)[:, 1]
        predicted = bundle["model"].predict(X).ravel().astype(int)
        values = patient_predictions(dataset.groups[use], y, p)
        m = metrics(y, p, predicted=predicted)
        report["test_metrics"][name] = m
        report["test_intervals"][name] = bootstrap_intervals(y, p, predicted=predicted)
        values["predicted"] = predicted  # This dataset is sorted by patient ID.
        values.to_csv(output / f"{name}_test_predictions.csv", index=False)
        predictions[name] = values
    winner = report["selected_model_by_development"]
    score = report["test_metrics"][winner]
    report["target_met"] = score["accuracy"] > 0.83 and score["accuracy"] > majority
    report["validated_superiority_over_source_paper"] = False
    report["interpretation"] = "A numeric comparison with the paper's 83% is descriptive because cohorts and feature sets differ."
    (output / "external_report.json").write_text(json.dumps(report, indent=2))
    save_figures(predictions, output)
    print(json.dumps({"winner": winner, "target_met": report["target_met"], "metrics": report["test_metrics"]}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run(evaluate=True)
