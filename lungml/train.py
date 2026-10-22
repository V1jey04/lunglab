"""Reserve the test patients first; select all parameters on development data."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score

from .data import TASK_LABELS, load_dataset
from .evaluation import (bootstrap_intervals, metrics, paired_auc_difference,
                         patient_predictions, save_figures)
from .models import candidates


def patient_folds(y, groups, n_splits=5, seed=42):
    patients = pd.DataFrame({"patient_id": groups, "y": y}).drop_duplicates("patient_id")
    if patients.y.value_counts().min() < n_splits:
        raise ValueError(f"Need at least {n_splits} development patients per class for CV.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, va in splitter.split(patients.patient_id, patients.y):
        train = np.flatnonzero(np.isin(groups, patients.iloc[tr].patient_id))
        valid = np.flatnonzero(np.isin(groups, patients.iloc[va].patient_id))
        if min(np.bincount(np.asarray(y)[train], minlength=2)) < 2:
            raise ValueError("Each training fold needs at least two examples per class for SMOTE.")
        folds.append((train, valid))
    return folds


def train(csv_path, output="artifacts/latest", task="subtype", seed=42,
          cv=5, quick=False, provenance="real", calibrate=True):
    dataset = load_dataset(csv_path, task)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory {output} is nonempty. Choose a new run directory.")
    patients = dataset.frame[["patient_id", "label"]].drop_duplicates("patient_id")
    if patients.label.value_counts().min() < 8:
        raise ValueError("Need at least eight independent patients per class for a train/test comparison.")
    dev_ids, test_ids = train_test_split(patients.patient_id, test_size=0.2,
                                         stratify=patients.label, random_state=seed)
    dev = np.flatnonzero(np.isin(dataset.groups, dev_ids))
    test = np.flatnonzero(np.isin(dataset.groups, test_ids))
    Xdev = dataset.X.iloc[dev]
    ydev = dataset.y[dev]
    groups_dev = dataset.groups[dev]
    folds = patient_folds(ydev, groups_dev, n_splits=cv, seed=seed)

    # GridSearch receives DataFrames with their original row indices, allowing
    # fold scores to aggregate nodules to patients exactly as the final metrics do.
    patient_lookup = pd.Series(dataset.groups, index=dataset.X.index)

    def patient_auc(estimator, X, y):
        p = estimator.predict_proba(X)[:, 1]
        values = patient_predictions(patient_lookup.loc[X.index].to_numpy(), y, p)
        return roc_auc_score(values.y_true, values.probability)

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "task": task, "labels": list(TASK_LABELS[task]), "positive_class": TASK_LABELS[task][1],
        "data_provenance": provenance, "seed": seed, "test_fraction": 0.2,
        "dataset_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "feature_names": dataset.X.columns.tolist(),
        "development_patients": len(dev_ids), "test_patients": len(test_ids),
        "cv_folds": cv, "threshold": 0.5,
        "evaluation_unit": "patient; mean probability across nodules",
        "versions": {p: version(p) for p in ("scikit-learn", "numpy", "pandas", "catboost", "imbalanced-learn")},
        "models": {},
    }
    provenance_file = Path(csv_path).with_suffix(".provenance.json")
    report["dataset_provenance"] = json.loads(provenance_file.read_text()) if provenance_file.exists() else None
    splits = dataset.frame[["patient_id", "sample_id", "label"]].copy()
    splits["split"] = np.where(np.isin(dataset.groups, dev_ids), "development", "test")
    splits.to_csv(output / "splits.csv", index=False)
    predictions = {}
    for name, (estimator, grid) in candidates(seed, quick).items():
        print(f"Training {name}: {cv}-fold patient-grouped search", flush=True)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ConvergenceWarning)
            search = GridSearchCV(estimator, grid, scoring=patient_auc, cv=folds,
                                  n_jobs=1, refit=True, error_score="raise")
            search.fit(Xdev, ydev)
        fitted = search.best_estimator_
        convergence = [str(w.message) for w in recorded if issubclass(w.category, ConvergenceWarning)]
        if convergence:
            warnings.warn(f"{name}: {len(convergence)} convergence warnings. See report.json.")
        base = fitted
        threshold = 0.5
        if name == "catboost" and calibrate:
            # Every calibrator's validation patients are held out from its base model.
            fitted = CalibratedClassifierCV(estimator=base, cv=folds, method="sigmoid", ensemble=True)
            fitted.fit(Xdev, ydev)
            # Nested, patient-grouped calibration gives out-of-fold development
            # probabilities on the same scale as the deployed calibrated model.
            # Hyperparameters were already selected on development data, so this
            # is a threshold-selection set, not unbiased generalization evidence.
            oof = np.zeros(len(ydev))
            for tr, va in folds:
                inner = patient_folds(ydev[tr], groups_dev[tr], n_splits=3, seed=seed)
                threshold_model = CalibratedClassifierCV(estimator=clone(base), cv=inner,
                                                          method="sigmoid", ensemble=True)
                threshold_model.fit(Xdev.iloc[tr], ydev[tr])
                oof[va] = threshold_model.predict_proba(Xdev.iloc[va])[:, 1]
            development = patient_predictions(groups_dev, ydev, oof)
            choices = np.unique(np.r_[0.5, development.probability.to_numpy()])
            threshold = float(max(choices, key=lambda t: (
                balanced_accuracy_score(development.y_true, development.probability >= t),
                -abs(float(t) - 0.5))))
            development.to_csv(output / "catboost_development_oof.csv", index=False)
        p = fitted.predict_proba(dataset.X.iloc[test])[:, 1]
        values = patient_predictions(dataset.groups[test], dataset.y[test], p)
        predictions[name] = values
        values.to_csv(output / f"{name}_test_predictions.csv", index=False)
        selected = (dataset.X.columns[base.named_steps["select"].indices_].tolist()
                    if name == "paper_mlp" else dataset.X.columns.tolist())
        report["models"][name] = {
            "cv_selection_auc": float(search.best_score_),
            "cv_selection_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "best_parameters": search.best_params_,
            "selected_features": selected,
            "calibration": "grouped sigmoid ensemble" if name == "catboost" and calibrate else "none",
            "threshold": threshold,
            "threshold_selection": "development OOF balanced accuracy" if name == "catboost" and calibrate else "fixed 0.5",
            "test_metrics_at_0_5": metrics(values.y_true, values.probability),
            "test_metrics": metrics(values.y_true, values.probability, threshold),
            "test_ci95": bootstrap_intervals(values.y_true, values.probability, seed, threshold=threshold),
            "convergence_warnings": sorted(set(convergence)),
        }
        bundle = {
            "model": fitted, "task": task, "feature_names": report["feature_names"],
            "labels": report["labels"], "threshold": threshold, "data_provenance": provenance,
            "name": name, "training_patient_ids": dev_ids.tolist(),
            "dataset_provenance": report["dataset_provenance"],
        }
        joblib.dump(bundle, output / f"{name}.joblib")
        if name == "catboost":
            # Importance refers to the refit, uncalibrated training model.
            importance = base.named_steps["model"].feature_importances_
        else:
            importance = base.named_steps["select"].scores_
        pd.DataFrame({"feature": report["feature_names"], "importance": importance}).sort_values(
            "importance", ascending=False).to_csv(output / f"{name}_feature_importance.csv", index=False)
        pd.DataFrame(search.cv_results_).to_csv(output / f"{name}_cv_results.csv", index=False)
    # Deployment choice is made by development CV alone, never test accuracy.
    report["selected_model_by_cv"] = max(report["models"], key=lambda n: report["models"][n]["cv_selection_auc"])
    baseline, candidate = predictions["paper_mlp"], predictions["catboost"]
    if not baseline.patient_id.equals(candidate.patient_id):
        raise RuntimeError("Comparison patient order mismatch.")
    report["catboost_minus_mlp_test_auc"] = paired_auc_difference(
        baseline.y_true, baseline.probability, candidate.probability, seed)
    report["interpretation"] = (
        "Synthetic demonstration only. These metrics are not evidence of cancer detection performance."
        if provenance == "synthetic" else
        "Single held-out cohort comparison. CV scores used for selection are optimistic estimates; "
        "external validation is needed. No claim of clinical diagnosis or superiority to the paper.")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    save_figures(predictions, output)
    summary = pd.DataFrame({n: v["test_metrics"] for n, v in report["models"].items()}).T
    summary.drop(columns="confusion_matrix").to_csv(output / "metrics.csv", index_label="model")
    print(f"Saved models, patient predictions and metrics to {output}", flush=True)
    return report
