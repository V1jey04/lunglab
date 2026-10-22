"""Exploratory classification of the source's recorded yes/no recurrence field.

This endpoint has no time horizon. It does not turn last-known-alive dates into
recurrence-free follow-up, and does not unlock the competing-risk protocol.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import CLINICAL_FEATURES, numeric_features
from .predict import load_bundle, predict_frame
from .quality import create_profile, sha256
from .recurrence_predict import read_feature_csv

TASK = "recorded_recurrence"
LABELS = ["no_recorded_recurrence", "recorded_recurrence"]
DEFAULT_OUTPUT = Path("artifacts/recurrence_recorded")
CONTEXT = (
    "Exploratory estimate of recurrence recorded in the source study's variable follow-up. "
    "Requires available surgical pathology information. "
    "No fixed time horizon; a source 'no' does not establish future freedom from recurrence. "
    "Not a validated clinical risk estimate."
)
LIMITATIONS = [
    "The yes/no target reflects the source records, not verified recurrence-free surveillance.",
    "Follow-up length, competing death and scan/surgery timing are not modelled by this classifier.",
    "Small retrospective cohort; no independent clinical validation or individual confidence intervals.",
    "The preserved test patients were inspected in previous project experiments; scores are exploratory.",
    "CT measurements require the same extraction protocol as the training source.",
]
MODEL_NAMES = {
    "clinical_logistic": "Clinical logistic · postoperative profile",
    "clinical_radiomics_logistic": "Clinical + CT radiomics logistic",
}
PATHOLOGY_FEATURES = ["clinical_pathological_t", "clinical_pathological_n", "clinical_grade",
                      "clinical_lymphovascular_invasion", "clinical_pleural_invasion"]
GRADE_VALUES = {"G1 Well differentiated": 1., "Other, Type I: Well to moderately differentiated": 1.5,
                "G2 Moderately differentiated": 2., "Other, Type II: Moderately to poorly differentiated": 2.5,
                "G3 Poorly differentiated": 3.}


def _write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def _unique_ids(frame, column, context):
    if column not in frame or frame[column].isna().any():
        raise ValueError(f"{context} requires nonmissing {column}.")
    frame[column] = frame[column].astype(str).str.strip()
    if frame[column].eq("").any() or frame[column].duplicated().any():
        raise ValueError(f"{context} requires unique, nonempty patient IDs.")


def prepare_recorded_cohort(features_path, clinical_path):
    features_path, clinical_path = Path(features_path), Path(clinical_path)
    features = read_feature_csv(features_path)
    clinical = pd.read_csv(clinical_path, dtype={"Case ID": str})
    clinical.columns = clinical.columns.str.strip()
    _unique_ids(clinical, "Case ID", "Clinical source")
    if "Recurrence" not in clinical:
        raise ValueError("Clinical source requires its explicit Recurrence yes/no field.")
    split_path = features_path.with_suffix(".splits.csv")
    splits = pd.read_csv(split_path, dtype={"patient_id": str})
    _unique_ids(splits, "patient_id", "Preserved split")
    if ("split" not in splits or set(splits.split) != {"development", "test"}
            or set(splits.patient_id) != set(features.patient_id)):
        raise ValueError("The preserved split must assign every source patient exactly once.")
    clinical_names = list(CLINICAL_FEATURES) + PATHOLOGY_FEATURES
    clinical_by_id = clinical.set_index("Case ID")
    aligned = clinical_by_id.reindex(features.patient_id).reset_index(drop=True)
    required_pathology = {"Pathological T stage", "Pathological N stage", "Histopathological Grade",
                          "Lymphovascular invasion", "Pleural invasion (elastic, visceral, or parietal)"}
    if not required_pathology <= set(aligned):
        raise ValueError("Clinical source is missing the required surgical pathology fields.")
    features["clinical_pathological_t"] = aligned["Pathological T stage"].map(
        {"Tis": 0., **{f"T{i}{suffix}": float(i) for i in range(1, 5) for suffix in ("", "a", "b")}})
    features["clinical_pathological_n"] = aligned["Pathological N stage"].map({f"N{i}": float(i) for i in range(4)})
    features["clinical_grade"] = aligned["Histopathological Grade"].map(GRADE_VALUES)
    features["clinical_lymphovascular_invasion"] = aligned["Lymphovascular invasion"].map({"Absent": 0., "Present": 1.})
    features["clinical_pleural_invasion"] = aligned["Pleural invasion (elastic, visceral, or parietal)"].map({"No": 0., "Yes": 1.})
    radiomics_names = [name for name in features if name.startswith("original_")]
    if not radiomics_names:
        raise ValueError("Original radiomic features are required for the CT candidate.")
    names = clinical_names + radiomics_names
    X = numeric_features(features, names)
    labels = features.patient_id.map(clinical.set_index("Case ID").Recurrence)
    labels = labels.astype("string").str.strip().str.casefold()
    known = labels.isin(["yes", "no"]).fillna(False)
    frame = features[["patient_id"]].copy()
    if "sample_id" in features:
        frame["sample_id"] = features.sample_id
    frame = pd.concat([frame, X], axis=1)
    frame["label"] = labels.map({"no": 0, "yes": 1})
    frame["split"] = frame.patient_id.map(splits.set_index("patient_id").split)
    frame = frame.loc[known].reset_index(drop=True)
    frame["label"] = frame.label.astype(int)
    if frame.empty or set(frame.label) != {0, 1}:
        raise ValueError("Both recorded recurrence classes are required.")
    for split in ("development", "test"):
        if set(frame.loc[frame.split.eq(split), "label"]) != {0, 1}:
            raise ValueError(f"Both recurrence classes are required in {split}.")
    return frame, {"clinical_logistic": clinical_names,
                   "clinical_radiomics_logistic": names}, {
        "features_path": str(features_path), "features_sha256": sha256(features_path),
        "clinical_path": str(clinical_path), "clinical_sha256": sha256(clinical_path),
        "split_path": str(split_path), "split_sha256": sha256(split_path),
        "source_patients": len(features), "excluded_unknown_recurrence": int((~known).sum()),
    }


def _pipeline(name, feature_count):
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("scale", StandardScaler())]
    if name == "clinical_radiomics_logistic":
        steps.append(("select", SelectKBest(f_classif, k=min(10, feature_count))))
    steps.append(("model", LogisticRegression(C=0.1, max_iter=3000, random_state=42)))
    return Pipeline(steps)


def classification_metrics(y, probability, threshold=.5):
    y, probability = np.asarray(y), np.asarray(probability, dtype=float)
    predicted = probability >= threshold
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "brier_score": float(brier_score_loss(y, probability)),
            "recurrence_recall": float(recall_score(y, predicted, zero_division=0)),
            "no_recorded_recurrence_recall": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
            "recurrence_precision": float(precision_score(y, predicted, zero_division=0)),
            "confusion_matrix": matrix.tolist(), "threshold": float(threshold)}


def development_threshold(y, scores):
    """Choose an operating point from development predictions only."""
    candidates = np.linspace(.1, .6, 51)
    return float(max(candidates, key=lambda t: (
        balanced_accuracy_score(y, np.asarray(scores) >= t), -abs(t - .5))))


def validate_clinical_inputs(bundle, frame):
    """Reject impossible clinical encodings before any model is called."""
    X = numeric_features(frame, bundle["feature_names"])
    for name, low, high in [("clinical_age_at_diagnosis", 0, 120), ("clinical_pack_years", 0, 250),
                            ("clinical_pathological_t", 0, 4), ("clinical_pathological_n", 0, 3),
                            ("clinical_grade", 1, 3)]:
        observed = X[name].dropna()
        if not observed.between(low, high).all():
            raise ValueError(f"{name} must be between {low} and {high}.")
    for name in ["clinical_male", "clinical_smoking_never", "clinical_smoking_former", "clinical_smoking_current",
                 "clinical_lymphovascular_invasion", "clinical_pleural_invasion"]:
        if not X[name].dropna().isin([0., 1.]).all():
            raise ValueError(f"{name} must be 0, 1 or missing.")
    for name in ["clinical_pathological_t", "clinical_pathological_n"]:
        if not X[name].dropna().mod(1).eq(0).all():
            raise ValueError(f"{name} must be an integer category.")
    smoking = X[["clinical_smoking_never", "clinical_smoking_former", "clinical_smoking_current"]]
    if smoking.isna().any().any() or not smoking.sum(axis=1).eq(1).all():
        raise ValueError("Smoking indicators must identify exactly one known smoking category.")
    if (X.clinical_smoking_never.eq(1) & X.clinical_pack_years.fillna(0).gt(0)).any():
        raise ValueError("A never-smoker must have zero or missing pack-years.")


def run_recorded_recurrence(
        features_path="data/radiogenomics_clinical_subtype.csv",
        clinical_path="data/raw/nsclc_radiogenomics/clinical.csv",
        output=DEFAULT_OUTPUT):
    """Freeze two prespecified pipelines using development evidence, then score test."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; frozen recurrence results cannot be overwritten.")
    frame, feature_sets, provenance = prepare_recorded_cohort(features_path, clinical_path)
    train = frame.loc[frame.split.eq("development")].reset_index(drop=True)
    test = frame.loc[frame.split.eq("test")].reset_index(drop=True)
    if train.label.value_counts().min() < 5:
        raise ValueError("At least five development patients of each class are required.")
    if frame.loc[frame.split.eq("development"), feature_sets["clinical_radiomics_logistic"]].isna().all().any():
        raise ValueError("A model input is completely missing in development patients.")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, train.label))
    oof = train[["patient_id", "label"]].copy()
    baseline_oof = np.zeros(len(train))
    for fitting, validation in folds:
        baseline_oof[validation] = train.label.iloc[fitting].mean()
    development, fitted, thresholds = {}, {}, {}
    for name, names in feature_sets.items():
        pipeline = _pipeline(name, len(names))
        scores = np.zeros(len(train))
        for fitting, validation in folds:
            candidate = clone(pipeline).fit(train[names].iloc[fitting], train.label.iloc[fitting])
            scores[validation] = candidate.predict_proba(train[names].iloc[validation])[:, 1]
        thresholds[name] = development_threshold(train.label, scores)
        development[name] = classification_metrics(train.label, scores, thresholds[name])
        oof[f"{name}_probability"] = scores
        fitted[name] = pipeline.fit(train[names], train.label)
    selected = min(development, key=lambda name: (development[name]["brier_score"],
                                                 -development[name]["roc_auc"], len(feature_sets[name])))
    output.mkdir(parents=True, exist_ok=True)
    cohort_path = output / "cohort.csv"
    frame.to_csv(cohort_path, index=False)
    frame[["patient_id", "split"]].to_csv(output / "splits.csv", index=False)
    oof["prevalence_reference_probability"] = baseline_oof
    oof.to_csv(output / "development_predictions.csv", index=False)
    selection = {"schema_version": 1, "task": TASK,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "Source-recorded recurrence yes/no during variable study follow-up",
        "prediction_context": CONTEXT, "limitations": LIMITATIONS,
        "dataset_path": str(cohort_path), "cohort_sha256": sha256(cohort_path),
        "source_provenance": provenance, "development_patients": len(train), "test_patients": len(test),
        "recorded_recurrences": int(frame.label.sum()),
        "development_recorded_recurrences": int(train.label.sum()),
        "test_recorded_recurrences": int(test.label.sum()),
        "selection_rule": "Lowest five-fold development out-of-fold Brier score; then AUC, then fewer inputs.",
        "protocol": "L2 logistic pipelines, C=0.1. Threshold maximizes development out-of-fold balanced accuracy on a 0.10–0.60 grid. Exploratory evaluation; initial age/smoking-only test scores were already inspected.",
        "thresholds": thresholds,
        "development_metrics": development, "selected_model_by_development": selected,
        "prevalence_reference_development": classification_metrics(train.label, baseline_oof),
        "frozen_models": {}, "test_patient_ids": test.patient_id.tolist(),
        "previously_inspected_holdout": True, "fixed_horizon_supported": False}
    for name, model in fitted.items():
        bundle = {"schema_version": 1, "task": TASK, "name": name, "model": model,
                  "feature_names": feature_sets[name], "labels": LABELS, "threshold": thresholds[name],
                  "data_provenance": "public NSCLC Radiogenomics · recorded recurrence labels",
                  "dataset_provenance": {"output_sha256": sha256(cohort_path)},
                  "training_patient_ids": train.patient_id.tolist(),
                  "development_patient_ids": train.patient_id.tolist(),
                  "test_patient_ids": test.patient_id.tolist(), "prediction_context": CONTEXT,
                  "limitations": LIMITATIONS, "fixed_horizon_supported": False}
        path = output / f"{name}.joblib"
        joblib.dump(bundle, path)
        create_profile(path, cohort_path)
        selection["frozen_models"][name] = {"model_sha256": sha256(path), "feature_names": feature_sets[name]}
    # Model choice and hashes are saved before the test labels are scored.
    _write(output / "selection.json", selection)
    test_predictions = test[["patient_id", "label"]].copy()
    test_metrics = {}
    for name, model in fitted.items():
        probability = model.predict_proba(test[feature_sets[name]])[:, 1]
        test_predictions[f"{name}_probability"] = probability
        test_predictions[f"{name}_prediction"] = (probability >= thresholds[name]).astype(int)
        test_metrics[name] = classification_metrics(test.label, probability, thresholds[name])
    baseline_test = np.full(len(test), train.label.mean())
    test_predictions["prevalence_reference_probability"] = baseline_test
    test_predictions.to_csv(output / "test_predictions.csv", index=False)
    report = {**selection, "test_metrics": test_metrics,
              "prevalence_reference_test": classification_metrics(test.label, baseline_test),
              "evaluation": "Preserved patient split: development-only selection; previously inspected test patients scored after freezing.",
              "interpretation": CONTEXT}
    _write(output / "report.json", report)
    return report


def verify_recorded_models(output, selection=None):
    output = Path(output)
    if selection is None:
        selection = json.loads((output / "selection.json").read_text())
    if selection.get("task") != TASK or set(selection.get("frozen_models", {})) != set(MODEL_NAMES):
        raise ValueError("Invalid recorded-recurrence selection.")
    for name, value in selection["frozen_models"].items():
        path = output / f"{name}.joblib"
        if not path.is_file() or sha256(path) != value.get("model_sha256"):
            raise ValueError(f"Frozen recorded-recurrence model changed or is missing: {name}.")
    return selection


def predict_recorded_csv(model_path, data_path, output_path):
    model_path, data_path, output_path = Path(model_path), Path(data_path), Path(output_path)
    if output_path.resolve() in {model_path.resolve(), data_path.resolve()} or output_path.exists():
        raise ValueError("Use a new prediction path; source files and existing predictions cannot be overwritten.")
    verify_recorded_models(model_path.parent)
    bundle = load_bundle(model_path)
    if bundle["task"] != TASK:
        raise ValueError("Select a recorded-recurrence classifier.")
    from .dashboard_runtime import validate_queue
    frame = read_feature_csv(data_path)
    validate_queue(bundle, frame)
    result = predict_frame(bundle, frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def run_training_fit_experiment(source=DEFAULT_OUTPUT,
                                output="artifacts/recurrence_training_fit", patients=40):
    """Deliberately overfit a development subset; report training and test separately."""
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; training-fit results cannot be overwritten.")
    selection = verify_recorded_models(source)
    cohort = Path(selection["dataset_path"])
    if sha256(cohort) != selection["cohort_sha256"]:
        raise ValueError("The source cohort no longer matches its frozen experiment.")
    rows = read_feature_csv(cohort)
    development = rows.loc[rows.split.eq("development")].copy()
    test = rows.loc[rows.split.eq("test")].copy()
    if set(test.patient_id) != set(selection["test_patient_ids"]):
        raise ValueError("The source test patient split has changed.")
    if not 12 <= patients < len(development):
        raise ValueError("Choose at least 12 patients and fewer than all development patients.")
    train, unused = train_test_split(development, train_size=patients, stratify=development.label, random_state=42)
    names = selection["frozen_models"]["clinical_radiomics_logistic"]["feature_names"]
    model = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("tree", DecisionTreeClassifier(random_state=42))])
    model.fit(numeric_features(train, names), train.label)
    training_probability = model.predict_proba(numeric_features(train, names))[:, 1]
    test_probability = model.predict_proba(numeric_features(test, names))[:, 1]
    output.mkdir(parents=True, exist_ok=True)
    context = "Intentional overfitting experiment. Training accuracy measures memorization of fitted patients; test accuracy is reported separately. " + CONTEXT
    bundle = {"schema_version": 1, "task": TASK, "name": "small_cohort_decision_tree",
              "model": model, "feature_names": names, "labels": LABELS, "threshold": .5,
              "data_provenance": "public NSCLC Radiogenomics · intentional training-fit experiment",
              "training_patient_ids": train.patient_id.tolist(), "test_patient_ids": test.patient_id.tolist(),
              "prediction_context": context, "experiment_type": "intentional_overfit"}
    path = output / "decision_tree.joblib"
    joblib.dump(bundle, path)
    report = {"schema_version": 1, "task": "recurrence_training_fit", "experiment_type": "intentional_overfit",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "model": "Unrestricted decision tree",
              "source_cohort_sha256": sha256(cohort), "source_selection_sha256": sha256(source / "selection.json"),
              "model_sha256": sha256(path), "seed": 42, "threshold": .5,
              "training_patients": len(train), "test_patients": len(test), "unused_development_patients": len(unused),
              "training_recurrences": int(train.label.sum()), "test_recurrences": int(test.label.sum()),
              "training_patient_ids": train.patient_id.tolist(), "test_patient_ids": test.patient_id.tolist(),
              "feature_names": names, "training_metrics": classification_metrics(train.label, training_probability),
              "test_metrics": classification_metrics(test.label, test_probability),
              "prediction_context": context,
              "interpretation": "Training accuracy is resubstitution performance, not an estimate for new patients. "
                                "The same previously inspected test split is scored once with no retries or test-driven tuning."}
    _write(output / "report.json", report)
    for split, frame, probability in [("training", train, training_probability), ("test", test, test_probability)]:
        predictions = frame[["patient_id", "label"]].copy()
        predictions["probability_recorded_recurrence"] = probability
        predictions["prediction"] = (probability >= .5).astype(int)
        predictions.to_csv(output / f"{split}_predictions.csv", index=False)
    return report
