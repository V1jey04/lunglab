"""Accuracy-focused search on preserved patient splits, with nested threshold tuning.

All development predictions come from an outer fold. Hyperparameters and the
decision threshold are selected using only inner-fold predictions. Existing
holdouts have already been inspected in earlier experiments; this module
explicitly labels any further holdout assessment as exploratory.
"""
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .data import CLINICAL_FEATURES, TASK_LABELS, load_dataset
from .evaluation import bootstrap_intervals, metrics, save_figures
from .improve import CorrelationPruner
from .models import paper_mlp
from .train import patient_folds

THRESHOLDS = (0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85)


def clears_accuracy_target(accuracy, majority, target=0.83):
    return bool(accuracy > target and accuracy > majority)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clinical_features(c):
    """Read the explicit demographic/exposure whitelist independently of labels."""
    required = ["Case ID", "Age at Histological Diagnosis", "Gender", "Pack Years", "Smoking status"]
    c = c[required].copy()  # Never import histology, mutations, stage, treatment or outcomes.
    if c["Case ID"].isna().any() or c["Case ID"].duplicated().any():
        raise ValueError("Clinical records require unique patient IDs.")
    features = pd.DataFrame({
        "patient_id": c["Case ID"],
        "clinical_age_at_diagnosis": pd.to_numeric(c["Age at Histological Diagnosis"], errors="raise"),
        "clinical_male": c.Gender.map({"Male": 1.0, "Female": 0.0}),
        "clinical_pack_years": pd.to_numeric(c["Pack Years"].replace("Not Collected", np.nan), errors="raise"),
    })
    for status, column in [("Nonsmoker", "clinical_smoking_never"),
                           ("Former", "clinical_smoking_former"),
                           ("Current", "clinical_smoking_current")]:
        features[column] = np.where(c["Smoking status"].isin(["Nonsmoker", "Former", "Current"]),
                                    (c["Smoking status"] == status).astype(float), np.nan)
    return features


def prepare_clinical(data="data/radiogenomics_subtype.csv",
                     clinical="data/raw/nsclc_radiogenomics/clinical.csv",
                     output="data/radiogenomics_clinical_subtype.csv"):
    """Join only age, sex and smoking history, preserving the original split."""
    source = load_dataset(data, feature_mode="all")
    c = pd.read_csv(clinical, dtype={"Case ID": str})
    c.columns = c.columns.str.strip()
    features = clinical_features(c)
    # Retain exactly the radiomic input allowlist and existing labels/IDs.
    frame = source.frame[["patient_id", "sample_id", "label", *source.X.columns]].merge(
        features, on="patient_id", how="left", validate="one_to_one", indicator=True)
    if not frame._merge.eq("both").all():
        raise ValueError("Every radiomics patient must have a clinical record.")
    frame = frame.drop(columns="_merge")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    split_source = Path(data).with_suffix(".splits.csv")
    if not split_source.exists():
        raise ValueError("A preserved split manifest is required; no new random split is created.")
    if target.exists():
        existing = pd.read_csv(target, dtype={"patient_id": str, "sample_id": str})
        pd.testing.assert_frame_equal(existing, frame, check_dtype=False, check_exact=False)
        if target.with_suffix(".splits.csv").read_bytes() != split_source.read_bytes():
            raise ValueError("Clinical experiment split differs from the original split.")
        return target
    frame.to_csv(target, index=False)
    target.with_suffix(".splits.csv").write_bytes(split_source.read_bytes())
    provenance = json.loads(Path(data).with_suffix(".provenance.json").read_text())
    provenance.update({
        "clinical_source_sha256": sha256(clinical), "radiomics_dataset_sha256": sha256(data),
        "output_sha256": sha256(target), "clinical_features": list(CLINICAL_FEATURES),
        "total_features": len(source.X.columns) + len(CLINICAL_FEATURES),
        "clinical_input_requirements": "Age around diagnosis, sex, smoking status and pack years",
        "clinical_notes": [
            "Source age is recorded at histological diagnosis; availability at CT must be confirmed for deployment.",
            "Unknown smoking status remains missing; unknown pack years are imputed within training folds.",
            "Histology, pathological stage, molecular tests, treatment and outcomes are excluded.",
            "Additional clinical inputs change the input requirements relative to radiomics-only models.",
        ],
    })
    target.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    return target


def prepare_expanded_clinical(data="data/radiogenomics_subtype.csv",
                              clinical="data/raw/nsclc_radiogenomics/clinical.csv",
                              output="data/radiogenomics_expanded_clinical.csv"):
    """Add all other eligible clinical patients to development, keeping test IDs."""
    source = load_dataset(data)
    c = pd.read_csv(clinical, dtype={"Case ID": str})
    c.columns = c.columns.str.strip()
    features = clinical_features(c)
    histology_map = {"Adenocarcinoma": TASK_LABELS["subtype"][0],
                    "Squamous cell carcinoma": TASK_LABELS["subtype"][1]}
    labels = c[["Case ID", "Histology"]].rename(columns={"Case ID": "patient_id"})
    labels["label"] = labels.Histology.map(histology_map)
    labels = labels.dropna(subset=["label"])[["patient_id", "label"]]
    reference = source.frame[["patient_id", "label"]].merge(labels, on="patient_id", suffixes=("_original", "_clinical"), validate="one_to_one")
    if len(reference) != len(source.frame) or not reference.label_original.eq(reference.label_clinical).all():
        raise ValueError("Original patient labels differ from official clinical labels.")
    frame = labels.merge(features, on="patient_id", validate="one_to_one").sort_values("patient_id").reset_index(drop=True)
    frame.insert(1, "sample_id", frame.patient_id + "-clinical")
    original_splits = pd.read_csv(Path(data).with_suffix(".splits.csv"), dtype={"patient_id": str})
    test_ids = set(original_splits.loc[original_splits.split == "test", "patient_id"])
    if not test_ids <= set(frame.patient_id):
        raise ValueError("Expanded clinical cohort is missing an original test patient.")
    splits = frame[["patient_id", "sample_id", "label"]].copy()
    splits["split"] = np.where(splits.patient_id.isin(test_ids), "test", "development")
    original_assignments = original_splits.set_index("patient_id").split
    if not splits.set_index("patient_id").split.loc[original_assignments.index].equals(original_assignments):
        raise ValueError("An original patient changed split.")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        pd.testing.assert_frame_equal(pd.read_csv(target), frame, check_dtype=False)
        pd.testing.assert_frame_equal(pd.read_csv(target.with_suffix(".splits.csv")), splits, check_dtype=False)
        return target
    frame.to_csv(target, index=False)
    splits.to_csv(target.with_suffix(".splits.csv"), index=False)
    provenance = json.loads(Path(data).with_suffix(".provenance.json").read_text())
    provenance.update({"output_sha256": sha256(target), "clinical_source_sha256": sha256(clinical),
                       "patients": len(frame), "class_counts": frame.label.value_counts().to_dict(),
                       "source": "Official TCIA NSCLC Radiogenomics clinical release",
                       "original_features": 0, "total_features": len(CLINICAL_FEATURES),
                       "clinical_features": list(CLINICAL_FEATURES), "radiomic_features": 0,
                       "additional_development_patients": len(frame) - len(source.frame),
                       "input_requirements": "Clinical-only: age around diagnosis, sex, smoking status and pack years.",
                       "notes": ["Extra AC/SCC clinical records are development-only; the original test IDs remain fixed.",
                                 "Age is recorded at diagnosis; availability at CT requires confirmation.",
                                 "This model has no imaging input and is a clinical subtype benchmark, not a CT detector.",
                                 "Histology supplies the target only; stage, molecular tests, treatment and outcomes are excluded."]})
    target.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    return target


def expanded_clinical_specs(seed=42, memory=None):
    base = candidate_specs(seed, memory)
    result = {f"expanded_{name}": base[name] for name in ["clinical_accuracy_logistic", "clinical_accuracy_catboost"]}
    def pipe(model):
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scale", StandardScaler()), ("model", model)], memory=memory)
    result["expanded_clinical_hist_boosting"] = (pipe(HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, random_state=seed)), {
            "model__max_leaf_nodes": [3, 7], "model__min_samples_leaf": [5, 10],
            "model__l2_regularization": [3, 20]}, "clinical", True)
    result["expanded_clinical_extra_trees"] = (pipe(ExtraTreesClassifier(
        n_estimators=300, n_jobs=2, random_state=seed)), {
            "model__min_samples_leaf": [5, 10], "model__max_features": [0.7, 1.0]}, "clinical", True)
    voting = VotingClassifier(estimators=[
        ("logistic", LogisticRegression(C=0.1, max_iter=4000, random_state=seed)),
        ("boosting", CatBoostClassifier(iterations=300, depth=2, l2_leaf_reg=20,
                                       learning_rate=0.05, loss_function="Logloss", random_seed=seed,
                                       thread_count=2, verbose=False, allow_writing_files=False))], voting="soft")
    result["expanded_clinical_voting"] = (pipe(voting), {"model__weights": [[2, 1], [1, 1], [1, 2]]}, "clinical", True)
    return result


def candidate_specs(seed=42, memory=None):
    def pipe(model, quantile=False, clinical=False):
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if not clinical:
            steps.append(("prune", CorrelationPruner()))
        steps.extend([
            ("scale", QuantileTransformer(n_quantiles=30, output_distribution="normal", random_state=seed)
             if quantile else StandardScaler()),
            ("select", SelectKBest(f_classif, k="all")), ("model", model),
        ])
        return Pipeline(steps, memory=memory)

    def boosting():
        return CatBoostClassifier(iterations=300, learning_rate=0.05, loss_function="Logloss",
                                  random_seed=seed, thread_count=2, verbose=False, allow_writing_files=False)

    logistic = LogisticRegression(max_iter=4000, random_state=seed)
    svm = SVC(kernel="rbf", probability=True, random_state=seed)
    boost_grid = {"model__depth": [2, 3, 4], "model__l2_leaf_reg": [3, 20]}
    return {
        "paper_mlp_accuracy": (paper_mlp(seed), {"select__alpha": [0.001, 0.01]}, "original", False),
        "original_accuracy_logistic": (pipe(logistic), {
            "select__k": [5, 15, 40, "all"], "model__C": [0.01, 0.1, 1, 10]}, "original", True),
        "rich_accuracy_logistic": (pipe(logistic, True), {
            "select__k": [10, 40, 100], "model__C": [0.01, 0.1, 1]}, "rich", True),
        "original_accuracy_svm": (pipe(svm, True), {
            "select__k": [10, 30, "all"], "model__C": [0.3, 3, 30],
            "model__gamma": ["scale", 0.01, 0.1]}, "original", True),
        "rich_accuracy_svm": (pipe(svm, True), {
            "select__k": [10, 40, 100], "model__C": [0.3, 3, 30],
            "model__gamma": ["scale", 0.01]}, "rich", True),
        "original_accuracy_lda": (pipe(LinearDiscriminantAnalysis(solver="lsqr"), True), {
            "select__k": [10, 30, "all"], "model__shrinkage": ["auto", 0.1, 0.5, 0.9]}, "original", True),
        "original_accuracy_extra_trees": (pipe(ExtraTreesClassifier(
            n_estimators=300, n_jobs=2, random_state=seed)), {
                "select__k": [10, 30, "all"], "model__min_samples_leaf": [2, 5],
                "model__max_features": [0.3, 1.0]}, "original", True),
        "original_accuracy_catboost": (pipe(boosting()), {**boost_grid, "select__k": [10, "all"]}, "original", True),
        "clinical_accuracy_logistic": (pipe(logistic, clinical=True), {
            "model__C": [0.01, 0.1, 1, 10]}, "clinical", True),
        "clinical_accuracy_catboost": (pipe(boosting(), clinical=True), boost_grid, "clinical", True),
        "fusion_accuracy_logistic": (pipe(logistic), {
            "select__k": [10, 30, "all"], "model__C": [0.01, 0.1, 1, 10]}, "fusion", True),
        "fusion_accuracy_catboost": (pipe(boosting()), {**boost_grid, "select__k": [10, "all"]}, "fusion", True),
    }


def view_columns(X, view):
    original = [c for c in X if c.startswith("original_")]
    clinical = [c for c in X if c in CLINICAL_FEATURES]
    views = {"original": original, "rich": [c for c in X if c not in CLINICAL_FEATURES],
             "clinical": clinical, "fusion": original + clinical}
    columns = views[view]
    if not columns:
        raise ValueError(f"No features available for {view} view.")
    return columns


def choose_threshold(y, probabilities, thresholds=THRESHOLDS):
    choices = []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        choices.append((float(accuracy_score(y, predictions)),
                        float(balanced_accuracy_score(y, predictions)),
                        -abs(threshold - 0.5), float(threshold)))
    accuracy, balanced, _, threshold = max(choices)
    return threshold, accuracy, balanced


def inner_search(estimator, grid, X, y, groups, tune_threshold=True, seed=42):
    """Tune parameters and threshold on inner out-of-fold predictions only."""
    folds = patient_folds(y, groups, 3, seed)
    best = None
    rows = []
    for parameters in ParameterGrid(grid):
        probabilities = np.empty(len(y))
        native = np.empty(len(y), dtype=int)
        for training, validation in folds:
            model = clone(estimator).set_params(**parameters)
            model.fit(X.iloc[training], y[training])
            probabilities[validation] = model.predict_proba(X.iloc[validation])[:, 1]
            native[validation] = model.predict(X.iloc[validation])
        if tune_threshold:
            threshold, accuracy, balanced = choose_threshold(y, probabilities)
            prediction_rule = "probability_threshold"
        else:
            threshold = 0.5
            accuracy = float(accuracy_score(y, native))
            balanced = float(balanced_accuracy_score(y, native))
            prediction_rule = "native_predict"
        auc = float(roc_auc_score(y, probabilities))
        result = {"parameters": parameters, "threshold": threshold, "prediction_rule": prediction_rule,
                  "inner_accuracy": accuracy, "inner_balanced_accuracy": balanced, "inner_auc": auc}
        rows.append(result)
        key = (accuracy, balanced, auc, -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, result)
    fitted = clone(estimator).set_params(**best[1]["parameters"]).fit(X, y)
    return fitted, best[1], rows


def predict_selected(model, selection, X):
    probabilities = model.predict_proba(X)[:, 1]
    predicted = (model.predict(X).ravel().astype(int) if selection["prediction_rule"] == "native_predict"
                 else (probabilities >= selection["threshold"]).astype(int))
    return probabilities, predicted


def run_accuracy(data="data/radiogenomics_clinical_subtype.csv",
                 output="artifacts/radiogenomics_accuracy", evaluate=False, specs=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        if evaluate:
            return evaluate_accuracy(output)
        raise ValueError("Completed selection exists; retain it and its original results.")
    dataset = load_dataset(data, feature_mode="clinical")
    if len(set(dataset.groups)) != len(dataset.groups):
        raise ValueError("Accuracy comparison requires one primary tumour per patient.")
    split_path = Path(data).with_suffix(".splits.csv")
    splits = pd.read_csv(split_path, dtype={"patient_id": str})
    if (splits.patient_id.duplicated().any() or set(splits.patient_id) != set(dataset.groups)
            or set(splits.split) != {"development", "test"}):
        raise ValueError("Preserved split must assign every patient exactly once.")
    use = dataset.frame.patient_id.isin(splits.loc[splits.split == "development", "patient_id"]).to_numpy()
    X = dataset.X.loc[use].reset_index(drop=True)
    y, groups = dataset.y[use], dataset.groups[use]
    test_ids = sorted(splits.loc[splits.split == "test", "patient_id"])
    memory = joblib.Memory(output / "training_cache", verbose=0)
    specs = candidate_specs(memory=memory) if specs is None else specs
    protocol = {
        "dataset_path": str(data), "dataset_sha256": sha256(data), "split_sha256": sha256(split_path),
        "seed": 42, "development_patients": len(y), "test_patients": len(test_ids),
        "outer_folds": 5, "inner_folds": 3, "thresholds": list(THRESHOLDS),
        "selection_metric": "nested CV accuracy; balanced accuracy then AUC break ties",
        "target_accuracy": 0.83, "clinical_features": list(CLINICAL_FEATURES),
        "previously_inspected_holdout": True, "test_patient_ids_excluded_from_selection": test_ids,
        "candidates": {name: {"estimator": repr(estimator), "grid": grid, "view": view,
                              "tune_threshold": tune} for name, (estimator, grid, view, tune) in specs.items()},
    }
    protocol_path = output / "protocol.json"
    # Commit the finite search before fitting. A restart may resume the same search.
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol changed after fitting started; do not revise the search using outcomes.")
    protocol_path.write_text(json.dumps(protocol, indent=2))
    source_provenance = json.loads(Path(data).with_suffix(".provenance.json").read_text())
    report = {**protocol, "models": {}, "feature_mode": "clinical", "test_evaluated": False,
              "development_majority_accuracy": float(max(np.bincount(y)) / len(y))}
    folds = patient_folds(y, groups, 5, 42)
    warnings.filterwarnings("ignore", message="n_quantiles .* is greater")
    warnings.filterwarnings("ignore", message="k=.* is greater than n_features")
    with threadpool_limits(limits=2):
        for name, (estimator, grid, view, tune) in specs.items():
            result_path = output / f"{name}_selection.json"
            if result_path.exists():
                report["models"][name] = json.loads(result_path.read_text())
                print(f"Resuming: retained frozen {name}", flush=True)
                continue
            columns = view_columns(X, view)
            inputs = X[columns]
            print(f"Accuracy search: {name}; {len(columns)} inputs; {len(ParameterGrid(grid))} configurations", flush=True)
            probabilities = np.empty(len(y))
            predicted = np.empty(len(y), dtype=int)
            selections = []
            for number, (tr, va) in enumerate(folds, 1):
                fitted, selected, rows = inner_search(estimator, grid, inputs.iloc[tr], y[tr], groups[tr], tune)
                p, decision = predict_selected(fitted, selected, inputs.iloc[va])
                probabilities[va], predicted[va] = p, decision
                selections.append(selected)
                pd.DataFrame(rows).to_json(output / f"{name}_fold{number}_search.json", orient="records", indent=2)
                print(f"  outer fold {number}/5: accuracy {accuracy_score(y[va],decision):.3f}; threshold {selected['threshold']:.2f}", flush=True)
            values = metrics(y, probabilities, predicted=predicted)
            result = {"nested_cv_accuracy": values["accuracy"], "nested_cv_balanced_accuracy": values["balanced_accuracy"],
                      "nested_cv_auc": values["roc_auc"], "nested_cv_metrics": values, "fold_selections": selections,
                      "input_view": view, "input_features": len(columns), "feature_names": columns}
            pd.DataFrame({"patient_id": groups, "y_true": y, "probability": probabilities,
                          "predicted": predicted}).to_csv(output / f"{name}_development_oof.csv", index=False)
            # Final parameters/threshold use all development OOF predictions, never holdout outcomes.
            fitted, selected, rows = inner_search(estimator, grid, inputs, y, groups, tune)
            result.update({"frozen_parameters": selected["parameters"], "threshold": selected["threshold"],
                           "prediction_rule": selected["prediction_rule"]})
            pd.DataFrame(rows).to_json(output / f"{name}_final_search.json", orient="records", indent=2)
            bundle = {"model": fitted, "feature_names": columns, "labels": list(TASK_LABELS["subtype"]),
                      "task": "subtype", "name": name, "threshold": selected["threshold"],
                      "prediction_rule": selected["prediction_rule"], "data_provenance": "real",
                      "training_patient_ids": groups.tolist(), "dataset_provenance": source_provenance}
            joblib.dump(bundle, output / f"{name}.joblib")
            result_path.write_text(json.dumps(result, indent=2))
            report["models"][name] = result
            (output / "selection_progress.json").write_text(json.dumps(report, indent=2))
            print(f"  {name}: nested development accuracy {values['accuracy']:.1%}, balanced accuracy {values['balanced_accuracy']:.1%}", flush=True)
    report["selected_model_by_development"] = max(report["models"], key=lambda n: (
        report["models"][n]["nested_cv_accuracy"], report["models"][n]["nested_cv_balanced_accuracy"],
        report["models"][n]["nested_cv_auc"]))
    report["development_target_met"] = clears_accuracy_target(
        report["models"][report["selected_model_by_development"]]["nested_cv_accuracy"],
        report["development_majority_accuracy"])
    (output / "selection.json").write_text(json.dumps(report, indent=2))
    (output / "splits.csv").write_bytes(split_path.read_bytes())
    print(f"Frozen development-selected winner: {report['selected_model_by_development']}", flush=True)
    if evaluate:
        return evaluate_accuracy(output)
    return report


def evaluate_accuracy(output):
    output = Path(output)
    if (output / "external_report.json").exists():
        raise ValueError("Accuracy experiment has already been scored; retain its original results.")
    selection = json.loads((output / "selection.json").read_text())
    if sha256(selection["dataset_path"]) != selection["dataset_sha256"]:
        raise ValueError("Dataset changed after model selection.")
    if sha256(output / "splits.csv") != selection["split_sha256"]:
        raise ValueError("Split changed after model selection.")
    dataset = load_dataset(selection["dataset_path"], feature_mode="clinical")
    splits = pd.read_csv(output / "splits.csv", dtype={"patient_id": str})
    use = dataset.frame.patient_id.isin(splits.loc[splits.split == "test", "patient_id"]).to_numpy()
    y, groups = dataset.y[use], dataset.groups[use]
    if len(groups) != selection["test_patients"] or set(groups) != set(selection["test_patient_ids_excluded_from_selection"]):
        raise ValueError("Held-out patients changed.")
    report = {**selection, "test_evaluated": True, "test_metrics": {}, "test_intervals": {},
              "task": "subtype", "labels": list(TASK_LABELS["subtype"]), "data_provenance": "real",
              "positive_class": TASK_LABELS["subtype"][1], "feature_names": dataset.X.columns.tolist(),
              "majority_baseline_accuracy": float(max(np.bincount(y)) / len(y)),
              "evaluation": f"Preserved {len(y)}-patient Radiogenomics holdout; inspected in earlier experiments, so this new assessment is exploratory.",
              "validated_superiority_over_source_paper": False}
    predictions = {}
    for name, result in report["models"].items():
        bundle = joblib.load(output / f"{name}.joblib")
        if set(groups) & set(bundle["training_patient_ids"]):
            raise ValueError("Model was fitted on held-out patients.")
        p, decision = predict_selected(bundle["model"], bundle, dataset.X.loc[use, bundle["feature_names"]])
        report["test_metrics"][name] = metrics(y, p, predicted=decision)
        report["test_intervals"][name] = bootstrap_intervals(y, p, predicted=decision)
        values = pd.DataFrame({"patient_id": groups, "y_true": y, "probability": p, "predicted": decision})
        values.to_csv(output / f"{name}_test_predictions.csv", index=False)
        predictions[name] = values
    winner = report["selected_model_by_development"]
    accuracy = report["test_metrics"][winner]["accuracy"]
    report["target_met"] = clears_accuracy_target(accuracy, report["majority_baseline_accuracy"])
    report["observed_above_target_models"] = [n for n, m in report["test_metrics"].items() if m["accuracy"] > 0.83]
    report["interpretation"] = (
        "Accuracy-focused parameters and thresholds were chosen from development data only. "
        "The preserved holdout was inspected during earlier model comparisons, so this follow-up result is exploratory. "
        "The paper's 83% used different patients; any numerical comparison is descriptive. "
        "Clinical/fusion models also require age, sex and smoking history. New independent patients are needed for confirmation.")
    (output / "external_report.json").write_text(json.dumps(report, indent=2))
    save_figures(predictions, output)
    pd.DataFrame(report["test_metrics"]).T.drop(columns="confusion_matrix").to_csv(output / "holdout_metrics.csv", index_label="model")
    print(json.dumps({"winner": winner, "target_met": report["target_met"],
                      "winner_metrics": report["test_metrics"][winner],
                      "all_accuracies": {n: m["accuracy"] for n, m in report["test_metrics"].items()}}, indent=2), flush=True)
    return report
