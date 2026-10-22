import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lungml.accuracy_experiment import (choose_threshold, clears_accuracy_target, evaluate_accuracy,
                                       inner_search, prepare_clinical, prepare_expanded_clinical, run_accuracy)
from lungml.data import CLINICAL_FEATURES, load_dataset
from lungml.reports import load_report


def fixture_data(tmp_path):
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(rng.normal(size=(40, 10)), columns=[f"original_firstorder_{i}" for i in range(10)])
    frame["patient_id"] = [f"P{i:02d}" for i in range(len(frame))]
    frame["sample_id"] = frame.patient_id + "-tumour"
    frame["label"] = np.where(frame.original_firstorder_0 > 0, "squamous_cell_carcinoma", "adenocarcinoma")
    path = tmp_path / "source.csv"
    frame.to_csv(path, index=False)
    splits = frame[["patient_id", "sample_id", "label"]].copy()
    splits["split"] = np.where(np.arange(40) < 30, "development", "test")
    splits.to_csv(path.with_suffix(".splits.csv"), index=False)
    path.with_suffix(".provenance.json").write_text("{}")
    clinical = pd.DataFrame({"Case ID": frame.patient_id, "Age at Histological Diagnosis": 60,
                             "Gender": "Male", "Pack Years": "Not Collected", "Smoking status": "Former",
                             "Histology": frame.label.map({"adenocarcinoma": "Adenocarcinoma",
                                                          "squamous_cell_carcinoma": "Squamous cell carcinoma"}),
                             "Survival Status": frame.label,
                             "Pathological T stage": frame.label})
    clinical.loc[0, "Pack Years"] = "0"
    clinical_path = tmp_path / "clinical.csv"
    clinical.to_csv(clinical_path, index=False)
    return path, clinical_path


def test_clinical_join_preserves_patients_split_and_excludes_diagnosis(tmp_path):
    source, clinical = fixture_data(tmp_path)
    output = prepare_clinical(source, clinical, tmp_path / "joined.csv")
    data = load_dataset(output, feature_mode="clinical")
    assert set(CLINICAL_FEATURES) <= set(data.X)
    assert not {"Histology", "Survival Status", "Pathological T stage"} & set(data.frame)
    assert output.with_suffix(".splits.csv").read_bytes() == source.with_suffix(".splits.csv").read_bytes()
    assert data.X.clinical_pack_years.iloc[0] == 0
    assert data.X.clinical_pack_years.iloc[1:].isna().all()
    frame = pd.read_csv(output)
    frame["clinical_histology_code"] = np.arange(40)
    frame.to_csv(output, index=False)
    assert "clinical_histology_code" not in load_dataset(output, feature_mode="clinical").X


def test_threshold_selection_can_improve_accuracy_without_changing_scores():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.1, 0.2, 0.6, 0.8])
    threshold, accuracy, balanced = choose_threshold(y, p)
    assert threshold == 0.65
    assert accuracy == balanced == 1


def test_majority_prediction_does_not_count_as_accuracy_improvement():
    assert not clears_accuracy_target(0.837, 0.837)
    assert not clears_accuracy_target(0.83, 0.8)
    assert clears_accuracy_target(31 / 35, 28 / 35)


def test_expanded_clinical_patients_are_development_only_and_duplicates_are_valid(tmp_path):
    source, clinical = fixture_data(tmp_path)
    records = pd.read_csv(clinical)
    additional = records.iloc[:4].copy()
    additional["Case ID"] = [f"P{i}" for i in range(40, 44)]
    records = pd.concat([records, additional], ignore_index=True)
    records.to_csv(clinical, index=False)
    joined = prepare_expanded_clinical(source, clinical, tmp_path / "expanded.csv")
    cohort = load_dataset(joined, feature_mode="clinical")
    assert cohort.X.shape == (44, 6)
    original = pd.read_csv(source.with_suffix(".splits.csv"))
    expanded = pd.read_csv(joined.with_suffix(".splits.csv"))
    assert set(expanded.loc[expanded.split == "test", "patient_id"]) == set(original.loc[original.split == "test", "patient_id"])
    assert expanded.loc[expanded.patient_id.isin(additional["Case ID"]), "split"].eq("development").all()
    assert not {"Histology", "Survival Status"} & set(cohort.frame)


def test_nested_accuracy_runner_excludes_holdout_and_report_preserves_threshold(tmp_path):
    source, clinical = fixture_data(tmp_path)
    data = prepare_clinical(source, clinical, tmp_path / "joined.csv")
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500))])
    specifications = {"test_logistic": (model, {"model__C": [0.1, 1]}, "original", True)}
    output = tmp_path / "run"
    result = run_accuracy(data, output, evaluate=True, specs=specifications)
    import joblib
    bundle = joblib.load(output / "test_logistic.joblib")
    assert not set(bundle["training_patient_ids"]) & set(result["test_patient_ids_excluded_from_selection"])
    assert len(bundle["training_patient_ids"]) == 30
    assert result["previously_inspected_holdout"] is True
    assert result["validated_superiority_over_source_paper"] is False
    loaded = load_report(output)
    assert loaded["models"]["test_logistic"]["prediction_rule"] == "probability_threshold"
    assert loaded["models"]["test_logistic"]["threshold"] == bundle["threshold"]
    with pytest.raises(ValueError, match="already been scored"):
        evaluate_accuracy(output)
    frozen = json.loads((output / "selection.json").read_text())
    assert frozen["test_evaluated"] is False


def test_inner_selection_is_invariant_to_dataframe_index():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(30, 3)))
    y = (X[0] > 0).astype(int).to_numpy()
    groups = np.array([f"P{i}" for i in range(30)])
    estimator = LogisticRegression(max_iter=500)
    _, selected, _ = inner_search(estimator, {"C": [0.1, 1]}, X, y, groups)
    shifted = X.copy()
    shifted.index = np.arange(100, 130)
    _, other, _ = inner_search(estimator, {"C": [0.1, 1]}, shifted, y, groups)
    assert selected == other
