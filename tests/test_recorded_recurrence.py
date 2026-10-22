"""Real endpoint boundaries: no future-field leakage, frozen inference and UI/API."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from lungml.api import create_app
from lungml.dashboard_runtime import stream_inference, validate_queue
from lungml.explanations import explain_case
from lungml.quality import load_profile
from lungml.recurrence_classifier import (MODEL_NAMES, classification_metrics, predict_recorded_csv,
                                         run_recorded_recurrence, verify_recorded_models)
from lungml.reports import classification_runs
from lungml.service import discover_models, run_prediction, trusted_bundle


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("recorded-recurrence")
    rng = np.random.default_rng(13)
    count = 100
    signal = rng.normal(size=count)
    smoking = np.arange(count) % 3
    features = pd.DataFrame({"patient_id": [f"{i:04d}" for i in range(count)],
        "clinical_age_at_diagnosis": rng.integers(45, 85, count), "clinical_male": np.arange(count) % 2,
        "clinical_pack_years": np.where(smoking == 0, 0, rng.integers(5, 50, count)),
        "clinical_smoking_never": smoking == 0, "clinical_smoking_former": smoking == 1,
        "clinical_smoking_current": smoking == 2,
        "original_firstorder_signal": signal, "original_firstorder_noise": rng.normal(size=count),
        "Date of Recurrence": "2099-01-01", "Pathological N stage": "LEAKAGE_SENTINEL"})
    features_path = root / "features.csv"
    features.to_csv(features_path, index=False)
    pd.DataFrame({"patient_id": features.patient_id, "split": ["development"] * 80 + ["test"] * 20}).to_csv(
        features_path.with_suffix(".splits.csv"), index=False)
    clinical = pd.DataFrame({"Case ID": features.patient_id, "Recurrence": np.where(signal > 0, "yes", "no"),
        "Pathological T stage": np.where(signal > 0, "T2a", "T1a"),
        "Pathological N stage": np.where(np.arange(count) % 4 == 0, "N1", "N0"),
        "Histopathological Grade": np.where(np.arange(count) % 3 == 0, "G1 Well differentiated", "G2 Moderately differentiated"),
        "Lymphovascular invasion": np.where(np.arange(count) % 7 == 0, "Present", "Absent"),
        "Pleural invasion (elastic, visceral, or parietal)": np.where(np.arange(count) % 5 == 0, "Yes", "No"), "Date of Death": "2099-01-01"})
    clinical_path = root / "clinical.csv"
    clinical.to_csv(clinical_path, index=False)
    output = root / "artifacts" / "recurrence_recorded"
    report = run_recorded_recurrence(features_path, clinical_path, output)
    frame = pd.read_csv(output / "cohort.csv", dtype={"patient_id": str})
    return root, features_path, clinical_path, output, frame, report


def test_frozen_predictions_match_measured_test_scores_and_exact_explanations(recorded_run):
    _, _, _, output, frame, report = recorded_run
    selection = verify_recorded_models(output)
    assert selection["fixed_horizon_supported"] is False
    assert not classification_runs(output.parent)
    test = frame.loc[frame.split.eq("test")].sample(frac=1, random_state=3)
    for name in MODEL_NAMES:
        bundle = trusted_bundle({"path": output / f"{name}.joblib", "task": "recorded_recurrence"})
        assert not set(bundle["training_patient_ids"]) & set(test.patient_id)
        assert not any("date" in key.lower() or key == "label" for key in bundle["feature_names"])
        predicted = run_prediction(bundle, test)
        assert predicted.patient_id.tolist() == test.patient_id.tolist()
        assert predicted.p_recorded_recurrence.between(0, 1).all()
        assert "No fixed time horizon" in predicted.prediction_context.iloc[0]
        metrics = classification_metrics(test.label, predicted.p_recorded_recurrence, bundle["threshold"])
        for key, value in metrics.items():
            expected = report["test_metrics"][name][key]
            assert value == (pytest.approx(expected) if isinstance(value, float) else expected)
        explanation = explain_case(bundle, test.head(1), load_profile(output / f"{name}.joblib", bundle))
        assert explanation["reconstructed_probability"] == pytest.approx(predicted.p_recorded_recurrence.iloc[0])
        profile = load_profile(output / f"{name}.joblib", bundle)
        assert profile["training_patients"] == 80
    with pytest.raises(ValueError, match="cannot be overwritten"):
        run_recorded_recurrence(output=output)


def test_changes_to_test_labels_and_inputs_do_not_change_development_selection(recorded_run):
    root, features_path, clinical_path, output, frame, report = recorded_run
    changed = root / "changed"
    changed.mkdir()
    features = pd.read_csv(features_path, dtype={"patient_id": str})
    features.loc[80:, "original_firstorder_noise"] = 1e6
    new_features = changed / "features.csv"
    features.to_csv(new_features, index=False)
    new_features.with_suffix(".splits.csv").write_bytes(features_path.with_suffix(".splits.csv").read_bytes())
    clinical = pd.read_csv(clinical_path, dtype={"Case ID": str})
    clinical.loc[80:, "Recurrence"] = clinical.loc[80:, "Recurrence"].map({"yes": "no", "no": "yes"})
    new_clinical = changed / "clinical.csv"
    clinical.to_csv(new_clinical, index=False)
    changed_output = changed / "run"
    second = run_recorded_recurrence(new_features, new_clinical, changed_output)
    assert second["development_metrics"] == report["development_metrics"]
    assert second["thresholds"] == report["thresholds"]
    assert second["selected_model_by_development"] == report["selected_model_by_development"]
    for name in MODEL_NAMES:
        first_bundle = joblib.load(output / f"{name}.joblib")
        second_bundle = joblib.load(changed_output / f"{name}.joblib")
        np.testing.assert_array_equal(first_bundle["model"].steps[-1][1].coef_, second_bundle["model"].steps[-1][1].coef_)
        assert load_profile(output / f"{name}.joblib")["statistics"] == load_profile(changed_output / f"{name}.joblib")["statistics"]


def test_registry_api_exports_and_bad_input_rejection(recorded_run):
    root, _, _, output, frame, _ = recorded_run
    registry = discover_models(root / "artifacts")
    model_id = "recurrence_recorded/clinical_logistic"
    bundle = trusted_bundle(registry[model_id])
    case = frame.loc[frame.split.eq("test")].head(1)
    request = {"model_id": model_id, "records": json.loads(case.to_json(orient="records"))}
    with TestClient(create_app(root / "artifacts")) as client:
        response = client.post("/predict", json=request)
        assert response.status_code == 200, response.text
        assert response.json()["task"] == "recorded_recurrence"
        assert client.post("/explain", json=request).json()["method"] == "linear_log_odds"
        page = client.post("/report", json=request)
        assert page.status_code == 200
        assert "No fixed time horizon" in page.text
        assert client.get("/models/recurrence_recorded/clinical_logistic").json()["horizons_days"] is None
        assert client.get("/readiness").json()["extensions"][-1]["state"] == "trained_exploratory"
        request["records"][0]["clinical_pathological_n"] = 8
        assert client.post("/predict", json=request).status_code == 422
    duplicate = pd.concat([case, case], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        list(stream_inference(bundle, duplicate))
    bad = case.copy()
    bad["clinical_smoking_never"] = 1
    bad["clinical_smoking_former"] = 1
    with pytest.raises(ValueError, match="exactly one"):
        validate_queue(bundle, bad)
    source = root / "inference.csv"
    case.to_csv(source, index=False)
    result = predict_recorded_csv(output / "clinical_logistic.joblib", source, root / "predictions.csv")
    assert result.p_recorded_recurrence.iloc[0] == pytest.approx(run_prediction(bundle, case).p_recorded_recurrence.iloc[0])
    with pytest.raises(ValueError, match="cannot be overwritten"):
        predict_recorded_csv(output / "clinical_logistic.joblib", source, source)


def test_modified_model_is_rejected_before_loading(recorded_run, tmp_path):
    import shutil
    _, _, _, output, _, _ = recorded_run
    copied = tmp_path / "copy"
    shutil.copytree(output, copied)
    path = copied / "clinical_logistic.joblib"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed or is missing"):
        trusted_bundle({"path": path, "task": "recorded_recurrence"})


def test_small_cohort_fit_keeps_test_patients_separate_and_reports_both_scores(recorded_run, tmp_path):
    from lungml.recurrence_classifier import run_training_fit_experiment
    _, _, _, source, frame, original_report = recorded_run
    output = tmp_path / "training_fit"
    report = run_training_fit_experiment(source, output, patients=20)
    training_ids, test_ids = set(report["training_patient_ids"]), set(report["test_patient_ids"])
    assert len(training_ids) == 20
    assert training_ids <= set(frame.loc[frame.split.eq("development"), "patient_id"])
    assert test_ids == set(frame.loc[frame.split.eq("test"), "patient_id"])
    assert not training_ids & test_ids
    assert report["training_metrics"]["accuracy"] == 1.
    bundle = joblib.load(output / "decision_tree.joblib")
    test = frame.loc[frame.patient_id.isin(test_ids)]
    predictions = run_prediction(bundle, test)
    actual = classification_metrics(test.label, predictions.p_recorded_recurrence)
    assert actual["accuracy"] == report["test_metrics"]["accuracy"]
    assert actual["confusion_matrix"] == report["test_metrics"]["confusion_matrix"]
    assert "Training accuracy" in predictions.prediction_context.iloc[0]
    assert json.loads((source / "report.json").read_text()) == original_report
    assert not classification_runs(tmp_path)
    with pytest.raises(ValueError, match="cannot be overwritten"):
        run_training_fit_experiment(source, output, patients=20)
    with pytest.raises(ValueError, match="at least 12"):
        run_training_fit_experiment(source, tmp_path / "invalid", patients=5)
