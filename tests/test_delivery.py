"""Delivery boundaries: training-only references, exact explanations, API and export."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lungml.api import create_app
from lungml.case_report import build_case_report
from lungml.explanations import explain_case
from lungml.external_validation import validate_external
from lungml.improve import CorrelationPruner
from lungml.predict import predict_frame
from lungml.quality import assess_inputs, create_profile, load_profile, sha256


@pytest.fixture
def delivered(tmp_path):
    rng = np.random.default_rng(71)
    X = pd.DataFrame(rng.normal(size=(60, 12)), columns=[f"original_feature_{i}" for i in range(12)])
    X["original_feature_11"] = X.original_feature_0  # Exercise exact selector/name tracking.
    y = (X.original_feature_0 + X.original_feature_1 > 0).astype(int)
    frame = X.assign(patient_id=[f"{i:03d}" for i in range(60)], label=np.where(y, "squamous_cell_carcinoma", "adenocarcinoma"))
    model = Pipeline([("impute", SimpleImputer(keep_empty_features=True)), ("prune", CorrelationPruner()),
        ("scale", StandardScaler()), ("select", SelectKBest(k=5)), ("model", LogisticRegression())]).fit(X.iloc[:40], y.iloc[:40])
    bundle = {"model": model, "name": "linear", "feature_names": X.columns.tolist(), "task": "subtype",
              "labels": ["adenocarcinoma", "squamous_cell_carcinoma"], "threshold": .7,
              "data_provenance": "synthetic", "training_patient_ids": frame.patient_id[:40].tolist()}
    root = tmp_path / "artifacts"
    run = root / "fixture"
    run.mkdir(parents=True)
    path = run / "linear.joblib"
    joblib.dump(bundle, path)
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    (run / "report.json").write_text(json.dumps({"task": "subtype", "models": {"linear": {}},
        "selected_model_by_cv": "linear", "dataset_path": str(source)}))
    frame[["patient_id"]].assign(split=["development"] * 40 + ["test"] * 20).to_csv(run / "splits.csv", index=False)
    return bundle, frame, path, source, root


def test_profiles_use_training_patients_only_and_bind_to_model(delivered):
    bundle, frame, path, source, _ = delivered
    first = create_profile(path, source)
    frame.loc[40:, bundle["feature_names"]] = 100000
    frame.to_csv(source, index=False)
    second = create_profile(path, source)
    assert first["statistics"] == second["statistics"]
    assert first["training_patients"] == 40
    assert first["source_sha256"] != second["source_sha256"]
    assert load_profile(path, bundle)["model_sha256"] == sha256(path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="does not match"):
        load_profile(path, bundle)


def test_profiles_reject_unmatched_saved_source(delivered):
    bundle, _, path, source, _ = delivered
    bundle["dataset_provenance"] = {"output_sha256": "wrong"}
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="dataset hashes"):
        create_profile(path, source)
    with pytest.raises(ValueError, match="overwrite"):
        create_profile(path, source, output=path)


def test_quality_flags_outliers_and_missing_without_claiming_uncertainty(delivered):
    bundle, frame, path, source, _ = delivered
    profile = create_profile(path, source)
    case = frame.iloc[[40]].copy()
    case.loc[:, bundle["feature_names"][:3]] = 1e6
    case.loc[:, bundle["feature_names"][4]] = np.nan
    quality = assess_inputs(bundle, case, profile)[0]
    assert quality["outside_reference_count"] >= 3
    assert quality["missing_feature_count"] == 1
    assert quality["input_review_suggested"]
    unknown = assess_inputs(bundle, case)[0]
    assert unknown["outside_reference_fraction"] is None
    assert not unknown["reference_available"]


def test_linear_explanation_reconstructs_probability_and_feature_names(delivered):
    bundle, frame, _, _, _ = delivered
    case = frame.iloc[[42]]
    result = explain_case(bundle, case, limit=3)
    assert result["method"] == "linear_log_odds"
    expected = predict_frame(bundle, case).p_squamous_cell_carcinoma.iloc[0]
    assert result["reconstructed_probability"] == pytest.approx(expected, abs=1e-12)
    assert result["baseline"] + result["displayed_contribution_sum"] + result["remaining_contribution_sum"] == pytest.approx(result["total"])
    assert all(r["feature"] in bundle["feature_names"] for r in result["contributions"])
    assert "original_feature_11" not in [r["feature"] for r in result["contributions"]]
    with pytest.raises(ValueError, match="exactly one"):
        explain_case(bundle, frame.head(2))


def test_feature_replacement_uses_training_reference(delivered):
    bundle, frame, path, source, _ = delivered
    from sklearn.ensemble import RandomForestClassifier
    reference = create_profile(path, source)
    bundle["model"] = RandomForestClassifier(n_estimators=5, random_state=42).fit(
        frame[bundle["feature_names"]][:40], frame.label.eq(bundle["labels"][1])[:40])
    result = explain_case(bundle, frame.iloc[[42]], reference)
    assert result["method"] == "median_replacement_sensitivity"
    assert result["features_checked"] == 12
    assert explain_case(bundle, frame.iloc[[42]])["method"] == "unavailable"


@pytest.mark.parametrize("probabilities", [[[.2, .2]], [[np.nan, .5]], [[-.1, 1.1]]])
def test_invalid_model_probabilities_are_rejected(delivered, probabilities):
    bundle, frame, _, _, _ = delivered
    class InvalidModel:
        def predict_proba(self, X):
            return probabilities
    bundle["model"] = InvalidModel()
    with pytest.raises(ValueError, match="invalid binary probabilities"):
        predict_frame(bundle, frame.head(1))


def test_report_escapes_ids_and_preserves_reproducible_payload(delivered):
    bundle, frame, path, _, _ = delivered
    case = frame.iloc[[42]].copy()
    case["patient_id"] = '</script><script>alert("x")</script>'
    predictions = predict_frame(bundle, case)
    report = build_case_report(bundle, case, predictions, sha256(path), explanation=explain_case(bundle, case))
    assert '<script>alert("x")</script>' not in report
    assert "\\u003c/script\\u003e" in report
    assert sha256(path) in report
    assert "window.print()" in report
    predictions["patient_id"] = "different"
    with pytest.raises(ValueError, match="matching prediction IDs"):
        build_case_report(bundle, case, predictions, sha256(path))


def test_api_matches_local_inference_and_enforces_registry(delivered):
    bundle, frame, path, source, root = delivered
    create_profile(path, source)
    with TestClient(create_app(root)) as client:
        assert client.get("/health").json()["status"] == "ok"
        models = client.get("/models").json()["models"]
        assert models == [{"model_id": "fixture/linear", "task": "subtype", "selected": True}]
        schema = client.get("/models/fixture/linear").json()
        assert schema["feature_names"] == bundle["feature_names"]
        request = {"model_id": "fixture/linear", "records": frame.iloc[40:43].to_dict("records")}
        response = client.post("/predict", json=request)
        assert response.status_code == 200, response.text
        np.testing.assert_allclose([r["p_squamous_cell_carcinoma"] for r in response.json()["predictions"]],
                                   predict_frame(bundle, frame.iloc[40:43]).p_squamous_cell_carcinoma)
        assert response.json()["elapsed_ms"] > 0
        request["records"] = request["records"][:1]
        assert client.post("/explain", json=request).json()["method"] == "linear_log_odds"
        assert client.post("/report", json=request).headers["content-type"].startswith("text/html")
        request["model_id"] = str(path)
        assert client.post("/predict", json=request).status_code == 404
        request["model_id"] = "fixture/linear"
        request["records"][0].pop(bundle["feature_names"][0])
        assert client.post("/predict", json=request).status_code == 422
        assert all(not r["source_verified"] for r in client.get("/readiness").json()["extensions"])
        assert client.post("/predict", json={"model_id": "fixture/linear", "records": []}).status_code == 422


def test_api_validates_whole_batch_before_model_call(delivered, monkeypatch):
    _, frame, _, _, root = delivered
    request = {"model_id": "fixture/linear", "records": frame.iloc[40:43].to_dict("records")}
    request["records"][2]["patient_id"] = request["records"][0]["patient_id"]
    monkeypatch.setattr("lungml.service.predict_frame", lambda *args: pytest.fail("Invalid queue reached prediction"))
    with TestClient(create_app(root)) as client:
        assert client.post("/predict", json=request).status_code == 422


def test_external_validation_rejects_seen_patients_and_never_retunes(delivered, tmp_path):
    bundle, frame, path, source, _ = delivered
    with pytest.raises(ValueError, match="overlaps"):
        validate_external(path, source, tmp_path / "invalid", "Fixture provenance")
    assert not (tmp_path / "invalid").exists()
    external = frame.iloc[40:].copy()
    external.patient_id = "new-" + external.patient_id
    data = tmp_path / "external.csv"
    external.to_csv(data, index=False)
    before = sha256(path)
    report = validate_external(path, data, tmp_path / "validation", "Synthetic software test; not independent clinical evidence")
    assert report["frozen_threshold"] == .7
    assert report["patients"] == 20
    assert sha256(path) == before
    assert report["metrics"]["accuracy"] == pytest.approx(
        (predict_frame(bundle, external).prediction.to_numpy() == external.label.to_numpy()).mean())
    with pytest.raises(ValueError, match="cannot be overwritten"):
        validate_external(path, data, tmp_path / "validation", "Fixture")
