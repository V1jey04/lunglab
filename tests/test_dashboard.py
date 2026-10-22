import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest

from lungml.dashboard_runtime import input_signature, preserved_cases, stream_inference
from lungml.predict import predict_frame


@pytest.fixture
def model_and_cases():
    rng = np.random.default_rng(28)
    frame = pd.DataFrame(rng.normal(size=(36, 10)), columns=[f"original_firstorder_{i}" for i in range(10)])
    y = (frame.iloc[:, 0] > 0).astype(int)
    model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression()).fit(frame, y)
    frame["patient_id"] = [f"{i:04d}" for i in range(36)]
    frame["sample_id"] = [f"sample-{i}" for i in range(36)]
    bundle = {"model": model, "feature_names": frame.filter(like="original_").columns.tolist(),
              "labels": ["adenocarcinoma", "squamous_cell_carcinoma"], "task": "subtype",
              "threshold": 0.5, "data_provenance": "synthetic"}
    return bundle, frame


def test_streamed_calls_agree_with_batch_predictions_and_preserve_shuffled_ids(model_and_cases):
    bundle, frame = model_and_cases
    frame = frame.iloc[[8, 1, 5]]
    events = list(stream_inference(bundle, frame))
    assert [event.stage for event in events] == ["validated", "prediction", "prediction", "prediction", "complete"]
    assert [event.completed for event in events] == [0, 1, 2, 3, 3]
    assert [event.patient_id for event in events[1:-1]] == frame.patient_id.tolist()
    pd.testing.assert_frame_equal(events[-1].result, predict_frame(bundle, frame))
    assert all(event.case_ms >= 0 for event in events[1:-1])
    assert events[-1].compute_ms == pytest.approx(sum(event.case_ms for event in events[1:-1]))


@pytest.mark.parametrize("invalid", ["missing_column", "too_many_missing", "duplicate_id", "infinity", "empty"])
def test_whole_queue_is_validated_before_any_model_call(model_and_cases, invalid):
    bundle, frame = model_and_cases
    frame = frame.head(3).copy()
    if invalid == "missing_column":
        frame = frame.drop(columns=bundle["feature_names"][0])
    elif invalid == "too_many_missing":
        frame.loc[2, bundle["feature_names"][:3]] = np.nan
    elif invalid == "duplicate_id":
        frame.loc[2, "patient_id"] = frame.patient_id.iloc[0]
    elif invalid == "infinity":
        frame.loc[2, bundle["feature_names"][0]] = np.inf
    else:
        frame = frame.iloc[:0]
    class ForbiddenModel:
        def predict_proba(self, X):
            pytest.fail("Inference was called on a queue that failed validation")
    bundle["model"] = ForbiddenModel()
    with pytest.raises(ValueError):
        list(stream_inference(bundle, frame))


def test_example_queue_uses_id_join_and_only_test_patients(tmp_path, model_and_cases):
    _, frame = model_and_cases
    source = tmp_path / "features.csv"
    frame.sample(frac=1, random_state=2).to_csv(source, index=False)
    pd.DataFrame({"patient_id": frame.patient_id, "split": ["development"] * 24 + ["test"] * 12}).to_csv(
        tmp_path / "splits.csv", index=False)
    result = preserved_cases(tmp_path, {"dataset_path": str(source)})
    assert set(result.patient_id) == set(frame.patient_id.iloc[24:])
    assert result.patient_id.str.len().eq(4).all()


def test_signature_changes_with_input_or_model(tmp_path, model_and_cases):
    _, frame = model_and_cases
    path = tmp_path / "model.joblib"
    path.write_bytes(b"initial")
    first = input_signature(path, frame)
    changed = frame.copy()
    changed.iloc[0, 0] += 1
    assert input_signature(path, changed) != first
    assert input_signature(path, frame.sample(frac=1, random_state=3)) != first
    path.write_bytes(b"updated model")
    assert input_signature(path, frame) != first


def test_live_ui_runs_keeps_results_on_inspection_and_clears_them_for_new_inputs(
        tmp_path, monkeypatch, model_and_cases):
    bundle, frame = model_and_cases
    run, data = tmp_path / "artifacts" / "demo", tmp_path / "data"
    run.mkdir(parents=True)
    data.mkdir()
    source = data / "features.csv"
    frame.to_csv(source, index=False)
    pd.DataFrame({"patient_id": frame.patient_id, "split": ["development"] * 24 + ["test"] * 12}).to_csv(
        run / "splits.csv", index=False)
    joblib.dump(bundle, run / "logistic.joblib")
    (run / "report.json").write_text(json.dumps({
        "task": "subtype", "labels": bundle["labels"], "positive_class": bundle["labels"][1],
        "data_provenance": "synthetic", "selected_model_by_cv": "logistic", "development_patients": 24,
        "test_patients": 12, "feature_names": bundle["feature_names"], "dataset_path": str(source),
        "models": {"logistic": {"cv_selection_auc": 0.7, "test_metrics": {"accuracy": 0.75,
        "balanced_accuracy": 0.75, "confusion_matrix": [[4, 2], [1, 5]]}}}, "interpretation": "Synthetic fixture.",
    }))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(app_path)).run(timeout=15)
    assert not app.exception
    assert "live_run" not in app.session_state
    app.button(key="run_live").click().run(timeout=15)
    assert not app.exception
    assert not app.error
    saved = app.session_state["live_run"]
    pd.testing.assert_frame_equal(saved["result"], predict_frame(bundle, frame.iloc[24:29]))
    app.selectbox(key="inspect_case").select("0025").run(timeout=15)
    assert app.session_state["live_run"]["compute_ms"] == saved["compute_ms"]
    app.radio(key="input_source").set_value("Upload feature CSV").run(timeout=15)
    assert not app.exception
    assert "live_run" not in app.session_state
    assert app.button(key="run_live").disabled
