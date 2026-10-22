import hashlib
import io
import json
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest

from lungml.ct_transfer import ArrayUnpickler, normalize_crop, read_crops
from lungml.data import load_dataset
from lungml.evaluation import metrics
from lungml.improve import CorrelationPruner
from lungml.multiview import RadiomicsViewEnsemble
from lungml.predict import predict_frame
from lungml.reports import load_report
from lungml.rich_experiment import evaluate_frozen


def test_richer_feature_modes_exclude_diagnosis_and_patient_metadata(tmp_path):
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.normal(size=(30, 10)), columns=[f"original_firstorder_{i}" for i in range(10)])
    frame["patient_id"] = [f"P{i}" for i in range(30)]
    frame["label"] = np.where(np.arange(30) % 2, "squamous_cell_carcinoma", "adenocarcinoma")
    for column in ["wavelet-LLL_firstorder_Mean", "deep_resnet18_mean_000",
                   "diagnostics_Configuration_Settings", "HistologyCode", "Survival"]:
        frame[column] = np.arange(30)
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    original = load_dataset(path)
    rich = load_dataset(path, feature_mode="all")
    hybrid = load_dataset(path, feature_mode="hybrid")
    assert original.X.shape[1] == 10
    assert rich.X.shape[1] == 11
    assert hybrid.X.shape[1] == 12
    assert not {"HistologyCode", "Survival", "diagnostics_Configuration_Settings"} & set(hybrid.X)


def test_pruner_preserves_training_selection_on_changed_validation_correlations():
    x = np.arange(12, dtype=float)
    training = np.column_stack([x, 2 * x, np.ones(12), x % 2])
    pruner = CorrelationPruner().fit(training)
    validation = training.copy()
    validation[:, 1] = x % 3
    validation[:, 2] = x ** 2
    assert pruner.indices_.tolist() == [0, 3]
    np.testing.assert_array_equal(pruner.transform(validation), validation[:, [0, 3]])


def test_multiview_pipeline_handles_small_views_and_serializes(tmp_path):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(40, 18)), columns=[
        f"{family}_firstorder_{i}" for family in ["original", "wavelet-LLL", "gradient"] for i in range(6)])
    y = (X.original_firstorder_0 > 0).astype(int)
    model = RadiomicsViewEnsemble(k=20).fit(X, y)
    path = tmp_path / "ensemble.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)


def test_native_prediction_is_used_for_metrics_and_saved_bundle_inference():
    class NativeClassifier:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8], [0.7, 0.3]])
        def predict(self, X):
            return np.array([0, 1])
    result = metrics([0, 1], [0.8, 0.3], predicted=[0, 1])
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]
    assert result["precision"] == result["f1"] == result["accuracy"] == 1
    bundle = {"model": NativeClassifier(), "feature_names": ["original_firstorder_Mean"],
              "labels": ["adenocarcinoma", "squamous_cell_carcinoma"], "threshold": 0.5,
              "prediction_rule": "native_predict", "data_provenance": "synthetic"}
    output = predict_frame(bundle, pd.DataFrame({"original_firstorder_Mean": [1, 2]}))
    assert output.prediction.tolist() == bundle["labels"]
    assert output.review_suggested.all()


def test_downloaded_pickle_cannot_resolve_arbitrary_globals():
    with pytest.raises(pickle.UnpicklingError, match="Unsupported pickle global"):
        ArrayUnpickler(io.BytesIO(pickle.dumps(eval))).load()
    array = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(ArrayUnpickler(io.BytesIO(pickle.dumps(array))).load(), array)


def test_crop_loader_verifies_hash_shape_and_per_case_scaling(tmp_path):
    array = np.arange(224 * 224 * 5, dtype=np.float32).reshape(1, 224, 224, 5)
    path = tmp_path / "crop.pkl"
    path.write_bytes(pickle.dumps(array))
    with pytest.raises(ValueError, match="Incomplete or modified"):
        read_crops(path, "wrong", 1)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    result = read_crops(path, md5, 1)
    with pytest.raises(ValueError, match="Unexpected patient crop shape"):
        read_crops(path, md5, 2)
    crop = normalize_crop(result[0])
    assert crop.min() == 0 and crop.max() == 1
    np.testing.assert_allclose(crop, normalize_crop(2 * result[0] + 5), atol=1e-6)


def test_holdout_cannot_be_rescored_or_split_manifest_changed(tmp_path):
    (tmp_path / "external_report.json").write_text("{}")
    with pytest.raises(ValueError, match="already been evaluated"):
        evaluate_frozen("unused.csv", tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    data = other / "data.csv"
    data.write_text("unused")
    (other / "selection.json").write_text(json.dumps({
        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "split_sha256": "locked"}))
    (other / "splits.csv").write_text("changed")
    with pytest.raises(ValueError, match="split manifest changed"):
        evaluate_frozen(data, other)


def test_dashboard_reads_frozen_experiment_without_changing_selected_model(tmp_path):
    report = {"models": {"a": {"nested_cv_auc": 0.6, "nested_cv_balanced_accuracy": 0.55}},
              "test_metrics": {"a": {"accuracy": 0.5}}, "selected_model_by_development": "a"}
    path = tmp_path / "external_report.json"
    path.write_text(json.dumps(report))
    loaded = load_report(tmp_path)
    assert loaded["selected_model_by_cv"] == "a"
    assert loaded["models"]["a"]["test_metrics"]["accuracy"] == 0.5
    assert "cv_selection_auc" not in json.loads(path.read_text())["models"]["a"]
