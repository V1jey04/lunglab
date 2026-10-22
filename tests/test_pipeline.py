import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from lungml.data import load_dataset
from lungml.evaluation import metrics, patient_predictions
from lungml.models import AdaptiveSMOTE, catboost_model, paper_mlp
from lungml.predict import predict_frame
from lungml.train import patient_folds


@pytest.fixture
def feature_frame():
    X, y = make_classification(n_samples=60, n_features=12, n_informative=6,
                               n_redundant=2, random_state=7)
    frame = pd.DataFrame(X, columns=[f"original_firstorder_test{i}" for i in range(12)])
    frame.insert(0, "label", np.where(y, "squamous_cell_carcinoma", "adenocarcinoma"))
    frame.insert(0, "patient_id", [f"PATIENT-{i}" for i in range(60)])
    return frame


def test_clinical_fields_cannot_leak_into_features(tmp_path, feature_frame):
    feature_frame["HistologyCode"] = (feature_frame.label == "squamous_cell_carcinoma").astype(int)
    feature_frame["Survival.time"] = np.arange(len(feature_frame))
    path = tmp_path / "features.csv"
    feature_frame.to_csv(path, index=False)
    dataset = load_dataset(path)
    assert dataset.X.shape == (60, 12)
    assert "HistologyCode" not in dataset.X


def test_unknown_label_is_rejected(tmp_path, feature_frame):
    feature_frame.loc[0, "label"] = "normal"
    path = tmp_path / "bad.csv"
    feature_frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="requires both labels"):
        load_dataset(path)


def test_duplicate_patient_features_are_rejected(tmp_path, feature_frame):
    feature_frame.loc[1, feature_frame.filter(like="original_").columns] = feature_frame.loc[
        0, feature_frame.filter(like="original_").columns].to_numpy()
    path = tmp_path / "duplicate.csv"
    feature_frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="different patient IDs"):
        load_dataset(path)


def test_multiple_nodules_never_cross_patient_fold_boundaries():
    groups = np.repeat([f"P{i}" for i in range(30)], 3)
    y = np.repeat(np.arange(30) % 2, 3)
    folds = patient_folds(y, groups, 5)
    seen = []
    for tr, va in folds:
        assert not set(groups[tr]) & set(groups[va])
        assert len(np.unique(y[va])) == 2
        seen.extend(va)
    assert sorted(seen) == list(range(90))


def test_smote_handles_small_minority_without_touching_input():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.array([0] * 7 + [1] * 3)
    original = X.copy()
    resampled, labels = AdaptiveSMOTE().fit_resample(X, y)
    assert np.bincount(labels).tolist() == [7, 7]
    assert np.array_equal(X, original)
    assert resampled.shape[0] == 14


@pytest.mark.parametrize("builder", [paper_mlp, catboost_model])
def test_training_serialization_and_schema_checked_prediction(tmp_path, feature_frame, builder):
    import joblib
    from lungml.predict import load_bundle
    X = feature_frame.filter(like="original_")
    y = (feature_frame.label == "squamous_cell_carcinoma").astype(int)
    model = builder().fit(X, y)
    bundle = {"model": model, "feature_names": X.columns.tolist(),
              "labels": ["adenocarcinoma", "squamous_cell_carcinoma"],
              "task": "subtype", "threshold": 0.5, "data_provenance": "synthetic"}
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    result = predict_frame(load_bundle(path), feature_frame.head(3))
    assert len(result) == 3
    assert np.allclose(result.filter(like="p_").sum(axis=1), 1)
    with pytest.raises(ValueError, match="Missing required"):
        predict_frame(bundle, feature_frame.drop(columns=X.columns[0]))


def test_metrics_aggregate_patients_and_define_positive_class():
    values = patient_predictions(["A", "A", "B", "C"], [0, 0, 1, 1], [0.1, 0.3, 0.8, 0.2])
    assert len(values) == 3
    assert values.loc[values.patient_id == "A", "probability"].iloc[0] == pytest.approx(0.2)
    result = metrics(values.y_true, values.probability)
    assert result["confusion_matrix"] == [[1, 0], [1, 1]]
    assert result["specificity"] == 1
    assert result["sensitivity"] == 0.5


def test_review_flag_follows_threshold_selected_class(feature_frame):
    class FixedProbability:
        def predict_proba(self, X):
            return np.tile([0.28, 0.72], (len(X), 1))
    bundle = {"model": FixedProbability(), "feature_names": feature_frame.filter(like="original_").columns.tolist(),
              "labels": ["adenocarcinoma", "squamous_cell_carcinoma"],
              "threshold": 0.73, "data_provenance": "synthetic"}
    result = predict_frame(bundle, feature_frame.head(1))
    assert result.prediction.iloc[0] == "adenocarcinoma"
    assert result.review_suggested.iloc[0]
