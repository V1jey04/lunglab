import joblib
import json
import numpy as np
import pandas as pd
from sklearn.base import clone

from lungml.refinement import refinement_specs, summarize_refinement


def toy_radiomics():
    rng = np.random.default_rng(16)
    values = rng.normal(size=(60, 24))
    columns = [f"original_firstorder_{i}" for i in range(12)] + [f"wavelet-LLL_firstorder_{i}" for i in range(12)]
    frame = pd.DataFrame(values, columns=columns)
    return frame, (frame.iloc[:, 0] + 0.4 * frame.iloc[:, 15] > 0).astype(int).to_numpy()


def test_bagged_pipeline_accepts_array_subsamples_and_serializes(tmp_path):
    X, y = toy_radiomics()
    model = clone(refinement_specs(tmp_path)["bagged_rich_logistic"][0]).set_params(
        n_estimators=3, estimator__select__k=10, estimator__model__C=0.1)
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    assert len(model.estimators_) == 3
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    target = tmp_path / "bagged.joblib"
    joblib.dump(model, target)
    np.testing.assert_allclose(joblib.load(target).predict_proba(X), probabilities)


def test_stacking_keeps_preprocessing_inside_each_base_estimator(tmp_path):
    X, y = toy_radiomics()
    model = clone(refinement_specs(tmp_path)["stacked_radiomics"][0]).set_params(
        logistic__select__k=10, svm__select__k=10, trees__model__n_estimators=20)
    model.fit(X, y)
    assert all("impute" in estimator.named_steps for estimator in model.estimators_)
    assert model.named_estimators_.trees.named_steps["view"].transform(X).shape[1] == 12
    assert model.final_estimator_.named_steps["model"].n_features_in_ == 3
    assert model.predict_proba(X).shape == (60, 2)


def test_refinement_summary_respects_frozen_winner_instead_of_best_test_score(tmp_path):
    baseline_dir, candidate_dir = tmp_path / "baseline", tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline = {"dataset_sha256": "same", "split_sha256": "same",
                "selected_model_by_development": "baseline", "models": {
                    "baseline": {"nested_cv_accuracy": 0.81, "nested_cv_balanced_accuracy": 0.56, "nested_cv_auc": 0.7}},
                "test_metrics": {"baseline": {"accuracy": 0.886}}}
    candidate = {"dataset_sha256": "same", "split_sha256": "same",
                 "selected_model_by_development": "selected", "models": {
                     "selected": {"nested_cv_accuracy": 0.82, "nested_cv_balanced_accuracy": 0.58, "nested_cv_auc": 0.72},
                     "other": {"nested_cv_accuracy": 0.79, "nested_cv_balanced_accuracy": 0.5, "nested_cv_auc": 0.6}},
                 "test_metrics": {"selected": {"accuracy": 0.85}, "other": {"accuracy": 0.99}}}
    (baseline_dir / "external_report.json").write_text(json.dumps(baseline))
    (baseline_dir / "baseline.joblib").write_bytes(b"hash-only fixture")
    (candidate_dir / "external_report.json").write_text(json.dumps(candidate))
    summary = summarize_refinement(candidate_dir, baseline_dir)
    assert summary["selected_model_by_development"] == "selected"
    assert summary["development_selection_rank_improved"]
    assert not summary["observed_holdout_accuracy_improved"]
    assert not summary["refinement_selected_as_default"]
    assert not summary["independently_confirmed_improvement"]
    assert summary["best_observed_candidate"] == "other"
    assert summary["best_observed_holdout_accuracy"] == 0.99
    assert not summary["best_observed_candidate_is_development_selected"]
