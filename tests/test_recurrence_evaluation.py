import numpy as np
import pandas as pd
import pytest

from lungml import recurrence_train as trainer
from lungml.recurrence_evaluation import competing_risk_auc, horizon_metrics, reference_horizon_metrics
from lungml.recurrence_models import CensoringDistribution, ipcw_brier_score


TIME = np.array([100, 300, 900, 200])
EVENT = np.array([1, 2, 0, 0])
RISK = np.array([0.8, 0.6, 0.2, 0.5])
CURVES = {"by_horizon": {730: {"recurrence_cif": RISK}}}


def test_configured_at_risk_threshold_reaches_every_scorer(monkeypatch):
    training = pd.DataFrame({"time_days": np.r_[np.arange(1, 11), np.full(9, 1000)],
                             "event_type": np.r_[np.tile([1, 2], 5), np.zeros(9)]})
    monkeypatch.setattr(trainer, "MINIMUM_AT_RISK", 8)
    censoring = trainer._training_censoring(training)
    assert trainer._supported_horizons(training) == [365, 730]
    result = horizon_metrics(TIME, EVENT, CURVES, 730, censoring, require_supported=True)
    assert result["available"]
    assert result["ipcw_brier_score"] == pytest.approx(0.11)
    assert result["competing_risk_time_dependent_auc"] == 1
    reference = reference_horizon_metrics(
        training.time_days, training.event_type, TIME, EVENT, 730, censoring)
    assert reference["available"]
    assert trainer._paired_bootstrap_brier(
        TIME, EVENT, {"a": RISK, "b": RISK / 2}, 730, censoring, repetitions=10)["available"]

    monkeypatch.setattr(trainer, "MINIMUM_AT_RISK", 10)
    censoring = trainer._training_censoring(training)
    assert trainer._supported_horizons(training) == []
    assert not horizon_metrics(TIME, EVENT, CURVES, 730, censoring)["available"]
    with pytest.raises(ValueError, match="Unsupported"):
        ipcw_brier_score(TIME, EVENT, RISK, 730, censoring)
    with pytest.raises(ValueError, match="Unsupported"):
        competing_risk_auc(TIME, EVENT, RISK, 730, censoring)
    assert not trainer._paired_bootstrap_brier(
        TIME, EVENT, {"a": RISK, "b": RISK / 2}, 730, censoring, repetitions=10)["available"]


def test_configured_survival_threshold_controls_event_weights_and_horizon_support(monkeypatch):
    training = pd.DataFrame({"time_days": np.r_[np.ones(200), np.full(20, 1000)],
                             "event_type": np.r_[np.zeros(200), np.tile([1, 2], 10)]})
    monkeypatch.setattr(trainer, "MINIMUM_CENSORING_SURVIVAL", 0.08)
    censoring = trainer._training_censoring(training)
    assert trainer._supported_horizons(training) == [365, 730]
    result = horizon_metrics(TIME, EVENT, CURVES, 730, censoring, require_supported=True)
    assert result["ipcw_brier_score"] == pytest.approx(0.11 / (20 / 220))
    assert result["competing_risk_time_dependent_auc"] == 1
    assert reference_horizon_metrics(
        training.time_days, training.event_type, TIME, EVENT, 730, censoring)["available"]
    assert trainer._paired_bootstrap_brier(
        TIME, EVENT, {"a": RISK, "b": RISK / 2}, 730, censoring, repetitions=10)["available"]

    monkeypatch.setattr(trainer, "MINIMUM_CENSORING_SURVIVAL", 0.10)
    strict = trainer._training_censoring(training)
    assert trainer._supported_horizons(training) == []
    assert not horizon_metrics(TIME, EVENT, CURVES, 730, strict)["available"]
    with pytest.raises(ValueError, match="Unsupported"):
        horizon_metrics(TIME, EVENT, CURVES, 730, strict, require_supported=True)


def test_tied_censoring_changes_ipcw_weights_using_event_first_reference():
    # G(1)=1/2. The recurrence on day 1 uses G(1-)=1; the day-2 death
    # uses G(2-)=1/2. The earlier censor contributes no known outcome.
    censoring = CensoringDistribution(min_at_risk=1).fit([1, 1, 2], [1, 0, 2])
    score = ipcw_brier_score([1, 1, 2], [1, 0, 2], [0.2, 0.9, 0.4], 2, censoring)
    assert score == pytest.approx((0.8 ** 2 + 2 * 0.4 ** 2) / 3)
