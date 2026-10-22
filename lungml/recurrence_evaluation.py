"""Censoring-aware scores for first recurrence with death as a competing event."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .recurrence_models import AalenJohansenReference, CensoringDistribution, ipcw_brier_score


def competing_risk_auc(time, event_type, recurrence_risk, horizon: int,
                       censoring: CensoringDistribution) -> float:
    """IPCW cumulative/dynamic AUC with deaths counted among non-recurrence controls."""
    time = np.asarray(time, dtype=np.float64)
    event_type = np.asarray(event_type, dtype=np.int8)
    risk = np.asarray(recurrence_risk, dtype=np.float64)
    if not (len(time) == len(event_type) == len(risk)) or not np.isfinite(risk).all():
        raise ValueError("AUC inputs must be finite and have matching lengths.")
    censoring.require_support(horizon)
    cases, case_weights, controls, control_weights = [], [], [], []
    for i, (duration, kind) in enumerate(zip(time, event_type)):
        if duration <= horizon and kind == 1:
            g = censoring.survival_probability(duration, left_limit=True)
            if g >= censoring.min_survival:
                cases.append(i)
                case_weights.append(1.0 / g)
        elif duration <= horizon and kind == 2:
            g = censoring.survival_probability(duration, left_limit=True)
            if g >= censoring.min_survival:
                controls.append(i)
                control_weights.append(1.0 / g)
        elif duration > horizon:
            g = censoring.survival_probability(horizon)
            if g >= censoring.min_survival:
                controls.append(i)
                control_weights.append(1.0 / g)
        # An earlier censor has an unknown recurrence status at the horizon.
    if not cases or not controls:
        raise ValueError(f"{horizon}-day AUC is unavailable without observed cases and controls.")
    cases, controls = np.asarray(cases), np.asarray(controls)
    case_weights, control_weights = np.asarray(case_weights), np.asarray(control_weights)
    difference = risk[cases, None] - risk[controls][None, :]
    concordance = (difference > 0).astype(float) + 0.5 * (difference == 0)
    pair_weights = case_weights[:, None] * control_weights[None, :]
    return float(np.sum(pair_weights * concordance) / np.sum(pair_weights))


def horizon_metrics(time, event_type, curves, horizon: int,
                    censoring: CensoringDistribution, *, require_supported: bool = False) -> dict:
    values = np.asarray(curves["by_horizon"][int(horizon)]["recurrence_cif"], dtype=float)
    training_at_risk = int(np.count_nonzero(censoring.time_ >= horizon))
    censoring_support = censoring.survival_probability(horizon)
    try:
        censoring.require_support(horizon)
        brier = ipcw_brier_score(time, event_type, values, horizon, censoring)
    except ValueError as error:
        if require_supported:
            raise
        return {
            "horizon_days": int(horizon),
            "available": False,
            "unavailable_reason": str(error),
            "ipcw_brier_score": None,
            "competing_risk_time_dependent_auc": None,
            "censoring_survival_at_horizon": censoring_support,
            "training_patients_at_risk": training_at_risk,
        }
    try:
        auc = competing_risk_auc(time, event_type, values, horizon, censoring)
        auc_unavailable_reason = None
    except ValueError as error:
        auc = None
        auc_unavailable_reason = str(error)
    return {
        "horizon_days": int(horizon),
        "available": True,
        "ipcw_brier_score": brier,
        "competing_risk_time_dependent_auc": auc,
        "auc_unavailable_reason": auc_unavailable_reason,
        "censoring_survival_at_horizon": censoring_support,
        "training_patients_at_risk": training_at_risk,
    }


def calibration_by_tertile(time, event_type, recurrence_risk, horizon: int) -> list[dict]:
    """Descriptive observed/predicted recurrence CIF by up to three risk groups."""
    frame = pd.DataFrame({"time": time, "event": event_type, "risk": recurrence_risk})
    if frame.risk.nunique() < 2 or len(frame) < 9:
        return []
    frame["group"] = pd.qcut(frame.risk, q=3, labels=False, duplicates="drop")
    rows = []
    for group, subset in frame.groupby("group", observed=True):
        if subset.empty:
            continue
        estimate = AalenJohansenReference().fit(
            subset.time, subset.event, require_both_causes=False).predict_at(horizon)
        rows.append({
            "risk_tertile": int(group) + 1,
            "patients": int(len(subset)),
            "mean_predicted_recurrence_cif": float(subset.risk.mean()),
            "observed_aalen_johansen_recurrence_cif": estimate["recurrence_cif"],
            "observed_competing_death_cif": estimate["death_cif"],
        })
    return rows


def reference_horizon_metrics(training_time, training_event_type,
                              evaluation_time, evaluation_event_type, horizon: int,
                              censoring: CensoringDistribution) -> dict:
    estimator = AalenJohansenReference().fit(training_time, training_event_type)
    reference = estimator.predict_at(horizon)
    risk = np.full(len(evaluation_time), reference["recurrence_cif"], dtype=float)
    metrics = horizon_metrics(
        evaluation_time,
        evaluation_event_type,
        {"by_horizon": {int(horizon): {"recurrence_cif": risk}}},
        horizon,
        censoring,
    )
    return {
        "prediction": reference,
        **{key: value for key, value in metrics.items() if key != "horizon_days"},
    }
