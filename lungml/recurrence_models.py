"""Small competing-risk models for an experimental recurrence outcome."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .improve import CorrelationPruner

BASE_FEATURES = ("clinical_age_at_diagnosis", "tumor_volume_source_units")
IMAGE_PREFIX = "original_"
SHAPE_PREFIX = "original_shape_"


def _require_event_support(event_type: np.ndarray, minimum: int = 5) -> None:
    counts = {kind: int(np.count_nonzero(event_type == kind)) for kind in (1, 2)}
    if min(counts.values()) < minimum:
        raise ValueError(
            f"Each training fold needs at least {minimum} recurrences and competing deaths; "
            f"found {counts}."
        )


class CauseSpecificCox:
    """Ridge-regularized Cox model with Breslow tied-event baseline hazards."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 300):
        if alpha <= 0:
            raise ValueError("Cox ridge alpha must be positive.")
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)

    def fit(self, X, time, event):
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)
        event = np.asarray(event, dtype=bool)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("Cox inputs must be a finite two-dimensional numeric matrix.")
        if len(time) != len(X) or len(event) != len(X) or not np.isfinite(time).all() or (time <= 0).any():
            raise ValueError("Cox event times must be finite, positive, and match the input rows.")
        if event.sum() < 2:
            raise ValueError("Cox fitting requires at least two events of the requested cause.")

        event_times = np.unique(time[event])

        def objective(beta):
            eta = np.clip(X @ beta, -40.0, 40.0)
            loss = 0.5 * self.alpha * float(beta @ beta)
            gradient = self.alpha * beta.copy()
            for point in event_times:
                failures = event & (time == point)
                risk = time >= point
                d = int(failures.sum())
                log_risk = eta[risk]
                normalizer = logsumexp(log_risk)
                weights = np.exp(log_risk - normalizer)
                loss -= float(eta[failures].sum() - d * normalizer)
                gradient -= X[failures].sum(axis=0) - d * (weights @ X[risk])
            return loss, gradient

        result = minimize(objective, np.zeros(X.shape[1]), method="L-BFGS-B", jac=True,
                          options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-7})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Cause-specific Cox fit did not converge: {result.message}")
        self.coef_ = result.x
        self.n_features_in_ = X.shape[1]
        self.event_times_ = event_times
        eta = np.clip(X @ self.coef_, -40.0, 40.0)
        self.baseline_jump_ = np.asarray([
            int(np.count_nonzero(event & (time == point))) /
            float(np.exp(logsumexp(eta[time >= point])))
            for point in event_times
        ])
        self.n_iter_ = int(result.nit)
        self.objective_ = float(result.fun)
        return self

    def relative_risk(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Cox prediction inputs do not match the fitted feature schema.")
        return np.exp(np.clip(X @ self.coef_, -40.0, 40.0))


class RecurrenceCoxPair:
    """Competing recurrence/death Cox pair with fold-fitted preprocessing."""

    def __init__(self, alpha: float = 1.0, n_components: int = 0):
        self.alpha = float(alpha)
        self.n_components = int(n_components)
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if self.n_components not in (0, 1, 2):
            raise ValueError("n_components must be zero, one, or two.")

    @staticmethod
    def feature_names(X: pd.DataFrame) -> list[str]:
        missing = sorted(set(BASE_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Missing baseline recurrence features: {missing}")
        imaging = [name for name in X.columns
                   if name.startswith(IMAGE_PREFIX) and not name.startswith(SHAPE_PREFIX)]
        return [*BASE_FEATURES, *imaging]

    def _transform_training(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Recurrence models require a named feature DataFrame.")
        all_names = self.feature_names(X)
        if self.n_components and sum(name not in BASE_FEATURES for name in all_names) < self.n_components:
            raise ValueError("Not enough non-shape original radiomics for the requested PCA components.")
        names = all_names if self.n_components else list(BASE_FEATURES)
        numeric = X.loc[:, names].apply(pd.to_numeric, errors="raise").astype(float)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        raw_volume = numeric["tumor_volume_source_units"].to_numpy()
        if (raw_volume < 0).any() or not np.isfinite(raw_volume).all():
            raise ValueError("Tumour-volume source values must be finite and nonnegative.")
        numeric["tumor_volume_source_units"] = np.log1p(raw_volume)
        self.feature_names_in_ = tuple(names)
        self.baseline_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.baseline_scaler_ = StandardScaler()
        baseline = self.baseline_imputer_.fit_transform(numeric.loc[:, BASE_FEATURES])
        baseline = self.baseline_scaler_.fit_transform(baseline)
        self.image_names_ = tuple(name for name in names if name not in BASE_FEATURES)
        blocks = [baseline]
        if self.n_components:
            self.image_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            image = self.image_imputer_.fit_transform(numeric.loc[:, self.image_names_])
            self.correlation_pruner_ = CorrelationPruner(threshold=0.95).fit(image)
            image = self.correlation_pruner_.transform(image)
            self.image_scaler_ = StandardScaler().fit(image)
            image = self.image_scaler_.transform(image)
            self.pca_ = PCA(n_components=self.n_components, svd_solver="full").fit(image)
            blocks.append(self.pca_.transform(image))
        else:
            self.image_names_ = ()
        return np.column_stack(blocks)

    def _transform_prediction(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Recurrence models require a named feature DataFrame.")
        missing = sorted(set(self.feature_names_in_) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required recurrence features: {missing[:10]}")
        numeric = X.loc[:, self.feature_names_in_].apply(pd.to_numeric, errors="raise").astype(float)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        raw_volume = numeric["tumor_volume_source_units"].to_numpy()
        if (raw_volume < 0).any() or not np.isfinite(raw_volume).all():
            raise ValueError("Tumour-volume source values must be finite and nonnegative.")
        numeric["tumor_volume_source_units"] = np.log1p(raw_volume)
        baseline = self.baseline_scaler_.transform(
            self.baseline_imputer_.transform(numeric.loc[:, BASE_FEATURES]))
        blocks = [baseline]
        if self.n_components:
            image = self.image_imputer_.transform(numeric.loc[:, self.image_names_])
            image = self.correlation_pruner_.transform(image)
            image = self.image_scaler_.transform(image)
            blocks.append(self.pca_.transform(image))
        return np.column_stack(blocks)

    def fit(self, X: pd.DataFrame, time, event_type):
        time = np.asarray(time, dtype=np.float64)
        event_type = np.asarray(event_type, dtype=np.int8)
        if len(X) != len(time) or len(time) != len(event_type):
            raise ValueError("Feature, time, and event arrays must have identical row counts.")
        if not set(np.unique(event_type)) <= {0, 1, 2}:
            raise ValueError("event_type must encode censoring=0, recurrence=1, or death=2.")
        _require_event_support(event_type)
        self.columns_ = tuple(X.columns)
        Z = self._transform_training(X)
        self.recurrence_model_ = CauseSpecificCox(self.alpha).fit(Z, time, event_type == 1)
        self.death_model_ = CauseSpecificCox(self.alpha).fit(Z, time, event_type == 2)
        self.classes_ = np.asarray(["event_free", "recurrence", "death_before_recurrence"], dtype=object)
        return self

    def predict_curves(self, X: pd.DataFrame, horizons: Iterable[int] = (365, 730)) -> dict:
        Z = self._transform_prediction(X)
        rr = self.recurrence_model_.relative_risk(Z)
        rd = self.death_model_.relative_risk(Z)
        time = np.union1d(self.recurrence_model_.event_times_, self.death_model_.event_times_)
        recurrence_jumps = np.zeros(len(time), dtype=float)
        death_jumps = np.zeros(len(time), dtype=float)
        recurrence_jumps[np.searchsorted(time, self.recurrence_model_.event_times_)] = self.recurrence_model_.baseline_jump_
        death_jumps[np.searchsorted(time, self.death_model_.event_times_)] = self.death_model_.baseline_jump_
        survival = np.ones(len(X), dtype=float)
        cif_recurrence = np.zeros(len(X), dtype=float)
        cif_death = np.zeros(len(X), dtype=float)
        curve_rows = [{"time_days": 0, "recurrence_cif": cif_recurrence.copy(),
                       "death_cif": cif_death.copy(), "event_free_survival": survival.copy()}]
        requested = sorted(set(int(value) for value in horizons))
        if not requested or any(value <= 0 for value in requested):
            raise ValueError("Prediction horizons must be positive day counts.")
        sampled = {}
        for point, base_recurrence, base_death in zip(time, recurrence_jumps, death_jumps):
            for horizon in requested:
                if horizon not in sampled and horizon < point:
                    sampled[horizon] = {
                        "recurrence_cif": cif_recurrence.copy(),
                        "death_cif": cif_death.copy(),
                        "event_free_survival": survival.copy(),
                    }
            hr, hd = base_recurrence * rr, base_death * rd
            total = hr + hd
            event_probability = -np.expm1(-total)
            fraction_recurrence = np.divide(hr, total, out=np.zeros_like(hr), where=total > 0)
            fraction_death = np.divide(hd, total, out=np.zeros_like(hd), where=total > 0)
            cif_recurrence = cif_recurrence + survival * event_probability * fraction_recurrence
            cif_death = cif_death + survival * event_probability * fraction_death
            survival = survival * np.exp(-total)
            point = int(point)
            curve_rows.append({"time_days": point, "recurrence_cif": cif_recurrence.copy(),
                               "death_cif": cif_death.copy(), "event_free_survival": survival.copy()})
            for horizon in requested:
                if horizon not in sampled and point == horizon:
                    sampled[horizon] = {
                        "recurrence_cif": cif_recurrence.copy(),
                        "death_cif": cif_death.copy(),
                        "event_free_survival": survival.copy(),
                    }
        # The baseline hazard is a step function: its risk estimates stay fixed
        # after the last observed event time, and horizon support is audited
        # separately from this numerical lookup.
        last = curve_rows[-1]
        for horizon in requested:
            if horizon not in sampled:
                sampled[horizon] = {key: value.copy() for key, value in last.items() if key != "time_days"}
        for row in curve_rows:
            row["risk_sum_error"] = np.abs(
                row["recurrence_cif"] + row["death_cif"] + row["event_free_survival"] - 1.0)
        return {"by_horizon": sampled, "curves": curve_rows}


class AalenJohansenReference:
    """Nonparametric cumulative-incidence reference for training-fold outcomes."""

    def fit(self, time, event_type, *, require_both_causes: bool = True):
        time = np.asarray(time, dtype=np.float64)
        event_type = np.asarray(event_type, dtype=np.int8)
        if len(time) != len(event_type) or not np.isfinite(time).all() or (time <= 0).any():
            raise ValueError("AJ event times must be finite, positive, and match event types.")
        if require_both_causes:
            _require_event_support(event_type)
        self.event_times_ = np.unique(time[event_type != 0])
        survival = 1.0
        recurrence, death = 0.0, 0.0
        self.curves_ = [{"time_days": 0, "recurrence_cif": 0.0,
                         "death_cif": 0.0, "event_free_survival": 1.0}]
        for point in self.event_times_:
            at_risk = int(np.count_nonzero(time >= point))
            d_recurrence = int(np.count_nonzero((time == point) & (event_type == 1)))
            d_death = int(np.count_nonzero((time == point) & (event_type == 2)))
            recurrence += survival * d_recurrence / at_risk
            death += survival * d_death / at_risk
            survival *= 1.0 - (d_recurrence + d_death) / at_risk
            self.curves_.append({"time_days": int(point), "recurrence_cif": recurrence,
                                 "death_cif": death, "event_free_survival": survival})
        return self

    def predict_at(self, horizon: int) -> dict:
        available = [row for row in self.curves_ if row["time_days"] <= horizon]
        return dict(available[-1])


@dataclass
class CensoringDistribution:
    """Reverse Kaplan-Meier censoring estimate with a shared support policy."""

    min_survival: float = 0.10
    min_at_risk: int = 10

    def __post_init__(self):
        if not np.isfinite(self.min_survival) or not 0 < self.min_survival <= 1:
            raise ValueError("Minimum censoring survival must be in (0, 1].")
        if (not isinstance(self.min_at_risk, (int, np.integer))
                or self.min_at_risk < 1):
            raise ValueError("Minimum patients at risk must be a positive integer.")

    def fit(self, time, event_type):
        self.time_ = np.asarray(time, dtype=np.float64)
        self.censored_ = np.asarray(event_type, dtype=np.int8) == 0
        if len(self.time_) != len(self.censored_) or not np.isfinite(self.time_).all():
            raise ValueError("Censoring inputs must be finite and have matching lengths.")
        self.censor_times_ = np.unique(self.time_[self.censored_])
        survival = 1.0
        values = []
        for point in self.censor_times_:
            at_point = self.time_ == point
            # Observed failures precede censoring on a tied day, matching the
            # event-first risk sets used by the Aalen-Johansen reference.
            at_risk = int(np.count_nonzero(self.time_ >= point)
                          - np.count_nonzero(at_point & ~self.censored_))
            d_censored = int(np.count_nonzero(at_point & self.censored_))
            survival *= 1.0 - d_censored / at_risk
            values.append(survival)
        self.after_ = np.asarray(values, dtype=float)
        self.before_ = np.r_[1.0, self.after_[:-1]]
        return self

    def survival_probability(self, at: float, *, left_limit: bool = False) -> float:
        side = "left" if left_limit else "right"
        count = int(np.searchsorted(self.censor_times_, at, side=side))
        if count == 0:
            return 1.0
        return float(self.after_[count - 1])

    def require_support(self, horizon: int) -> float:
        support = self.survival_probability(horizon)
        at_risk = int(np.count_nonzero(self.time_ >= horizon))
        if support < self.min_survival or at_risk < self.min_at_risk:
            raise ValueError(
                f"Unsupported {horizon}-day horizon: censoring survival={support:.3f}, "
                f"training patients at risk={at_risk}; required survival>={self.min_survival:g} "
                f"and patients>={self.min_at_risk}."
            )
        return support


def ipcw_brier_score(time, event_type, recurrence_risk, horizon: int,
                     censoring: CensoringDistribution) -> float:
    """Censoring-weighted Brier score for recurrence cumulative incidence."""
    time = np.asarray(time, dtype=np.float64)
    event_type = np.asarray(event_type, dtype=np.int8)
    risk = np.asarray(recurrence_risk, dtype=np.float64)
    if not (len(time) == len(event_type) == len(risk)) or not np.isfinite(risk).all():
        raise ValueError("Brier inputs must be finite and have matching lengths.")
    if (risk < 0).any() or (risk > 1).any():
        raise ValueError("Recurrence probabilities must be between zero and one.")
    censoring.require_support(horizon)
    weighted_error = 0.0
    for i, (duration, kind, estimate) in enumerate(zip(time, event_type, risk)):
        if duration <= horizon and kind != 0:
            observed = float(kind == 1)
            probability = censoring.survival_probability(duration, left_limit=True)
            if probability < censoring.min_survival:
                raise ValueError(
                    f"Censoring support is below {censoring.min_survival:g} at an observed event.")
        elif duration > horizon:
            observed = 0.0
            probability = censoring.survival_probability(horizon)
        else:
            # Recurrence status is unknown at the horizon for an earlier censor.
            continue
        weighted_error += (estimate - observed) ** 2 / probability
    return float(weighted_error / len(time))
