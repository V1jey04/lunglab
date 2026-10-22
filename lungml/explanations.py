"""Exact linear log-odds contributions, or explicitly non-additive sensitivity."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data import numeric_features
from .predict import predict_frame


def explain_case(bundle, frame, profile=None, limit=15):
    if len(frame) != 1:
        raise ValueError("Explain exactly one case at a time.")
    if bundle["task"] not in {"subtype", "malignancy", "recorded_recurrence"}:
        raise ValueError("Case explanations currently support binary classification models only.")
    prediction = predict_frame(bundle, frame).iloc[0].to_dict()
    X = numeric_features(frame, bundle["feature_names"])
    model = bundle["model"]
    names = np.asarray(bundle["feature_names"], dtype=object)
    # Track selection without assuming every transformer implements sklearn's names API.
    if hasattr(model, "steps") and isinstance(model.steps[-1][1], LogisticRegression):
        values = X
        for _, step in model.steps[:-1]:
            if not hasattr(step, "transform"):
                continue  # Training-only samplers do not run at inference.
            values = step.transform(values)
            if hasattr(step, "indices_"):
                names = names[step.indices_]
            elif hasattr(step, "get_support"):
                names = names[step.get_support()]
            elif hasattr(step, "components_"):
                names = np.asarray([f"Transformed component {i + 1}" for i in range(values.shape[1])])
            elif values.shape[1] != len(names):
                names = np.asarray([f"Transformed input {i + 1}" for i in range(values.shape[1])])
        estimator = model.steps[-1][1]
        coef = np.asarray(estimator.coef_)
        if coef.shape == (1, len(names)) and list(estimator.classes_) == [0, 1]:
            contributions = np.asarray(values).ravel() * coef[0]
            baseline = float(estimator.intercept_[0])
            total = float(baseline + contributions.sum())
            rows = [{"feature": str(name), "contribution": float(value),
                     "transformed_value": float(transformed)}
                    for name, value, transformed in zip(names, contributions, np.asarray(values).ravel())]
            rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
            return {"method": "linear_log_odds", "positive_class": bundle["labels"][1],
                    "units": "log odds of the positive class", "baseline": baseline,
                    "total": total, "reconstructed_probability": float(1 / (1 + np.exp(-np.clip(total, -700, 700)))),
                    "displayed_contribution_sum": float(sum(r["contribution"] for r in rows[:limit])),
                    "remaining_contribution_sum": float(sum(r["contribution"] for r in rows[limit:])),
                    "contributions": rows[:limit], "prediction": prediction,
                    "interpretation": "Exact fitted linear score after frozen preprocessing. Positive values increase the positive-class log odds. These are associations, not causes or SHAP values."}
    if profile is None:
        return {"method": "unavailable", "contributions": [], "prediction": prediction,
                "interpretation": "Build a training-only input profile to enable feature replacement sensitivity for this model."}
    stats = pd.DataFrame.from_dict(profile["statistics"], orient="index").loc[X.columns]
    scale = (stats.q99 - stats.q01).replace(0, np.nan)
    deviation = ((X.iloc[0] - stats["median"]).abs() / scale).fillna(0)
    selected = deviation.nlargest(min(40, len(deviation))).index.tolist()
    variants = pd.concat([X] * len(selected), ignore_index=True)
    for i, name in enumerate(selected):
        variants.loc[i, name] = stats.loc[name, "median"]
    probabilities = np.asarray(model.predict_proba(variants))[:, 1]
    original = float(prediction[f"p_{bundle['labels'][1]}"])
    rows = [{"feature": name, "contribution": float(original - probability),
             "replacement_value": float(stats.loc[name, "median"])} for name, probability in zip(selected, probabilities)]
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return {"method": "median_replacement_sensitivity", "positive_class": bundle["labels"][1],
            "units": "positive-class probability difference", "contributions": rows[:limit],
            "features_checked": len(selected), "prediction": prediction,
            "interpretation": "Change when one input is replaced by its training median. Up to 40 inputs are chosen by normalized deviation. Effects do not add up; correlated replacements can be unrealistic. Not causal or SHAP values."}
