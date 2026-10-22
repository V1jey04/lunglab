"""Schema-checked inference from locally created model bundles."""
import joblib
import numpy as np
import pandas as pd

from .data import numeric_features


def load_bundle(path):
    # Only load bundles created locally; joblib is a Python serialization format.
    bundle = joblib.load(path)
    for key in ("model", "feature_names", "labels", "task", "threshold", "data_provenance"):
        if key not in bundle:
            raise ValueError(f"Invalid model bundle: missing {key}")
    return bundle


def predict_frame(bundle, frame):
    if frame.columns.duplicated().any():
        raise ValueError("Input columns must have unique names.")
    if not np.isfinite(bundle["threshold"]) or not 0 <= bundle["threshold"] <= 1:
        raise ValueError("Invalid model threshold.")
    X = numeric_features(frame, bundle["feature_names"])
    if len(X) == 0:
        raise ValueError("No input cases.")
    if X.isna().all(axis=1).any():
        raise ValueError("A case has no radiomic measurements.")
    if X.isna().mean(axis=1).gt(0.2).any():
        raise ValueError("More than 20% of radiomic features are missing in a case.")
    probabilities = np.asarray(bundle["model"].predict_proba(X), dtype=float)
    labels = bundle["labels"]
    if (len(labels) != 2 or probabilities.shape != (len(X), 2) or
            not np.isfinite(probabilities).all() or (probabilities < 0).any() or
            (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Model returned invalid binary probabilities.")
    if hasattr(bundle["model"], "classes_") and list(bundle["model"].classes_) != [0, 1]:
        raise ValueError("Model class order must be [0, 1].")
    predicted_class = (np.asarray(bundle["model"].predict(X)).ravel()
                       if bundle.get("prediction_rule") == "native_predict" else
                       (probabilities[:, 1] >= bundle["threshold"]))
    if predicted_class.shape != (len(X),) or not np.isin(predicted_class, [0, 1]).all():
        raise ValueError("Model returned invalid class indices.")
    predicted_class = predicted_class.astype(int)
    result = pd.DataFrame({"prediction": np.asarray(labels)[predicted_class],
                           f"p_{labels[0]}": probabilities[:, 0],
                           f"p_{labels[1]}": probabilities[:, 1]})
    selected_probability = probabilities[np.arange(len(probabilities)), predicted_class]
    result["review_suggested"] = ((selected_probability < 0.7) |
                                  (np.abs(probabilities[:, 1] - bundle["threshold"]) < 0.1))
    result["data_provenance"] = bundle["data_provenance"]
    if bundle.get("prediction_context"):
        result["prediction_context"] = bundle["prediction_context"]
    for column in ("patient_id", "sample_id"):
        if column in frame:
            result.insert(0, column, frame[column].to_numpy())
    return result
