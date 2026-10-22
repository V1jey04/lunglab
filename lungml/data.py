"""Explicit labels and feature allowlists prevent clinical-label leakage."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TASK_LABELS = {
    "subtype": ("adenocarcinoma", "squamous_cell_carcinoma"),
    "malignancy": ("benign", "malignant"),
}
RADIOMIC_PREFIXES = (
    "original_", "wavelet-LLL_", "wavelet-LLH_", "wavelet-LHL_", "wavelet-LHH_",
    "wavelet-HLL_", "wavelet-HLH_", "wavelet-HHL_", "wavelet-HHH_", "square_",
    "squareroot_", "logarithm_", "exponential_", "gradient_", "lbp-2D_",
    "lbp-3D-k_", "lbp-3D-m1_", "lbp-3D-m2_", "log-sigma-",
)
CLINICAL_FEATURES = (
    "clinical_age_at_diagnosis", "clinical_male", "clinical_pack_years",
    "clinical_smoking_never", "clinical_smoking_former", "clinical_smoking_current",
)


@dataclass
class Dataset:
    frame: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    task: str


def numeric_features(frame, features):
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required radiomic features: {missing[:10]}")
    X = frame.loc[:, features].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Feature values contain infinity.")
    return X


def load_dataset(path: str | Path, task="subtype", feature_mode="original") -> Dataset:
    frame = pd.read_csv(path, dtype={"patient_id": str, "sample_id": str})
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names.")
    for name in ("patient_id", "label"):
        if name not in frame or frame[name].isna().any():
            raise ValueError(f"Required, nonmissing column: {name}")
    frame["patient_id"] = frame["patient_id"].str.strip()
    if frame["patient_id"].eq("").any():
        raise ValueError("Empty patient ID.")
    labels = TASK_LABELS[task]
    if set(frame.label) != set(labels):
        raise ValueError(f"Task {task} requires both labels {labels}; found {sorted(set(frame.label))}")
    if (frame.groupby("patient_id").label.nunique() > 1).any():
        raise ValueError("Conflicting labels within a patient. Define a consistent patient-level task first.")
    if feature_mode not in {"original", "all", "hybrid", "clinical"}:
        raise ValueError("feature_mode must be original, all, hybrid, or clinical.")
    prefixes = ("original_",) if feature_mode == "original" else RADIOMIC_PREFIXES
    if feature_mode == "hybrid":
        prefixes += ("deep_resnet18_",)
    features = [c for c in frame if c.startswith(prefixes)
                or (feature_mode == "clinical" and c in CLINICAL_FEATURES)]
    clinical_only = feature_mode == "clinical" and set(features) == set(CLINICAL_FEATURES)
    if len(features) < 10 and not clinical_only:
        raise ValueError("Expected at least 10 allowed image feature columns.")
    X = numeric_features(frame, features)
    if X.isna().all(axis=0).any():
        raise ValueError("A feature is completely missing. Remove it before training.")
    # Identical ages/sex/smoking histories can legitimately occur in different
    # patients. Image fingerprints, rather than clinical covariates, are audited.
    image_columns = [c for c in features if c not in CLINICAL_FEATURES]
    if image_columns:
        fingerprints = pd.util.hash_pandas_object(X[image_columns], index=False)
        if (pd.DataFrame({"hash": fingerprints, "patient": frame.patient_id})
                .groupby("hash").patient.nunique() > 1).any():
            raise ValueError("Identical feature rows have different patient IDs; resolve possible duplicate patients.")
    if "sample_id" not in frame:
        frame["sample_id"] = [f"sample-{i:05d}" for i in range(len(frame))]
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique and nonmissing.")
    y = frame.label.map({labels[0]: 0, labels[1]: 1}).to_numpy(dtype=int)
    return Dataset(frame, X, y, frame.patient_id.to_numpy(), task)
