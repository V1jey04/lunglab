"""Patient-level metrics, interval estimates and comparison figures."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             roc_auc_score)


def patient_predictions(groups, y, p):
    rows = pd.DataFrame({"patient_id": groups, "y_true": y, "probability": p})
    if (rows.groupby("patient_id").y_true.nunique() > 1).any():
        raise ValueError("Conflicting patient labels.")
    # Multiple nodules belonging to a patient are averaged before scoring.
    return rows.groupby("patient_id", sort=True).agg(
        y_true=("y_true", "first"), probability=("probability", "mean")).reset_index()


def metrics(y, p, threshold=0.5, predicted=None):
    predicted = np.asarray(p) >= threshold if predicted is None else np.asarray(predicted)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def bootstrap_intervals(y, p, seed=42, repeats=1000, threshold=0.5, predicted=None):
    """Resample held-out patients, retaining both classes for AUC estimates."""
    rng = np.random.default_rng(seed)
    values = {k: [] for k in ("accuracy", "roc_auc", "sensitivity", "specificity", "f1")}
    y, p = np.asarray(y), np.asarray(p)
    predicted = None if predicted is None else np.asarray(predicted)
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) != 2:
            continue
        m = metrics(y[idx], p[idx], threshold, None if predicted is None else predicted[idx])
        for k in values:
            values[k].append(m[k])
    return {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in values.items() if v}


def paired_auc_difference(y, baseline, candidate, seed=42, repeats=1000):
    rng = np.random.default_rng(seed)
    y, baseline, candidate = map(np.asarray, (y, baseline, candidate))
    differences = []
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            differences.append(roc_auc_score(y[idx], candidate[idx]) -
                               roc_auc_score(y[idx], baseline[idx]))
    return {
        "estimate": float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline)),
        "ci95": np.percentile(differences, [2.5, 97.5]).tolist(),
        "method": "paired percentile bootstrap over held-out patients",
    }


def save_figures(predictions, output):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, frame in predictions.items():
        y, p = frame.y_true, frame.probability
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        observed, predicted = calibration_curve(y, p, n_bins=5, strategy="quantile")
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
        axes[2].plot(predicted, observed, "o-", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3)
    for ax, title, xlabel, ylabel in zip(
            axes, ["ROC", "Precision-recall", "Calibration"],
            ["False positive rate", "Recall", "Predicted probability"],
            ["True positive rate", "Precision", "Observed positive fraction"]):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    fig.suptitle("Held-out patients — research comparison")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
