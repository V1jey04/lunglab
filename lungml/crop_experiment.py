"""A new CT crop benchmark with frozen CNN features and a locked 25% split."""
import pandas as pd
from sklearn.svm import SVC

from .improve import search_candidates
from .rich_experiment import run


def crop_candidates():
    base = search_candidates()
    logistic, grid = base["robust_logistic"]
    grid = {"model__C": [0.01, 0.1, 1], "select__k": [10, 20]}
    svm, _ = base["quantile_svm"]
    svm.set_params(model=SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))
    return {
        "paper_mlp": base["paper_mlp"],
        "cnn_logistic": (logistic, grid),
        "hybrid_logistic": (logistic, grid),
        "hybrid_svm": (svm, {"model__C": [0.1, 1, 10], "model__gamma": ["scale"], "select__k": [10, 20]}),
    }


def run_crops(output="artifacts/tcga_crops_hybrid", evaluate=False):
    data = "data/tcga_crops_features.csv"
    columns = pd.read_csv(data, nrows=0).columns
    return run(data=data, output=output, evaluate=evaluate,
        feature_mode="hybrid", candidate_specs=crop_candidates(),
        model_columns={"cnn_logistic": [c for c in columns if c.startswith("deep_resnet18_")]},
        evaluation_description=("25% held-out unique published CT patient arrays within TCGA-LUAD/LUSC. "
                                "Clinical ID mapping is unavailable; collection/class confounding is possible. "
                                "The MLP is an adaptation using crop textures, not original 3D HU radiomics."))


if __name__ == "__main__":
    run_crops(evaluate=True)
