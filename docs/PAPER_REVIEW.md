# Review of the supplied paper

Selvam et al., *Radiomics for differentiating adenocarcinoma and squamous cell
carcinoma in non-small cell lung cancer beyond nodule morphology in chest CT*,
Scientific Reports 14, 32088 (2024).
[DOI and article](https://doi.org/10.1038/s41598-024-83786-6).

## What the study actually predicts

It distinguishes two histologically confirmed NSCLC subtypes, AC and SCC,
within already identified solid lung nodules. It does not classify healthy
versus cancer scans, find nodules automatically, or classify other cancers.

Of 1,280 scans reviewed, the authors selected 74 primary NSCLC cases: 46 AC and
28 SCC. The selected lesions were solid pulmonary nodules 3–30 mm, collected
at one centre during 2017–2023. Mixed histologies, subsolid nodules and certain
unsuitable lesions were excluded (Methods, pages 2–3).

## Reported pipeline

| Component | Paper | Implementation |
| --- | --- | --- |
| Imaging | HRCT; 0.65–0.80 mm reconstructed slice thickness | 3D CT with HU and geometry retained |
| Segmentation | ITK-SNAP; radiologist delineation and review | User-supplied tumour masks; alignment checks |
| Features | PyRadiomics, 101 ROI features | 107 available original features; mismatch disclosed |
| Selection | LASSO, mRMR and Random Forest compared; LASSO best | LASSO ranks top 10 within each training fold |
| Imbalance | SMOTE | Training-only adaptive SMOTE |
| Best classifier | MLP, ReLU, hidden layer (100), learning rate 0.001 | Those architecture and learning-rate settings |
| Evaluation | Accuracy, sensitivity, specificity, precision, F1, AUC | Those metrics plus balanced accuracy, PR-AUC, Brier and patient CIs |
| Explanation | SHAP | Global LASSO/boosting importance; local SHAP left as an extension |
| Decision analysis | Decision curves | Not reproduced; requires an explicitly defined action and costs |

The text says 13 classifiers. Table 1 lists KNN, linear SVM, RBF SVM, decision
tree, Random Forest, AdaBoost, Naive Bayes, LDA, QDA and three MLP activation
variants (ReLU, identity, tanh). XGBoost appears in the methods but has no row
in the main table. This project implements the reported winning MLP and one
additional classifier, rather than claiming to recreate every paper experiment.

Figure 5 identifies Imc2, Kurtosis, MinorAxisLength, Sphericity, MajorAxisLength,
Complexity, LargeAreaHighGrayLevelEmphasis, LargeDependenceLowGrayLevelEmphasis,
Flatness and RunVariance. Their PyRadiomics equivalents span GLCM, first-order,
shape, NGTDM, GLSZM, GLDM and GLRLM families. We relearn the top 10 on training
data, since the public cohort differs from the study cohort.

The paper reports 83% accuracy, 86% sensitivity, 81% specificity, 83% precision,
83% F1 and 88% AUC for the ReLU model. The tanh MLP also rounds to 83% accuracy
in Table 1; ReLU is identified as the authors' preferred model.

## Reproduction limits and explicit assumptions

The study dataset is available from the corresponding author upon reasonable
request (page 11), not bundled with this project. The main article does not
fully specify the feature-extraction configuration, complete 101-feature list,
exact patient split, seeds or all optimizer/training parameters. It references
supplementary hyperparameter tables; those are not included in the local PDF.
The descriptions of metric variability as standard deviations versus 95%
intervals also need clarification before claiming exact replication.

Our assumptions are median imputation, StandardScaler, Lasso regression with
alpha selected from 0.001/0.01, sklearn's Adam optimizer, seed 42, up to 5,000
MLP iterations, up to 100,000 LASSO iterations, and an 80/20 stratified patient
split with development CV. Feature ranking uses original column order to break
ties. SMOTE's neighbours are reduced when a training fold is small.

The article's narrative does not establish whether every preprocessing step
was performed within training folds. That ambiguity is not proof of leakage;
this implementation explicitly prevents it. Paper metrics must not be assigned
to the new models, and synthetic or public-cohort performance must not be
described as a replication of the original 74 cases.

## Why CatBoost is the alternative

CatBoost is absent from the methods/model comparison. A shallow, regularized
boosted-tree model is a practical alternative for a small numeric feature
table; it can learn nonlinear interactions without training a CT CNN from
scratch. The hypothesis is reduced overfitting and improved calibration,
assessed on the same patients as the MLP.

The original Lung1 CatBoost comparison supports improved Brier score only;
it does not demonstrate improved AUC or balanced accuracy. The subsequent
accuracy-focused Radiogenomics experiment selected a richer-radiomics logistic
classifier from development CV and observed 88.6% holdout accuracy, versus
71.4% for the accuracy-selected paper MLP implementation on those same patients.
This is an exploratory follow-up on previously inspected test patients, not
validated superiority over the original paper. See
[complete results](IMPROVEMENT_RESULTS.md).

Implementation references:
[CatBoost classifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier),
[PyRadiomics extraction settings](https://pyradiomics.readthedocs.io/en/latest/customization.html).
