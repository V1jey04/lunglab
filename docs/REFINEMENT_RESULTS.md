# Further accuracy refinement

The highest observed experimental score is **91.4% (32/35 patients)** from
`bagged_rich_logistic`. The development-selected model remains the baseline
rich logistic model at **88.6% (31/35)**. Bagging tied nested development
accuracy but had lower balanced accuracy, so it did not win the frozen
selection. The original saved 88.6% model is unchanged and remains the dashboard
default.

These are exploratory subtype results on previously inspected patients.
They do not establish independent superiority over the paper's 83% accuracy
on its different private cohort. This task classifies a known segmented tumour
as adenocarcinoma (AC) or squamous cell carcinoma (SCC); it does not detect
cancer in an unlabelled full CT.

## Protocol and all results

The finite seven-model protocol was saved before fitting. This refinement
uses the same dataset and patient split as the preceding accuracy run:
139 patients, 1,682 radiomic inputs; 104 development patients (83 AC, 21 SCC)
and 35 test patients (28 AC, 7 SCC). Dataset and split SHA-256 values match.
No test patient enters model fitting, preprocessing, selection or threshold
search. The experiment was designed after earlier scores had been observed;
freezing this round does not make the reused holdout independent.

Five outer stratified patient folds provide out-of-fold development scores.
Three inner folds choose hyperparameters and a threshold from
0.35/0.45/0.50/0.55/0.65/0.75/0.85. Model selection ranks nested accuracy,
then balanced accuracy, then AUC. Final models and thresholds are frozen using
development data before follow-up scoring. Calibration and stacking use their
own internal three-fold splits, with preprocessing inside each base estimator.
One primary tumour row per patient makes these internal splits patient-safe.

| Model | Nested CV accuracy | Nested CV balanced accuracy | Test accuracy | Test balanced accuracy | Test AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline rich logistic (development-selected) | 80.8% | 55.9% | 88.6% | 76.8% | 0.852 |
| Sigmoid-calibrated logistic | 78.8% | 49.4% | 80.0% | 50.0% | 0.796 |
| Bagged rich logistic (highest observed test accuracy) | 80.8% | 52.4% | 91.4% | 78.6% | 0.847 |
| Elastic-net radiomics | 79.8% | 50.0% | 85.7% | 64.3% | 0.811 |
| PCA + logistic | 78.8% | 49.4% | 82.9% | 62.5% | 0.796 |
| PCA + SVM | 79.8% | 50.0% | 82.9% | 57.1% | 0.827 |
| Stacked logistic/SVM/trees | 78.8% | 49.4% | 80.0% | 50.0% | 0.811 |

Majority prediction scores **80%** on the test patients. Calibration and
stacking predict AC for every test patient at their frozen thresholds. All
unsuccessful candidates remain in the report. The reproduced baseline has the
same nested scores and test predictions as the preceding selected model.

## The 91.4% experimental candidate

Bagging trains **12 bootstrap pipelines**, each sampling 80% of development
rows with replacement. Each pipeline independently fits median imputation,
correlation pruning at 0.95, a normal quantile transform, ANOVA feature selection
of 40 features, and L2 logistic regression with **C=0.3**. The selected features
can differ between bootstrap estimators. The ensemble averages SCC
probabilities and applies the development-chosen **0.55 threshold**. It uses
radiomics only and requires no clinical inputs.

Its test confusion matrix, with rows and columns ordered AC/SCC, is:

| Actual / predicted | AC | SCC |
| --- | ---: | ---: |
| AC | 28 | 0 |
| SCC | 3 | 4 |

AC recall is **100% in these 28 patients**, SCC recall is **57.1% (4/7)**,
balanced accuracy is **78.6%**, and AUC is **0.847**. The baseline's AUC is
slightly higher at 0.852. The extra correct classification improves AC recall;
the count of correctly classified SCC patients remains unchanged. All three
remaining bagged-model errors are SCC patients classified as AC.

The patient-bootstrap 95% accuracy interval is **82.7%–100%**, which includes
83%. Bootstrap intervals describe uncertainty within this small cohort and
do not account for adaptive experimentation or generalization to another site.
The ensemble's nested development accuracy is **80.8%**, balanced accuracy
**52.4%**, and SCC recall **4.8%**. The baseline's nested balanced accuracy is
**55.9%**, with SCC recall **14.3%**. These weak development SCC recalls explain
why the higher test point score is insufficient to promote the bagged model.

## Use and reproduce

In the dashboard, select `radiogenomics_refinement` and then choose
`bagged_rich_logistic` under Predict cases. The comparison explicitly separates
the development-selected baseline from the highest observed experimental score.

```bash
.venv/bin/lung-ml predict \
  --model artifacts/radiogenomics_refinement/bagged_rich_logistic.joblib \
  --data data/radiogenomics_subtype.csv \
  --output bagged_predictions.csv

# Reproduction on the same patients requires a fresh output directory.
.venv/bin/lung-ml refine --output artifacts/new_refinement_run --evaluate
```

Full-data inference includes training cases and is a demonstration, not new
validation. Repeating this command recreates the benchmark; it does not add
independent evidence. Saved evidence is under `artifacts/radiogenomics_refinement`:
`protocol.json`, `selection.json`, `external_report.json`, per-fold searches,
development out-of-fold predictions, held-out predictions, model bundles and
ROC/PR/calibration plots.

## Where further improvement is most useful

1. Add independently labelled SCC cases and evaluate on fresh patients from
   another collection or site. Both leading candidates still miss 3/7 SCC
   cases on this test cohort; overall accuracy alone hides this limitation.
2. Re-extract raw CT features with one documented configuration, and assess
   sensitivity to segmentation and acquisition differences. Existing downloaded
   third-party features have incompletely documented extraction settings.
3. Compare radiomics, pretrained 3D CT embeddings and their fusion on those
   consistently prepared patients, with a locked final evaluation.

The completed refinement improves the observed accuracy point score by one
patient. Fresh data and stronger minority-class discrimination are the next
steps for a defensible improvement claim.
