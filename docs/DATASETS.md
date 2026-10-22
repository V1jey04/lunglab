# Dataset selection and download guidance

## Accuracy-focused clinical joins

`data/radiogenomics_clinical_subtype.csv` preserves all 139 radiomics patients
and the original 104/35 development/test assignment. It adds six explicitly
allowed demographic/exposure columns: age at diagnosis, sex, pack years and
three smoking-status indicators. Diagnosis, pathological stage, molecular
tests, treatment and outcomes are excluded from predictors.

`data/radiogenomics_expanded_clinical.csv` contains 207 AC/SCC patients from
the official clinical release. The original 35 test IDs remain fixed; all
68 additional eligible clinical patients are development-only. This separate
benchmark has six clinical inputs and no images. Identical age/sex/smoking
records can legitimately occur in different patients and are not treated as
duplicate scans. Source age timing needs confirmation before prospective use.

The selected 88.6% model uses radiomics only and can receive
`data/radiogenomics_subtype.csv` directly. It does not require clinical fields.

## Exact cohort from the supplied paper

Public availability checked on 1 October 2026. The
[original article](https://www.nature.com/articles/s41598-024-83786-6)
used 74 patients from Sri Ramachandra Institute of Higher Education and
Research, Chennai: 46 adenocarcinoma and 28 squamous cell carcinoma cases,
collected during 2017–2023. Its data availability statement offers the study
datasets from the corresponding author upon reasonable request.

No verified public download of this exact cohort was found in the publisher
record or searches of GitHub, Figshare, Zenodo and Mendeley Data. The article's
public XML links one supplementary PDF; downloading that PDF could not be
completed, so its contents have not been verified. The downloaded public
cohorts below are alternatives, not these same 74 patients. No email or other
contact was made; the user requested that authors not be emailed.

## Additional datasets downloaded

`data/radiogenomics_subtype.csv` includes 139 officially labelled TCIA patients
with 1,682 independently published PyRadiomics features. Source commits,
checksums, extraction logs, class counts and limitations accompany it. Use
`lung-ml download-rich` to prepare this cohort on a new installation.

`data/tcga_crops_features.csv` includes 61 unique published five-slice CT stacks
after one exact duplicate was excluded. It combines crop textures with frozen
ResNet18 features. Its clinical ID mapping and physical CT geometry are
unavailable; it is a separate benchmark with weaker provenance and potential
source/class confounding. Use `lung-ml download-crops` to fetch the verified
TCGA arrays and weights. Radiogenomics crop arrays were not joined to labels
because their patient order could not be verified.

## Post-surgical recurrence audit

`.venv/bin/lung-ml recurrence prepare` joins the 139 Radiogenomics feature rows
to the official clinical table without changing either source. It writes
`data/recurrence_features.csv`, `data/recurrence_outcomes.csv`, the preserved
`data/recurrence_cohort.splits.csv`, and `artifacts/recurrence_audit/audit.json`.
The current provisional 1–90-day M0 cohort has 115 patients: 26 recorded
recurrences, 24 deaths before recurrence, and 65 right-censored records. The
audit excludes 24 cases for metastatic disease, long CT-to-surgery intervals,
or chronology/status issues. Split and horizon counts are in the audit JSON;
per-patient checks are in `case_audit.csv`.
The prespecified 1–180-day sensitivity cohort has 120 patients, including 31
recorded recurrences. At 730 days, 55 development and 18 test cases have a
recorded event or censoring time at/after the horizon.

The feature table uses the published `original_shape_MeshVolume` as its tumour
volume input; the source unit interpretation and whether it matches the exact
radiomic CT used for extraction are not independently verified. Recurrence-
negative status is censored at last-known-alive, but the source does not say
whether recurrence was assessed through that date. These definitions keep
model fitting disabled. The audit records source references supplied through
`--surgery-date-evidence` and `--recurrence-followup-evidence`; provide them
only after checking the cited material, and adapt the cohort code if actual
patient-level recurrence-assessment dates become available. The prediction code and frozen protocol are
implemented, but there are no trained recurrence models or performance
results from these labels. Do not interpret the provisional dates or event
counts as clinically verified follow-up.

See [complete improvement results](IMPROVEMENT_RESULTS.md) before interpreting
either benchmark as a comparison against the paper.

## Dataset acquired for this project

The [TCIA NSCLC-Radiomics (Lung1) collection](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/)
contains 422 patients with pretreatment CT, manually delineated primary tumour
volumes and clinical data. Its labels include AC and SCC, so it matches the
paper's subtype task. It is a different cohort and includes larger tumours,
which limits direct comparisons with the paper's small solid nodules.

We downloaded the official clinical CSV and
[a third-party Slicer feature release](https://github.com/hhfchoi/Radiomics),
pinned to commit `841592dc855172061d42b4dfd47f831fee12d44d`. We selected only
the exact primary `_segment_GTV-1` mask and joined official histology by
PatientID. No clinical variables are used as radiomic inputs. The aggregate
feature table is used solely to recover feature-family names for duplicate
headers, not as input values.

The source has 419 primary-tumour patients; 202 have the requested histologies.
Others are excluded for large-cell, NOS, unknown or missing histology. The
result contains 51 AC and 151 SCC patients, 107 features and no missing feature
values. See `data/nsclc_subtype.provenance.json` for the complete audit.

Files downloaded under `data/raw/nsclc_radiomics/`:

| File | Purpose |
| --- | --- |
| clinical.csv | TCIA clinical ground-truth labels |
| features_unpadded.csv | Third-party primary and other tumour ROI features |
| features.csv | Family-qualified header mapping only |
| feature_source_readme.md | Extraction notes from the feature publisher |

The source feature extraction settings/version are incompletely documented.
Treat these as an initial real-data benchmark. For a defensible final study,
extract all radiomics from the original CT with a recorded configuration.

Underlying TCIA data are licensed CC BY-NC 3.0. The third-party repository has
no explicit license; do not assume its code or packaged tables are covered by
a permissive software license. Keep attribution to TCIA, Aerts et al., Clark
et al. and the feature publisher in the project report. For redistribution,
prefer links and reproducible download instructions.

## Download the original images when needed

The full Lung1 DICOM/segmentation collection is about 33 GB. Use the collection
page's NBIA Data Retriever manifest or the
[IDC collection interface](https://portal.imaging.datacommons.cancer.gov/collections/nsclc_radiomics/).
Filter to the patient IDs in `data/nsclc_subtype.csv` to reduce the download.

The compact feature command has not downloaded those raw voxels. Original
volumes should go under `data/raw/`. Choose the appropriate CT series and
convert the tumour `GTV-1` RTSTRUCT/SEG to NIfTI or NRRD using 3D Slicer or
dcmqi. Verify alignment rather than relying on a mask filename. TCIA notes
some series contain missing slices and that LUNG1-128 has no primary GTV-1;
review collection release notes before image extraction.

Use a single extraction configuration for every training and inference case.
Record HU conversion, reconstruction kernel, spacing, resampling, binning,
tumour mask label and extractor version. Public-cohort scans should not be
mixed with the paper's reported features without an extraction audit.

## Lymph-node and recurrence targets

The downloaded Radiogenomics clinical table also contains pathological nodal
stage and postoperative recurrence fields. These are labels, not subtype
predictors. The nodal join contains 139 source patients: 113 N0 and 26 N1/N2.
The provisional 1–90 day preoperative window retains 120 patients, including
24 node-positive patients. The preserved split retains 88 development and
32 test patients. Scan correspondence and interval direction still need
verification before fitting a preoperative model.

The separate recurrence audit retains 115 provisional patients. It requires
both scan/surgery date verification and documented recurrence assessment
through censoring. A last-known-alive date alone does not establish that
recurrence was absent through that date. Neither extension has a trained
model for the current cohort. See [clinical extension instructions](CLINICAL_EXTENSIONS.md)
and the generated audits under `artifacts/nodal_audit/` and
`artifacts/recurrence_audit/`.

## Suitable next collections

| Collection | Use | Important limitation |
| --- | --- | --- |
| NSCLC Radiogenomics | Independent subtype validation; CT, masks and clinical histology | Download and harmonize extraction; no benign controls |
| LIDC-IDRI | Nodule localization/segmentation; nodule malignancy research | Reader malignancy ratings are not automatically biopsy-confirmed cancer labels |
| LUNA16 | Nodule detector evaluation | Nodule annotations, not a standalone cancer-versus-normal ground truth |

For a full detection system, obtain benign/malignant nodule diagnoses and
normal scans with trustworthy patient IDs and confirmed label definitions.
Do not relabel every non-AC/SCC Lung1 patient as normal or benign: those patients
also have lung cancer. Avoid symptom survey datasets, histopathology images
and PNG screenshot collections as substitutes for this paper's 3D CT input.

An external holdout must remain untouched during model design. Split by
patient and, where practical, hospital/scanner or acquisition time. Audit
repeated studies, nodule masks and duplicate scans across collections.
