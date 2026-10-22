# Run and inspect LungLab

Run all commands from `/Users/vijey/Documents/ml_project_new`.
The installed environment is `.venv`. See the README to rebuild it.

## Start the application

In one terminal:

```sh
.venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open [the dashboard](http://127.0.0.1:8501).
Use **Live inference** to run preserved feature-table cases.
Select a completed case to inspect its inputs, explanation and report.
The report downloads as self-contained HTML. Its print button supports PDF export.

## Run the trained recurrence classifier

Open **Recurrence**. The **Clinical logistic · postoperative profile** model
is selected from development evidence. Choose **Example patients**, **Enter
patient details**, or **Upload feature CSV**, then click **Predict recurrence**.
The CT candidate supports examples and feature CSVs. Manual entry requires
known pathological T/N categories, grade, and invasion findings.

The output estimates **source-recorded recurrence during variable follow-up**.
It has no fixed time horizon and is not a validated clinical risk estimate.
Inspect **Measured recurrence performance and limitations** beside the output.
The default model finds 8/10 test recurrences with 16 false positives.
Its 48.6% accuracy and 58.0% balanced accuracy are separate from subtype scores.

The saved default threshold is 22%, chosen from development predictions.
A score below 50% can therefore produce a positive research classification.
This threshold is not a treatment or surveillance rule.

The frozen run is in `artifacts/recurrence_recorded/`. For a new research run:

```sh
.venv/bin/lung-ml recurrence train-recorded --output artifacts/my_recorded_recurrence
```

For local batch inference with the delivered clinical model:

```sh
.venv/bin/lung-ml recurrence predict-recorded --data my_patient_features.csv --output new_recurrence_predictions.csv
```

Use the dashboard's CSV template for the exact input columns. The API model
IDs are `recurrence_recorded/clinical_logistic` and
`recurrence_recorded/clinical_radiomics_logistic`. Both support `/predict`,
single-case `/explain` and `/report`. The existing nodal and time-based
recurrence verification gates remain unresolved.

In a second terminal:

```sh
.venv/bin/lung-ml serve --port 8000
```

The API binds to loopback. It has no public deployment or authentication layer.
Open [interactive API documentation](http://127.0.0.1:8000/docs).
Stop each process with Control-C in its terminal.

| Route | Result |
| --- | --- |
| `GET /health` | Service version and health |
| `GET /readiness` | Actual nodal and recurrence source gates |
| `GET /models` | Registered saved models |
| `GET /models/{run}/{name}` | Required inputs, model hash and frozen decision rule |
| `POST /predict` | Predictions, input review and measured duration |
| `POST /explain` | One subtype/malignancy case explanation |
| `POST /report` | One printable HTML research report |

Requests use a registered `model_id` and `records`, with one row per patient.
Patient IDs must be nonempty and unique. Preserve leading zeros as strings.
Supply every required feature column. Use JSON `null` for missing values.
The whole queue is checked before any prediction. Queues support at most 500 patients.
Clients cannot upload models or select arbitrary filesystem paths.
Only load locally trusted joblib bundles. Joblib can execute Python during loading.

```python
import requests
from lungml.recurrence_predict import read_feature_csv

frame = read_feature_csv("data/radiogenomics_subtype.csv").head(1)
request = {
    "model_id": "radiogenomics_accuracy/rich_accuracy_logistic",
    "records": frame.drop(columns=["label"]).to_dict("records"),
}
response = requests.post("http://127.0.0.1:8000/predict", json=request, timeout=30)
response.raise_for_status()
print(response.json())
```

This example runs an existing public-cohort case. It does not add validation evidence.

## Input reference ranges

Build a profile for another classification model:

```sh
.venv/bin/lung-ml profile \
  --model artifacts/radiogenomics_accuracy/rich_accuracy_logistic.joblib \
  --data data/radiogenomics_subtype.csv
```

The command selects only the bundle's fitted training patients.
It uses no outcome values or test-patient statistics.
When the bundle records source dataset hashes, the command requires a matching hash.
The saved profile is bound to the model SHA-256 and feature schema.
The default, bagged candidate and converged Lung1 models have profiles in this workspace.

Values outside training 1st–99th percentile ranges trigger a per-feature flag.
A case receives an input review flag above 10% flagged features, or for missing inputs.
These fixed rules are heuristics. They do not measure clinical uncertainty.
A model without a profile reports that the range check is unavailable.

## Explanations and case reports

The default logistic model reports exact contributions to its transformed linear score.
All contributions plus its intercept reconstruct the positive-class probability.
The display names selected radiomic features and reports contributions in log odds.
This is an explanation of the fitted model, not SHAP or biological causation.

Other classifiers use training-median replacement sensitivity when a profile exists.
The method checks up to 40 features selected by normalized input deviation.
Its probability changes do not add together. Correlated replacements can be unrealistic.
The interface identifies the method and its limits beside each result.

Export a case from the CLI:

```sh
.venv/bin/lung-ml case-report \
  --model artifacts/radiogenomics_accuracy/rich_accuracy_logistic.joblib \
  --data data/radiogenomics_subtype.csv \
  --patient-id R01-002 \
  --output artifacts/my_report/case.html
```

Use a new output path. The report includes model/input hashes, probabilities,
quality checks, explanation, timestamp and an embedded JSON record.
It uses no remote assets. Patient text is escaped before insertion into HTML.

## CT and mask workbench

Open **CT & mask**. The initial example is an explicitly labelled synthetic phantom.
It produces no cancer prediction. Use local paths or upload NIfTI/NRRD volumes.
A local DICOM directory must contain one image series. Convert RTSTRUCT/SEG to a matching label volume first.

The viewer checks scalar 3D geometry, spacing, origin, direction and mask extent.
It reports disconnected regions and boundary contact for review.
Slice controls provide three planes, CT window presets and overlay opacity.
The 3D view shows mask boundary samples in physical LPS coordinates, in millimetres.
Display permutation/flips preserve alignment and do not modify extraction inputs.
Oblique acquisition remains an image-grid view and is labelled accordingly.
An intensity-range check cannot establish HU calibration. Verify CT units at the source.

Extraction exports the measured features and a protocol/geometry provenance record.
The packaged protocols use 1 mm resampling and explicit 25-HU bins.
They differ from the third-party extraction behind the default classifier.
Train a matching new-cohort model before interpreting these features as subtype predictions.

**Segmentation stability** re-extracts original, 1 mm eroded and 1 mm dilated masks.
It reports physical volumes and relative feature changes. Invalid small masks are recorded.
One-case perturbations are not an ICC estimate or model-performance validation.

CLI inspection:

```sh
.venv/bin/lung-ml inspect-ct \
  --image /absolute/path/ct.nii.gz \
  --mask /absolute/path/mask.nii.gz \
  --output artifacts/ct_inspection --stability
```

## Evaluate a fresh external classification cohort

```sh
.venv/bin/lung-ml validate-external \
  --model artifacts/radiogenomics_accuracy/rich_accuracy_logistic.joblib \
  --data data/independent_compatible_features.csv \
  --source-evidence 'Documented independent cohort and matching feature extraction reference' \
  --output artifacts/independent_validation
```

This command preserves the model and threshold. It performs no fitting or selection.
It rejects IDs from fitted patients and the local experiment's preserved split.
It requires both supported labels and at least two patients per class.
It exports metrics, patient-bootstrap intervals, predictions, quality checks and hashes.
Use a new output directory. Earlier evaluation outputs are not overwritten.
ID checks cannot detect renamed patients. The supplied evidence must establish actual independence and extraction compatibility.
The example command describes required data. That fresh cohort is not currently present.
Nodal and recurrence use their endpoint-specific frozen evaluation protocols.

## Verify software and inspect remaining gates

```sh
.venv/bin/lung-ml doctor
.venv/bin/python -m pytest -q
```

The current acceptance suite covers real extraction, task gates, patient separation,
frozen artifacts, inference, report escaping, model explanations and HTTP contracts.
Synthetic fixtures test software only. They do not create extension performance evidence.

Implementation references: [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/),
[Streamlit figures](https://docs.streamlit.io/develop/api-reference/charts/st.pyplot),
and [SimpleITK metadata](https://simpleitk.readthedocs.io/en/master/link_DicomImagePrintTags_docs.html).
