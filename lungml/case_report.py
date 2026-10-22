"""Portable, printable HTML with reproducible model and input identities."""
from datetime import datetime, timezone
from html import escape
import hashlib
import json


def build_case_report(bundle, frame, predictions, model_sha256, quality=None, explanation=None, image_data_uri=None):
    patient_ids = frame.patient_id.astype(str).tolist()
    if len(frame) != 1 or "patient_id" not in predictions or set(predictions.patient_id.astype(str)) != set(patient_ids):
        raise ValueError("A case report requires one patient and matching prediction IDs.")
    payload = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "patient_id": patient_ids[0], "task": bundle["task"],
               "model_name": bundle.get("name", bundle.get("model_family", "saved model")),
               "model_sha256": model_sha256,
               "input_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
               "input_features": len(bundle["feature_names"]),
               "predictions": json.loads(predictions.to_json(orient="records")),
               "quality": quality, "explanation": explanation,
               "context": "Research use only. Exploratory public-cohort evidence. No treatment recommendation or clinical validation."
                          + (" " + bundle["prediction_context"] if bundle.get("prediction_context") else "")}
    def value(item):
        if isinstance(item, float):
            return f"{item:.6g}"
        return str(item)
    rows = "".join(f"<tr><th>{escape(str(key))}</th><td>{escape(value(item))}</td></tr>"
                   for record in payload["predictions"] for key, item in record.items())
    def section(title, item):
        return f"<h2>{title}</h2><pre>{escape(json.dumps(item, indent=2, ensure_ascii=False, allow_nan=False))}</pre>" if item else ""
    overlay = ""
    if image_data_uri:
        if not image_data_uri.startswith("data:image/png;base64,"):
            raise ValueError("The report overlay must be an embedded PNG.")
        overlay = f'<h2>CT and mask inspection</h2><img alt="CT slice with tumour mask" src="{escape(image_data_uri, quote=True)}">'
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LungLab case · {escape(patient_ids[0])}</title><style>
body{{font:15px/1.6 system-ui,sans-serif;color:#203950;background:#f2f6fa;margin:0;padding:32px}}
main{{max-width:940px;margin:auto;background:white;padding:36px;border-radius:20px;border-top:6px solid #0b9488}}
h1{{font-size:34px;margin:6px 0}}h2{{font-size:20px;margin-top:30px}}small{{color:#62778b}}table{{border-collapse:collapse;width:100%}}
td,th{{text-align:left;padding:10px;border-bottom:1px solid #e4edf1;overflow-wrap:anywhere}}th{{width:40%}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fb;padding:20px;font-size:12px}}img{{max-width:100%}}
.context{{background:#eaf7f3;padding:16px;border-radius:10px}}button{{background:#0b9488;color:white;padding:10px 20px;border:0;border-radius:8px;cursor:pointer}}
@media print{{body{{padding:0;background:white}}main{{padding:10px}}button{{display:none}}pre{{font-size:10px}}}}
</style><main><small>LUNGLAB / RESEARCH CASE REPORT</small><h1>{escape(patient_ids[0])}</h1>
<p class="context">{escape(payload['context'])}</p><button onclick="window.print()">Print / save as PDF</button>
<h2>Model output</h2><table>{rows}</table>{overlay}{section('Input quality', quality)}{section('Case explanation', explanation)}
<h2>Reproducibility</h2><p>Task: {escape(payload['task'])}<br>Model: {escape(payload['model_name'])}<br>
Created: {escape(payload['created_utc'])}</p><pre>Model SHA-256: {escape(model_sha256)}\nInput SHA-256: {payload['input_sha256']}\nFeatures: {payload['input_features']}</pre>
<p>Feature extraction must match the training procedure. Model probability is not clinical certainty. Feature contributions do not establish biological causation.</p>
<script type="application/json" id="case-data">{encoded}</script></main></html>'''
