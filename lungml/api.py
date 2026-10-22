"""Loopback research API. Clients select registered models, never filesystem paths."""
import json
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from .case_report import build_case_report
from .explanations import explain_case
from .quality import assess_inputs, load_profile, sha256
from .service import discover_models, extension_readiness, run_prediction, trusted_bundle


class CaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=200)
    records: list[dict[str, float | str | None]] = Field(min_length=1, max_length=500)


def create_app(artifact_root="artifacts"):
    app = FastAPI(title="LungLab research API", version="0.2.0",
                  description="Local inference on known segmented tumours. Exploratory evidence; no clinical or treatment claims.")

    def resolve(model_id):
        registry = discover_models(artifact_root)
        if model_id not in registry:
            raise HTTPException(404, "Unknown registered model ID.")
        entry = registry[model_id]
        return entry, trusted_bundle(entry)

    def evaluate(request):
        entry, bundle = resolve(request.model_id)
        frame = pd.DataFrame(request.records)
        started = perf_counter()
        predictions = run_prediction(bundle, frame)
        profile = load_profile(entry["path"], bundle)
        quality = assess_inputs(bundle, frame, profile)
        return entry, bundle, frame, predictions, quality, (perf_counter() - started) * 1000, profile

    @app.get("/health")
    def health():
        return {"status": "ok", "context": "local research service", "version": "0.2.0"}

    @app.get("/readiness")
    def readiness():
        return {"extensions": extension_readiness(artifact_root)}

    @app.get("/models")
    def models():
        return {"models": [{"model_id": name, "task": entry["task"], "selected": entry["selected"]}
                           for name, entry in discover_models(artifact_root).items()]}

    @app.get("/models/{run}/{name}")
    def schema(run: str, name: str):
        try:
            entry, bundle = resolve(f"{run}/{name}")
            return {"model_id": f"{run}/{name}", "task": bundle["task"], "feature_names": bundle["feature_names"],
                    "model_sha256": sha256(entry["path"]), "maximum_cases": 500,
                    "labels": bundle.get("labels"), "threshold": bundle.get("threshold"),
                    "horizons_days": bundle.get("horizons_days"),
                    "prediction_context": bundle.get("prediction_context")}
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/predict")
    def predict(request: CaseRequest):
        try:
            entry, bundle, frame, predictions, quality, elapsed_ms, _ = evaluate(request)
            return {"model_id": request.model_id, "model_sha256": sha256(entry["path"]),
                    "task": bundle["task"], "cases": len(frame), "elapsed_ms": elapsed_ms,
                    "predictions": json.loads(predictions.to_json(orient="records")), "quality": quality,
                    "context": "Research output. Probability is not validated clinical certainty."}
        except (ValueError, OSError, KeyError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/explain")
    def explain(request: CaseRequest):
        try:
            _, bundle, frame, _, _, _, profile = evaluate(request)
            return explain_case(bundle, frame, profile)
        except (ValueError, OSError, KeyError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/report", response_class=HTMLResponse)
    def report(request: CaseRequest):
        try:
            entry, bundle, frame, predictions, quality, _, profile = evaluate(request)
            explanation = explain_case(bundle, frame, profile) if bundle["task"] in {"subtype", "malignancy", "recorded_recurrence"} else None
            return build_case_report(bundle, frame, predictions, sha256(entry["path"]), quality, explanation)
        except (ValueError, OSError, KeyError) as error:
            raise HTTPException(422, str(error)) from error

    return app


app = create_app()
