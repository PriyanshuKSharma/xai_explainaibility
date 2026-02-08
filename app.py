from __future__ import annotations

# Purpose: Unified inference API used by local Docker, Vertex AI, and SageMaker.
# Run locally for testing:
#   uvicorn app:app --host 0.0.0.0 --port 8080
# Endpoints:
#   - GET /health (Vertex health check)
#   - GET /ping (SageMaker health check)
#   - POST /predict (Vertex predictions)
#   - POST /invocations (SageMaker predictions)

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("random_forest_model.pkl")
ENCODER_PATH = Path("label_encoder.pkl")

FEATURE_COLUMNS = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


class PredictRequest(BaseModel):
    instances: list[dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries. Each item must include all 30 feature keys.",
        min_length=1,
    )


app = FastAPI(title="XAI Breast Cancer Inference API")


@app.on_event("startup")
def load_artifacts() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run: python3 train_model.py"
        )

    app.state.model = joblib.load(MODEL_PATH)
    app.state.encoder = joblib.load(ENCODER_PATH) if ENCODER_PATH.exists() else None


def _to_dataframe(instances: list[dict[str, float]]) -> pd.DataFrame:
    missing_by_row: list[str] = []
    extra_by_row: list[str] = []

    for idx, row in enumerate(instances):
        missing = [c for c in FEATURE_COLUMNS if c not in row]
        extra = [k for k in row.keys() if k not in FEATURE_COLUMNS]
        if missing:
            missing_by_row.append(f"row {idx}: missing {missing}")
        if extra:
            extra_by_row.append(f"row {idx}: extra {extra}")

    if missing_by_row or extra_by_row:
        details = "; ".join(missing_by_row + extra_by_row)
        raise HTTPException(status_code=400, detail=f"Invalid features: {details}")

    return pd.DataFrame(instances, columns=FEATURE_COLUMNS)


def _predict(instances: list[dict[str, float]]) -> dict[str, Any]:
    df = _to_dataframe(instances)
    model = app.state.model
    encoder = app.state.encoder

    pred_int = model.predict(df).tolist()
    pred_proba = model.predict_proba(df).tolist()

    labels: list[str] | None = None
    if encoder is not None:
        labels = encoder.inverse_transform(pred_int).tolist()

    return {
        "predictions": pred_int,
        "predicted_labels": labels,
        "prediction_probabilities": pred_proba,
        "feature_order": FEATURE_COLUMNS,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    return _predict(request.instances)


@app.post("/invocations")
def invocations(request: PredictRequest) -> dict[str, Any]:
    return _predict(request.instances)
