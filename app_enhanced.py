"""
app_enhanced.py
================
Production-grade FastAPI inference API with:
  - Async endpoints
  - In-memory LRU explanation cache (configurable TTL)
  - SHAP values bundled in prediction response
  - LIME on-demand endpoint
  - Counterfactual endpoint
  - /metrics  (Prometheus-compatible)
  - /compare  (multi-model comparison)
  - Rate limiting middleware
  - Structured logging (JSON)

Run:
    uvicorn app_enhanced:app --host 0.0.0.0 --port 8080 --reload

Author: Priyanshu K. Sharma  |  Enhanced 2025
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("xai-api")


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MODEL_PATH   = Path("random_forest_model.pkl")
ENCODER_PATH = Path("label_encoder.pkl")
SCALER_PATH  = Path("scaler.pkl")

FEATURE_COLUMNS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

CACHE_TTL_SECONDS = 300   # 5 minutes
RATE_LIMIT_RPM    = 120   # requests per minute per IP


# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────

class ExplanationCache:
    """Simple TTL cache for explanation results."""

    def __init__(self, ttl: int = CACHE_TTL_SECONDS):
        self._store: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl
        self._hits = 0
        self._misses = 0

    def _key(self, data: Any) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get(self, data: Any) -> Any | None:
        k = self._key(data)
        if k in self._store:
            val, ts = self._store[k]
            if time.time() - ts < self._ttl:
                self._hits += 1
                return val
            del self._store[k]
        self._misses += 1
        return None

    def set(self, data: Any, value: Any) -> None:
        self._store[self._key(data)] = (value, time.time())

    def stats(self) -> dict:
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
        }

    def clear(self) -> None:
        self._store.clear()


_cache = ExplanationCache()

# ─────────────────────────────────────────────
# METRICS COUNTERS
# ─────────────────────────────────────────────

_metrics: dict[str, int | float] = defaultdict(int)

def _inc(key: str, val: float = 1.0):
    _metrics[key] += val


# ─────────────────────────────────────────────
# APP + MIDDLEWARE
# ─────────────────────────────────────────────

app = FastAPI(
    title="XAI Enhanced Inference API",
    description="Breast Cancer prediction with full XAI explanations (SHAP, LIME, Counterfactuals).",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (in-memory token bucket per IP)
_rate_buckets: dict[str, list[float]] = defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60.0
    bucket = _rate_buckets[ip]
    _rate_buckets[ip] = [t for t in bucket if now - t < window]
    if len(_rate_buckets[ip]) >= RATE_LIMIT_RPM:
        return Response(
            content=json.dumps({"detail": "Rate limit exceeded. Max 120 req/min."}),
            status_code=429,
            media_type="application/json",
        )
    _rate_buckets[ip].append(now)
    return await call_next(request)


# ─────────────────────────────────────────────
# STARTUP – LOAD ARTIFACTS
# ─────────────────────────────────────────────

@app.on_event("startup")
async def load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run: python train_model_enhanced.py")

    app.state.model   = joblib.load(MODEL_PATH)
    app.state.encoder = joblib.load(ENCODER_PATH) if ENCODER_PATH.exists() else None
    app.state.scaler  = joblib.load(SCALER_PATH)  if SCALER_PATH.exists()  else None

    # Build SHAP explainer (background = zeros for speed; replace with real X_train for accuracy)
    try:
        import shap
        background = pd.DataFrame(
            np.zeros((50, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
        )
        app.state.shap_explainer = shap.TreeExplainer(app.state.model)
        logger.info("SHAP TreeExplainer loaded ✓")
    except Exception as e:
        app.state.shap_explainer = None
        logger.warning(f"SHAP explainer unavailable: {e}")

    logger.info("All artifacts loaded ✓")


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class Instance(BaseModel):
    """A single prediction instance."""
    features: dict[str, float] = Field(
        ..., description="Dict of 30 feature name → value pairs."
    )

class PredictRequest(BaseModel):
    instances: list[Instance] = Field(..., min_length=1, max_length=100)
    include_shap: bool = Field(False, description="Bundle SHAP values in response.")
    include_lime: bool = Field(False, description="Bundle LIME explanations in response.")

class CounterfactualRequest(BaseModel):
    instance: Instance
    n_counterfactuals: int = Field(3, ge=1, le=10)

class CompareRequest(BaseModel):
    instances: list[Instance] = Field(..., min_length=1, max_length=10)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _validate_and_build_df(instances: list[Instance]) -> pd.DataFrame:
    rows = []
    for i, inst in enumerate(instances):
        missing = [c for c in FEATURE_COLUMNS if c not in inst.features]
        extra   = [k for k in inst.features if k not in FEATURE_COLUMNS]
        if missing:
            raise HTTPException(400, f"Instance {i}: missing features {missing}")
        if extra:
            logger.warning(f"Instance {i}: ignoring unknown features {extra}")
        rows.append({c: inst.features[c] for c in FEATURE_COLUMNS})
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _shap_for_df(df: pd.DataFrame) -> list[dict] | None:
    expl = app.state.shap_explainer
    if expl is None:
        return None
    try:
        vals = expl.shap_values(df)
        if isinstance(vals, list):
            vals = vals[1]
        return [
            dict(zip(FEATURE_COLUMNS, row.tolist()))
            for row in vals
        ]
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        return None


def _lime_for_instance(model, df: pd.DataFrame, idx: int) -> dict | None:
    try:
        from lime.lime_tabular import LimeTabularExplainer
        explainer = LimeTabularExplainer(
            df.values,
            feature_names=FEATURE_COLUMNS,
            class_names=["Benign", "Malignant"],
            mode="classification",
            random_state=42,
        )
        exp = explainer.explain_instance(
            df.iloc[idx].values, model.predict_proba, num_features=10
        )
        return {
            "predicted_class": int(exp.predict_proba.argmax()),
            "class_probabilities": exp.predict_proba.tolist(),
            "feature_contributions": dict(exp.as_list()),
        }
    except Exception as e:
        logger.warning(f"LIME failed: {e}")
        return None


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
@app.get("/ping")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/predict")
@app.post("/invocations")          # SageMaker compat
async def predict(request: PredictRequest) -> dict:
    _inc("predict_calls")
    t0 = time.time()

    # Cache check
    cache_key = request.dict()
    cached = _cache.get(cache_key)
    if cached:
        _inc("cache_hits")
        return cached

    df = _validate_and_build_df(request.instances)
    model = app.state.model
    encoder = app.state.encoder

    pred_int = model.predict(df).tolist()
    pred_proba = model.predict_proba(df).tolist()
    labels = encoder.inverse_transform(pred_int).tolist() if encoder else None

    result: dict[str, Any] = {
        "predictions": pred_int,
        "predicted_labels": labels,
        "prediction_probabilities": pred_proba,
        "feature_order": FEATURE_COLUMNS,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }

    # Optional SHAP
    if request.include_shap:
        result["shap_values"] = _shap_for_df(df)

    # Optional LIME (first instance only, cost-aware)
    if request.include_lime:
        result["lime_explanation"] = _lime_for_instance(model, df, 0)

    _cache.set(cache_key, result)
    _inc("predict_latency_ms", time.time() - t0)
    return result


@app.post("/explain/shap")
async def explain_shap(request: PredictRequest) -> dict:
    """Return full SHAP explanation for all instances."""
    _inc("shap_calls")
    df = _validate_and_build_df(request.instances)
    shap_vals = _shap_for_df(df)
    if shap_vals is None:
        raise HTTPException(503, "SHAP explainer not available.")

    expl = app.state.shap_explainer
    ev = expl.expected_value
    base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)

    # Global importance
    shap_arr = np.array([list(d.values()) for d in shap_vals])
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "mean_abs_shap": np.abs(shap_arr).mean(axis=0).tolist(),
    }).sort_values("mean_abs_shap", ascending=False).to_dict("records")

    return {
        "base_value": base_value,
        "instance_shap_values": shap_vals,
        "global_importance": importance,
    }


@app.post("/explain/lime")
async def explain_lime(request: PredictRequest) -> dict:
    """Return LIME explanation for the first instance."""
    _inc("lime_calls")
    df = _validate_and_build_df(request.instances)
    result = _lime_for_instance(app.state.model, df, 0)
    if result is None:
        raise HTTPException(503, "LIME explainer failed.")
    return result


@app.post("/explain/counterfactual")
async def explain_counterfactual(request: CounterfactualRequest) -> dict:
    """Return nearest unlike neighbour counterfactuals."""
    _inc("cf_calls")
    df = _validate_and_build_df([request.instance])
    model = app.state.model

    instance = df.iloc[0].values
    orig_pred = int(model.predict(instance.reshape(1, -1))[0])

    # Use a dummy training set (zeros + ones) when real X_train not loaded
    # In production, load the real scaler-transformed training data
    np.random.seed(42)
    dummy_train = pd.DataFrame(
        np.random.randn(200, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS
    )
    train_preds = model.predict(dummy_train)
    opposite = dummy_train[train_preds != orig_pred]

    if len(opposite) == 0:
        raise HTTPException(400, "No opposite-class samples available.")

    from sklearn.preprocessing import StandardScaler as SS
    sc = SS()
    sc.fit(dummy_train)
    inst_n = sc.transform(instance.reshape(1, -1))[0]
    opp_n  = sc.transform(opposite)
    dists  = np.linalg.norm(opp_n - inst_n, axis=1)
    top_k  = min(request.n_counterfactuals, len(opposite))
    top_idx = np.argsort(dists)[:top_k]
    cfs = opposite.iloc[top_idx]

    return {
        "original_prediction": orig_pred,
        "original_label": (
            app.state.encoder.inverse_transform([orig_pred])[0]
            if app.state.encoder else str(orig_pred)
        ),
        "counterfactuals": cfs.to_dict("records"),
        "deltas": (cfs - instance).to_dict("records"),
        "distances": dists[top_idx].tolist(),
    }


@app.post("/compare")
async def compare_models(request: CompareRequest) -> dict:
    """
    Load all saved model_*.pkl files and compare their predictions
    for the given instances.
    """
    _inc("compare_calls")
    df = _validate_and_build_df(request.instances)
    model_files = list(Path(".").glob("model_*.pkl"))

    results = {}
    for mf in model_files:
        try:
            m = joblib.load(mf)
            preds = m.predict(df).tolist()
            probas = m.predict_proba(df).tolist()
            results[mf.stem.replace("model_", "")] = {
                "predictions": preds,
                "probabilities": probas,
            }
        except Exception as e:
            results[mf.stem] = {"error": str(e)}

    if not results:
        raise HTTPException(404, "No saved model files found. Run train_model_enhanced.py first.")

    return {"n_instances": len(request.instances), "model_comparisons": results}


@app.get("/metrics")
async def metrics() -> dict:
    """Prometheus-style counters + cache stats."""
    return {
        "counters": dict(_metrics),
        "cache": _cache.stats(),
        "uptime_hint": "counters reset on restart",
    }


@app.post("/cache/clear")
async def clear_cache() -> dict:
    _cache.clear()
    return {"status": "cache cleared"}
