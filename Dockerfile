# ─────────────────────────────────────────────────────────────────────────────
# XAI Explainability v3.0 — Production Docker Image
#
# Build:
#   docker build -t xai-explainability:v3.0 .
#
# Run – Streamlit Dashboard (default, port 8501):
#   docker run --rm -p 8501:8501 xai-explainability:v3.0
#
# Run – FastAPI inference server (port 8080):
#   docker run --rm -p 8080:8080 -e APP_MODE=api xai-explainability:v3.0
#
# Run – both (separate containers recommended; or override CMD):
#   docker run --rm -p 8501:8501 xai-explainability:v3.0 dashboard
#   docker run --rm -p 8080:8080 xai-explainability:v3.0 api
# ─────────────────────────────────────────────────────────────────────────────

# Use Python 3.11-slim: stable numpy/shap/torch wheels, well-tested on this stack
FROM python:3.11-slim

# ── Build-time metadata ───────────────────────────────────────────────────────
LABEL maintainer="Priyanshu K. Sharma"
LABEL version="3.0"
LABEL description="XAI Explainability Framework – SHAP, LIME, Anchors, IG, Counterfactuals"

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Default mode: 'dashboard' | 'api'
    APP_MODE=dashboard

WORKDIR /app

# ── System dependencies (for lightgbm, shap native libs) ─────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to maximise Docker layer cache reuse
COPY requirements.txt ./

# Step 1: upgrade pip + build tools
RUN pip install --upgrade pip setuptools wheel

# Step 2: pin numpy>=2.0 FIRST (ensures binary wheel, avoids alibi downgrade conflict)
RUN pip install "numpy>=2.0"

# Step 3: install all other dependencies (excluding alibi – handled separately)
RUN grep -v '^alibi' requirements.txt | \
    grep -v '^#.*no-deps' | \
    pip install --no-cache-dir -r /dev/stdin

# Step 4: install alibi without deps (alibi requires numpy<2, but we use its
#         code only; all actual deps are already installed above)
RUN pip install alibi --no-deps || echo "[WARN] alibi unavailable – using fallback Anchors"

# ── Application source ────────────────────────────────────────────────────────
COPY breast-cancer.csv            ./
COPY xai_techniques.py            ./
COPY train_model_enhanced.py      ./
COPY train_model.py               ./
COPY app_enhanced.py              ./
COPY app.py                       ./
COPY dashboard.py                 ./

# ── Pre-train model artifacts during image build ──────────────────────────────
# This makes the container self-contained: no training needed at runtime.
RUN python3 train_model_enhanced.py && \
    echo "[Build] Model artifacts created successfully."

# ── Ports ─────────────────────────────────────────────────────────────────────
# 8501 = Streamlit dashboard
# 8080 = FastAPI inference server
EXPOSE 8501 8080

# ── Entrypoint script (selects mode via APP_MODE env var or first CLI arg) ─────
# Usage:
#   docker run ... (uses $APP_MODE, default=dashboard)
#   docker run ... dashboard
#   docker run ... api
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD []
