#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# docker-entrypoint.sh — XAI Explainability v3.0
#
# Selects run mode:
#   1. First CLI argument: 'dashboard' or 'api'
#   2. $APP_MODE environment variable (default: dashboard)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="${1:-${APP_MODE:-dashboard}}"

case "$MODE" in
  dashboard|streamlit)
    echo "[Entrypoint] Starting Streamlit dashboard on port 8501 …"
    exec streamlit run dashboard.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false
    ;;

  api|fastapi|uvicorn)
    echo "[Entrypoint] Starting FastAPI server on port 8080 …"
    exec uvicorn app_enhanced:app \
        --host 0.0.0.0 \
        --port 8080 \
        --workers 2 \
        --loop uvloop \
        --log-level info
    ;;

  *)
    echo "[Entrypoint] ERROR: Unknown mode '${MODE}'. Use 'dashboard' or 'api'."
    echo ""
    echo "Usage:"
    echo "  docker run -p 8501:8501 xai-explainability:v3.0              # dashboard (default)"
    echo "  docker run -p 8080:8080 -e APP_MODE=api xai-explainability:v3.0  # API"
    echo "  docker run xai-explainability:v3.0 dashboard"
    echo "  docker run xai-explainability:v3.0 api"
    exit 1
    ;;
esac
