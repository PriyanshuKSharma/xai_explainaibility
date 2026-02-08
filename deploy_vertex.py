from __future__ import annotations

# Purpose: Python-only Vertex AI deployment flow using Vertex prebuilt sklearn container.
# Prerequisites:
#   1) python train_model.py
#   2) gcloud auth login
#   3) gcloud auth application-default login
# Run:
#   python deploy_vertex.py
# Optional env overrides:
#   GCP_PROJECT_ID, GCP_REGION, GCS_BUCKET_NAME, MODEL_FILE,
#   VERTEX_MODEL_DISPLAY_NAME, VERTEX_ENDPOINT_DISPLAY_NAME

import os
from datetime import datetime, timezone
from pathlib import Path

from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import aiplatform, storage

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "mqi-ims").strip()
REGION = os.getenv("GCP_REGION", "us-central1").strip()
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "xai_explanability").strip()
MODEL_PATH = Path(os.getenv("MODEL_FILE", "random_forest_model.pkl"))
MODEL_DISPLAY_NAME = os.getenv("VERTEX_MODEL_DISPLAY_NAME", "breast-cancer-xai-model").strip()
ENDPOINT_DISPLAY_NAME = os.getenv("VERTEX_ENDPOINT_DISPLAY_NAME", "breast-cancer-endpoint").strip()

if not PROJECT_ID or not REGION or not BUCKET_NAME:
    raise ValueError(
        "Missing config. Set GCP_PROJECT_ID, GCP_REGION, and GCS_BUCKET_NAME."
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. Run: python3 train_model.py"
    )

try:
    aiplatform.init(project=PROJECT_ID, location=REGION)
    storage_client = storage.Client(project=PROJECT_ID)
except DefaultCredentialsError as exc:
    raise RuntimeError(
        "Google Cloud credentials not found.\n"
        "Run:\n"
        "  gcloud auth login\n"
        "  gcloud auth application-default login\n"
        f"  gcloud config set project {PROJECT_ID}\n"
        "Or set GOOGLE_APPLICATION_CREDENTIALS to a service account key file."
    ) from exc

bucket = storage_client.bucket(BUCKET_NAME)
model_run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
model_prefix = f"models/{model_run_id}"
blob = bucket.blob(f"{model_prefix}/model.pkl")
blob.upload_from_filename(str(MODEL_PATH))

try:
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=f"gs://{BUCKET_NAME}/{model_prefix}/",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )

    endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1,
    )
except GoogleAPIError as exc:
    raise RuntimeError(f"Vertex AI deployment failed: {exc}") from exc

print(f"Model deployed to endpoint: {endpoint.resource_name}")
