#!/usr/bin/env bash
set -euo pipefail

# Purpose: Build Docker image and deploy it to Vertex AI endpoint.
# Run after training artifacts are available:
#   python train_model.py
# Required env vars:
#   PROJECT_ID, REGION, REPO_NAME, IMAGE_NAME, IMAGE_TAG,
#   MODEL_DISPLAY_NAME, ENDPOINT_DISPLAY_NAME
# Run:
#   ./scripts/deploy_vertex.sh

# Required environment variables:
# PROJECT_ID, REGION, REPO_NAME, IMAGE_NAME, IMAGE_TAG, MODEL_DISPLAY_NAME, ENDPOINT_DISPLAY_NAME

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${REPO_NAME:?Set REPO_NAME}"
: "${IMAGE_NAME:?Set IMAGE_NAME}"
: "${IMAGE_TAG:?Set IMAGE_TAG}"
: "${MODEL_DISPLAY_NAME:?Set MODEL_DISPLAY_NAME}"
: "${ENDPOINT_DISPLAY_NAME:?Set ENDPOINT_DISPLAY_NAME}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

gcloud config set project "${PROJECT_ID}"
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com

gcloud artifacts repositories describe "${REPO_NAME}" \
  --location "${REGION}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Docker repo for XAI inference images"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"

gcloud ai models upload \
  --region="${REGION}" \
  --display-name="${MODEL_DISPLAY_NAME}" \
  --container-image-uri="${IMAGE_URI}" \
  --container-predict-route="/predict" \
  --container-health-route="/health"

MODEL_ID=$(gcloud ai models list \
  --region="${REGION}" \
  --filter="displayName=${MODEL_DISPLAY_NAME}" \
  --sort-by="~createTime" \
  --limit=1 \
  --format="value(name)" | awk -F'/' '{print $NF}')

ENDPOINT_ID=$(gcloud ai endpoints list \
  --region="${REGION}" \
  --filter="displayName=${ENDPOINT_DISPLAY_NAME}" \
  --format="value(name)" | head -n1 | awk -F'/' '{print $NF}')

if [[ -z "${ENDPOINT_ID}" ]]; then
  gcloud ai endpoints create \
    --region="${REGION}" \
    --display-name="${ENDPOINT_DISPLAY_NAME}"

  ENDPOINT_ID=$(gcloud ai endpoints list \
    --region="${REGION}" \
    --filter="displayName=${ENDPOINT_DISPLAY_NAME}" \
    --sort-by="~createTime" \
    --limit=1 \
    --format="value(name)" | awk -F'/' '{print $NF}')
fi

gcloud ai endpoints deploy-model "${ENDPOINT_ID}" \
  --region="${REGION}" \
  --model="${MODEL_ID}" \
  --display-name="${MODEL_DISPLAY_NAME}" \
  --machine-type="n1-standard-2" \
  --traffic-split=0=100

echo "Vertex AI deployment complete."
echo "MODEL_ID=${MODEL_ID}"
echo "ENDPOINT_ID=${ENDPOINT_ID}"
