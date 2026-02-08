#!/usr/bin/env bash
set -euo pipefail

# Required environment variables:
# AWS_REGION, AWS_ACCOUNT_ID, ECR_REPO_NAME, IMAGE_NAME, IMAGE_TAG,
# SAGEMAKER_ROLE_ARN, ENDPOINT_NAME

: "${AWS_REGION:?Set AWS_REGION}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
: "${ECR_REPO_NAME:?Set ECR_REPO_NAME}"
: "${IMAGE_NAME:?Set IMAGE_NAME}"
: "${IMAGE_TAG:?Set IMAGE_TAG}"
: "${SAGEMAKER_ROLE_ARN:?Set SAGEMAKER_ROLE_ARN}"
: "${ENDPOINT_NAME:?Set ENDPOINT_NAME}"

INSTANCE_TYPE="${INSTANCE_TYPE:-ml.m5.large}"
MODEL_NAME="${MODEL_NAME:-xai-rf-model-$(date +%Y%m%d%H%M%S)}"
ENDPOINT_CONFIG_NAME="${ENDPOINT_CONFIG_NAME:-xai-rf-config-$(date +%Y%m%d%H%M%S)}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}/${IMAGE_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
aws ecr create-repository --repository-name "${ECR_REPO_NAME}/${IMAGE_NAME}" --region "${AWS_REGION}" >/dev/null

aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build -t "${ECR_URI}" .
docker push "${ECR_URI}"

aws sagemaker create-model \
  --region "${AWS_REGION}" \
  --model-name "${MODEL_NAME}" \
  --execution-role-arn "${SAGEMAKER_ROLE_ARN}" \
  --primary-container "Image=${ECR_URI}"

aws sagemaker create-endpoint-config \
  --region "${AWS_REGION}" \
  --endpoint-config-name "${ENDPOINT_CONFIG_NAME}" \
  --production-variants "VariantName=AllTraffic,ModelName=${MODEL_NAME},InitialInstanceCount=1,InstanceType=${INSTANCE_TYPE},InitialVariantWeight=1"

if aws sagemaker describe-endpoint --region "${AWS_REGION}" --endpoint-name "${ENDPOINT_NAME}" >/dev/null 2>&1; then
  aws sagemaker update-endpoint \
    --region "${AWS_REGION}" \
    --endpoint-name "${ENDPOINT_NAME}" \
    --endpoint-config-name "${ENDPOINT_CONFIG_NAME}"
else
  aws sagemaker create-endpoint \
    --region "${AWS_REGION}" \
    --endpoint-name "${ENDPOINT_NAME}" \
    --endpoint-config-name "${ENDPOINT_CONFIG_NAME}"
fi

echo "SageMaker deployment started."
echo "MODEL_NAME=${MODEL_NAME}"
echo "ENDPOINT_CONFIG_NAME=${ENDPOINT_CONFIG_NAME}"
echo "ENDPOINT_NAME=${ENDPOINT_NAME}"
