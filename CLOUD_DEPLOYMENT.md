# Cloud Deployment Guide (GCP + AWS)

This file is dedicated to cloud deployment for this project.

## Why Use Cloud Services for This Project

Running this model on cloud platforms is useful when you need:
- Always-on inference endpoints instead of local notebooks.
- Team/shared access with stable URLs and managed authentication.
- Scalable serving (increase or decrease instances based on load).
- Centralized monitoring/logging for debugging and reliability.
- Production-style deployment workflows (image registry, endpoint rollout, updates).

## What Is Deployed

The deployed service is a REST inference API built from:
- `app.py` (prediction server)
- `Dockerfile` (container image build)
- `train_model.py` output (`random_forest_model.pkl`, `label_encoder.pkl`)

Prediction routes:
- Vertex AI: `POST /predict`, `GET /health`
- SageMaker: `POST /invocations`, `GET /ping`

## GCP: Vertex AI Purpose and Flow

### Why Vertex AI here
- Managed model hosting and endpoint lifecycle.
- Built-in integration with Artifact Registry and Cloud Logging.
- Standard way to expose an online ML endpoint on GCP.

### What the script does
Using `scripts/deploy_vertex.sh`:
1. Ensures Artifact Registry repo exists.
2. Builds/pushes container image.
3. Uploads Vertex model using that image.
4. Creates endpoint (if needed).
5. Deploys model to endpoint and routes traffic.

Using `deploy_vertex.py` (alternate flow):
1. Uploads model artifact (`model.pkl`) to GCS.
2. Registers model with Vertex prebuilt sklearn container.
3. Creates endpoint and deploys model.

## AWS: SageMaker Purpose and Flow

### Why SageMaker here
- Managed ML endpoint hosting on AWS.
- Tight integration with ECR, IAM, and CloudWatch.
- Easy endpoint updates for new model/container versions.

### What the script does
Using `scripts/deploy_sagemaker.sh`:
1. Ensures ECR repo exists.
2. Builds/pushes container image to ECR.
3. Creates SageMaker model referencing the image.
4. Creates endpoint config (instance/traffic settings).
5. Creates or updates the endpoint.

## What You Get After Deployment

- A production-style HTTPS endpoint for inference.
- JSON response with:
  - `predictions`
  - `predicted_labels`
  - `prediction_probabilities`
- Logs and health status in the cloud console.

## When to Choose Which Cloud

- Choose Vertex AI if your stack is already on GCP and you want GCP-native IAM/ops.
- Choose SageMaker if your stack is on AWS and you want AWS-native IAM/ops.
- For this project, both serve the same containerized inference logic.

## Quick Start Commands

### Vertex AI
```bash
python train_model.py
./scripts/deploy_vertex.sh
```

### SageMaker
```bash
python train_model.py
./scripts/deploy_sagemaker.sh
```
