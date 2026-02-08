# Manual Cloud Console Steps (No Local Scripts)

This guide explains how to do build, training, testing, and analysis directly from cloud consoles.

## 1. Google Cloud (Vertex AI) Manual Flow

### A. Initial setup in console
1. Open Google Cloud Console.
2. Select your project.
3. Go to `APIs & Services -> Library` and enable:
- Vertex AI API
- Artifact Registry API
- Cloud Storage API
- Cloud Build API
4. Go to `IAM & Admin -> IAM` and ensure your user/service account has Vertex AI Admin, Storage Admin, and Artifact Registry Admin (or equivalent least-privilege roles).

### B. Build and register model serving image
1. Go to `Artifact Registry -> Repositories` and create a Docker repo (for example, `xai-images`) in `us-central1`.
2. Go to `Cloud Build -> Triggers` and connect your source repo (GitHub/Cloud Source Repositories).
3. Create a trigger that builds `Dockerfile` from your repo and pushes to Artifact Registry.
4. Run the trigger manually once.
5. Verify image exists in `Artifact Registry -> xai-images`.

### C. Train and analyze (Colab-equivalent) in Vertex Workbench
1. Go to `Vertex AI -> Workbench` and create a notebook instance (Python 3 environment).
2. Open JupyterLab from Workbench.
3. Upload:
- `XAI_algo.ipynb`
- `breast-cancer.csv`
- `requirements.txt` (or install only required packages for notebook use)
4. In notebook terminal or first cell, run:
```bash
pip install --upgrade pip setuptools wheel
pip install pandas numpy scikit-learn matplotlib seaborn shap lime joblib plotly ipywidgets
```
5. Run notebook cells end-to-end for:
- data preprocessing
- model training
- SHAP global/local explanations
- LIME local explanations
6. Save output figures/notebook to GCS if required.

### D. Deploy endpoint manually from console
1. Go to `Vertex AI -> Models` and click `Upload`.
2. Choose `Import as container` and select your pushed image URI from Artifact Registry.
3. Set predict route `/predict` and health route `/health`.
4. After upload, click `Deploy to endpoint`.
5. Create a new endpoint (or choose existing), choose machine type, and deploy.

### E. Test endpoint manually in console
1. Go to `Vertex AI -> Endpoints -> <your-endpoint> -> Test & use`.
2. Paste a JSON body under `instances` with 30 feature fields.
3. Run prediction.
4. Confirm response includes:
- `predictions`
- `predicted_labels`
- `prediction_probabilities`

### F. Monitor and inspect
1. In endpoint page, open `Logs` and monitoring charts.
2. For deeper logs, go to `Logging -> Logs Explorer` and filter by Vertex endpoint resource.

## 2. AWS (SageMaker) Manual Flow

### A. Initial setup in console
1. Open AWS Console and select region (for example, `ap-south-1`).
2. Go to IAM and ensure you have permissions for SageMaker, ECR, S3, and CloudWatch.
3. Create/verify a SageMaker execution role.

### B. Create storage and image registry
1. Go to `S3` and create a bucket for artifacts/data.
2. Go to `ECR -> Repositories` and create repo path for your image (for example, `xai/inference`).

### C. Build and push image (console-centric with CloudShell)
1. Open `AWS CloudShell` from the console.
2. Set environment variables in CloudShell (replace placeholders):
```bash
export AWS_REGION="ap-south-1"
export AWS_ACCOUNT_ID="<your-aws-account-id>"
export ECR_REPO_NAME="xai"
export IMAGE_NAME="inference"
export IMAGE_TAG="v1"
```
3. Clone your repo:
```bash
git clone https://github.com/PriyanshuKSharma/xai_explainaibility.git
cd xai_explainaibility
```
4. Build model artifact first (if not already generated in repo):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-serving.txt
python train_model.py
```
5. Create ECR repository (if it does not already exist):
```bash
aws ecr describe-repositories \
  --region "${AWS_REGION}" \
  --repository-names "${ECR_REPO_NAME}/${IMAGE_NAME}" >/dev/null 2>&1 || \
aws ecr create-repository \
  --region "${AWS_REGION}" \
  --repository-name "${ECR_REPO_NAME}/${IMAGE_NAME}"
```
6. Authenticate Docker to ECR:
```bash
aws ecr get-login-password --region "${AWS_REGION}" | \
docker login \
  --username AWS \
  --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
```
7. Build and push container image:
```bash
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

docker build -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"
```
8. Verify image was pushed:
```bash
aws ecr describe-images \
  --region "${AWS_REGION}" \
  --repository-name "${ECR_REPO_NAME}/${IMAGE_NAME}" \
  --query "sort_by(imageDetails,& imagePushedAt)[-1].[imageTags[0],imagePushedAt,imageDigest]" \
  --output table
```
9. Console verification:
- Go to `ECR -> Repositories -> xai/inference`.
- Open `Images` tab.
- Confirm tag (`v1`) and latest push timestamp are visible.

### CloudShell storage troubleshooting (`No space left on device`)
If you see errors like:
- `Building wheel ... failed: [Errno 28] No space left on device`
- `Failed to build installable wheels`

Use this recovery flow in CloudShell:

```bash
# Check disk usage
df -h

# Clean common caches
rm -rf ~/.cache/pip ~/.cache/pypoetry ~/.npm ~/.cache/yarn
docker system prune -af || true

# Force pip to avoid persistent cache
export PIP_NO_CACHE_DIR=1
export PIP_CACHE_DIR=/tmp/pip-cache
mkdir -p /tmp/pip-cache
```

For CloudShell, install only minimum dependencies needed to create model artifacts:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --no-cache-dir pandas scikit-learn joblib
python train_model.py
```

Then continue Docker build/push steps. Avoid running full `pip install -r requirements.txt` in CloudShell.

### D. Train and analyze (Colab-equivalent) in SageMaker Studio
1. Go to `SageMaker -> Studio` and open/create a Studio domain.
2. Launch JupyterLab inside Studio.
3. Upload:
- `XAI_algo.ipynb`
- `breast-cancer.csv`
4. Install notebook dependencies in terminal:
```bash
pip install --upgrade pip setuptools wheel
pip install pandas numpy scikit-learn matplotlib seaborn shap lime joblib plotly ipywidgets
```
5. Run notebook cells for full EDA + SHAP + LIME analysis.
6. Save notebooks/plots back to S3 if needed.

### E. Deploy endpoint manually from SageMaker console
1. Go to `SageMaker -> Inference -> Models` and create model.
2. Select container image from ECR and set execution role.
3. Go to `Endpoint configurations` and create config (instance type/count).
4. Go to `Endpoints` and create endpoint using that config.

### F. Test endpoint manually in console
1. Go to `SageMaker -> Endpoints -> <your-endpoint> -> Test inference`.
2. Use `application/json` and send request with `instances` list.
3. Confirm response has predictions/probabilities.

### G. Monitor and inspect
1. Go to endpoint monitoring tabs in SageMaker.
2. Open `CloudWatch -> Logs` for container runtime logs.

## 3. What matches your Colab workflow

The following Colab activities are preserved on both clouds via managed notebooks:
- dataset upload and preprocessing
- model training and evaluation
- SHAP explainability plots
- LIME local explanations

The endpoint deployment is the production extension of that notebook workflow.
