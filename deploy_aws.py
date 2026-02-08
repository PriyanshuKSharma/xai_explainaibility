# Purpose: Alternate SageMaker SDK deployment example (Python-based).
# Note: This file is optional and mainly for Studio/Notebook usage.
# Recommended CLI path for this repo:
#   ./scripts/deploy_sagemaker.sh
# If you run this file, ensure:
#   1) random_forest_model.pkl exists (python train_model.py)
#   2) bucket/role values below are valid

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn import SKLearnModel

# Set your AWS region and role
region = 'ap-south-1'  # Replace with your region
role = get_execution_role()  # Or specify your role ARN

# Create SageMaker session
sagemaker_session = sagemaker.Session()

# Upload model to S3
bucket = 'xai-explanability-trial'  # Replace with your S3 bucket
prefix = 'models'
model_data = sagemaker_session.upload_data(path='random_forest_model.pkl', bucket=bucket, key_prefix=prefix)

# Create SKLearn model
sklearn_model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point='inference.py',  # Need to create this script
    framework_version='1.0-1',
    py_version='py3',
)

# Deploy to endpoint
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='breast-cancer-endpoint',
)

print(f"Model deployed to endpoint: {predictor.endpoint_name}")
