import joblib
import os
import numpy as np

def model_fn(model_dir):
    """Load the model from the model_dir"""
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    model = joblib.load(model_path)
    return model

def predict_fn(input_data, model):
    """Make predictions"""
    predictions = model.predict(input_data)
    return predictions

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        import json
        data = json.loads(request_body)
        return np.array(data['instances'])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def output_fn(prediction, content_type):
    """Format output"""
    return str(prediction)