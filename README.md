# XAI Explainability Project

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)](https://colab.research.google.com/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)](https://shap.readthedocs.io/)
[![LIME](https://img.shields.io/badge/LIME-Interpretability-lightgreen)](https://lime-ml.readthedocs.io/)

A comprehensive implementation of Explainable AI (XAI) techniques using SHAP and LIME for machine learning model interpretability. This project provides hands-on experience with state-of-the-art explainability methods for understanding black-box machine learning models.

## 📋 Overview

Explainable AI (XAI) is crucial for building trust in machine learning systems, especially in high-stakes domains like healthcare, finance, and legal systems. This project demonstrates practical implementation of two leading XAI techniques:

- **SHAP (SHapley Additive exPlanations)**: Provides unified framework for interpreting model predictions based on game theory
- **LIME (Local Interpretable Model-agnostic Explanations)**: Explains individual predictions by learning local surrogate models

The project uses a breast cancer diagnostic dataset to showcase how these techniques can provide insights into medical diagnosis predictions, making AI decisions transparent and interpretable.

## 🚀 Features

### Core Functionality
- **Complete ML Pipeline**: End-to-end machine learning workflow from data preprocessing to model deployment
- **Interactive Data Upload**: Google Colab-compatible file upload system for seamless data integration
- **Multiple Model Support**: Compatible with various ML algorithms (Random Forest, SVM, Neural Networks)
- **Real-time Explanations**: Generate explanations for individual predictions instantly

### XAI Capabilities
- **Global Interpretability**: Understand overall model behavior and feature importance
- **Local Explanations**: Explain individual predictions with feature contributions
- **Feature Interaction Analysis**: Discover how features interact to influence predictions
- **Visualization Suite**: Rich set of plots and charts for explanation visualization

### Production Features
- **Model Persistence**: Save and load trained models with explanations
- **Cloud Deployment Ready**: Vertex AI compatible for production deployment
- **Scalable Architecture**: Designed for handling large datasets and multiple models
- **Documentation**: Comprehensive LaTeX documentation included

## 📁 Project Structure

```
Sem-6/
├── 📓 XAI_algo.ipynb              # Main implementation notebook
│   ├── Data Upload & Preprocessing
│   ├── Model Training & Evaluation
│   ├── SHAP Implementation
│   ├── LIME Implementation
│   └── Model Deployment
├── 📊 breast-cancer.csv           # Breast cancer diagnostic dataset
├── 📖 README.md                   # This comprehensive guide
├── 📄 xai_explainability.tex      # Academic documentation
├── 📋 PriyanshuKSharma_EH_MidTerm (1).pdf  # Project report
└── 🔧 .qodo/                      # Development tools configuration
    ├── agents/
    └── workflows/
```

## 🛠️ Installation & Setup

### Prerequisites

- **Python**: 3.7 or higher
- **Environment**: Jupyter Notebook, JupyterLab, or Google Colab
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: At least 1GB free space

### Quick Start (Google Colab)

1. **Open in Colab**: Click the "Open in Colab" button or upload the notebook
2. **Runtime Setup**: Select GPU runtime for faster processing (optional)
3. **Install Dependencies**: Run the installation cell (dependencies auto-installed)
4. **Upload Dataset**: Use the built-in file upload widget

### Local Installation

#### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Sem-6
```

#### Step 2: Create Virtual Environment
```bash
# Using conda
conda create -n xai-env python=3.8
conda activate xai-env

# Using venv
python -m venv xai-env
source xai-env/bin/activate  # Linux/Mac
# or
xai-env\Scripts\activate     # Windows
```

#### Step 3: Install Dependencies
```bash
# Core ML libraries
pip install pandas==1.5.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.1
pip install seaborn==0.12.2

# XAI libraries
pip install shap==0.42.1
pip install lime==0.2.0.1

# Additional utilities
pip install joblib==1.3.1
pip install plotly==5.15.0
pip install ipywidgets==8.0.7

# For Colab compatibility (optional)
pip install google-colab
```

#### Step 4: Launch Jupyter
```bash
jupyter notebook XAI_algo.ipynb
# or
jupyter lab XAI_algo.ipynb
```

## 📊 Dataset Information

### Breast Cancer Diagnostic Dataset

The project uses the Wisconsin Breast Cancer Diagnostic dataset, a well-established benchmark in medical AI:

**Dataset Characteristics:**
- **Samples**: 569 instances
- **Features**: 30 numerical features
- **Target**: Binary classification (Malignant/Benign)
- **Missing Values**: None
- **Feature Types**: Real-valued measurements

**Feature Categories:**
1. **Radius**: Mean distances from center to perimeter points
2. **Texture**: Standard deviation of gray-scale values
3. **Perimeter**: Tumor perimeter measurements
4. **Area**: Tumor area calculations
5. **Smoothness**: Local variation in radius lengths
6. **Compactness**: Perimeter² / area - 1.0
7. **Concavity**: Severity of concave portions
8. **Concave Points**: Number of concave portions
9. **Symmetry**: Tumor symmetry measurements
10. **Fractal Dimension**: "Coastline approximation" - 1

**Statistical Summary:**
```
Feature Statistics:
- Mean radius: 14.13 ± 3.52
- Mean texture: 19.29 ± 4.30
- Mean perimeter: 91.97 ± 24.30
- Target distribution: 357 Benign (62.7%), 212 Malignant (37.3%)
```

## 🔧 Usage Guide

### Google Colab Workflow

#### 1. Environment Setup
```python
# The notebook automatically handles Colab setup
from google.colab import files
import pandas as pd
import numpy as np
```

#### 2. Data Upload
```python
# Interactive file upload
uploaded = files.upload()
for filename in uploaded.keys():
    print(f'Uploaded: {filename} ({len(uploaded[filename])} bytes)')
```

#### 3. Model Training
```python
# Automated model training pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Data preprocessing and model training handled automatically
```

### Local Development Workflow

#### 1. Data Loading
```python
import pandas as pd

# Load dataset
df = pd.read_csv('breast-cancer.csv')
print(f"Dataset shape: {df.shape}")
print(f"Features: {df.columns.tolist()}")
```

#### 2. Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## 📈 Detailed Implementation Guide

### 1. Data Preprocessing Pipeline

#### Data Quality Assessment
```python
# Comprehensive data analysis
def analyze_dataset(df):
    print("=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
    
    print("\n=== Data Types ===")
    print(df.dtypes.value_counts())
    
    print("\n=== Target Distribution ===")
    print(df['target'].value_counts(normalize=True))
```

#### Feature Engineering
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    # Handle categorical variables
    le = LabelEncoder()
    if df['target'].dtype == 'object':
        df['target'] = le.fit_transform(df['target'])
    
    # Feature scaling
    scaler = StandardScaler()
    feature_cols = df.columns.drop('target')
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df, scaler, le
```

### 2. Advanced Model Training

#### Multi-Model Comparison
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_multiple_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
    
    return models, results
```

#### Hyperparameter Optimization
```python
from sklearn.model_selection import GridSearchCV

def optimize_model(X_train, y_train):
    # Random Forest hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_
```

### 3. SHAP Implementation Deep Dive

#### Global Explanations
```python
import shap
import matplotlib.pyplot as plt

def generate_shap_explanations(model, X_train, X_test, feature_names):
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, use class 1 (positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Global feature importance
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Global Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Feature importance bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return explainer, shap_values
```

#### Local Explanations
```python
def explain_individual_prediction(explainer, X_test, feature_names, instance_idx=0):
    # Get SHAP values for specific instance
    shap_values = explainer.shap_values(X_test.iloc[[instance_idx]])
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Waterfall plot for individual prediction
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[0],
        X_test.iloc[instance_idx],
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}')
    plt.tight_layout()
    plt.show()
    
    # Force plot
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[0],
        X_test.iloc[instance_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot - Instance {instance_idx}')
    plt.tight_layout()
    plt.show()
```

#### Advanced SHAP Analysis
```python
def advanced_shap_analysis(explainer, X_test, feature_names):
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Dependence plots for top features
    feature_importance = np.abs(shap_values).mean(0)
    top_features = np.argsort(feature_importance)[-5:]  # Top 5 features
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature_idx in enumerate(top_features):
        plt.sca(axes[i])
        shap.dependence_plot(
            feature_idx, shap_values, X_test,
            feature_names=feature_names,
            show=False
        )
        plt.title(f'Dependence Plot: {feature_names[feature_idx]}')
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.show()
```

### 4. LIME Implementation Deep Dive

#### Tabular Explainer Setup
```python
import lime
import lime.lime_tabular

def setup_lime_explainer(X_train, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True,
        random_state=42
    )
    return explainer
```

#### Individual Explanations
```python
def explain_with_lime(lime_explainer, model, X_test, instance_idx=0, num_features=10):
    # Generate explanation for specific instance
    explanation = lime_explainer.explain_instance(
        X_test.iloc[instance_idx].values,
        model.predict_proba,
        num_features=num_features
    )
    
    # Show in notebook
    explanation.show_in_notebook(show_table=True)
    
    # Save as HTML
    explanation.save_to_file(f'lime_explanation_instance_{instance_idx}.html')
    
    # Get explanation as list
    exp_list = explanation.as_list()
    print(f"\nTop {len(exp_list)} feature contributions:")
    for feature, contribution in exp_list:
        print(f"{feature}: {contribution:.4f}")
    
    return explanation
```

#### Batch Explanations
```python
def batch_lime_explanations(lime_explainer, model, X_test, num_instances=10, num_features=5):
    explanations = []
    
    for i in range(min(num_instances, len(X_test))):
        exp = lime_explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba,
            num_features=num_features
        )
        explanations.append(exp)
        
        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f"Generated {i + 1}/{num_instances} explanations")
    
    return explanations
```

### 5. Model Deployment & Persistence

#### Model Serialization
```python
import joblib
import json
from datetime import datetime

def save_model_with_metadata(model, scaler, feature_names, model_metrics, filename_prefix="xai_model"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"{filename_prefix}_{timestamp}.joblib"
    joblib.dump(model, model_filename)
    
    # Save scaler
    scaler_filename = f"scaler_{timestamp}.joblib"
    joblib.dump(scaler, scaler_filename)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': type(model).__name__,
        'feature_names': feature_names,
        'metrics': model_metrics,
        'model_file': model_filename,
        'scaler_file': scaler_filename
    }
    
    metadata_filename = f"metadata_{timestamp}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved: {model_filename}")
    print(f"Scaler saved: {scaler_filename}")
    print(f"Metadata saved: {metadata_filename}")
    
    return model_filename, scaler_filename, metadata_filename
```

#### Model Loading
```python
def load_model_with_metadata(metadata_filename):
    # Load metadata
    with open(metadata_filename, 'r') as f:
        metadata = json.load(f)
    
    # Load model and scaler
    model = joblib.load(metadata['model_file'])
    scaler = joblib.load(metadata['scaler_file'])
    
    print(f"Loaded model: {metadata['model_type']}")
    print(f"Training timestamp: {metadata['timestamp']}")
    print(f"Model metrics: {metadata['metrics']}")
    
    return model, scaler, metadata
```

#### Vertex AI Deployment Preparation
```python
def prepare_vertex_ai_deployment(model, scaler, feature_names):
    """
    Prepare model for Google Cloud Vertex AI deployment
    """
    
    # Create prediction function
    def predict_fn(instances):
        # Convert to DataFrame
        df = pd.DataFrame(instances, columns=feature_names)
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Make predictions
        predictions = model.predict_proba(df_scaled)
        
        return predictions.tolist()
    
    # Save deployment artifacts
    deployment_artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'predict_function': predict_fn
    }
    
    joblib.dump(deployment_artifacts, 'vertex_ai_deployment.joblib')
    
    # Create requirements.txt for deployment
    requirements = [
        'scikit-learn==1.3.0',
        'pandas==1.5.3',
        'numpy==1.24.3',
        'joblib==1.3.1'
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    
    print("Vertex AI deployment artifacts created:")
    print("- vertex_ai_deployment.joblib")
    print("- requirements.txt")
```

## 🎯 Advanced XAI Techniques

### SHAP Advanced Features

#### 1. Interaction Values
```python
def analyze_feature_interactions(explainer, X_test, feature_names):
    # Calculate interaction values
    interaction_values = explainer.shap_interaction_values(X_test[:100])  # Subset for performance
    
    # Plot interaction heatmap
    plt.figure(figsize=(12, 10))
    interaction_sum = np.abs(interaction_values).sum(0)
    sns.heatmap(interaction_sum, 
                xticklabels=feature_names, 
                yticklabels=feature_names,
                annot=True, 
                cmap='viridis')
    plt.title('Feature Interaction Heatmap')
    plt.tight_layout()
    plt.show()
```

#### 2. Clustering Analysis
```python
from sklearn.cluster import KMeans

def shap_clustering_analysis(shap_values, X_test, n_clusters=3):
    # Cluster based on SHAP values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(shap_values)
    
    # Visualize clusters
    plt.figure(figsize=(15, 5))
    
    for i in range(n_clusters):
        plt.subplot(1, n_clusters, i+1)
        cluster_mask = clusters == i
        shap.summary_plot(shap_values[cluster_mask], 
                         X_test[cluster_mask], 
                         plot_type="bar", 
                         show=False)
        plt.title(f'Cluster {i+1} (n={cluster_mask.sum()})')
    
    plt.tight_layout()
    plt.show()
    
    return clusters
```

### LIME Advanced Features

#### 1. Submodular Pick
```python
from lime import submodular_pick

def lime_submodular_pick(lime_explainer, model, X_test, num_features=5, num_exps_desired=10):
    # Select diverse set of explanations
    sp_obj = submodular_pick.SubmodularPick(
        lime_explainer,
        X_test.values,
        model.predict_proba,
        num_features=num_features,
        num_exps_desired=num_exps_desired
    )
    
    # Show selected explanations
    for i, exp in enumerate(sp_obj.explanations):
        print(f"\n=== Explanation {i+1} ===")
        exp.show_in_notebook(show_table=True)
    
    return sp_obj
```

#### 2. Stability Analysis
```python
def lime_stability_analysis(lime_explainer, model, instance, num_runs=50, num_features=10):
    explanations = []
    
    for _ in range(num_runs):
        exp = lime_explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=num_features
        )
        explanations.append(dict(exp.as_list()))
    
    # Analyze stability
    all_features = set()
    for exp in explanations:
        all_features.update(exp.keys())
    
    stability_scores = {}
    for feature in all_features:
        values = [exp.get(feature, 0) for exp in explanations]
        stability_scores[feature] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'frequency': sum(1 for v in values if v != 0) / len(values)
        }
    
    return stability_scores
```

## 📚 Comprehensive Learning Resources

### Theoretical Background

#### SHAP Theory
- **Game Theory Foundation**: Based on Shapley values from cooperative game theory
- **Efficiency**: Sum of SHAP values equals difference between prediction and expected value
- **Symmetry**: Features with equal contributions have equal SHAP values
- **Dummy**: Features that don't affect prediction have zero SHAP value
- **Additivity**: SHAP values are additive across features

#### LIME Theory
- **Local Surrogate Models**: Approximates black-box model locally
- **Interpretable Representations**: Uses simpler feature representations
- **Locality**: Focuses on neighborhood around instance of interest
- **Model Agnostic**: Works with any machine learning model

### Best Practices

#### When to Use SHAP vs LIME

**Use SHAP when:**
- You need theoretically grounded explanations
- Global and local explanations are both important
- You're working with tree-based models
- Consistency across explanations is crucial

**Use LIME when:**
- You need model-agnostic explanations
- Working with complex models (deep learning, ensembles)
- Local explanations are the primary focus
- You need explanations for different data types (text, images)

#### Explanation Quality Assessment

```python
def assess_explanation_quality(model, X_test, y_test, explanations):
    # Fidelity: How well do explanations predict model behavior
    fidelity_scores = []
    
    for i, exp in enumerate(explanations):
        # Create simplified model based on explanation
        important_features = [f for f, _ in exp.as_list()]
        
        # Calculate local fidelity
        # Implementation depends on specific explanation format
        pass
    
    # Stability: Consistency across similar instances
    # Comprehensibility: Human understanding metrics
    # Coverage: Fraction of instances that can be explained
    
    return {
        'fidelity': np.mean(fidelity_scores),
        'stability': calculate_stability(explanations),
        'coverage': calculate_coverage(explanations)
    }
```

## 🔍 Troubleshooting & FAQ

### Common Issues

#### 1. Memory Issues with Large Datasets
```python
# Solution: Use sampling for SHAP calculations
def memory_efficient_shap(explainer, X_test, sample_size=1000):
    if len(X_test) > sample_size:
        sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_idx]
    else:
        X_sample = X_test
    
    return explainer.shap_values(X_sample)
```

#### 2. LIME Timeout Issues
```python
# Solution: Reduce number of samples
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    num_samples=1000,  # Reduce from default 5000
    random_state=42
)
```

#### 3. Visualization Issues in Colab
```python
# Solution: Use matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# For SHAP plots
shap.initjs()  # Initialize JavaScript visualizations
```

### Performance Optimization

#### 1. Parallel Processing
```python
from joblib import Parallel, delayed

def parallel_lime_explanations(lime_explainer, model, X_test, n_jobs=-1):
    def explain_instance(i):
        return lime_explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba,
            num_features=10
        )
    
    explanations = Parallel(n_jobs=n_jobs)(
        delayed(explain_instance)(i) for i in range(len(X_test))
    )
    
    return explanations
```

#### 2. Caching Results
```python
import pickle
from functools import lru_cache

# Cache SHAP values
def cache_shap_values(explainer, X_test, cache_file='shap_cache.pkl'):
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        shap_values = explainer.shap_values(X_test)
        with open(cache_file, 'wb') as f:
            pickle.dump(shap_values, f)
        return shap_values
```

## 🚀 Production Deployment

### Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
```

### Flask API Wrapper

```python
# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model artifacts
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
explainer = joblib.load('explainer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Preprocess
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict_proba(df_scaled)[0]
    
    # Explain
    shap_values = explainer.shap_values(df_scaled)[0]
    
    return jsonify({
        'prediction': prediction.tolist(),
        'explanation': shap_values.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Vertex AI Custom Container

```python
# predictor.py for Vertex AI
from google.cloud.aiplatform.prediction.sklearn.predictor import SklearnPredictor
import joblib
import pandas as pd

class XAIPredictor(SklearnPredictor):
    def __init__(self):
        super().__init__()
        self._explainer = None
    
    def load(self, artifacts_uri):
        super().load(artifacts_uri)
        self._explainer = joblib.load(f"{artifacts_uri}/explainer.joblib")
    
    def predict(self, instances):
        predictions = super().predict(instances)
        
        # Add explanations
        explanations = []
        for instance in instances:
            shap_vals = self._explainer.shap_values([instance])[0]
            explanations.append(shap_vals.tolist())
        
        return {
            'predictions': predictions,
            'explanations': explanations
        }
```

## 📊 Evaluation Metrics

### Model Performance Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def comprehensive_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("=== Model Performance ===")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return metrics
```

### Explanation Quality Metrics

```python
def explanation_faithfulness(model, X_test, shap_values, num_samples=100):
    """
    Measure how faithful SHAP explanations are to model predictions
    """
    faithfulness_scores = []
    
    for i in range(min(num_samples, len(X_test))):
        # Original prediction
        original_pred = model.predict_proba([X_test.iloc[i]])[0][1]
        
        # Remove top features and predict
        feature_importance = np.abs(shap_values[i])
        top_features = np.argsort(feature_importance)[-3:]  # Top 3 features
        
        modified_instance = X_test.iloc[i].copy()
        modified_instance.iloc[top_features] = 0  # Remove top features
        
        modified_pred = model.predict_proba([modified_instance])[0][1]
        
        # Calculate faithfulness (larger change = more faithful)
        faithfulness = abs(original_pred - modified_pred)
        faithfulness_scores.append(faithfulness)
    
    return np.mean(faithfulness_scores)
```

## 🤝 Contributing Guidelines

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd xai-explainability

# Create development environment
conda create -n xai-dev python=3.8
conda activate xai-dev

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all functions
- Include unit tests for new features

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_shap_explanations.py
```

## 📄 License & Citation

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Citation
If you use this project in your research, please cite:

```bibtex
@misc{xai_explainability_2024,
  title={XAI Explainability Project: Comprehensive Implementation of SHAP and LIME},
  author={CFIS Semester 6 Team},
  year={2024},
  howpublished={\url{https://github.com/your-repo/xai-explainability}}
}
```

### Acknowledgments

- SHAP library developers: https://github.com/slundberg/shap
- LIME library developers: https://github.com/marcotcr/lime
- Scikit-learn community: https://scikit-learn.org/
- Google Colab team for providing free GPU access

## 📞 Support & Contact

### Getting Help

1. **Documentation**: Check this README and inline code comments
2. **Issues**: Open a GitHub issue for bugs or feature requests
3. **Discussions**: Use GitHub Discussions for questions
4. **Email**: Contact the development team at [email]

### Reporting Issues

When reporting issues, please include:
- Python version and environment details
- Complete error traceback
- Minimal code example to reproduce the issue
- Expected vs actual behavior

### Feature Requests

We welcome feature requests! Please provide:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Any relevant examples or references

---

**Project Status**: Active Development | **Last Updated**: 2024 | **Version**: 1.0.0

**Keywords**: Explainable AI, XAI, SHAP, LIME, Machine Learning, Interpretability, Healthcare AI, Model Explanation, Feature Importance, Google Colab, Vertex AI
