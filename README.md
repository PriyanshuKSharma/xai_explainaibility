# XAI Explainability — Enhanced v2.0

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-green)](https://shap.readthedocs.io/)
[![LIME](https://img.shields.io/badge/LIME-0.2.0-lightgreen)](https://lime-ml.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-blue)](https://lightgbm.readthedocs.io/)

A **comprehensive, production-ready** implementation of Explainable AI (XAI) techniques applied to breast cancer diagnosis. This enhanced version adds five new explanation methods, a full-featured interactive dashboard, multi-model support, and a robust async API.

---

## What's New in v2.0

| Area | v1.0 | v2.0 |
|---|---|---|
| XAI Methods | SHAP, LIME | + **Anchors**, **Integrated Gradients**, **Counterfactuals** |
| Models | Random Forest | + **XGBoost**, **LightGBM**, **PyTorch MLP**, Gradient Boosting |
| Dashboard | None | **Streamlit app** with 8 interactive tabs |
| API | Basic FastAPI | **Async**, SHAP-in-response, **LRU cache**, rate limiting, `/compare`, `/metrics` |
| Hyperparameter tuning | Grid Search | + **Random Search**, **Bayesian (Optuna)** |
| Feature engineering | Raw features | + derived ratio / growth features |
| Explanation quality | None | **Faithfulness**, **Completeness**, **Complexity** metrics |

---

## Project Structure

```
xai_explainability/
├── 📓 XAI_algo.ipynb              # Main notebook (original)
│
├── 🔬 xai_techniques.py           # NEW: All XAI methods
│   ├── SHAPExplainer              # TreeExplainer / KernelExplainer
│   ├── LIMEExplainer              # Tabular + stability + submodular pick
│   ├── AnchorsExplainer           # Rule-based IF-THEN (alibi)
│   ├── IntegratedGradientsExplainer  # NumPy approx + captum (PyTorch)
│   ├── CounterfactualExplainer    # DiCE + NUN fallback
│   └── ExplanationEvaluator       # Faithfulness / completeness / complexity
│
├── 🤖 train_model_enhanced.py     # NEW: Multi-model training pipeline
│   ├── load_and_preprocess()      # Feature engineering + scaling
│   ├── get_model_registry()       # RF, GB, XGBoost, LightGBM, LR
│   ├── MLPClassifierTorch         # PyTorch MLP (sklearn-compatible)
│   ├── optimise_model()           # Grid / Random / Bayesian (Optuna)
│   ├── train_and_evaluate()       # Cross-validated comparison
│   └── save_artifacts()           # Persist models + metadata
│
├── 🌐 app_enhanced.py             # NEW: Production FastAPI
│   ├── POST /predict              # Sync prediction + optional SHAP/LIME
│   ├── POST /explain/shap         # Full SHAP breakdown
│   ├── POST /explain/lime         # LIME for first instance
│   ├── POST /explain/counterfactual  # Counterfactual instances
│   ├── POST /compare              # Multi-model comparison
│   └── GET  /metrics              # Prometheus-style counters + cache stats
│
├── 📊 dashboard.py                # NEW: Streamlit interactive dashboard
│   ├── Tab 1: Data Explorer       # EDA, distributions, correlation heatmap
│   ├── Tab 2: Model Training      # ROC curves, metric comparison
│   ├── Tab 3: SHAP Explorer       # Global importance, waterfall, dependence
│   ├── Tab 4: LIME Explorer       # Per-instance rules + stability
│   ├── Tab 5: Anchors / IG        # Rule anchors + Integrated Gradients
│   ├── Tab 6: Counterfactuals     # "What would flip the prediction?"
│   ├── Tab 7: Explanation Quality # Faithfulness, completeness, complexity
│   └── Tab 8: Live Predict        # Real-time prediction + full XAI report
│
├── app.py                         # Original API (preserved for compatibility)
├── train_model.py                 # Original trainer (preserved)
├── breast-cancer.csv              # Wisconsin Breast Cancer dataset
├── requirements.txt               # Updated dependencies
├── Dockerfile                     # Container build
├── Research Paper/                # LaTeX documentation
└── scripts/                       # Cloud deployment scripts
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/PriyanshuKSharma/xai_explainaibility.git
cd xai_explainaibility

python -m venv xai-env
source xai-env/bin/activate   # Windows: xai-env\Scripts\activate

pip install -r requirements.txt
```

### 2. Train all models

```bash
python train_model_enhanced.py
# Optional flags:
#   --nn            include PyTorch MLP
#   --optimise xgboost   run Bayesian HPO for xgboost
```

### 3. Launch the interactive dashboard

```bash
streamlit run dashboard.py
# Opens http://localhost:8501
```

### 4. Start the enhanced API

```bash
uvicorn app_enhanced:app --host 0.0.0.0 --port 8080 --reload
# Docs: http://localhost:8080/docs
```

---

## XAI Methods — Theory & Usage

### SHAP (SHapley Additive exPlanations)

**Theory:** Based on Shapley values from cooperative game theory. Each feature's contribution is its average marginal contribution across all possible feature coalitions.

**Key properties:**
- **Efficiency** — SHAP values sum to the difference between the prediction and the expected output
- **Symmetry** — Features with equal contributions receive equal SHAP values
- **Dummy** — Features that change nothing receive SHAP = 0
- **Additivity** — Additive across features

**When to use:** Tree models (fast exact TreeExplainer), any model needing global + local explanations, when consistency is critical.

```python
from xai_techniques import SHAPExplainer

explainer = SHAPExplainer(model, X_train, feature_names)
importance = explainer.global_importance(X_test)      # ranked DataFrame
local = explainer.local_explanation(X_test, idx=5)    # dict with shap values
iv = explainer.interaction_values(X_test)             # pairwise interactions
```

---

### LIME (Local Interpretable Model-agnostic Explanations)

**Theory:** Fits a simple interpretable surrogate model (linear) in the neighbourhood of the instance of interest, using perturbed samples weighted by proximity.

**When to use:** Any model type, when local fidelity matters more than global consistency, text/image data.

```python
from xai_techniques import LIMEExplainer

explainer = LIMEExplainer(X_train, feature_names)
exp = explainer.explanation_as_dict(model, X_test, idx=3)

# Stability: how consistent are explanations?
stability = explainer.stability_analysis(model, X_test, idx=3, n_runs=20)

# Submodular pick: select diverse representative explanations
sp_exps = explainer.submodular_pick(model, X_test, num_exps=5)
```

---

### Anchors

**Theory:** Produces high-precision IF-THEN rules that "anchor" the prediction — the rule is sufficient for the prediction to hold with high probability regardless of the other feature values.

**Advantages over LIME:** Rules are human-readable, have defined precision and coverage statistics.

```python
from xai_techniques import AnchorsExplainer

explainer = AnchorsExplainer(X_train, feature_names)
result = explainer.explain(model, X_test, idx=0, threshold=0.95)
# result: { anchor_rules, precision, coverage, prediction }
```

*Requires:* `pip install alibi`

---

### Integrated Gradients

**Theory:** Attributes the prediction of a neural network (or any differentiable function) to its input features. Computes the integral of gradients along a straight path from a baseline input to the actual input.

**Axiomatic properties:** Sensitivity, Implementation Invariance, Completeness (attributions sum to prediction − baseline output).

```python
from xai_techniques import IntegratedGradientsExplainer

explainer = IntegratedGradientsExplainer(model, feature_names, baseline="mean")
result = explainer.explain(X_test, idx=0, n_steps=50, target_class=1)
# result: { feature_attributions, attribution_sum }
```

Works with sklearn models (finite-difference approximation) and PyTorch models (captum auto-diff).

---

### Counterfactual Explanations

**Theory:** Answers "What is the minimal change to the input that would flip the model's decision?" Counterfactuals provide actionable recourse.

**Two strategies:**
- **DiCE** — Diverse Counterfactual Explanations via gradient-based optimisation (when `dice-ml` is installed)
- **NUN fallback** — Nearest Unlike Neighbour in normalised feature space (always available)

```python
from xai_techniques import CounterfactualExplainer

explainer = CounterfactualExplainer(model, X_train, feature_names)
result = explainer.explain(X_test, idx=0, n_counterfactuals=3)
# result: { original_prediction, counterfactuals, deltas, distances }
sparsity = explainer.sparsity(result["deltas"][0])  # fraction of unchanged features
```

---

### Explanation Quality Metrics

```python
from xai_techniques import ExplanationEvaluator

quality = ExplanationEvaluator.summary(model, X_test, shap_explainer, feature_names)
# {
#   "faithfulness":  0.231   ← mean |Δpred| when removing top-5 SHAP features
#   "completeness":  0.0003  ← mean |base + Σshap - pred| (lower is better)
#   "complexity":    8.4     ← mean # non-zero SHAP features per instance
# }
```

---

## API Reference

### `POST /predict`

```json
{
  "instances": [{ "radius_mean": 17.99, "texture_mean": 10.38, "..." : "..." }],
  "include_shap": true,
  "include_lime": false
}
```

Response includes `predictions`, `predicted_labels`, `prediction_probabilities`, `shap_values` (if requested), `latency_ms`.

### `POST /explain/counterfactual`

```json
{ "instance": { "..." }, "n_counterfactuals": 3 }
```

### `POST /compare`

Compare all saved `model_*.pkl` files on the same instances.

### `GET /metrics`

```json
{
  "counters": { "predict_calls": 42, "cache_hits": 15, ... },
  "cache": { "size": 12, "hits": 15, "misses": 27, "hit_rate": 0.357 }
}
```

Full OpenAPI docs at `http://localhost:8080/docs`.

---

## Model Support

| Model | Type | XAI Support |
|---|---|---|
| Random Forest | Tree ensemble | SHAP TreeExplainer (fast, exact) |
| Gradient Boosting | Tree ensemble | SHAP TreeExplainer |
| XGBoost | Tree ensemble | SHAP TreeExplainer |
| LightGBM | Tree ensemble | SHAP TreeExplainer |
| Logistic Regression | Linear | SHAP LinearExplainer |
| PyTorch MLP | Neural network | SHAP GradientExplainer, IG via captum |

All models support LIME, Anchors, and Counterfactuals (model-agnostic).

---

## Cloud Deployment

See `CLOUD_DEPLOYMENT.md` for full Vertex AI and AWS SageMaker instructions. The `app_enhanced.py` is drop-in compatible with both platforms via `/predict` (Vertex) and `/invocations` + `/ping` (SageMaker).

---

## Dataset

Wisconsin Breast Cancer Diagnostic (UCI), 569 instances, 30 real-valued features, binary target (B/M). Feature categories: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each for mean, SE, and worst measurements.

---

## Research Paper

The `Research Paper/` folder contains LaTeX source for the accompanying academic write-up covering the theoretical background, experimental results, and comparison of XAI methods.

---

## Citation

```bibtex
@misc{xai_explainability_2025,
  title   = {XAI Explainability Project v2.0: SHAP, LIME, Anchors, Integrated Gradients and Counterfactuals},
  author  = {Priyanshu K. Sharma},
  year    = {2025},
  url     = {https://github.com/PriyanshuKSharma/xai_explainaibility}
}
```

## Acknowledgements

- [SHAP](https://github.com/slundberg/shap) — Scott Lundberg et al.
- [LIME](https://github.com/marcotcr/lime) — Marco Tulio Ribeiro et al.
- [alibi](https://github.com/SeldonIO/alibi) — Seldon
- [DiCE](https://github.com/interpretml/DiCE) — Microsoft Research
- [captum](https://captum.ai) — PyTorch / Meta AI
- [Optuna](https://optuna.org) — Preferred Networks
