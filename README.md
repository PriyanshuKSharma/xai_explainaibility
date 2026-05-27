# XAI Explainability — Enhanced v3.0

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-green)](https://shap.readthedocs.io/)
[![LIME](https://img.shields.io/badge/LIME-0.2.0-lightgreen)](https://lime-ml.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-blue)](https://lightgbm.readthedocs.io/)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Compliant-purple)](https://artificialintelligenceact.eu/)

A **comprehensive, production-ready** implementation of Explainable AI (XAI) techniques applied to breast cancer diagnosis. v3.0 adds a five-method Hybrid Explainer, feature engineering ablation study, regulatory compliance analysis (EU AI Act / GDPR), a new Hybrid XAI dashboard tab, and a substantially enhanced research paper.

---

## What's New in v3.0

| Area | v2.0 | v3.0 |
|---|---|---|
| XAI Methods | SHAP, LIME, Anchors, IG, CF | + **HybridExplainer** (5-method ensemble) |
| Benchmarking | None | **XAIBenchmark** (timing + agreement metrics) |
| Feature Engineering | 4 engineered features | + **Ablation study** (incremental impact) |
| Dashboard | 8 tabs | + **Tab 9: Hybrid XAI** (ensemble + heatmap) |
| Research Paper | SHAP + LIME only | **All 5 methods** + ablation + EU AI Act/GDPR |
| Regulatory | Not addressed | **EU AI Act & GDPR** compliance analysis |
| Bibliography | 14 references | **26 references** (Anchors, IG, CF, LLM-XAI) |
| Accuracy | 94.7% (RF) | **97.4%** (hybrid ensemble) |

---

## Project Structure

```
xai_explainability/
├── 📓 XAI_algo.ipynb              # Main notebook (original)
│
├── 🔬 xai_techniques.py           # All XAI methods (v3.0: +HybridExplainer, +XAIBenchmark)
│   ├── SHAPExplainer              # TreeExplainer / KernelExplainer
│   ├── LIMEExplainer              # Tabular + stability + submodular pick
│   ├── AnchorsExplainer           # Rule-based IF-THEN (alibi)
│   ├── IntegratedGradientsExplainer  # NumPy approx + captum (PyTorch)
│   ├── CounterfactualExplainer    # DiCE + NUN fallback
│   ├── ExplanationEvaluator       # Faithfulness / completeness / complexity
│   ├── HybridExplainer [v3.0]     # Confidence-weighted 5-method ensemble
│   └── XAIBenchmark    [v3.0]     # Timing + agreement benchmarking utility
│
├── 🤖 train_model_enhanced.py     # Multi-model training pipeline (v3.0: +ablation)
│   ├── load_and_preprocess()      # Feature engineering + scaling
│   ├── get_model_registry()       # RF, GB, XGBoost, LightGBM, LR
│   ├── MLPClassifierTorch         # PyTorch MLP (sklearn-compatible)
│   ├── optimise_model()           # Grid / Random / Bayesian (Optuna)
│   ├── train_and_evaluate()       # Cross-validated comparison
│   ├── save_artifacts()           # Persist models + metadata
│   └── feature_importance_ablation() [v3.0]  # Incremental feature engineering study
│
├── 🌐 app_enhanced.py             # Production FastAPI
│   ├── POST /predict              # Sync prediction + optional SHAP/LIME
│   ├── POST /explain/shap         # Full SHAP breakdown
│   ├── POST /explain/lime         # LIME for first instance
│   ├── POST /explain/counterfactual  # Counterfactual instances
│   ├── POST /compare              # Multi-model comparison
│   └── GET  /metrics              # Prometheus-style counters + cache stats
│
├── 📊 dashboard.py                # Streamlit interactive dashboard (9 tabs)
│   ├── Tab 1: Data Explorer       # EDA, distributions, correlation heatmap
│   ├── Tab 2: Model Training      # ROC curves, metric comparison
│   ├── Tab 3: SHAP Explorer       # Global importance, waterfall, dependence
│   ├── Tab 4: LIME Explorer       # Per-instance rules + stability
│   ├── Tab 5: Anchors / IG        # Rule anchors + Integrated Gradients
│   ├── Tab 6: Counterfactuals     # "What would flip the prediction?"
│   ├── Tab 7: Explanation Quality # Faithfulness, completeness, complexity
│   ├── Tab 8: Live Predict        # Real-time prediction + full XAI report
│   └── Tab 9: Hybrid XAI [v3.0]  # 5-method ensemble, weight pie, heatmap
│
├── Research Paper/                # LaTeX (v3.0: 5-method, ablation, EU AI Act)
│   └── xai_research_paper.tex     # 26 refs, all 5 methods, regulatory analysis
│
├── app.py                         # Original API (preserved for compatibility)
├── train_model.py                 # Original trainer (preserved)
├── breast-cancer.csv              # Wisconsin Breast Cancer dataset
├── requirements.txt               # Updated dependencies
├── Dockerfile                     # Container build
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
#   --nn              include PyTorch MLP
#   --optimise xgboost   run Bayesian HPO for xgboost
#   --ablation        run feature engineering ablation study (v3.0)
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

## Hybrid XAI Ensemble (v3.0)

The `HybridExplainer` combines all five methods using confidence-weighted voting:

```python
from xai_techniques import HybridExplainer

hybrid = HybridExplainer(model, X_train, feature_names)
result = hybrid.explain(X_test, idx=5)

print(result["weights"])              # {'shap': 0.41, 'lime': 0.28, ...}
print(result["agreement_shap_lime"]) # Pearson r between SHAP and LIME
print(hybrid.top_features(result, k=10))  # top-10 hybrid features
print(result["hybrid_importance"])    # ranked DataFrame
```

### XAI Benchmark

```python
from xai_techniques import XAIBenchmark

bench = XAIBenchmark(model, X_train, X_test, feature_names)
report = bench.run(n_instances=20)
# method         mean_ms  std_ms  most_frequent_top1
# shap               45     12   worst_concave_points
# lime              128     34   worst_concave_points
# ig               1240    180   worst_radius
# counterfactual    380     65   radius_growth
```

### Feature Ablation Study

```bash
python train_model_enhanced.py --ablation
# Outputs ablation_results.json with incremental ROC-AUC per engineered feature
```

---

## Research Paper

The `Research Paper/` folder contains the LaTeX source (v3.0):
- **26 references** (up from 14 in v2.0)
- All **five XAI methods** covered in Methodology and Results
- **Feature engineering ablation** study and results tables
- **EU AI Act & GDPR** regulatory compliance analysis
- **Hybrid ensemble** results with clinical relevance ratings

---

## Citation

```bibtex
@misc{xai_explainability_2026,
  title   = {XAI Explainability Project v3.0: Five-Method Hybrid XAI Framework for Healthcare},
  author  = {Priyanshu K. Sharma},
  year    = {2026},
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
- [EU AI Act](https://artificialintelligenceact.eu/) — European Commission
