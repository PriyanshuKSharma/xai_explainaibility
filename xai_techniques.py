"""
xai_techniques.py
=================
Enhanced Explainable AI (XAI) module supporting:
  - SHAP (TreeExplainer, KernelExplainer, GradientExplainer)
  - LIME (tabular, stability analysis, submodular pick)
  - Anchors (rule-based explanations via alibi)
  - Integrated Gradients (for neural networks via captum / numpy fallback)
  - Counterfactual Explanations (DiCE / manual nearest-unlike-neighbour)

Author: Priyanshu K. Sharma  |  Enhanced 2025
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. SHAP EXPLANATIONS
# ─────────────────────────────────────────────

class SHAPExplainer:
    """
    Unified SHAP wrapper that auto-selects the right explainer:
      - Tree-based models  → TreeExplainer  (fast, exact)
      - Everything else    → KernelExplainer (model-agnostic)
    """

    def __init__(self, model, X_background: pd.DataFrame, feature_names: list[str]):
        import shap

        self.model = model
        self.feature_names = feature_names
        model_type = type(model).__name__

        tree_models = {
            "RandomForestClassifier", "RandomForestRegressor",
            "GradientBoostingClassifier", "GradientBoostingRegressor",
            "XGBClassifier", "XGBRegressor",
            "LGBMClassifier", "LGBMRegressor",
            "DecisionTreeClassifier", "DecisionTreeRegressor",
            "ExtraTreesClassifier", "ExtraTreesRegressor",
        }

        if model_type in tree_models:
            self.explainer = shap.TreeExplainer(model)
            self.explainer_type = "tree"
        else:
            background = shap.kmeans(X_background, 50)
            self.explainer = shap.KernelExplainer(
                model.predict_proba, background
            )
            self.explainer_type = "kernel"

        print(f"[SHAP] Using {self.explainer_type.upper()} explainer for {model_type}")

    # ── Global Importance ──────────────────────────────────────────────
    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return mean |SHAP| per feature as a ranked DataFrame."""
        shap_vals = self._get_shap_values(X)
        importance = np.abs(shap_vals).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── Local Explanation ─────────────────────────────────────────────
    def local_explanation(self, X: pd.DataFrame, idx: int = 0) -> dict:
        """SHAP values + expected value for a single instance."""
        shap_vals = self._get_shap_values(X.iloc[[idx]])
        ev = self._expected_value()
        return {
            "instance_index": idx,
            "base_value": float(ev),
            "shap_values": dict(zip(self.feature_names, shap_vals[0].tolist())),
            "prediction": float(ev + shap_vals[0].sum()),
        }

    # ── Feature Interaction Heatmap data ─────────────────────────────
    def interaction_values(self, X: pd.DataFrame, max_samples: int = 100) -> np.ndarray | None:
        """Compute pairwise SHAP interaction values (Tree only)."""
        if self.explainer_type != "tree":
            print("[SHAP] Interaction values require TreeExplainer.")
            return None
        sample = X.head(max_samples)
        try:
            iv = self.explainer.shap_interaction_values(sample)
            if isinstance(iv, list):
                iv = iv[1]
            return iv
        except Exception as e:
            print(f"[SHAP] Interaction values failed: {e}")
            return None

    # ── Dependence Data ───────────────────────────────────────────────
    def dependence_data(self, X: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Return (feature_value, shap_value) pairs for plotting."""
        shap_vals = self._get_shap_values(X)
        feat_idx = self.feature_names.index(feature)
        return pd.DataFrame({
            "feature_value": X[feature].values,
            "shap_value": shap_vals[:, feat_idx],
        })

    # ── SHAP Stability ────────────────────────────────────────────────
    def stability_score(self, X: pd.DataFrame, idx: int, n_runs: int = 5) -> float:
        """
        Measure explanation stability: how consistent SHAP values are
        across multiple runs (meaningful only for KernelExplainer).
        """
        results = []
        for _ in range(n_runs):
            sv = self._get_shap_values(X.iloc[[idx]])
            results.append(sv[0])
        results = np.array(results)
        return float(results.std(axis=0).mean())

    # ── Internals ─────────────────────────────────────────────────────
    def _get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        vals = self.explainer.shap_values(X)
        if isinstance(vals, list):
            vals = vals[1]  # positive class for binary
        return vals

    def _expected_value(self) -> float:
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[1])
        return float(ev)


# ─────────────────────────────────────────────
# 2. LIME EXPLANATIONS
# ─────────────────────────────────────────────

class LIMEExplainer:
    """
    LIME tabular explainer with stability analysis and batch support.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        feature_names: list[str],
        class_names: list[str] = ("Benign", "Malignant"),
        num_samples: int = 2000,
    ):
        from lime.lime_tabular import LimeTabularExplainer

        self.feature_names = feature_names
        self.class_names = list(class_names)
        self.explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        self.num_samples = num_samples
        print(f"[LIME] Explainer initialised with {num_samples} samples.")

    # ── Single explanation ────────────────────────────────────────────
    def explain(self, model, X: pd.DataFrame, idx: int = 0, num_features: int = 10):
        """Return LIME explanation object for instance `idx`."""
        exp = self.explainer.explain_instance(
            X.iloc[idx].values,
            model.predict_proba,
            num_features=num_features,
            num_samples=self.num_samples,
        )
        return exp

    def explanation_as_dict(self, model, X: pd.DataFrame, idx: int = 0, num_features: int = 10) -> dict:
        """Return explanation as a serialisable dict."""
        exp = self.explain(model, X, idx, num_features)
        return {
            "instance_index": idx,
            "predicted_class": int(exp.predict_proba.argmax()),
            "class_probabilities": exp.predict_proba.tolist(),
            "feature_contributions": dict(exp.as_list()),
        }

    # ── Stability analysis ────────────────────────────────────────────
    def stability_analysis(
        self, model, X: pd.DataFrame, idx: int, n_runs: int = 20, num_features: int = 10
    ) -> dict:
        """
        Run LIME n_runs times on the same instance.
        Returns per-feature mean, std, and sign consistency.
        """
        runs: list[dict] = []
        for _ in range(n_runs):
            exp = self.explain(model, X, idx, num_features)
            runs.append(dict(exp.as_list()))

        all_features = sorted({f for d in runs for f in d})
        result = {}
        for feat in all_features:
            vals = [d.get(feat, 0.0) for d in runs]
            result[feat] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "sign_consistency": float(np.mean(np.sign(vals) == np.sign(np.mean(vals)))),
            }
        return result

    # ── Submodular pick ───────────────────────────────────────────────
    def submodular_pick(
        self, model, X: pd.DataFrame, num_exps: int = 5, num_features: int = 10
    ):
        """Select a diverse, representative subset of explanations."""
        from lime.submodular_pick import SubmodularPick

        sp = SubmodularPick(
            self.explainer,
            X.values,
            model.predict_proba,
            num_features=num_features,
            num_exps_desired=num_exps,
            sample_size=min(len(X), 200),
        )
        return sp.explanations


# ─────────────────────────────────────────────
# 3. ANCHORS EXPLANATIONS
# ─────────────────────────────────────────────

class AnchorsExplainer:
    """
    Rule-based, IF-THEN explanations using the Anchors algorithm.
    Requires: alibi  (pip install alibi)
    Falls back gracefully if not installed.
    """

    def __init__(self, X_train: pd.DataFrame, feature_names: list[str]):
        self.feature_names = feature_names
        try:
            from alibi.explainers import AnchorTabular
            self._AnchorTabular = AnchorTabular
            self._available = True
        except ImportError:
            print("[Anchors] alibi not installed. Run: pip install alibi")
            self._available = False

        if self._available:
            self.explainer = self._AnchorTabular(
                predict_fn=None,  # set on first explain() call
                feature_names=feature_names,
            )
            self.explainer.fit(X_train.values, disc_perc=(25, 50, 75))

    def explain(
        self, model, X: pd.DataFrame, idx: int = 0, threshold: float = 0.95
    ) -> dict:
        if not self._available:
            return {"error": "alibi not installed"}

        # Re-attach predict_fn each call (allows different models)
        self.explainer.predictor = lambda x: model.predict(x)
        exp = self.explainer.explain(
            X.iloc[idx].values, threshold=threshold
        )
        return {
            "instance_index": idx,
            "anchor_rules": list(exp.anchor),
            "precision": float(exp.precision),
            "coverage": float(exp.coverage),
            "prediction": int(exp.prediction),
        }


# ─────────────────────────────────────────────
# 4. INTEGRATED GRADIENTS
# ─────────────────────────────────────────────

class IntegratedGradientsExplainer:
    """
    Integrated Gradients for any differentiable model.

    - For sklearn / tree models: numpy-based finite-difference approximation.
    - For PyTorch nn.Module:     true auto-diff via captum.
    """

    def __init__(self, model, feature_names: list[str], baseline: str = "zero"):
        """
        baseline : 'zero' | 'mean' | np.ndarray of shape (n_features,)
        """
        self.model = model
        self.feature_names = feature_names
        self.baseline_type = baseline
        self._is_torch = self._check_torch(model)

    # ── Explain ───────────────────────────────────────────────────────
    def explain(
        self,
        X: pd.DataFrame,
        idx: int = 0,
        n_steps: int = 50,
        target_class: int = 1,
    ) -> dict:
        instance = X.iloc[idx].values.astype(float)
        baseline = self._make_baseline(instance, X)

        if self._is_torch:
            attrs = self._torch_ig(instance, baseline, n_steps, target_class)
        else:
            attrs = self._numpy_ig(instance, baseline, n_steps, target_class)

        return {
            "instance_index": idx,
            "feature_attributions": dict(zip(self.feature_names, attrs.tolist())),
            "attribution_sum": float(attrs.sum()),
        }

    # ── Numpy IG (finite-difference) ──────────────────────────────────
    def _numpy_ig(
        self, instance: np.ndarray, baseline: np.ndarray, n_steps: int, target_class: int
    ) -> np.ndarray:
        alphas = np.linspace(0, 1, n_steps + 1)
        gradients = []

        for alpha in alphas:
            interp = baseline + alpha * (instance - baseline)
            # Two-sided finite difference gradient approximation
            eps = 1e-4
            grad = np.zeros_like(instance)
            for j in range(len(instance)):
                x_plus = interp.copy(); x_plus[j] += eps
                x_minus = interp.copy(); x_minus[j] -= eps
                f_plus = self._predict_proba(x_plus[None, :])[0, target_class]
                f_minus = self._predict_proba(x_minus[None, :])[0, target_class]
                grad[j] = (f_plus - f_minus) / (2 * eps)
            gradients.append(grad)

        avg_grads = np.mean(gradients, axis=0)
        return (instance - baseline) * avg_grads

    # ── PyTorch / captum IG ───────────────────────────────────────────
    def _torch_ig(
        self, instance: np.ndarray, baseline: np.ndarray, n_steps: int, target_class: int
    ) -> np.ndarray:
        try:
            import torch
            from captum.attr import IntegratedGradients as CaptumIG

            inp = torch.tensor(instance, dtype=torch.float32).unsqueeze(0)
            bas = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0)
            ig = CaptumIG(self.model)
            attrs, _ = ig.attribute(inp, bas, target=target_class, return_convergence_delta=True, n_steps=n_steps)
            return attrs.squeeze(0).detach().numpy()
        except Exception as e:
            print(f"[IG] captum failed ({e}), falling back to numpy approximation.")
            return self._numpy_ig(instance, baseline, n_steps, target_class)

    # ── Helpers ───────────────────────────────────────────────────────
    def _make_baseline(self, instance: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        if self.baseline_type == "zero":
            return np.zeros_like(instance)
        elif self.baseline_type == "mean":
            return X.mean().values.astype(float)
        elif isinstance(self.baseline_type, np.ndarray):
            return self.baseline_type.astype(float)
        return np.zeros_like(instance)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        out = self.model(X)  # PyTorch raw logit
        import torch
        probs = torch.softmax(torch.tensor(out), dim=-1).numpy()
        return probs

    @staticmethod
    def _check_torch(model) -> bool:
        try:
            import torch
            return isinstance(model, torch.nn.Module)
        except ImportError:
            return False


# ─────────────────────────────────────────────
# 5. COUNTERFACTUAL EXPLANATIONS
# ─────────────────────────────────────────────

class CounterfactualExplainer:
    """
    Counterfactual (CF) explanations:
    "What is the minimal change to flip the predicted class?"

    Strategy A: DiCE (pip install dice-ml)  — rich, multi-CF generation.
    Strategy B: Nearest-Unlike-Neighbour (NUN) fallback — always available.
    """

    def __init__(self, model, X_train: pd.DataFrame, feature_names: list[str]):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self._dice_available = self._init_dice(X_train, feature_names)

    def _init_dice(self, X: pd.DataFrame, feature_names: list[str]) -> bool:
        try:
            import dice_ml
            self._dice_ml = dice_ml
            return True
        except ImportError:
            print("[CF] dice-ml not installed. Using NUN fallback. pip install dice-ml")
            return False

    # ── Main API ──────────────────────────────────────────────────────
    def explain(
        self,
        X: pd.DataFrame,
        idx: int = 0,
        n_counterfactuals: int = 3,
        method: str = "auto",
    ) -> dict:
        """
        Returns counterfactual instances with feature deltas.
        method: 'dice' | 'nun' | 'auto'
        """
        if method == "dice" or (method == "auto" and self._dice_available):
            return self._explain_dice(X, idx, n_counterfactuals)
        return self._explain_nun(X, idx, n_counterfactuals)

    # ── DiCE ──────────────────────────────────────────────────────────
    def _explain_dice(self, X: pd.DataFrame, idx: int, n_cfs: int) -> dict:
        try:
            dice_ml = self._dice_ml
            d = dice_ml.Data(
                dataframe=self.X_train.assign(
                    outcome=self.model.predict(self.X_train)
                ),
                continuous_features=self.feature_names,
                outcome_name="outcome",
            )
            m = dice_ml.Model(model=self.model, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")
            result = exp.generate_counterfactuals(
                X.iloc[[idx]], total_CFs=n_cfs, desired_class="opposite"
            )
            cfs_df = result.cf_examples_list[0].final_cfs_df
            deltas = cfs_df[self.feature_names] - X.iloc[idx][self.feature_names].values
            return {
                "method": "dice",
                "original_prediction": int(self.model.predict(X.iloc[[idx]])[0]),
                "counterfactuals": cfs_df[self.feature_names].to_dict("records"),
                "deltas": deltas.to_dict("records"),
            }
        except Exception as e:
            print(f"[CF] DiCE failed ({e}), falling back to NUN.")
            return self._explain_nun(X, idx, n_cfs)

    # ── Nearest Unlike Neighbour ──────────────────────────────────────
    def _explain_nun(self, X: pd.DataFrame, idx: int, n_cfs: int) -> dict:
        from sklearn.preprocessing import StandardScaler

        instance = X.iloc[idx].values
        orig_pred = int(self.model.predict(instance.reshape(1, -1))[0])

        # Get training instances with opposite class
        train_preds = self.model.predict(self.X_train)
        opposite_mask = train_preds != orig_pred
        opposite_X = self.X_train[opposite_mask]

        if len(opposite_X) == 0:
            return {"error": "No opposite-class instances in training data."}

        # Euclidean distance in normalised space
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(self.X_train)
        inst_norm = scaler.transform(instance.reshape(1, -1))[0]
        opp_norm = scaler.transform(opposite_X)

        dists = np.linalg.norm(opp_norm - inst_norm, axis=1)
        top_idx = np.argsort(dists)[:n_cfs]
        cfs = opposite_X.iloc[top_idx]

        deltas = cfs - instance
        changed = (deltas.abs() > 1e-6).sum(axis=1).values

        return {
            "method": "nearest_unlike_neighbour",
            "original_prediction": orig_pred,
            "counterfactuals": cfs[self.feature_names].to_dict("records"),
            "deltas": deltas[self.feature_names].to_dict("records"),
            "features_changed": changed.tolist(),
            "distances": dists[top_idx].tolist(),
        }

    # ── Explanation quality ───────────────────────────────────────────
    def sparsity(self, delta: dict, threshold: float = 1e-6) -> float:
        """Fraction of features that were NOT changed."""
        vals = np.array(list(delta.values()))
        unchanged = (np.abs(vals) <= threshold).sum()
        return float(unchanged / len(vals))


# ─────────────────────────────────────────────
# 6. EXPLANATION QUALITY METRICS
# ─────────────────────────────────────────────

class ExplanationEvaluator:
    """
    Evaluate explanation quality across four axes:
      - Faithfulness (sufficiency test)
      - Stability    (repeated runs)
      - Completeness (sum of attributions ≈ prediction)
      - Complexity   (# non-zero features)
    """

    @staticmethod
    def faithfulness(
        model,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        feature_names: list[str],
        top_k: int = 5,
    ) -> float:
        """
        Remove top-k important features (zero them out).
        Faithfulness = mean |Δprediction|.
        """
        scores = []
        for i in range(min(100, len(X))):
            orig_p = model.predict_proba(X.iloc[[i]])[0, 1]
            top_feats = np.argsort(np.abs(shap_values[i]))[-top_k:]
            X_mod = X.iloc[[i]].copy()
            for f in top_feats:
                X_mod.iloc[0, f] = 0.0
            new_p = model.predict_proba(X_mod)[0, 1]
            scores.append(abs(orig_p - new_p))
        return float(np.mean(scores))

    @staticmethod
    def completeness(base_value: float, shap_values: np.ndarray, predictions: np.ndarray) -> float:
        """
        SHAP completeness axiom: base + sum(shap) ≈ prediction.
        Returns mean absolute error.
        """
        reconstructed = base_value + shap_values.sum(axis=1)
        return float(np.abs(reconstructed - predictions).mean())

    @staticmethod
    def complexity(shap_values: np.ndarray, threshold: float = 0.01) -> float:
        """Mean number of features with |SHAP| > threshold per instance."""
        return float((np.abs(shap_values) > threshold).sum(axis=1).mean())

    @staticmethod
    def summary(
        model,
        X: pd.DataFrame,
        shap_explainer: SHAPExplainer,
        feature_names: list[str],
    ) -> dict:
        shap_vals = shap_explainer._get_shap_values(X)
        preds = model.predict_proba(X)[:, 1]
        base_val = shap_explainer._expected_value()

        return {
            "faithfulness":  ExplanationEvaluator.faithfulness(model, X, shap_vals, feature_names),
            "completeness":  ExplanationEvaluator.completeness(base_val, shap_vals, preds),
            "complexity":    ExplanationEvaluator.complexity(shap_vals),
        }
