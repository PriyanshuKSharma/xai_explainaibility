"""
xai_techniques.py
=================
Enhanced Explainable AI (XAI) module supporting:
  - SHAP (TreeExplainer, KernelExplainer, GradientExplainer)
  - LIME (tabular, stability analysis, submodular pick)
  - Anchors (rule-based explanations via alibi)
  - Integrated Gradients (for neural networks via captum / numpy fallback)
  - Counterfactual Explanations (DiCE / manual nearest-unlike-neighbour)
  - HybridExplainer (confidence-weighted ensemble of all 5 methods)  [v3.0]
  - XAIBenchmark   (timing + comparison utility for all methods)       [v3.0]

Author: Priyanshu K. Sharma  |  Enhanced v3.0 (2026)
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


# ─────────────────────────────────────────────
# 7. HYBRID EXPLAINER  (v3.0)
# ─────────────────────────────────────────────

class HybridExplainer:
    """
    Confidence-weighted ensemble of all five XAI methods:
      SHAP + LIME + Anchors + Integrated Gradients + Counterfactuals

    Usage
    -----
    hybrid = HybridExplainer(model, X_train, feature_names)
    result = hybrid.explain(X_test, idx=0)
    # result keys: weights, hybrid_importance, agreement_shap_lime, method_importances
    """

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        feature_names: list[str],
        class_names: list[str] = ("Benign", "Malignant"),
        lime_samples: int = 1000,
        ig_steps: int = 30,
    ):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.class_names = list(class_names)
        self.ig_steps = ig_steps

        # Build sub-explainers
        self.shap_exp  = SHAPExplainer(model, X_train, feature_names)
        self.lime_exp  = LIMEExplainer(X_train, feature_names, class_names, num_samples=lime_samples)
        self.ig_exp    = IntegratedGradientsExplainer(model, feature_names, baseline="mean")
        self.cf_exp    = CounterfactualExplainer(model, X_train, feature_names)
        # Anchors (optional — graceful fallback)
        self.anchor_exp = AnchorsExplainer(X_train, feature_names)

        print("[Hybrid] All sub-explainers initialised.")

    # ── Main API ─────────────────────────────────────────────────────
    def explain(
        self,
        X: pd.DataFrame,
        idx: int = 0,
        num_lime_features: int = 10,
        target_class: int = 1,
    ) -> dict:
        """
        Generate a unified hybrid explanation for instance `idx`.

        Returns
        -------
        dict with keys:
          - method_importances: dict[method_name -> np.ndarray]
          - weights: dict[method_name -> float]
          - hybrid_importance: pd.DataFrame  (feature, hybrid_score)
          - agreement_shap_lime: float  (Pearson r)
          - prediction_info: dict
        """
        p = len(self.feature_names)
        method_scores: dict[str, np.ndarray] = {}

        # 1. SHAP
        try:
            sv = self.shap_exp._get_shap_values(X.iloc[[idx]])  # shape (1, p)
            method_scores["shap"] = np.abs(sv[0])
        except Exception as e:
            print(f"[Hybrid] SHAP failed: {e}")

        # 2. LIME
        try:
            exp = self.lime_exp.explain(self.model, X, idx, num_features=num_lime_features)
            lime_dict = dict(exp.as_list())
            # Map LIME feature strings back to feature_names (LIME adds bin labels)
            lime_vec = np.zeros(p)
            for feat_str, coef in lime_dict.items():
                for i, fn in enumerate(self.feature_names):
                    if fn in feat_str:
                        lime_vec[i] += abs(coef)
                        break
            method_scores["lime"] = lime_vec
        except Exception as e:
            print(f"[Hybrid] LIME failed: {e}")

        # 3. Integrated Gradients
        try:
            ig_result = self.ig_exp.explain(X, idx=idx, n_steps=self.ig_steps,
                                             target_class=target_class)
            ig_vec = np.array([abs(ig_result["feature_attributions"][f])
                               for f in self.feature_names])
            method_scores["integrated_gradients"] = ig_vec
        except Exception as e:
            print(f"[Hybrid] IG failed: {e}")

        # 4. Counterfactuals (delta magnitude as importance proxy)
        try:
            cf_result = self.cf_exp.explain(X, idx=idx, n_counterfactuals=1)
            if "deltas" in cf_result and cf_result["deltas"]:
                delta = cf_result["deltas"][0]
                cf_vec = np.array([abs(delta.get(f, 0.0)) for f in self.feature_names])
                method_scores["counterfactual"] = cf_vec
        except Exception as e:
            print(f"[Hybrid] CF failed: {e}")

        # 5. Anchors → convert rule to binary importance vector
        try:
            anc_result = self.anchor_exp.explain(self.model, X, idx)
            if "anchor_rules" in anc_result:
                anc_vec = np.zeros(p)
                for rule in anc_result["anchor_rules"]:
                    for i, fn in enumerate(self.feature_names):
                        if fn in rule:
                            anc_vec[i] = 1.0
                            break
                method_scores["anchors"] = anc_vec
        except Exception as e:
            print(f"[Hybrid] Anchors failed: {e}")

        # ── Confidence-weighted combination ───────────────────────────
        confidences = {
            m: float(v.sum()) for m, v in method_scores.items() if v.sum() > 0
        }
        total_conf = sum(confidences.values()) or 1.0
        weights = {m: c / total_conf for m, c in confidences.items()}

        hybrid_vec = np.zeros(p)
        for m, vec in method_scores.items():
            hybrid_vec += weights.get(m, 0.0) * vec

        hybrid_df = pd.DataFrame({
            "feature": self.feature_names,
            "hybrid_score": hybrid_vec,
        }).sort_values("hybrid_score", ascending=False).reset_index(drop=True)

        # ── Agreement between SHAP and LIME ───────────────────────────
        agreement = 0.0
        if "shap" in method_scores and "lime" in method_scores:
            s, l = method_scores["shap"], method_scores["lime"]
            if s.std() > 0 and l.std() > 0:
                agreement = float(np.corrcoef(s, l)[0, 1])

        # ── Prediction info ───────────────────────────────────────────
        pred_proba = self.model.predict_proba(X.iloc[[idx]])[0]
        pred_class = int(pred_proba.argmax())
        pred_label = self.class_names[pred_class] if pred_class < len(self.class_names) else str(pred_class)

        return {
            "instance_index": idx,
            "prediction_info": {
                "predicted_class": pred_class,
                "predicted_label": pred_label,
                "class_probabilities": pred_proba.tolist(),
            },
            "method_importances": {
                m: v.tolist() for m, v in method_scores.items()
            },
            "weights": weights,
            "hybrid_importance": hybrid_df,
            "agreement_shap_lime": round(agreement, 4),
        }

    def top_features(self, result: dict, k: int = 10) -> list[str]:
        """Return top-k feature names from hybrid explanation result."""
        return result["hybrid_importance"]["feature"].head(k).tolist()


# ─────────────────────────────────────────────
# 8. XAI BENCHMARK  (v3.0)
# ─────────────────────────────────────────────

class XAIBenchmark:
    """
    Benchmarks all five XAI methods on a sample of instances.
    Reports per-method timing, stability, and feature agreement.

    Usage
    -----
    bench = XAIBenchmark(model, X_train, X_test, feature_names)
    report = bench.run(n_instances=20)
    print(report)  # DataFrame: method × metric
    """

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: list[str],
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names

    def run(self, n_instances: int = 20, random_state: int = 42) -> pd.DataFrame:
        """
        Run all XAI methods on `n_instances` random test instances.
        Returns a summary DataFrame: method × (mean_ms, std_ms, mean_top1).
        """
        import time
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(self.X_test), size=min(n_instances, len(self.X_test)),
                              replace=False)

        shap_exp = SHAPExplainer(self.model, self.X_train, self.feature_names)
        lime_exp = LIMEExplainer(self.X_train, self.feature_names, num_samples=500)
        ig_exp   = IntegratedGradientsExplainer(self.model, self.feature_names, baseline="mean")
        cf_exp   = CounterfactualExplainer(self.model, self.X_train, self.feature_names)

        results: dict[str, list[float]] = {
            "shap": [], "lime": [], "ig": [], "counterfactual": []
        }
        top1_counts: dict[str, dict[str, int]] = {m: {} for m in results}

        for idx in indices:
            idx = int(idx)

            # SHAP
            t0 = time.perf_counter()
            try:
                sv = shap_exp._get_shap_values(self.X_test.iloc[[idx]])
                top1 = self.feature_names[int(np.abs(sv[0]).argmax())]
                top1_counts["shap"][top1] = top1_counts["shap"].get(top1, 0) + 1
            except Exception:
                pass
            results["shap"].append((time.perf_counter() - t0) * 1000)

            # LIME
            t0 = time.perf_counter()
            try:
                exp = lime_exp.explain(self.model, self.X_test, idx, num_features=10)
                top1 = exp.as_list()[0][0]
                top1_counts["lime"][top1] = top1_counts["lime"].get(top1, 0) + 1
            except Exception:
                pass
            results["lime"].append((time.perf_counter() - t0) * 1000)

            # IG
            t0 = time.perf_counter()
            try:
                ig_r = ig_exp.explain(self.X_test, idx=idx, n_steps=20)
                top1 = max(ig_r["feature_attributions"], key=lambda k: abs(ig_r["feature_attributions"][k]))
                top1_counts["ig"][top1] = top1_counts["ig"].get(top1, 0) + 1
            except Exception:
                pass
            results["ig"].append((time.perf_counter() - t0) * 1000)

            # Counterfactual
            t0 = time.perf_counter()
            try:
                cf_r = cf_exp.explain(self.X_test, idx=idx, n_counterfactuals=1)
                if cf_r.get("deltas"):
                    delta = cf_r["deltas"][0]
                    top1 = max(delta, key=lambda k: abs(delta[k]))
                    top1_counts["counterfactual"][top1] = top1_counts["counterfactual"].get(top1, 0) + 1
            except Exception:
                pass
            results["counterfactual"].append((time.perf_counter() - t0) * 1000)

        rows = []
        for method, times in results.items():
            if times:
                most_common_top1 = max(top1_counts[method], key=top1_counts[method].get) \
                    if top1_counts[method] else "N/A"
                rows.append({
                    "method": method,
                    "mean_ms": round(float(np.mean(times)), 1),
                    "std_ms":  round(float(np.std(times)), 1),
                    "min_ms":  round(float(np.min(times)), 1),
                    "max_ms":  round(float(np.max(times)), 1),
                    "most_frequent_top1": most_common_top1,
                })

        report = pd.DataFrame(rows).set_index("method")
        print("\n[XAIBenchmark] Results:")
        print(report.to_string())
        return report
