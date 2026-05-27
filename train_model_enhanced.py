"""
train_model_enhanced.py
========================
Multi-model training pipeline supporting:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - Logistic Regression
  - PyTorch Neural Network (MLP)

Includes:
  - Automated preprocessing + feature engineering
  - Bayesian / Grid hyperparameter optimisation
  - Model comparison dashboard
  - Persisting models + SHAP explainers

Author: Priyanshu K. Sharma  |  Enhanced 2025
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def load_and_preprocess(
    filepath: str = "breast-cancer.csv",
    target_col: str = "diagnosis",
    drop_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Load CSV, engineer features, and return train/test splits.
    Returns: X_train, X_test, y_train, y_test, feature_names, scaler, encoder
    """
    df = pd.read_csv(filepath)
    print(f"[Data] Loaded {df.shape[0]} rows × {df.shape[1]} cols from '{filepath}'")

    # Drop irrelevant columns
    if drop_cols:
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.drop(columns=["id", "Unnamed: 32"], errors="ignore", inplace=True)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].values)
    X = df.drop(columns=[target_col])

    # ── Feature Engineering ──────────────────────────────────────────
    X = _engineer_features(X)

    feature_names = list(X.columns)
    print(f"[Data] {len(feature_names)} features after engineering")
    print(f"[Data] Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names, scaler, le


def _engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add computed ratio / interaction features."""
    X = X.copy()
    # Nuclear shape ratios
    if {"perimeter_mean", "area_mean"}.issubset(X.columns):
        X["perimeter_area_ratio"] = X["perimeter_mean"] / (X["area_mean"] + 1e-9)
    if {"concavity_mean", "concave points_mean"}.issubset(X.columns):
        X["concavity_to_points"] = X["concavity_mean"] / (X["concave points_mean"] + 1e-9)
    if {"radius_worst", "radius_mean"}.issubset(X.columns):
        X["radius_growth"] = X["radius_worst"] - X["radius_mean"]
    if {"compactness_worst", "compactness_mean"}.issubset(X.columns):
        X["compactness_growth"] = X["compactness_worst"] - X["compactness_mean"]
    return X


# ─────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────

def get_model_registry() -> dict:
    """Return all supported model constructors."""
    registry: dict[str, Any] = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42
        ),
        "logistic_regression": LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=42
        ),
    }

    # Optional: XGBoost
    try:
        from xgboost import XGBClassifier
        registry["xgboost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
        print("[Models] XGBoost available ✓")
    except ImportError:
        print("[Models] XGBoost not installed (pip install xgboost)")

    # Optional: LightGBM
    try:
        from lightgbm import LGBMClassifier
        registry["lightgbm"] = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )
        print("[Models] LightGBM available ✓")
    except ImportError:
        print("[Models] LightGBM not installed (pip install lightgbm)")

    return registry


# ─────────────────────────────────────────────
# PYTORCH MLP
# ─────────────────────────────────────────────

class MLPClassifierTorch:
    """
    Lightweight PyTorch MLP with an sklearn-compatible interface
    (fit / predict / predict_proba).
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.classes_ = np.array([0, 1])
        self._build()

    def _build(self):
        try:
            import torch
            import torch.nn as nn

            layers: list[nn.Module] = []
            dims = [self.input_dim] + list(self.hidden_dims)
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(self.dropout)]
            layers += [nn.Linear(dims[-1], 2)]
            self.net = nn.Sequential(*layers).to(self.device)
            self._torch = torch
            self._nn = nn
            self._available = True
        except ImportError:
            print("[MLP] PyTorch not installed. pip install torch")
            self._available = False

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray):
        if not self._available:
            return self
        torch = self._torch
        nn = self._nn

        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.astype(int)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                opt.zero_grad()
                loss = criterion(self.net(xb.to(self.device)), yb.to(self.device))
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                print(f"  [MLP] Epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.4f}")
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self._available:
            raise RuntimeError("PyTorch not available.")
        torch = self._torch
        X_arr = X.values if hasattr(X, "values") else X
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X_arr, dtype=torch.float32).to(self.device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ─────────────────────────────────────────────
# HYPERPARAMETER OPTIMISATION
# ─────────────────────────────────────────────

def optimise_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    method: str = "grid",
) -> Any:
    """
    Run hyperparameter search for the given model.
    method: 'grid' | 'random' | 'bayes'  (bayes requires optuna)
    """
    from sklearn.ensemble import RandomForestClassifier

    param_grids = {
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "gradient_boosting": {
            "n_estimators": [100, 150, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5],
        },
        "xgboost": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 4, 6],
            "subsample": [0.7, 0.9],
        },
        "lightgbm": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 63],
            "min_child_samples": [20, 50],
        },
    }

    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"Model '{model_name}' not in registry.")

    base_model = registry[model_name]
    params = param_grids.get(model_name, {})

    if method == "bayes":
        return _optuna_search(base_model, model_name, X_train, y_train)

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if method == "random":
        searcher = RandomizedSearchCV(base_model, params, n_iter=20, cv=cv,
                                      scoring="f1", n_jobs=-1, random_state=42)
    else:
        searcher = GridSearchCV(base_model, params, cv=cv,
                                scoring="f1", n_jobs=-1)

    searcher.fit(X_train, y_train)
    print(f"[Optim] Best params for {model_name}: {searcher.best_params_}")
    print(f"[Optim] Best CV F1: {searcher.best_score_:.4f}")
    return searcher.best_estimator_


def _optuna_search(base_model, model_name: str, X_train, y_train):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[Optim] optuna not installed. pip install optuna — falling back to random search")
        return optimise_model(model_name, X_train, y_train, method="random")

    def objective(trial):
        from sklearn.model_selection import cross_val_score
        if model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                random_state=42, n_jobs=-1,
            )
        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            m = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                learning_rate=trial.suggest_float("lr", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                eval_metric="logloss", random_state=42,
            )
        else:
            return 0.0
        scores = cross_val_score(m, X_train, y_train, cv=5, scoring="f1")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print(f"[Optuna] Best F1: {study.best_value:.4f}  Params: {study.best_params}")

    # Refit with best params
    registry = get_model_registry()
    best = registry[model_name]
    best.set_params(**study.best_params)
    best.fit(X_train, y_train)
    return best


# ─────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    include_nn: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """
    Train all available models and return comparison table.
    Returns: (trained_models dict, metrics_df)
    """
    registry = get_model_registry()

    if include_nn:
        registry["neural_network"] = MLPClassifierTorch(
            input_dim=len(feature_names), epochs=100
        )

    trained: dict[str, Any] = {}
    rows: list[dict] = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in registry.items():
        print(f"\n[Train] ── {name.upper()} ──")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

        metrics = {
            "model": name,
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std":  round(cv_scores.std(), 4),
        }
        rows.append(metrics)
        trained[name] = model

        print(f"  Accuracy={metrics['accuracy']}  F1={metrics['f1']}  ROC-AUC={metrics['roc_auc']}")

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    print("\n[Results] Model Comparison:")
    print(metrics_df.to_string(index=False))
    return trained, metrics_df


# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

def save_artifacts(
    models: dict,
    scaler: StandardScaler,
    encoder: LabelEncoder,
    feature_names: list[str],
    metrics_df: pd.DataFrame,
    out_dir: str = ".",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save best model (highest ROC-AUC)
    best_name = metrics_df.iloc[0]["model"]
    best_model = models[best_name]
    joblib.dump(best_model, out / "random_forest_model.pkl")   # keep original name for API compat
    joblib.dump(best_model, out / f"best_model_{ts}.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    joblib.dump(encoder, out / "label_encoder.pkl")

    # Save all models
    for name, m in models.items():
        try:
            joblib.dump(m, out / f"model_{name}.pkl")
        except Exception:
            pass  # MLP torch state dict

    # Save metadata
    meta = {
        "timestamp": ts,
        "best_model": best_name,
        "feature_names": feature_names,
        "metrics": metrics_df.to_dict("records"),
    }
    with open(out / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Save] Artifacts saved to '{out}' — best model: {best_name}")
    return out


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XAI Enhanced Model Trainer")
    parser.add_argument("--data", default="breast-cancer.csv")
    parser.add_argument("--out", default=".")
    parser.add_argument("--nn", action="store_true", help="Include PyTorch MLP")
    parser.add_argument("--optimise", default=None,
                        help="Run hyperparameter search for model name")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, features, scaler, encoder = \
        load_and_preprocess(args.data)

    if args.optimise:
        best_model = optimise_model(args.optimise, X_train, y_train)
        registry = {args.optimise: best_model}
        from sklearn.metrics import classification_report as cr
        y_pred = best_model.predict(X_test)
        print(cr(y_test, y_pred))
    else:
        models, metrics_df = train_and_evaluate(
            X_train, X_test, y_train, y_test, features, include_nn=args.nn
        )
        save_artifacts(models, scaler, encoder, features, metrics_df, out_dir=args.out)
