"""
dashboard.py  –  XAI Interactive Explorer (Streamlit)
======================================================
Run:  streamlit run dashboard.py

Tabs:
  1. 🔬 Data Explorer       – EDA, class balance, feature distributions
  2. 🤖 Model Training      – Train & compare all models, ROC curves
  3. 🧠 SHAP Explorer       – Global importance, waterfall, dependence plots
  4. 🟡 LIME Explorer       – Per-instance rule explanations + stability
  5. ⚓ Anchors / IG        – Rule anchors & Integrated Gradients attribution
  6. 🔄 Counterfactuals     – "What would need to change?" explorer
  7. 📊 Explanation Quality – Faithfulness, completeness, complexity scores
  8. 🩺 Predict             – Live prediction with full XAI report
  9. 🔄 Hybrid XAI          – Five-method ensemble & agreement analysis  [v3.0]

Author: Priyanshu K. Sharma  |  Enhanced v3.0 (2026)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Streamlit config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: #0d0f18;
    color: #e2e8f0;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6ee7f7 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0;
    font-weight: 300;
}

.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #1a1d2e 100%);
    border: 1px solid #2d3154;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #6ee7f7;
}

.metric-label {
    color: #94a3b8;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.section-header {
    font-family: 'Space Mono', monospace;
    color: #a78bfa;
    font-size: 1.1rem;
    border-bottom: 1px solid #2d3154;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}

div[data-testid="stSidebar"] {
    background: #111320;
    border-right: 1px solid #2d3154;
}

div[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #111320;
    border-radius: 12px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.85rem;
    color: #94a3b8 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6ee7f7 0%, #a78bfa 100%) !important;
    color: #0d0f18 !important;
    font-weight: 600;
}

.stButton > button {
    background: linear-gradient(135deg, #6ee7f7, #a78bfa);
    color: #0d0f18;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}

.stButton > button:hover { opacity: 0.85; }

.prediction-card-m {
    background: linear-gradient(135deg, #3b1a2e, #1a1030);
    border: 2px solid #f472b6;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}

.prediction-card-b {
    background: linear-gradient(135deg, #0d2b1a, #0d1e2e);
    border: 2px solid #6ee7f7;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}

.pred-label {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
}

.info-box {
    background: #1a1d2e;
    border-left: 3px solid #a78bfa;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    color: #cbd5e1;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    paper_bgcolor="#0d0f18",
    plot_bgcolor="#0d0f18",
    font=dict(color="#e2e8f0", family="Sora"),
    colorway=["#6ee7f7","#a78bfa","#f472b6","#34d399","#fbbf24","#f87171"],
    xaxis=dict(gridcolor="#1e2130", zerolinecolor="#2d3154"),
    yaxis=dict(gridcolor="#1e2130", zerolinecolor="#2d3154"),
)


# ═══════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ═══════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df.drop(columns=["id", "Unnamed: 32"], errors="ignore", inplace=True)
    return df

@st.cache_resource
def get_pipeline(df: pd.DataFrame):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    le = LabelEncoder()
    y = le.fit_transform(df["diagnosis"])
    X = df.drop(columns=["diagnosis"])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X, X_scaled, X_train, X_test, y_train, y_test, scaler, le, list(X.columns)

@st.cache_resource
def train_models(_X_train, _y_train, selected: tuple):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    registry = {
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        registry["XGBoost"] = XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42, n_jobs=-1)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        registry["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        pass

    trained = {}
    for name in selected:
        if name in registry:
            registry[name].fit(_X_train, _y_train)
            trained[name] = registry[name]
    return trained

@st.cache_resource
def get_shap_explainer(_model, _X_train):
    import shap
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        bg = shap.kmeans(_X_train, 30)
        return shap.KernelExplainer(_model.predict_proba, bg)

@st.cache_data
def compute_shap_values(_explainer, _X_test):
    vals = _explainer.shap_values(_X_test)
    if isinstance(vals, list):
        vals = vals[1]
    return vals


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p class="main-title">XAI<br>Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Explainable AI Dashboard v2.0</p>', unsafe_allow_html=True)
    st.divider()

    data_path = st.text_input("Dataset path", value="breast-cancer.csv")
    st.divider()

    st.markdown("**Models to train**")
    model_choices = st.multiselect(
        "Select models",
        ["Random Forest", "Gradient Boosting", "Logistic Regression", "XGBoost", "LightGBM"],
        default=["Random Forest", "Gradient Boosting", "Logistic Regression"],
    )
    primary_model = st.selectbox("Primary model for XAI", model_choices or ["Random Forest"])

    st.divider()
    n_lime_features = st.slider("LIME: max features", 5, 20, 10)
    n_cf = st.slider("Counterfactuals: count", 1, 10, 3)
    st.divider()
    st.markdown(
        '<div class="info-box">Using Wisconsin Breast Cancer dataset. '
        '569 instances · 30 features · Binary (B/M).</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════

try:
    df = load_data(data_path)
except FileNotFoundError:
    st.error(f"❌ Dataset not found at '{data_path}'. Please check the path.")
    st.stop()

X_raw, X_scaled, X_train, X_test, y_train, y_test, scaler, le, feature_names = get_pipeline(df)

# Train models
with st.spinner("Training models …"):
    trained_models = train_models(X_train, y_train, tuple(model_choices))

if not trained_models:
    st.warning("No models trained. Select at least one model in the sidebar.")
    st.stop()

model = trained_models.get(primary_model) or list(trained_models.values())[0]
shap_explainer = get_shap_explainer(model, X_train)
shap_values_all = compute_shap_values(shap_explainer, X_test)


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "🔬 Data Explorer",
    "🤖 Model Training",
    "🧠 SHAP Explorer",
    "🟡 LIME Explorer",
    "⚓ Anchors / IG",
    "🔄 Counterfactuals",
    "📊 Explanation Quality",
    "🩺 Predict",
    "✴️ Hybrid XAI",
])


# ───────────────────────────────────────────────────────────────────
# TAB 1 – DATA EXPLORER
# ───────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Samples", len(df)),
        (c2, "Features", len(feature_names)),
        (c3, "Benign", int((df["diagnosis"] == "B").sum())),
        (c4, "Malignant", int((df["diagnosis"] == "M").sum())),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.pie(
            df, names="diagnosis",
            color="diagnosis",
            color_discrete_map={"B": "#6ee7f7", "M": "#f472b6"},
            title="Class Distribution",
        )
        fig.update_layout(**PLOTLY_DARK, title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        feat_sel = st.selectbox("Feature distribution", feature_names)
        fig2 = px.histogram(
            df, x=feat_sel, color="diagnosis",
            color_discrete_map={"B": "#6ee7f7", "M": "#f472b6"},
            barmode="overlay", opacity=0.7,
            title=f"Distribution: {feat_sel}",
        )
        fig2.update_layout(**PLOTLY_DARK, title_font_size=14)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-header">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
    top_feats = (
        pd.DataFrame({"f": feature_names, "i": np.abs(shap_values_all).mean(0)})
        .nlargest(15, "i")["f"].tolist()
    )
    corr = X_scaled[top_feats].corr()
    fig3 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        title="Top-15 Feature Correlations (by SHAP importance)",
    )
    fig3.update_layout(**PLOTLY_DARK, title_font_size=14, height=500)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📄 Raw dataset sample"):
        st.dataframe(df.head(20), use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 2 – MODEL TRAINING
# ───────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)

    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score, roc_curve,
    )

    rows = []
    roc_data = {}
    for name, m in trained_models.items():
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        })
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)

    metrics_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)

    # Colour best row
    def highlight_best(s):
        return ["background-color: #1a2a1a; color: #6ee7f7" if s["Model"] == metrics_df.iloc[0]["Model"]
                else "" for _ in s]

    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=1).format(precision=4),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        colors = ["#6ee7f7", "#a78bfa", "#f472b6", "#34d399", "#fbbf24"]
        for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
            auc = [r["ROC-AUC"] for r in rows if r["Model"] == name][0]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc})",
                                     line=dict(color=colors[i % len(colors)], width=2)))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="#4b5563"))
        fig.update_layout(**PLOTLY_DARK, title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            metrics_df, x="Model", y=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            barmode="group", title="Metric Comparison",
            color_discrete_sequence=["#6ee7f7","#a78bfa","#f472b6","#34d399","#fbbf24"],
        )
        fig2.update_layout(**PLOTLY_DARK, height=400)
        st.plotly_chart(fig2, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 3 – SHAP EXPLORER
# ───────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<p class="section-header">SHAP Explanations</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        shap_mode = st.radio("View", ["Global Importance", "Local Waterfall", "Dependence Plot"])
        instance_idx = st.slider("Instance index", 0, len(X_test) - 1, 0)
        dep_feat = st.selectbox("Dependence feature", feature_names)

    with col1:
        if shap_mode == "Global Importance":
            importance = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_values_all).mean(0),
            }).sort_values("mean_abs_shap", ascending=True).tail(20)
            fig = px.bar(
                importance, x="mean_abs_shap", y="feature", orientation="h",
                color="mean_abs_shap", color_continuous_scale=["#1e2130","#6ee7f7","#a78bfa"],
                title="Global Feature Importance (mean |SHAP|)",
            )
            fig.update_layout(**PLOTLY_DARK, height=600, showlegend=False,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        elif shap_mode == "Local Waterfall":
            sv = shap_values_all[instance_idx]
            ev = shap_explainer.expected_value
            base = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)
            pred = base + sv.sum()

            feat_df = pd.DataFrame({
                "feature": feature_names, "shap": sv,
                "value": X_test.iloc[instance_idx].values,
            }).sort_values("shap", key=abs, ascending=True).tail(15)

            colors = ["#f472b6" if v > 0 else "#6ee7f7" for v in feat_df["shap"]]
            fig = go.Figure(go.Bar(
                x=feat_df["shap"], y=[f"{r.feature} = {r.value:.3f}" for _, r in feat_df.iterrows()],
                orientation="h", marker_color=colors,
            ))
            fig.update_layout(
                **PLOTLY_DARK,
                title=f"SHAP Waterfall — Instance {instance_idx}  (base={base:.3f}, pred={pred:.3f})",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            true_label = le.inverse_transform([y_test[instance_idx]])[0]
            pred_label = le.inverse_transform([model.predict(X_test.iloc[[instance_idx]])[0]])[0]
            st.info(f"True label: **{true_label}** | Predicted: **{pred_label}**")

        else:  # Dependence
            fi = feature_names.index(dep_feat)
            dep_df = pd.DataFrame({
                "feature_value": X_test[dep_feat].values,
                "shap_value": shap_values_all[:, fi],
                "label": le.inverse_transform(y_test),
            })
            fig = px.scatter(
                dep_df, x="feature_value", y="shap_value", color="label",
                color_discrete_map={"B": "#6ee7f7", "M": "#f472b6"},
                title=f"SHAP Dependence: {dep_feat}",
                trendline="lowess",
            )
            fig.update_layout(**PLOTLY_DARK, height=450)
            st.plotly_chart(fig, use_container_width=True)

    # SHAP beeswarm summary
    st.markdown('<p class="section-header">SHAP Value Distribution (top 15 features)</p>',
                unsafe_allow_html=True)
    top15 = np.argsort(np.abs(shap_values_all).mean(0))[-15:]
    sv15 = shap_values_all[:, top15]
    feat15 = [feature_names[i] for i in top15]

    fig4 = go.Figure()
    for i, feat in enumerate(feat15):
        fig4.add_trace(go.Violin(
            x=sv15[:, i], name=feat, orientation="h",
            side="positive", line_color="#a78bfa",
            fillcolor="#1e1b38", width=0.8, showlegend=False,
            points=False,
        ))
    fig4.update_layout(**PLOTLY_DARK, height=600,
                       title="SHAP Distribution per Feature",
                       xaxis_title="SHAP value")
    st.plotly_chart(fig4, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 4 – LIME EXPLORER
# ───────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-header">LIME Explanations</p>', unsafe_allow_html=True)

    lime_idx = st.slider("Instance index (LIME)", 0, len(X_test) - 1, 0, key="lime_idx")
    run_stability = st.checkbox("Run stability analysis (20 runs)")

    if st.button("Generate LIME Explanation"):
        with st.spinner("Running LIME …"):
            try:
                from lime.lime_tabular import LimeTabularExplainer

                lime_exp = LimeTabularExplainer(
                    X_train.values, feature_names=feature_names,
                    class_names=["Benign", "Malignant"], mode="classification",
                    random_state=42,
                )
                exp = lime_exp.explain_instance(
                    X_test.iloc[lime_idx].values, model.predict_proba,
                    num_features=n_lime_features,
                )
                exp_list = exp.as_list()
                feat_names_l = [x[0] for x in exp_list]
                feat_vals_l  = [x[1] for x in exp_list]
                colors_l = ["#f472b6" if v > 0 else "#6ee7f7" for v in feat_vals_l]

                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = go.Figure(go.Bar(
                        x=feat_vals_l,
                        y=feat_names_l,
                        orientation="h",
                        marker_color=colors_l,
                    ))
                    fig.update_layout(
                        **PLOTLY_DARK,
                        title=f"LIME Feature Contributions — Instance {lime_idx}",
                        height=420,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    probs = exp.predict_proba
                    st.metric("P(Benign)", f"{probs[0]:.1%}")
                    st.metric("P(Malignant)", f"{probs[1]:.1%}")
                    true_l = le.inverse_transform([y_test[lime_idx]])[0]
                    pred_l = le.inverse_transform([model.predict(X_test.iloc[[lime_idx]])[0]])[0]
                    st.metric("True Label", true_l)
                    st.metric("Predicted", pred_l)

                if run_stability:
                    with st.spinner("Stability analysis …"):
                        all_runs = []
                        for _ in range(20):
                            e = lime_exp.explain_instance(
                                X_test.iloc[lime_idx].values, model.predict_proba,
                                num_features=n_lime_features,
                            )
                            all_runs.append(dict(e.as_list()))

                        stab_feats = sorted({f for d in all_runs for f in d})
                        means = [np.mean([d.get(f, 0) for d in all_runs]) for f in stab_feats]
                        stds  = [np.std([d.get(f, 0) for d in all_runs]) for f in stab_feats]

                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(
                            x=stab_feats, y=means,
                            error_y=dict(type="data", array=stds, visible=True),
                            marker_color="#a78bfa",
                        ))
                        fig2.update_layout(
                            **PLOTLY_DARK, height=400,
                            title="LIME Stability (mean ± std over 20 runs)",
                        )
                        st.plotly_chart(fig2, use_container_width=True)

            except ImportError:
                st.error("LIME not installed. Run: `pip install lime`")


# ───────────────────────────────────────────────────────────────────
# TAB 5 – ANCHORS / INTEGRATED GRADIENTS
# ───────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<p class="section-header">Anchors & Integrated Gradients</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⚓ Anchor Rules")
        anchor_idx = st.slider("Instance (Anchors)", 0, min(49, len(X_test)-1), 0)
        if st.button("Compute Anchors"):
            with st.spinner("Running Anchors (alibi) …"):
                try:
                    from alibi.explainers import AnchorTabular
                    anchor_exp = AnchorTabular(
                        predictor=model.predict,
                        feature_names=feature_names,
                    )
                    anchor_exp.fit(X_train.values, disc_perc=(25, 50, 75))
                    result = anchor_exp.explain(X_test.iloc[anchor_idx].values)

                    st.success("Anchor found!")
                    st.markdown(f"**Precision:** `{result.precision:.2%}`")
                    st.markdown(f"**Coverage:** `{result.coverage:.2%}`")
                    st.markdown("**Rules (IF … THEN predict):**")
                    for rule in result.anchor:
                        st.markdown(f"- `{rule}`")

                except ImportError:
                    st.warning("alibi not installed. Install with: `pip install alibi`")

                    # Fallback: simple rule extraction from SHAP
                    sv = shap_values_all[anchor_idx]
                    top3 = np.argsort(np.abs(sv))[-3:][::-1]
                    st.info("**Fallback: Top-3 SHAP rules** (install alibi for true Anchors)")
                    for fi in top3:
                        direction = "above" if X_test.iloc[anchor_idx, fi] > X_test[feature_names[fi]].mean() else "below"
                        st.markdown(f"- `{feature_names[fi]}` is **{direction}** mean → SHAP = {sv[fi]:.4f}")

    with col2:
        st.markdown("#### 🌊 Integrated Gradients")
        ig_idx = st.slider("Instance (IG)", 0, len(X_test)-1, 0, key="ig_idx")
        ig_steps = st.slider("Integration steps", 20, 100, 50)
        ig_baseline = st.selectbox("Baseline", ["zero", "mean"])

        if st.button("Compute Integrated Gradients"):
            with st.spinner("Computing IG …"):
                instance = X_test.iloc[ig_idx].values.astype(float)
                if ig_baseline == "zero":
                    baseline = np.zeros_like(instance)
                else:
                    baseline = X_train.mean().values.astype(float)

                alphas = np.linspace(0, 1, ig_steps + 1)
                gradients = []
                for alpha in alphas:
                    interp = baseline + alpha * (instance - baseline)
                    eps = 1e-4
                    grad = np.zeros_like(instance)
                    for j in range(len(instance)):
                        xp = interp.copy(); xp[j] += eps
                        xm = interp.copy(); xm[j] -= eps
                        fp = model.predict_proba(xp.reshape(1, -1))[0, 1]
                        fm = model.predict_proba(xm.reshape(1, -1))[0, 1]
                        grad[j] = (fp - fm) / (2 * eps)
                    gradients.append(grad)

                avg_grads = np.mean(gradients, axis=0)
                ig_attrs = (instance - baseline) * avg_grads

                ig_df = pd.DataFrame({
                    "feature": feature_names,
                    "attribution": ig_attrs,
                }).sort_values("attribution", key=abs, ascending=True).tail(15)

                fig = go.Figure(go.Bar(
                    x=ig_df["attribution"], y=ig_df["feature"],
                    orientation="h",
                    marker_color=["#f472b6" if v > 0 else "#6ee7f7" for v in ig_df["attribution"]],
                ))
                fig.update_layout(
                    **PLOTLY_DARK, height=460,
                    title=f"Integrated Gradients — Instance {ig_idx} (baseline={ig_baseline})",
                )
                st.plotly_chart(fig, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 6 – COUNTERFACTUALS
# ───────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<p class="section-header">Counterfactual Explanations</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Counterfactuals answer: <em>"What minimal change to the input '
        'would flip the prediction?"</em></div>', unsafe_allow_html=True
    )

    cf_idx = st.slider("Instance index", 0, len(X_test)-1, 0, key="cf_idx")
    orig_pred = int(model.predict(X_test.iloc[[cf_idx]])[0])
    orig_label = le.inverse_transform([orig_pred])[0]
    st.markdown(f"**Original prediction:** `{orig_label}` (class {orig_pred})")

    if st.button("Find Counterfactuals"):
        with st.spinner("Searching for counterfactuals …"):
            instance = X_test.iloc[cf_idx].values
            train_preds = model.predict(X_train)
            opposite = X_train[train_preds != orig_pred]

            if len(opposite) == 0:
                st.error("No opposite-class training samples found.")
            else:
                from sklearn.preprocessing import StandardScaler as SS
                sc = SS()
                sc.fit(X_train)
                inst_n = sc.transform(instance.reshape(1, -1))[0]
                opp_n  = sc.transform(opposite)
                dists  = np.linalg.norm(opp_n - inst_n, axis=1)
                top_k  = min(n_cf, len(opposite))
                top_idx = np.argsort(dists)[:top_k]
                cfs = opposite.iloc[top_idx].reset_index(drop=True)

                for i, (_, cf_row) in enumerate(cfs.iterrows()):
                    delta = cf_row.values - instance
                    changed = np.where(np.abs(delta) > 0.01)[0]
                    cf_pred = le.inverse_transform(model.predict(cf_row.values.reshape(1, -1)))[0]

                    with st.expander(f"Counterfactual {i+1}  →  predicted: **{cf_pred}**  |  distance: {dists[top_idx[i]]:.3f}"):
                        delta_df = pd.DataFrame({
                            "Feature": [feature_names[j] for j in changed],
                            "Original": [round(instance[j], 4) for j in changed],
                            "Counterfactual": [round(cf_row.values[j], 4) for j in changed],
                            "Δ": [round(delta[j], 4) for j in changed],
                        }).sort_values("Δ", key=abs, ascending=False)

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            fig = go.Figure(go.Bar(
                                x=delta_df["Δ"], y=delta_df["Feature"], orientation="h",
                                marker_color=["#f472b6" if v > 0 else "#6ee7f7" for v in delta_df["Δ"]],
                            ))
                            fig.update_layout(**PLOTLY_DARK, height=350, title="Feature Deltas")
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.dataframe(delta_df, use_container_width=True)
                            st.metric("Features changed", len(changed))
                            st.metric("Sparsity", f"{(len(feature_names)-len(changed))/len(feature_names):.0%}")


# ───────────────────────────────────────────────────────────────────
# TAB 7 – EXPLANATION QUALITY
# ───────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<p class="section-header">Explanation Quality Metrics</p>',
                unsafe_allow_html=True)

    if st.button("Compute Quality Metrics"):
        with st.spinner("Evaluating explanation quality …"):
            # Faithfulness
            faith_scores = []
            for i in range(min(100, len(X_test))):
                orig_p = model.predict_proba(X_test.iloc[[i]])[0, 1]
                top5 = np.argsort(np.abs(shap_values_all[i]))[-5:]
                X_mod = X_test.iloc[[i]].copy()
                X_mod.iloc[0, top5] = 0.0
                new_p = model.predict_proba(X_mod)[0, 1]
                faith_scores.append(abs(orig_p - new_p))

            # Completeness
            ev = shap_explainer.expected_value
            base = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)
            preds = model.predict_proba(X_test)[:, 1]
            reconstructed = base + shap_values_all.sum(axis=1)
            completeness = float(np.abs(reconstructed - preds).mean())

            # Complexity
            complexity = float((np.abs(shap_values_all) > 0.01).sum(axis=1).mean())

            c1, c2, c3 = st.columns(3)
            for col, label, val, desc in [
                (c1, "Faithfulness", f"{np.mean(faith_scores):.4f}", "Mean |Δpred| when removing top-5 SHAP features. Higher = more faithful."),
                (c2, "Completeness", f"{completeness:.6f}", "Mean |base + Σshap - pred|. Lower = more complete (SHAP axiom)."),
                (c3, "Complexity",   f"{complexity:.2f} feats", "Mean # non-zero SHAP features per instance. Lower = simpler."),
            ]:
                col.markdown(
                    f'<div class="metric-card"><div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'<div style="color:#64748b;font-size:0.75rem;margin-top:0.5rem">{desc}</div></div>',
                    unsafe_allow_html=True,
                )

            st.divider()

            # Per-feature faithfulness
            per_feat = []
            for j, feat in enumerate(feature_names):
                scores = []
                for i in range(min(50, len(X_test))):
                    orig_p = model.predict_proba(X_test.iloc[[i]])[0, 1]
                    X_mod = X_test.iloc[[i]].copy()
                    X_mod.iloc[0, j] = 0.0
                    new_p = model.predict_proba(X_mod)[0, 1]
                    scores.append(abs(orig_p - new_p))
                per_feat.append({"feature": feat, "faithfulness": np.mean(scores)})

            pf_df = pd.DataFrame(per_feat).sort_values("faithfulness", ascending=True).tail(15)
            fig = px.bar(
                pf_df, x="faithfulness", y="feature", orientation="h",
                color="faithfulness",
                color_continuous_scale=["#1e2130","#6ee7f7","#a78bfa"],
                title="Per-Feature Faithfulness Score",
            )
            fig.update_layout(**PLOTLY_DARK, height=500, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 8 – LIVE PREDICT
# ───────────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown('<p class="section-header">Live Prediction with XAI Report</p>',
                unsafe_allow_html=True)

    st.markdown("Adjust the sliders to set feature values and get a full explanation.")

    with st.form("predict_form"):
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(feature_names):
            col = cols[i % 3]
            mn, mx = float(X_raw[feat].min()), float(X_raw[feat].max())
            med = float(X_raw[feat].median())
            inputs[feat] = col.slider(feat, mn, mx, med, key=f"inp_{feat}")

        submitted = st.form_submit_button("🔬 Predict & Explain")

    if submitted:
        with st.spinner("Computing prediction + SHAP …"):
            row_df = pd.DataFrame([inputs], columns=feature_names)
            row_scaled = pd.DataFrame(scaler.transform(row_df), columns=feature_names)
            pred_int = int(model.predict(row_scaled)[0])
            pred_prob = model.predict_proba(row_scaled)[0]
            pred_label = le.inverse_transform([pred_int])[0]

            # Prediction card
            card_class = "prediction-card-m" if pred_label == "M" else "prediction-card-b"
            color = "#f472b6" if pred_label == "M" else "#6ee7f7"
            meaning = "Malignant ⚠️" if pred_label == "M" else "Benign ✅"
            st.markdown(
                f'<div class="{card_class}"><div class="pred-label" style="color:{color}">{meaning}</div>'
                f'<div style="color:#94a3b8;margin-top:0.5rem">P(Malignant) = {pred_prob[1]:.1%} | '
                f'P(Benign) = {pred_prob[0]:.1%}</div></div>',
                unsafe_allow_html=True,
            )
            st.divider()

            # SHAP explanation
            sv_row = shap_explainer.shap_values(row_scaled)
            if isinstance(sv_row, list): sv_row = sv_row[1]
            sv_row = sv_row[0]

            sv_df = pd.DataFrame({
                "feature": feature_names,
                "shap": sv_row,
                "value": list(inputs.values()),
            }).sort_values("shap", key=abs, ascending=True).tail(15)

            fig = go.Figure(go.Bar(
                x=sv_df["shap"],
                y=[f"{r.feature}={r.value:.3f}" for _, r in sv_df.iterrows()],
                orientation="h",
                marker_color=["#f472b6" if v > 0 else "#6ee7f7" for v in sv_df["shap"]],
            ))
            fig.update_layout(
                **PLOTLY_DARK, height=500,
                title="SHAP Explanation for This Prediction",
                xaxis_title="SHAP value (→ increases malignant probability)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Risk summary
            top_positive = sv_df[sv_df["shap"] > 0].nlargest(3, "shap")
            top_negative = sv_df[sv_df["shap"] < 0].nsmallest(3, "shap")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔴 Top risk-increasing features:**")
                for _, r in top_positive.iterrows():
                    st.markdown(f"- `{r.feature}` = {r.value:.3f}  (SHAP +{r.shap:.4f})")
            with c2:
                st.markdown("**🔵 Top risk-reducing features:**")
                for _, r in top_negative.iterrows():
                    st.markdown(f"- `{r.feature}` = {r.value:.3f}  (SHAP {r.shap:.4f})")


# ───────────────────────────────────────────────────────────────────
# TAB 9 – HYBRID XAI  (v3.0)
# ───────────────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown('<p class="section-header">Hybrid XAI — Five-Method Ensemble Explanation</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">The Hybrid Explainer combines SHAP, LIME, Integrated Gradients, '
        'Counterfactuals, and Anchors using confidence-weighted voting. '
        'Each method\'s weight is proportional to the total magnitude of its feature attributions.</div>',
        unsafe_allow_html=True,
    )

    hyb_idx = st.slider("Instance index (Hybrid)", 0, len(X_test) - 1, 0, key="hyb_idx")
    n_ig_steps = st.slider("IG integration steps", 10, 50, 20, key="hyb_ig_steps")
    n_lime_feats = st.slider("LIME features", 5, 20, 10, key="hyb_lime_feats")

    if st.button("✴️ Run Hybrid Explanation", key="run_hybrid"):
        with st.spinner("Running all five XAI methods … (may take ~30s)"):

            p = len(feature_names)
            method_scores: dict[str, np.ndarray] = {}

            # 1. SHAP
            try:
                sv = shap_values_all[hyb_idx]
                method_scores["SHAP"] = np.abs(sv)
            except Exception as e:
                st.warning(f"SHAP skipped: {e}")

            # 2. LIME
            try:
                from lime.lime_tabular import LimeTabularExplainer
                lime_exp_h = LimeTabularExplainer(
                    X_train.values, feature_names=feature_names,
                    class_names=["Benign", "Malignant"], mode="classification",
                    random_state=42,
                )
                lime_result = lime_exp_h.explain_instance(
                    X_test.iloc[hyb_idx].values, model.predict_proba,
                    num_features=n_lime_feats,
                )
                lime_dict = dict(lime_result.as_list())
                lime_vec = np.zeros(p)
                for feat_str, coef in lime_dict.items():
                    for i, fn in enumerate(feature_names):
                        if fn in feat_str:
                            lime_vec[i] += abs(coef)
                            break
                method_scores["LIME"] = lime_vec
            except Exception as e:
                st.warning(f"LIME skipped: {e}")

            # 3. Integrated Gradients
            try:
                instance_h = X_test.iloc[hyb_idx].values.astype(float)
                baseline_h = X_train.mean().values.astype(float)
                alphas_h = np.linspace(0, 1, n_ig_steps + 1)
                grads_h = []
                for alpha in alphas_h:
                    interp = baseline_h + alpha * (instance_h - baseline_h)
                    eps = 1e-4
                    grad = np.zeros(p)
                    for j in range(p):
                        xp = interp.copy(); xp[j] += eps
                        xm = interp.copy(); xm[j] -= eps
                        fp = model.predict_proba(xp.reshape(1, -1))[0, 1]
                        fm = model.predict_proba(xm.reshape(1, -1))[0, 1]
                        grad[j] = (fp - fm) / (2 * eps)
                    grads_h.append(grad)
                ig_attrs = np.abs((instance_h - baseline_h) * np.mean(grads_h, axis=0))
                method_scores["Integrated Gradients"] = ig_attrs
            except Exception as e:
                st.warning(f"IG skipped: {e}")

            # 4. Counterfactual deltas
            try:
                from sklearn.preprocessing import StandardScaler as SS
                inst_cf = X_test.iloc[hyb_idx].values
                orig_pred_cf = int(model.predict(inst_cf.reshape(1, -1))[0])
                tr_preds_cf = model.predict(X_train)
                opp_cf = X_train[tr_preds_cf != orig_pred_cf]
                if len(opp_cf) > 0:
                    sc_cf = SS(); sc_cf.fit(X_train)
                    inst_n_cf = sc_cf.transform(inst_cf.reshape(1, -1))[0]
                    opp_n_cf = sc_cf.transform(opp_cf)
                    dist_cf = np.linalg.norm(opp_n_cf - inst_n_cf, axis=1)
                    best_cf = opp_cf.iloc[dist_cf.argmin()].values
                    cf_vec = np.abs(best_cf - inst_cf)
                    method_scores["Counterfactual"] = cf_vec
            except Exception as e:
                st.warning(f"Counterfactual skipped: {e}")

            # ── Confidence-weighted hybrid ─────────────────────────────────
            confidences = {m: float(v.sum()) for m, v in method_scores.items() if v.sum() > 0}
            total_conf = sum(confidences.values()) or 1.0
            weights = {m: c / total_conf for m, c in confidences.items()}

            hybrid_vec = np.zeros(p)
            for m, vec in method_scores.items():
                hybrid_vec += weights.get(m, 0.0) * vec

            hybrid_df = pd.DataFrame({
                "feature": feature_names,
                "hybrid_score": hybrid_vec,
            }).sort_values("hybrid_score", ascending=False).reset_index(drop=True)

            # ── Agreement SHAP ⇔ LIME ─────────────────────────────────────
            agreement = 0.0
            if "SHAP" in method_scores and "LIME" in method_scores:
                s_v, l_v = method_scores["SHAP"], method_scores["LIME"]
                if s_v.std() > 0 and l_v.std() > 0:
                    agreement = float(np.corrcoef(s_v, l_v)[0, 1])

        # ── Layout ────────────────────────────────────────────────────
        pred_label_h = le.inverse_transform([model.predict(X_test.iloc[[hyb_idx]])[0]])[0]
        pred_prob_h  = model.predict_proba(X_test.iloc[[hyb_idx]])[0]
        true_label_h = le.inverse_transform([y_test[hyb_idx]])[0]
        color_h = "#f472b6" if pred_label_h == "M" else "#6ee7f7"

        st.markdown(
            f'<div class="metric-card" style="margin-bottom:1rem">'
            f'<span style="color:{color_h};font-size:1.4rem;font-weight:700">'
            f'Predicted: {pred_label_h}</span> '
            f'&nbsp;|&nbsp; True: {true_label_h} '
            f'&nbsp;|&nbsp; P(M) = {pred_prob_h[1]:.1%} '
            f'&nbsp;|&nbsp; 🤝 SHAP⇔LIME: ρ = {agreement:.3f}'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([3, 1])

        with col_left:
            # Hybrid importance bar chart (top 20)
            top20 = hybrid_df.head(20).sort_values("hybrid_score", ascending=True)
            colors_h = [
                f"rgba(167,139,250,{0.4 + 0.6 * (i / len(top20))})"
                for i in range(len(top20))
            ]
            fig_hyb = go.Figure(go.Bar(
                x=top20["hybrid_score"],
                y=top20["feature"],
                orientation="h",
                marker_color=colors_h,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_hyb.update_layout(
                **PLOTLY_DARK, height=550,
                title=f"Hybrid Feature Importance — Instance {hyb_idx}",
                xaxis_title="Weighted hybrid score",
            )
            st.plotly_chart(fig_hyb, use_container_width=True)

        with col_right:
            # Method weight pie chart
            if weights:
                pie_fig = go.Figure(go.Pie(
                    labels=list(weights.keys()),
                    values=list(weights.values()),
                    hole=0.45,
                    marker_colors=["#6ee7f7", "#a78bfa", "#f472b6", "#34d399", "#fbbf24"],
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{percent}<extra></extra>",
                ))
                pie_fig.update_layout(
                    **PLOTLY_DARK, height=300,
                    title="Method Weights",
                    showlegend=False,
                    margin=dict(t=40, b=0, l=0, r=0),
                )
                st.plotly_chart(pie_fig, use_container_width=True)

            st.markdown("**Top 5 hybrid features:**")
            for _, row_h in hybrid_df.head(5).iterrows():
                st.markdown(f"- `{row_h['feature']}` → {row_h['hybrid_score']:.4f}")

        st.divider()
        st.markdown('<p class="section-header">Method Attribution Heatmap (top 15 features)</p>',
                    unsafe_allow_html=True)

        top15_feats = hybrid_df["feature"].head(15).tolist()
        top15_idx   = [feature_names.index(f) for f in top15_feats]

        heat_data, heat_methods = [], []
        for m_name, m_vec in method_scores.items():
            heat_data.append([m_vec[i] for i in top15_idx])
            heat_methods.append(m_name)

        if heat_data:
            heat_arr = np.array(heat_data)
            # Normalise each method row to [0, 1]
            row_max = heat_arr.max(axis=1, keepdims=True)
            row_max[row_max == 0] = 1
            heat_norm = heat_arr / row_max

            fig_heat = go.Figure(go.Heatmap(
                z=heat_norm,
                x=top15_feats,
                y=heat_methods,
                colorscale=[[0, "#0d0f18"], [0.5, "#6ee7f7"], [1, "#a78bfa"]],
                text=[[f"{heat_arr[i,j]:.3f}" for j in range(len(top15_feats))]
                      for i in range(len(heat_methods))],
                texttemplate="%{text}",
                hovertemplate="%{y} | %{x}: %{text}<extra></extra>",
                showscale=True,
                colorbar=dict(title="Normalised", tickfont=dict(color="#e2e8f0")),
            ))
            fig_heat.update_layout(
                **PLOTLY_DARK, height=320,
                title="Per-Method Normalised Importance (top 15 hybrid features)",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

