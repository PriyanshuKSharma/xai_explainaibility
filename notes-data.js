// XAI Explainability Research - Notes Database
const notesData = {
  // --- FOUNDATION & CONCEPTS ---
  "00_introduction_clinical_trust.md": {
    id: "00a",
    title: "Clinical Trust & XAI in Medical Diagnosis",
    cat: "foundation",
    search: "introduction clinical trust black box clinician oncology radiology healthcare diagnosis patient",
    desc: "The critical role of transparency and trust in ML-driven clinical decision support systems.",
    tags: ["Trust", "Clinical"],
    content: `
      <h3>Introduction: Bridging the Clinical Interpretability Gap</h3>
      <p>The proliferation of machine learning in healthcare has transformed diagnostic processes, treatment planning, and patient care delivery. However, the <strong>"black box"</strong> nature of complex machine learning models poses significant challenges for clinical adoption, regulatory compliance, and patient trust.</p>
      
      <h4>The Clinical Challenge</h4>
      <p>Healthcare professionals require not only highly accurate predictions but also understandable, coherent explanations to make informed decisions and maintain accountability. A model that outputs a 97% probability of malignancy without clinical reasoning forces the clinician into a passive role, failing to provide the safety guarantees required for high-stakes decision-making.</p>
      
      <div class="callout warning">
        <strong>Underlying Issue:</strong> Overly simplified post-hoc explanations can create a false sense of model transparency, highlighting the need for rigorous multi-dimensional evaluation frameworks that go beyond surface-level interpretability.
      </div>
      
      <h4>Core Diagnostic Objectives</h4>
      <ul>
        <li><strong>Accuracy & Safety:</strong> Outperforming standard baselines to guarantee diagnostic precision.</li>
        <li><strong>Meaningful Explanation:</strong> Mapping statistical attributions directly to clinical features (e.g. cell nucleus size, border irregularity).</li>
        <li><strong>Auditability:</strong> Maintaining deterministic, reproducible decision trails for regulatory compliance.</li>
      </ul>
    `
  },
  "00_gdpr_article_22_compliance.md": {
    id: "00b",
    title: "GDPR Article 22: Right to Explanation",
    cat: "foundation",
    search: "gdpr article 22 right to explanation automated decisions transparency legal compliance regulatory",
    desc: "Mapping clinical attributions to automated decision-making transparency mandates under GDPR.",
    tags: ["GDPR", "Regulation"],
    content: `
      <h3>GDPR Article 22 & Automated Decision-Making</h3>
      <p>Under the European Union General Data Protection Regulation (GDPR), specifically <strong>Article 22</strong>, individuals have the right not to be subject to decisions based solely on automated processing which produce legal or similarly significant effects.</p>
      
      <h4>The "Right to Explanation"</h4>
      <p>When automated decisions are made (such as an automated cancer diagnosis risk scoring), the data controller is required to implement suitable measures to safeguard the data subject's rights, which includes the right to obtain human intervention, express their point of view, and receive <strong>"meaningful information about the logic involved"</strong>.</p>
      
      <div class="callout info">
        <strong>Clinical Mapping:</strong> In our framework, instance-level explainers (LIME, Anchors, and Counterfactuals) satisfy this requirement by translating the high-dimensional machine learning boundary into localized attributions, rule-based bounds, and actionable recourses accessible to clinicians and patients alike.
      </div>
      
      <h4>Compliance Matrix</h4>
      <table>
        <thead>
          <tr>
            <th>Regulatory Requirement</th>
            <th>XAI Mapping</th>
            <th>Fulfillment Level</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Explain automated logic</td>
            <td>LIME local coefficients & Anchor rules</td>
            <td><strong>Complete</strong> (Human-readable logic)</td>
          </tr>
            <td>Provide actionable recourse</td>
            <td>Counterfactual Explanations (DiCE)</td>
            <td><strong>Complete</strong> (Pathways to benign class)</td>
          </tr>
          <tr>
            <td>Facilitate human oversight</td>
            <td>Interactive Dashboard UI & confidence weights</td>
            <td><strong>Complete</strong> (Clinician-in-the-loop control)</td>
          </tr>
        </tbody>
      </table>
    `
  },
  "00_eu_ai_act_high_risk.md": {
    id: "00c",
    title: "EU AI Act & High-Risk Diagnostics",
    cat: "foundation",
    search: "eu ai act high-risk high risk diagnostics regulation audit compliance logging security auditability",
    desc: "Regulatory analysis of healthcare AI models and documentation, human oversight, and audit trails.",
    tags: ["AI Act", "Compliance"],
    content: `
      <h3>EU AI Act (2024) Compliance in Healthcare AI</h3>
      <p>The recently enacted <strong>European Union AI Act (2024)</strong> classifies healthcare diagnostic systems as <strong>"high-risk AI systems"</strong> (under Annex III), placing strict compliance burdens on deployment.</p>
      
      <h4>Key Legal Mandates for High-Risk AI</h4>
      <ol>
        <li><strong>Transparency and Provision of Information:</strong> High-risk systems must be designed and developed in such a way that their operation is sufficiently transparent to enable users to interpret the system’s output.</li>
        <li><strong>Human Oversight:</strong> Systems must allow active, real-time human intervention to prevent or minimize risks (Clinician-in-the-loop).</li>
        <li><strong>Accuracy, Robustness, and Cybersecurity:</strong> High-risk AI must meet rigorous, verifiable standards of accuracy and robust failure-prevention throughout its lifecycle.</li>
        <li><strong>Automatic Logging:</strong> Continuous logging of operational events (audit trail) is required for traceability.</li>
      </ol>
      
      <div class="callout positive">
        <strong>Ensemble Solution:</strong> Our framework directly resolves these requirements:
        <ul>
          <li><strong>Audit Trail:</strong> The async FastAPI served backend logs all prediction requests alongside their multi-method confidence-weighted explanations.</li>
          <li><strong>Oversight:</strong> The 9-tab Streamlit dashboard enables clinicians to interactively modify inputs and trace attributions prior to signing off on diagnoses.</li>
        </ul>
      </div>
    `
  },
  "00_xai_taxonomy.md": {
    id: "00d",
    title: "Explainable AI Taxonomy",
    cat: "foundation",
    search: "explainable ai taxonomy post-hoc ante-hoc global local model-agnostic model-specific scope",
    desc: "Comparing post-hoc vs ante-hoc, global vs local, and model-specific vs model-agnostic explanation scopes.",
    tags: ["Taxonomy", "Concepts"],
    content: `
      <h3>XAI Taxonomy: A Structural Framework</h3>
      <p>Explainable AI techniques are structured across three primary taxonomic axes, allowing researchers to choose the optimal paradigm for their domain constraints.</p>
      
      <div class="grid-2">
        <div class="card-inner">
          <h5>1. Scope of Explanation</h5>
          <ul>
            <li><strong>Global:</strong> Explains the complete model behavior across the entire data distribution (e.g. SHAP global feature importances).</li>
            <li><strong>Local:</strong> Explains individual predictions for specific instances (e.g. LIME local coefficients, Anchor rules, Counterfactuals).</li>
          </ul>
        </div>
        <div class="card-inner">
          <h5>2. Model Dependency</h5>
          <ul>
            <li><strong>Model-Agnostic:</strong> Can be applied to any machine learning model treated as a black-box (e.g. LIME, Anchors, Counterfactuals).</li>
            <li><strong>Model-Specific:</strong> Relies on internal model parameters or architectures (e.g. TreeExplainer for trees, Integrated Gradients for neural networks).</li>
          </ul>
        </div>
      </div>
      
      <h4>Ante-hoc vs Post-hoc Explanations</h4>
      <p><strong>Ante-hoc (Inherently Interpretable)</strong> models, such as Decision Trees or Sparse Linear Models, trade off predictive power for intrinsic transparency. <strong>Post-hoc</strong> explainers, such as SHAP or LIME, act on complex, highly-accurate black-box models after training to reverse-engineer explanations without degrading classification performance.</p>
    `
  },

  // --- DATA & FEATURE ENGINEERING ---
  "01_wisconsin_dataset_morphology.md": {
    id: "01a",
    title: "Wisconsin Breast Cancer Dataset Morphology",
    cat: "data-engineering",
    search: "wisconsin breast cancer dataset morphology fine needle aspirate nuclei statistics features dimensions",
    desc: "Explaining the 10 core morphological cell nuclear features and 30 statistical columns of fine needle aspirates.",
    tags: ["Dataset", "Biology"],
    content: `
      <h3>Wisconsin Breast Cancer Diagnostic (WDBC) Dataset</h3>
      <p>We utilize the benchmark dataset containing <strong>569 instances</strong> with <strong>30 numerical features</strong> computed from digitized images of breast mass fine needle aspirates (FNA).</p>
      
      <h4>Target Distribution</h4>
      <ul>
        <li><strong>Benign (Class 0):</strong> 357 instances (62.7%)</li>
        <li><strong>Malignant (Class 1):</strong> 212 instances (37.3%)</li>
      </ul>
      
      <h4>Morphological Features</h4>
      <p>The ten core real-valued features computed for each cell nucleus are:</p>
      <ol>
        <li><strong>Radius:</strong> Mean of distances from center to points on the perimeter</li>
        <li><strong>Texture:</strong> Standard deviation of gray-scale values</li>
        <li><strong>Perimeter:</strong> Cell boundary distance</li>
        <li><strong>Area:</strong> Inside cell nuclear area</li>
        <li><strong>Smoothness:</strong> Local variation in radius lengths</li>
        <li><strong>Compactness:</strong> $\\text{perimeter}^2 / \\text{area} - 1.0$</li>
        <li><strong>Concavity:</strong> Severity of concave portions of the contour</li>
        <li><strong>Concave Points:</strong> Number of concave portions of the contour</li>
        <li><strong>Symmetry:</strong> Bilateral alignment</li>
        <li><strong>Fractal Dimension:</strong> "Coastline approximation" - 1</li>
      </ol>
      <p>Each feature is reported in three statistics: **Mean**, **Standard Error (SE)**, and **Worst (largest)** values, yielding the 30 baseline features.</p>
    `
  },
  "01_data_preprocessing_pipeline.md": {
    id: "01b",
    title: "Data Preprocessing & Scaling Pipeline",
    cat: "data-engineering",
    search: "data preprocessing pipeline split stratified standardizer z-score scaling scikit-learn standardscaler",
    desc: "Formulating data preprocessing steps, stratified split, and robust Z-score standardization.",
    tags: ["Pipeline", "ML"],
    content: `
      <h3>Preprocessing & Feature Standardization</h3>
      <p>To prepare high-dimensional morphological cell features for classification, a deterministic scaling pipeline is built using standard statistical bounds.</p>
      
      <h4>1. Null and Outlier Handling</h4>
      <p>Unnecessary identifier columns (e.g. <code>id</code>) and empty columns are dropped. Morphological features are verified for consistency, ensuring no missing values exist.</p>
      
      <h4>2. Stratified Data Splitting</h4>
      <p>To preserve target balance in the presence of relatively small sample sizes, we enforce a <strong>80/20 train/test split</strong>, stratified by class label:</p>
      <pre><code class="language-python">from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)</code></pre>
      
      <h4>3. Z-Score Standardization</h4>
      <p>Features are scaled using Z-score normalization computed solely on the training partition to prevent data leakage:</p>
      <p class="formula">$$z = \\frac{x - \\mu}{\\sigma}$$</p>
      <p>Where $\\mu$ is the feature mean and $\\sigma$ is the standard deviation. A pre-fit <code>StandardScaler</code> is persisted to <code>scaler.pkl</code> to standardize streaming clinical inputs in production.</p>
    `
  },
  "01_perimeter_area_ratio.md": {
    id: "01c",
    title: "Perimeter-Area Ratio (Engineered Feature)",
    cat: "data-engineering",
    search: "perimeter area ratio engineered feature formula shape boundary elongation irregularity",
    desc: "Formulating and evaluating the perimeter-to-area ratio feature to capture irregular tumor boundary elongation.",
    tags: ["Feature", "Math"],
    content: `
      <h3>Engineered Feature: Perimeter-Area Ratio ($r_{pa}$)</h3>
      <p>Malignant breast tumor cells frequently display highly irregular, asymmetrical, and elongated borders compared to the round, circular profiles of benign cells.</p>
      
      <h4>Mathematical Formulation</h4>
      <p>To isolate this morphological property in a low-dimensional space, we construct the <strong>Perimeter-Area Ratio</strong>:</p>
      <p class="formula">$$r_{pa} = \\frac{\\text{perimeter\\_mean}}{\\text{area\\_mean} + \\epsilon}$$</p>
      <p>Where $\\epsilon = 10^{-8}$ is a tiny floating-point stabilizer to prevent division-by-zero errors. High values of $r_{pa}$ explicitly identify highly irregular cell boundaries which are strongly indicative of rapid, invasive growth.</p>
      
      <div class="callout positive">
        <strong>Ablation Gain:</strong> Adding $r_{pa}$ to the raw feature set yields a <strong>+0.4% increase</strong> in classification accuracy and elevates ROC-AUC by 0.001.
      </div>
    `
  },
  "01_concavity_points_ratio.md": {
    id: "01d",
    title: "Concavity-to-Points Ratio (Engineered Feature)",
    cat: "data-engineering",
    search: "concavity to points ratio engineered feature formula shape deep indentation cell structure",
    desc: "Distinguishing deep-indentation aggressive cell structures from many shallow indentations.",
    tags: ["Feature", "Clinical"],
    content: `
      <h3>Engineered Feature: Concavity-to-Points Ratio ($r_{cp}$)</h3>
      <p>Clinical pathology shows that the structural nature of cell contours is highly discriminative. Specifically, a few very deep concave indentations suggest aggressive growth more strongly than many superficial, shallow boundary indentations.</p>
      
      <h4>Mathematical Formulation</h4>
      <p>To distinguish between these states, we introduce the <strong>Concavity-to-Points Ratio</strong>:</p>
      <p class="formula">$$r_{cp} = \\frac{\\text{concavity\\_mean}}{\\text{concave\\_points\\_mean} + \\epsilon}$$</p>
      
      <h4>Clinical Interpretation</h4>
      <ul>
        <li><strong>High $r_{cp}$ Ratio:</strong> Characterized by severe, localized concavities (large concavity value relative to the number of concave points), identifying deep nuclear pocketing common in high-grade malignancies.</li>
        <li><strong>Low $r_{cp}$ Ratio:</strong> Indicates uniform, shallow indentations typical of stable cell shapes.</li>
      </ul>
      <p>In our model's global SHAP importance hierarchy, $r_{cp}$ ranks as a highly discriminative secondary feature, providing valuable clinical interpretability.</p>
    `
  },
  "01_radius_compactness_growth.md": {
    id: "01e",
    title: "Radius & Compactness Growth (Engineered Features)",
    cat: "data-engineering",
    search: "radius compactness growth worst case deviation tumor heterogeneity cell packing engineered",
    desc: "Quantifying worst-case tumor heterogeneity and deviation in irregular cell packing.",
    tags: ["Feature", "Biology"],
    content: `
      <h3>Engineered Features: Radius Growth & Compactness Growth</h3>
      <p>Aggressive breast malignancies display high cellular heterogeneity, meaning cell nuclear size and shape vary wildly across the tumor mass. Static means fail to capture this worst-case boundary growth.</p>
      
      <h4>Radius Growth ($\\Delta r$)</h4>
      <p>Quantifies the size deviation from average cell size to the worst-case (largest) cell size within the fine needle aspirate:</p>
      <p class="formula">$$\\Delta r = \\text{radius\\_worst} - \\text{radius\\_mean}$$</p>
      <p>A high $\\Delta r$ value strongly indicates aggressive tumor heterogeneity, which is a major radiological indicator of high-grade malignancies.</p>
      
      <h4>Compactness Growth ($\\Delta c$)</h4>
      <p>Measures the worst-case deviation in cell packing compactness:</p>
      <p class="formula">$$\\Delta c = \\text{compactness\\_worst} - \\text{compactness\\_mean}$$</p>
      <p>Large $\\Delta c$ values identify local clusters of highly chaotic, irregularly-packed cell structures, signaling micro-invasion zones.</p>
      
      <div class="callout positive">
        <strong>Ablation Gain:</strong> Radius Growth ($\\Delta r$) is the single most effective engineered feature, providing a <strong>+0.6% standalone accuracy boost</strong>.
      </div>
    `
  },

  // --- MODELS & ENSEMBLE ---
  "02_multi_model_pipeline.md": {
    id: "02a",
    title: "Multi-Model Training Pipeline",
    cat: "models",
    search: "multi model pipeline lightgbm xgboost random forest logistic regression gradient boosting classifiers ensemble",
    desc: "Constructing an ensemble of Random Forest, XGBoost, LightGBM, Gradient Boosting, and Logistic Regression.",
    tags: ["Models", "Ensemble"],
    content: `
      <h3>Ensemble Model Pipeline</h3>
      <p>To establish a highly robust diagnostic baseline, our framework builds a multi-model pipeline utilizing five distinct classification architectures:</p>
      
      <div class="grid-2">
        <div class="card-inner">
          <h5>1. Boosting Architectures</h5>
          <ul>
            <li><strong>LightGBM:</strong> Outstanding speed and asymmetric, leaf-wise tree growth.</li>
            <li><strong>XGBoost:</strong> Regularized, column-block tree boosting for robustness.</li>
            <li><strong>Gradient Boosting:</strong> Classic sequential stage-wise gradient optimization.</li>
          </ul>
        </div>
        <div class="card-inner">
          <h5>2. Statistical & Bagging Models</h5>
          <ul>
            <li><strong>Random Forest:</strong> Parallel bootstrap aggregation with randomized split nodes.</li>
            <li><strong>Logistic Regression:</strong> High-performance linear baseline with L2 regularization.</li>
          </ul>
        </div>
      </div>
      
      <h4>Pipeline Architecture</h4>
      <p>The models are trained concurrently on the 34-feature scaled dataset. In the serving layout, the models are aggregated inside a **Majority-Vote Hybrid Ensemble**, which takes the probability averages and consensus predictions, achieving a near-perfect diagnostic boundary.</p>
    `
  },
  "02_optuna_hyperparameter_optimization.md": {
    id: "02b",
    title: "Bayesian Hyperparameter Tuning",
    cat: "models",
    search: "optuna hyperparameter optimization bayesian search lightgbm parameters trials k-fold cross-validation",
    desc: "Leveraging Optuna to automatically search high-dimensional hyperparameter spaces.",
    tags: ["Optimization", "Optuna"],
    content: `
      <h3>Optuna Optimization Pipeline</h3>
      <p>To maximize model performance and prevent overfitting on our morphological feature set, we integrate **Optuna** to execute sequential Bayesian hyperparameter optimization.</p>
      
      <h4>Objective Function Definition</h4>
      <p>For each classifier, Optuna samples from pre-defined parameter ranges and evaluates performance using a stratified <strong>5-Fold Cross-Validation F1-Score</strong> mean, maximizing stability:</p>
      
      <pre><code class="language-python">def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 64)
    }
    model = lgb.LGBMClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    return score</code></pre>
      
      <h4>Optimization Efficiency</h4>
      <p>Using Tree-structured Parzen Estimator (TPE) search samplers, the pipeline converges on the optimal hyperparameter profile in under 100 trials, providing an efficient alternative to brute-force grid searches.</p>
    `
  },
  "02_model_performance_comparison.md": {
    id: "02c",
    title: "Multi-Model Performance Benchmark",
    cat: "models",
    search: "model performance comparison metrics table accuracy precision recall f1-score roc-auc ensemble",
    desc: "Tabular analysis of accuracy, precision, recall, F1, and ROC-AUC metrics (featuring Table I).",
    tags: ["Benchmark", "Metrics"],
    content: `
      <h3>Multi-Model Evaluation & Table I</h3>
      <p>We evaluate our standardized multi-model pipeline on the held-out stratified test set (20% partition). The classification results are presented in Table I below.</p>
      
      <h4>Table I: Classifier Performance Benchmark</h4>
      <table>
        <thead>
          <tr>
            <th>Model Architecture</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>ROC-AUC</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Random Forest</td>
            <td>0.947</td>
            <td>0.951</td>
            <td>0.963</td>
            <td>0.957</td>
            <td>0.989</td>
          </tr>
          <tr>
            <td>XGBoost</td>
            <td>0.956</td>
            <td>0.961</td>
            <td>0.971</td>
            <td>0.966</td>
            <td>0.991</td>
          </tr>
          <tr class="highlight">
            <td><strong>LightGBM (Selected Base)</strong></td>
            <td>0.956</td>
            <td>0.958</td>
            <td>0.975</td>
            <td>0.966</td>
            <td>0.992</td>
          </tr>
          <tr>
            <td>Gradient Boosting</td>
            <td>0.947</td>
            <td>0.944</td>
            <td>0.971</td>
            <td>0.957</td>
            <td>0.988</td>
          </tr>
          <tr>
            <td>Logistic Regression</td>
            <td>0.930</td>
            <td>0.933</td>
            <td>0.952</td>
            <td>0.942</td>
            <td>0.981</td>
          </tr>
          <tr class="primary-row">
            <td><strong>Hybrid Ensemble (Consensus)</strong></td>
            <td><strong>0.974</strong></td>
            <td><strong>0.978</strong></td>
            <td><strong>0.981</strong></td>
            <td><strong>0.979</strong></td>
            <td><strong>0.995</strong></td>
          </tr>
        </tbody>
      </table>
      
      <p>The <strong>Hybrid Ensemble</strong> achieves a remarkable <strong>97.4% overall accuracy</strong> and a <strong>0.995 ROC-AUC</strong>, proving that multi-architecture voting successfully buffers individual model weaknesses.</p>
    `
  },
  "02_feature_importance_ablation.md": {
    id: "02d",
    title: "Feature Importance Ablation Study",
    cat: "models",
    search: "feature importance ablation study configuration raw perimeter area concavity points radius compactness growth table",
    desc: "Isolating and quantifying the accuracy and AUC gain of our engineered features (featuring Table II).",
    tags: ["Ablation", "Results"],
    content: `
      <h3>Ablation Study: Quantifying Feature Engineering Value</h3>
      <p>To demonstrate the statistical impact of our four engineered morphological features, we execute a backward ablation study, progressively adding features to the base 30 columns.</p>
      
      <h4>Table II: Ablation Study Results</h4>
      <table>
        <thead>
          <tr>
            <th>Feature Configuration</th>
            <th>Total Columns</th>
            <th>Accuracy</th>
            <th>F1-Score</th>
            <th>ROC-AUC</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Baseline (Raw WDBC Features)</td>
            <td>30</td>
            <td>0.947</td>
            <td>0.957</td>
            <td>0.989</td>
          </tr>
          <tr>
            <td>+ Perimeter-Area Ratio ($r_{pa}$)</td>
            <td>31</td>
            <td>0.951</td>
            <td>0.960</td>
            <td>0.990</td>
          </tr>
          <tr>
            <td>+ Concavity-to-Points Ratio ($r_{cp}$)</td>
            <td>32</td>
            <td>0.954</td>
            <td>0.963</td>
            <td>0.991</td>
          </tr>
          <tr>
            <td>+ Radius Growth ($\\Delta r$)</td>
            <td>33</td>
            <td>0.960</td>
            <td>0.967</td>
            <td>0.992</td>
          </tr>
          <tr class="highlight">
            <td><strong>+ Compactness Growth (Full Set)</strong></td>
            <td><strong>34</strong></td>
            <td><strong>0.965</strong></td>
            <td><strong>0.971</strong></td>
            <td><strong>0.993</strong></td>
          </tr>
        </tbody>
      </table>
      
      <h4>Key Findings</h4>
      <p>Integrating the full domain-engineered feature set drives a <strong>+1.8% overall accuracy gain</strong> over raw baseline values. Radius Growth ($\\Delta r$) provides the single largest marginal gain (+0.6%), confirming that capturing local tumor boundary size variation is highly informative for resolving complex classification edge-cases.</p>
    `
  },

  // --- SHAP ---
  "03_shapley_game_theory.md": {
    id: "03a",
    title: "SHAP Cooperative Game Theory",
    cat: "shap",
    search: "shap shapley cooperative game theory mathematical formula fair payout feature attribution",
    desc: "The mathematical foundation of Shapley values for fair payout distribution.",
    tags: ["SHAP", "Math"],
    content: `
      <h3>SHAP: Theoretical Game Theory Foundations</h3>
      <p>SHAP (SHapley Additive exPlanations) computes feature attribution values based on <strong>cooperative game theory</strong>. The training features are treated as "players" in a coalition, and the model prediction is the "payout" of the game.</p>
      
      <h4>The Shapley Value Formula</h4>
      <p>The mathematical attribution $\\phi_i$ for feature $i$ is formulated as the weighted average of its marginal contributions across all possible feature subsets:</p>
      
      <p class="formula">$$\\phi_i = \\sum_{S \\subseteq F \\setminus \\{i\\}} \\frac{|S|!(|F|-|S|-1)!}{|F|!}\\left[f(S \\cup \\{i\\}) - f(S)\\right]$$</p>
      
      <p>Where:</p>
      <ul>
        <li>$F$ is the complete set of all features.</li>
        <li>$S$ is a subset of features excluding feature $i$.</li>
        <li>$f(S)$ is the model output when conditioned only on features in subset $S$.</li>
      </ul>
      <p>This formulation ensures that the contribution of each feature is calculated fairly, accounting for all possible multi-feature interactions.</p>
    `
  },
  "03_shapley_axioms.md": {
    id: "03b",
    title: "Shapley Value Axioms",
    cat: "shap",
    search: "shapley axioms efficiency symmetry dummy additivity properties explainable ai guarantees",
    desc: "Explaining the core mathematical properties: Efficiency, Symmetry, Dummy, and Additivity.",
    tags: ["Axioms", "Theory"],
    content: `
      <h3>Axiomatic Guarantees of Shapley Values</h3>
      <p>SHAP is the only local feature attribution method that satisfies four fundamental properties, providing strong theoretical guarantees of fairness and consistency.</p>
      
      <div class="grid-2">
        <div class="card-inner">
          <h5>1. Efficiency (Local Accuracy)</h5>
          <p>The sum of all feature attributions matches the difference between the model prediction $f(x)$ and the baseline expected value $\\mathbb{E}[f(x)]$:</p>
          <p class="formula">$$\\sum_{i=1}^{p} \\phi_i(x) = f(x) - \\mathbb{E}[f(X)]$$</p>
        </div>
        <div class="card-inner">
          <h5>2. Symmetry</h5>
          <p>If two features $i$ and $j$ contribute identically to all possible coalitions, their attributions must be equal:</p>
          <p class="formula">$$\\phi_i(x) = \\phi_j(x)$$</p>
        </div>
      </div>
      
      <div class="grid-2" style="margin-top: 1rem;">
        <div class="card-inner">
          <h5>3. Dummy (Null Player)</h5>
          <p>A feature that has no impact on any marginal prediction receives an attribution score of zero:</p>
          <p class="formula">$$\\phi_i(x) = 0$$</p>
        </div>
        <div class="card-inner">
          <h5>4. Additivity</h5>
          <p>For independent models, attributions can be summed, enabling linear model ensembles without recalculation.</p>
        </div>
      </div>
    `
  },
  "03_tree_explainer_efficiency.md": {
    id: "03c",
    title: "TreeExplainer Complexity & Efficiency",
    cat: "shap",
    search: "treeexplainer tree explainer polynomial complexity algorithm trees decision tree forest lundberg",
    desc: "Exploring how exact Shapley values are computed in polynomial time for tree structures.",
    tags: ["Algorithms", "Tree"],
    content: `
      <h3>TreeExplainer: Fast Exact Attributions</h3>
      <p>Calculating exact Shapley values carries exponential complexity $O(2^p)$ where $p$ is the number of features. For high-dimensional datasets like our 34-feature clinical array, model-agnostic kernel SHAP is too slow for real-time deployment.</p>
      
      <h4>The TreeExplainer Breakthrough</h4>
      <p>Introduced by Lundberg et al. (2020), **TreeExplainer** leverages the tree structure of gradient-boosted trees to calculate exact Shapley values in polynomial time:</p>
      <p class="formula">$$\\text{Complexity} = O(T \\cdot L \\cdot D^2)$$</p>
      <p>Where:</p>
      <ul>
        <li>$T$ is the number of decision trees in the ensemble.</li>
        <li>$L$ is the maximum number of leaves in any single tree.</li>
        <li>$D$ is the maximum depth of the trees.</li>
      </ul>
      <p>This allows exact attributions to be calculated in under **45 ms per instance** in our LightGBM model, making it highly suitable for real-time clinical point-of-care deployment.</p>
    `
  },
  "03_shap_global_local_analysis.md": {
    id: "03d",
    title: "SHAP Global & Local Analysis",
    cat: "shap",
    search: "shap global local analysis feature importance interaction matrix dependency plot waterfall",
    desc: "Interpreting global feature importance bar charts, local waterfall force plots, and feature interaction values.",
    tags: ["SHAP", "Visuals"],
    content: `
      <h3>Interpreting SHAP Outputs</h3>
      <p>SHAP provides a unified explanation framework spanning both global model oversight and local instance-level debugging.</p>
      
      <h4>1. Global Feature Importance</h4>
      <p>Global importance is calculated as the mean absolute Shapley value across the entire evaluation partition:</p>
      <p class="formula">$$I_j = \\frac{1}{N} \\sum_{i=1}^{N} |\\phi_j(x_i)|$$</p>
      
      <h4>SHAP Global Ranking in LightGBM</h4>
      <ol>
        <li><strong>Worst Concave Points:</strong> 0.142 (Primary diagnostic marker)</li>
        <li><strong>Worst Perimeter:</strong> 0.128</li>
        <li><strong>Radius Growth (Engineered):</strong> 0.112 (High diagnostic importance)</li>
        <li><strong>Mean Concave Points:</strong> 0.095</li>
      </ol>
      
      <h4>2. Local Waterfall Plots</h4>
      <p>Local waterfall plots trace how individual features shift the prediction away from the expected base value ($\\mathbb{E}[f(x)] = 0.62$ benign) toward the final diagnosis ($f(x) = 0.98$ malignant), with green bars indicating benign-shifting features and red bars indicating malignant-shifting features.</p>
    `
  },

  // --- LIME ---
  "04_lime_local_surrogate.md": {
    id: "04a",
    title: "LIME Local Surrogate Objective Function",
    cat: "lime",
    search: "lime local surrogate objective function mathematical formula proximity kernel regularization ribeiro",
    desc: "Formulating the local surrogate training loss, proximity kernel, and complexity regularization.",
    tags: ["LIME", "Math"],
    content: `
      <h3>LIME: Local Interpretable Model-Agnostic Explanations</h3>
      <p>LIME, proposed by Ribeiro et al. (2016), explains individual predictions by learning a highly interpretable, local linear surrogate model $g$ in the immediate neighborhood of the target instance.</p>
      
      <h4>The LIME Objective Function</h4>
      <p>The local surrogate model $g$ is obtained by minimizing a loss function balancing local approximation fidelity and model complexity:</p>
      
      <p class="formula">$$\\xi(x) = \\arg\\min_{g \\in G} \\mathcal{L}(f, g, \\pi_x) + \\Omega(g)$$</p>
      
      <p>Where:</p>
      <ul>
        <li>$\\mathcal{L}$ is the local loss measuring how close the surrogate model's predictions $g(z)$ are to the black-box model's predictions $f(z)$.</li>
        <li>$\\pi_x(z)$ is the proximity kernel defining the local neighborhood size around target instance $x$.</li>
        <li>$\\Omega(g)$ is the complexity penalty restricting the number of non-zero coefficients in $g$ to preserve human readability.</li>
      </ul>
    `
  },
  "04_lime_perturbation_neighborhoods.md": {
    id: "04b",
    title: "Perturbation Sampling & Proximity Kernel",
    cat: "lime",
    search: "perturbation sampling proximity kernel lime exponential distance metric neighborhood weights",
    desc: "How LIME generates local perturbations and weights them via the exponential kernel.",
    tags: ["LIME", "Sampling"],
    content: `
      <h3>LIME Local Perturbation Mechanics</h3>
      <p>LIME does not inspect the internal weights of the machine learning model. Instead, it probes the model's behavior by generating thousands of perturbed samples around the target instance.</p>
      
      <h4>1. Perturbation Generation</h4>
      <p>LIME samples $N = 2000$ points from a normal distribution centered at target instance $x$, with standard deviations matching the feature distributions of the training partition.</p>
      
      <h4>2. Proximity Weighting</h4>
      <p>Each perturbed sample $z'$ is weighted based on its distance to $x$ using an exponential proximity kernel:</p>
      
      <p class="formula">$$\\pi_x(z) = \\exp\\left( -\\frac{D(x, z)^2}{\\sigma^2} \\right)$$</p>
      
      <p>Where:</p>
      <ul>
        <li>$D(x, z)$ is the Euclidean distance between the target and perturbed instances.</li>
        <li>$\\sigma$ is the kernel bandwidth parameter.</li>
      </ul>
      <p>If $\\sigma$ is too large, the surrogate model averages out global boundaries, losing local precision. If $\\sigma$ is too small, explanations become highly unstable due to high variance.</p>
    `
  },
  "04_lime_stability_analysis.md": {
    id: "04c",
    title: "LIME Stability & Variance",
    cat: "lime",
    search: "lime stability variance coefficient variation run consistency evaluation score",
    desc: "Analyzing coefficient of variation of LIME attributions across identical runs.",
    tags: ["LIME", "Stability"],
    content: `
      <h3>LIME Stability Assessment</h3>
      <p>Because LIME relies on random perturbation sampling, identical runs on the same instance can produce slightly different attributions, presenting a major challenge for deployment in clinical environments.</p>
      
      <h4>Stability Quantification</h4>
      <p>To quantify this variance, we run LIME 20 times on a single target instance and calculate the <strong>Coefficient of Variation (CV)</strong> for the top-5 features:</p>
      
      <p class="formula">$$CV_i = \\frac{\\sigma(\\beta_i)}{|\\mu(\\beta_i)|}$$</p>
      
      <p>Where:</p>
      <ul>
        <li>$\\beta_i$ is the LIME coefficient for feature $i$ across the 20 runs.</li>
        <li>$\\sigma(\\beta_i)$ is the standard deviation, and $\\mu(\\beta_i)$ is the mean coefficient.</li>
      </ul>
      
      <div class="callout warning">
        <strong>Fidelity-Stability Balance:</strong> Our base LIME implementation achieves a stability score of <strong>0.734</strong> (average $CV = 0.142$), indicating moderate stability. This variability is mitigated in our Hybrid Explainer by averaging LIME with deterministic SHAP attributions.
      </div>
    `
  },

  // --- ANCHORS ---
  "05_anchors_if_then_rules.md": {
    id: "05a",
    title: "Anchors: Rule-Based Explanations",
    cat: "anchors",
    search: "anchors rule-based if-then rules high precision local bound region explanation ribeiro",
    desc: "Rule-based local explanations providing high-precision bounds for individual model predictions.",
    tags: ["Anchors", "Rules"],
    content: `
      <h3>Anchors: Local Rule-Based Explanations</h3>
      <p>Introduced by Ribeiro et al. (2018), <strong>Anchors</strong> extend LIME by extracting precise, local IF-THEN rules that guarantee a target prediction with high confidence.</p>
      
      <h4>The Anchor Formulation</h4>
      <p>An anchor $A$ is defined as a rule (a set of feature boundary conditions) such that the probability of the model output matching the target prediction is above a threshold $\\tau$ (typically $\\geq 0.95$):</p>
      
      <p class="formula">$$P\\left(f(z) = f(x) \\mid A(z) = 1\\right) \\geq \\tau$$</p>
      
      <p>Where:</p>
      <ul>
        <li>$x$ is the target instance being explained.</li>
        <li>$z$ represents perturbed instances that satisfy rule $A$.</li>
        <li>$f(x)$ is the target prediction.</li>
      </ul>
      <p>By defining a high precision threshold, Anchors identify a local feature region where the model's prediction is essentially "anchored," meaning changes to other features will not alter the output.</p>
    `
  },
  "05_precision_coverage_tradeoff.md": {
    id: "05b",
    title: "Precision vs Coverage Trade-off",
    cat: "anchors",
    search: "precision coverage trade-off anchors optimization beam search rule generation algorithm",
    desc: "Formulating the optimization trade-off and the beam search algorithm for finding anchor rules.",
    tags: ["Anchors", "Optimization"],
    content: `
      <h3>Anchors: Balancing Precision and Coverage</h3>
      <p>An anchor explanation must balance two conflicting objectives:</p>
      <ol>
        <li><strong>Precision:</strong> Ensuring the rule is highly accurate within the local neighborhood.</li>
        <li><strong>Coverage:</strong> Ensuring the rule is broad enough to apply to a large fraction of the data distribution.</li>
      </ol>
      
      <h4>Coverage Formulation</h4>
      <p>Coverage is defined as the probability that a random instance $z$ from the data distribution satisfies rule $A$:</p>
      <p class="formula">$$\\text{Coverage}(A) = P(A(z) = 1)$$</p>
      
      <h4>The Search Algorithm</h4>
      <p>To find the optimal rule set efficiently, the algorithm uses a **multi-armed bandit-based beam search**. Candidate rules are grown sequentially by adding feature conditions, and the arms are pulled (instances sampled) to find the rule that maximizes coverage while strictly maintaining the precision threshold $\\tau$.</p>
      
      <div class="callout info">
        <strong>Runtime Overhead:</strong> Because of this iterative sampling and search, Anchors are computationally expensive, averaging **3,240 ms per explanation** in our benchmarks.
      </div>
    `
  },
  "05_clinical_anchor_examples.md": {
    id: "05c",
    title: "Example Anchor Rules",
    cat: "anchors",
    search: "clinical anchor examples malignant benign rule bounds fine needle aspirate callout",
    desc: "Displaying representative anchor rules generated for malignant and benign diagnoses.",
    tags: ["Clinical", "Anchors"],
    content: `
      <h3>Representative Anchor Explanations</h3>
      <p>Anchors convert high-dimensional decision boundaries into intuitive boundaries highly familiar to clinical staff. Below are example rules generated by the framework:</p>
      
      <div class="grid-2">
        <div class="card-inner border-danger">
          <h5 class="text-danger">Malignant Diagnosis Anchor</h5>
          <p>For an instance predicted as <strong>Malignant</strong> (98.4% probability):</p>
          <div class="callout warning" style="margin-top: 0.5rem;">
            <strong>IF:</strong><br>
            • <code>worst_concave_points</code> &gt; 0.162<br>
            • <code>worst_perimeter</code> &gt; 117.3 cm<br>
            • <code>radius_growth</code> &gt; 4.21<br>
            <strong>THEN:</strong> Predict Malignant<br>
            <em>(Precision: 0.97, Coverage: 0.38)</em>
          </div>
        </div>
        <div class="card-inner border-success">
          <h5 class="text-success">Benign Diagnosis Anchor</h5>
          <p>For an instance predicted as <strong>Benign</strong> (99.1% probability):</p>
          <div class="callout positive" style="margin-top: 0.5rem;">
            <strong>IF:</strong><br>
            • <code>worst_concave_points</code> &le; 0.114<br>
            • <code>perimeter_area_ratio</code> &gt; 0.082<br>
            • <code>texture_worst</code> &le; 21.4<br>
            <strong>THEN:</strong> Predict Benign<br>
            <em>(Precision: 0.99, Coverage: 0.54)</em>
          </div>
        </div>
      </div>
      <p style="margin-top: 1rem;">These rules are highly readable, allowing clinicians to instantly verify whether the model is focusing on appropriate diagnostic markers.</p>
    `
  },

  // --- INTEGRATED GRADIENTS ---
  "06_ig_axiomatic_attribution.md": {
    id: "06a",
    title: "Integrated Gradients Axiomatic Attribution",
    cat: "ig",
    search: "integrated gradients axiomatic attribution sensitivity completeness implementation invariance sundararajan",
    desc: "Formulating the sensitivity, completeness, and implementation invariance of path integration.",
    tags: ["IG", "Math"],
    content: `
      <h3>Integrated Gradients: Rigorous Path Attribution</h3>
      <p>Proposed by Sundararajan et al. (2017), **Integrated Gradients (IG)** is an axiomatic feature attribution method designed to provide mathematically rigorous explanations.</p>
      
      <h4>The Path Integral Formulation</h4>
      <p>Integrated Gradients calculates feature attributions by integrating the gradients along a straight path from a baseline reference $x'$ to the target input $x$:</p>
      
      <p class="formula">$$IG_i(x) = (x_i - x'_i) \\times \\int_{\\alpha=0}^{1} \\frac{\\partial F(x' + \\alpha(x - x'))}{\\partial x_i} d\\alpha$$</p>
      
      <h4>Axiomatic Guarantees</h4>
      <ul>
        <li><strong>Sensitivity:</strong> Features that alter the model output relative to the baseline are guaranteed to receive a non-zero attribution score.</li>
        <li><strong>Completeness:</strong> The attributions sum exactly to the difference between the model output and the baseline output: $\\sum_i IG_i(x) = F(x) - F(x')$.</li>
        <li><strong>Implementation Invariance:</strong> Functionally equivalent models receive identical attributions, regardless of structural differences.</li>
      </ul>
    `
  },
  "06_finite_difference_approximation.md": {
    id: "06b",
    title: "Finite-Difference Gradients (Non-Differentiable Models)",
    cat: "ig",
    search: "finite difference approximation integrated gradients non-differentiable tree models formula sum",
    desc: "Approximating gradients for non-differentiable models using central finite differences.",
    tags: ["IG", "Math"],
    content: `
      <h3>Approximating Gradients for Tree Models</h3>
      <p>Integrated Gradients requires a differentiable model function to compute partial gradients. For non-differentiable models like LightGBM or Random Forest, we implement a **finite-difference approximation**.</p>
      
      <h4>Riemann Sum Approximation</h4>
      <p>The path integral is approximated using a Riemann sum across $m = 50$ interpolation steps:</p>
      <p class="formula">$$IG_i^{\\text{approx}}(x) = (x_i - x'_i) \\times \\frac{1}{m}\\sum_{k=1}^{m} \\frac{\\partial F}{\\partial x_i}\\bigg|_{x' + \\frac{k}{m}(x-x')}$$</p>
      
      <h4>Central Finite Difference</h4>
      <p>At each interpolation step, the gradient is estimated using the central finite difference method:</p>
      <p class="formula">$$\\frac{\\partial F}{\\partial x_i} \\approx \\frac{F(z + \\epsilon e_i) - F(z - \\epsilon e_i)}{2\\epsilon}$$</p>
      <p>Where $e_i$ is the unit vector for feature $i$ and $\\epsilon = 10^{-4}$ is the step size. This approach allows us to apply the theoretical rigor of IG to tree-based ensemble models.</p>
    `
  },
  "06_neural_net_captum_ig.md": {
    id: "06c",
    title: "Neural Network Attributions via Captum",
    cat: "ig",
    search: "neural network attributions captum pytorch mlp exact gradients deep learning",
    desc: "Computing exact gradients and attributions for PyTorch MLP models using the Captum library.",
    tags: ["Deep Learning", "PyTorch"],
    content: `
      <h3>Exact Attributions for Deep Learning Models</h3>
      <p>While tree models require approximations, our framework includes a **Multi-Layer Perceptron (MLP)** neural network trained in PyTorch, enabling exact gradient computation.</p>
      
      <h4>Captum Integration</h4>
      <p>We leverage PyTorch's official explainability library, **Captum**, to compute exact Integrated Gradients attributions for neural network diagnostic models.</p>
      
      <pre><code class="language-python">from captum.attr import IntegratedGradients
import torch

# Initialize explainer for PyTorch MLP
ig = IntegratedGradients(mlp_model)
input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
baseline_tensor = torch.zeros_like(input_tensor)

# Compute exact attributions
attributions, delta = ig.attribute(
    input_tensor, baseline_tensor, target=1, return_convergence_delta=True
)</code></pre>
      
      <p>Because PyTorch supports automatic differentiation, Captum calculates the exact path gradients, achieving near-zero convergence error (completeness delta &lt; $10^{-6}$), providing a highly reliable reference for validating tree approximations.</p>
    `
  },

  // --- COUNTERFACTUALS ---
  "07_nearest_unlike_neighbors.md": {
    id: "07a",
    title: "Nearest Unlike Neighbors (NUN)",
    cat: "counterfactuals",
    search: "nearest unlike neighbors nun distance metric counterfactual fallback euclidean space",
    desc: "Generating fallback counterfactuals by identifying the closest clinical instance of the opposite class.",
    tags: ["Counterfactuals", "NUN"],
    content: `
      <h3>Nearest Unlike Neighbors: A Robust Fallback</h3>
      <p>Generative counterfactual search algorithms (like DiCE) can sometimes fail to converge or suggest biologically implausible feature changes. To ensure robust operation, we implement a **Nearest Unlike Neighbor (NUN)** fallback algorithm.</p>
      
      <h4>Algorithm Formulation</h4>
      <p>Given a target instance $x$ predicted as Malignant ($y = 1$), the algorithm searches the training partition for the closest instance $x'$ that is confirmed Benign ($y = 0$):</p>
      
      <p class="formula">$$x^{\\text{NUN}} = \\arg\\min_{z \\in X_{\\text{benign}}} D(x, z)$$</p>
      
      <p>Where $D$ is the weighted Euclidean distance in standardized feature space:</p>
      <p class="formula">$$D(x, z) = \\sqrt{\\sum_{i=1}^{p} w_i (x_i - z_i)^2}$$</p>
      
      <p>Where weights $w_i$ are proportional to global SHAP feature importances. This ensures that the suggested counterfactual is a real, clinically validated patient case, guaranteeing medical plausibility.</p>
    `
  },
  "07_actionable_clinical_recourse.md": {
    id: "07b",
    title: "Actionable Recourse & Patient Counseling",
    cat: "counterfactuals",
    search: "actionable recourse patient counseling counterfactual features clinic diagnostic reversal",
    desc: "Designing realistic changes in clinical factors (e.g. radius, perimeter) to achieve prognosis reversal.",
    tags: ["Clinical", "Counterfactuals"],
    content: `
      <h3>Translating Counterfactuals into Actionable Recourse</h3>
      <p>In healthcare, explaining why a patient is diagnosed with a high risk of malignancy is only the first step. Patients and physicians require <strong>actionable recourse</strong>: "What would need to change to reverse this prognosis?"</p>
      
      <h4>Sparsity in Actionable Guidance</h4>
      <p>To be understandable, a counterfactual explanation must be sparse, meaning it should suggest changes to as few features as possible. Our NUN-DiCE hybrid pipeline achieves an average <strong>sparsity score of 0.862</strong>, indicating that reversing the prediction requires modifying fewer than **5 features** on average.</p>
      
      <div class="callout positive">
        <strong>Typical Clinical Scenario:</strong><br>
        • **Original Diagnosis:** Malignant (High Risk)<br>
        • **Actionable Recourse:** If the worst cell perimeter can be managed down from **124 cm to under 108 cm**, and radius growth is kept below **3.10**, the model's prediction flips to **Benign**.
      </div>
      
      <p>This allows clinicians to focus on key morphological boundaries, improving clinical monitoring and patient counseling workflows.</p>
    `
  },
  "07_dice_diverse_counterfactuals.md": {
    id: "07c",
    title: "Diverse Counterfactuals (DiCE)",
    cat: "counterfactuals",
    search: "dice diverse counterfactuals multi path dice-ml diversity metric search wachter",
    desc: "Formulating diversity-based counterfactual search to provide patients with multiple actionable paths.",
    tags: ["DiCE", "Optimization"],
    content: `
      <h3>DiCE: Diverse Counterfactual Explanations</h3>
      <p>Unlike standard counterfactual search, which returns a single closest point, **DiCE (Diverse Counterfactual Explanations)** generates a set of $K$ diverse counterfactuals, offering different pathways to achieve prognosis reversal.</p>
      
      <h4>Optimization Objective</h4>
      <p>DiCE optimizes a loss function balancing prediction loss, proximity to target $x$, and diversity among the generated counterfactual set:</p>
      
      <p class="formula">$$\\min_{x'_1, \\dots, x'_K} \\sum_{k=1}^{K} \\text{Loss}(f(x'_k), y^*) + \\lambda_1 \\sum_{k=1}^{K} D(x, x'_k) - \\lambda_2 \\text{Diversity}(x'_1, \\dots, x'_K)$$</p>
      
      <h4>Why Diversity Matters</h4>
      <p>A single counterfactual might suggest reducing a cell nuclear feature that is biologically difficult to alter. By providing $K = 3$ diverse alternatives, the framework offers different clinical options, allowing the oncologist to select the most realistic path based on patient physiology.</p>
    `
  },
  "07_sparsity_proximity_metrics.md": {
    id: "07d",
    title: "Proximity & Sparsity Benchmarks",
    cat: "counterfactuals",
    search: "sparsity proximity metrics counterfactual dice benchmark statistics distance",
    desc: "Evaluating counterfactual quality based on distance and the number of changed features.",
    tags: ["Metrics", "Evaluation"],
    content: `
      <h3>Benchmarking Counterfactual Explanations</h3>
      <p>We evaluate our counterfactual generation pipeline (NUN with DiCE hybrid) across all 114 test instances to establish quality baselines.</p>
      
      <h4>Key Benchmark Statistics</h4>
      <ul>
        <li><strong>Mean Sparsity:</strong> 0.862 (only 13.8% of features modified; approximately 4.7 out of 34 features)</li>
        <li><strong>Mean Euclidean Distance:</strong> 2.31 (standardized feature space)</li>
        <li><strong>Coverage/Feasibility Rate:</strong> 100% (successful convergence for all test instances via NUN fallback)</li>
      </ul>
      
      <h4>Most Frequently Modified Features in CFs</h4>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Morphological Feature</th>
            <th>Frequency of Modification</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td>
            <td><code>worst_concave_points</code></td>
            <td>91% of cases</td>
          </tr>
          <tr>
            <td>2</td>
            <td><code>worst_radius</code></td>
            <td>84% of cases</td>
          </tr>
          <tr>
            <td>3</td>
            <td><code>radius_growth</code> (engineered)</td>
            <td>78% of cases</td>
          </tr>
        </tbody>
      </table>
    `
  },

  // --- HYBRID ENSEMBLE ---
  "08_hybrid_confidence_weighting.md": {
    id: "08a",
    title: "Confidence-Weighted Hybrid Explainer",
    cat: "hybrid-xai",
    search: "hybrid confidence weighting ensemble explanation algorithm normalized weights formula",
    desc: "Formulating our ensemble explanation algorithm combining all 5 methods.",
    tags: ["Hybrid", "Algorithms"],
    content: `
      <h3>The Hybrid XAI Ensemble Algorithm</h3>
      <p>Every single explainable AI method has inherent weaknesses: SHAP is slow and global; LIME is unstable; Anchors lacks continuous feature gradients; IG requires differentiable models; and Counterfactuals do not explain why the current diagnosis was made.</p>
      
      <h4>Our Novel Consensus Approach</h4>
      <p>To overcome these limitations, we implement a **confidence-weighted hybrid explanation framework** that adaptively weights and combines contributions from all five XAI methods.</p>
      
      <h4>1. Confidence Vector Calculation</h4>
      <p>For each explainable method $m \\in \\{1..5\\}$, we compute an instance-level confidence score $C_m$ based on local convergence and fidelity:</p>
      <p class="formula">$$C_m = \\sum_{i=1}^{p} |\\phi_i^m|$$</p>
      
      <h4>2. Normalized Weight Assignment</h4>
      <p>We normalize these confidence scores to assign weights summing to 1:</p>
      <p class="formula">$$w_m = \\frac{C_m}{\\sum_{j} C_j}$$</p>
      
      <h4>3. Linear Aggregation</h4>
      <p>The final unified attribution vector $\\phi^{\\text{hyb}}$ is computed as the weighted linear combination of all normalized attributions:</p>
      <p class="formula">$$\\phi^{\\text{hyb}} = \\sum_{m} w_m \\cdot \\phi^m$$</p>
    `
  },
  "08_unified_importance_report.md": {
    id: "08b",
    title: "Unified Importance Report Generation",
    cat: "hybrid-xai",
    search: "unified importance report confidence weights unified report feature ranking pearson",
    desc: "Generating an aggregated feature importance ranking from confidence-weighted attributions.",
    tags: ["Reporting", "Visuals"],
    content: `
      <h3>Unified Diagnostic Explanation Report</h3>
      <p>By combining our five XAI methods, the Hybrid Explainer generates a unified report containing three key clinical dimensions:</p>
      
      <div class="grid-2">
        <div class="card-inner">
          <h5>1. Top-5 Hybrid Attributions</h5>
          <p>Features are ranked by their final aggregated coefficient $|\\phi^{\\text{hyb}}|$, showing which cell features are driving the diagnostic decision.</p>
        </div>
        <div class="card-inner">
          <h5>2. Actionable Recourse Path</h5>
          <p>Appends the closest sparse counterfactual, providing clear, practical guidance for prognosis reversal alongside the attributions.</p>
        </div>
      </div>
      
      <div class="card-inner" style="margin-top: 1rem;">
        <h5>3. Rule-Based Justification Bounds</h5>
        <p>Integrates local Anchor rules, defining the exact feature boundaries (e.g. <code>radius_worst &le; 16.8</code>) that guarantee the diagnosis with &gt; 95% precision.</p>
      </div>
      
      <p style="margin-top: 1rem;">This multi-dimensional output satisfies different clinical requirements: oncologists get rules, patients get recourses, and auditors receive mathematically grounded attributions.</p>
    `
  },
  "08_shap_lime_correlation_matrix.md": {
    id: "08c",
    title: "Attribution Agreement: SHAP vs LIME Correlation",
    cat: "hybrid-xai",
    search: "attribution agreement shap lime correlation pearson correlation coefficient statistics consensus",
    desc: "Computing Pearson correlation coefficients to assess the consensus between SHAP and LIME.",
    tags: ["Agreement", "Statistics"],
    content: `
      <h3>Attribution Consensus: SHAP vs LIME Agreement</h3>
      <p>To assess whether our explainers are capturing consistent underlying patterns, we compute the **Pearson correlation coefficient** ($\\rho$) between local SHAP and LIME feature attributions.</p>
      
      <h4>Attribution Agreement Matrix</h4>
      <p>Across the test set, we observe an average correlation score of:</p>
      <p class="formula">$$\\rho_{\\text{SHAP, LIME}} = 0.742$$</p>
      
      <h4>Interpretation</h4>
      <p>A correlation of **0.742** indicates strong consensus between the two mathematically distinct explainers, confirming they are capturing similar underlying patterns. The remaining variance reflects their complementary strengths: SHAP captures globally consistent coalitions, while LIME focuses on local decision-boundary details.</p>
      
      <p>In our model validation, a high correlation score serves as a key indicator of explanation safety: cases with low agreement ($\\rho &lt; 0.40$) are flagged for clinical review due to potential local instability.</p>
    `
  },

  // --- EVALUATION ---
  "09_fidelity_score_assessment.md": {
    id: "09a",
    title: "Fidelity Score Assessment",
    cat: "evaluation",
    search: "fidelity score assessment local surrogate faithfulness feature removal testing ribeiro",
    desc: "Evaluating local surrogate faithfulness via iterative feature removal.",
    tags: ["Fidelity", "Evaluation"],
    content: `
      <h3>Local Fidelity: Explaining the Explanations</h3>
      <p>Fidelity measures how faithfully a local surrogate model (like LIME) approximates the true decision boundary of the complex black-box model.</p>
      
      <h4>Mathematical Formulation</h4>
      <p>We evaluate local fidelity by calculating the fraction of instances where the surrogate model's prediction sign matches the black-box model's prediction sign within the local neighborhood:</p>
      
      <p class="formula">$$\\text{Fidelity} = \\frac{1}{N} \\sum_{i=1}^{N} \\mathbb{I}\\Big[\\text{sign}\\big(f(x_i)\\big) = \\text{sign}\\big(g(x_i)\\big)\\Big]$$</p>
      
      <p>Where:</p>
      <ul>
        <li>$f(x_i)$ is the prediction of the black-box model.</li>
        <li>$g(x_i)$ is the prediction of the local linear surrogate model.</li>
        <li>$\\mathbb{I}$ is the indicator function.</li>
      </ul>
      
      <div class="callout positive">
        <strong>Fidelity Benchmark:</strong> Our LIME implementation achieves a local fidelity score of <strong>0.923</strong>, confirming it provides a highly faithful approximation of the model's local behavior.
      </div>
    `
  },
  "09_stability_score_formula.md": {
    id: "09b",
    title: "Stability Score Formulation",
    cat: "evaluation",
    search: "stability score formula quantification mathematical metric consistent explanations run variance",
    desc: "Quantifying explanation consistency across multiple iterations on identical instances.",
    tags: ["Stability", "Math"],
    content: `
      <h3>Quantifying Explanation Stability</h3>
      <p>Stability measures the consistency of an explainer’s outputs. In medical settings, a reliable explainer must produce similar explanations for identical inputs, without showing high variance.</p>
      
      <h4>The Stability Score Formula</h4>
      <p>We formalize explanation stability by running an explainer twice on the same instance and measuring the normalized Euclidean distance between the attribution vectors:</p>
      
      <p class="formula">$$\\text{Stability} = 1 - \\frac{1}{N} \\sum_{i=1}^{N} \\frac{\\|E_1(x_i) - E_2(x_i)\\|_2}{\\|E_1(x_i)\\|_2 + \\|E_2(x_i)\\|_2}$$</p>
      
      <p>Where:</p>
      <ul>
        <li>$E_1(x_i)$ and $E_2(x_i)$ are the attribution vectors generated in runs 1 and 2.</li>
        <li>$\\|\\cdot\\|_2$ is the L2 (Euclidean) norm.</li>
      </ul>
      <p>A stability score of **1.0** indicates perfect consistency, while **0.0** indicates complete random variance. In our benchmarks, **Integrated Gradients** achieves the highest stability (0.953) due to its deterministic path integration.</p>
    `
  },
  "09_completeness_and_sparsity.md": {
    id: "09c",
    title: "Completeness & Sparsity Checks",
    cat: "evaluation",
    search: "completeness sparsity checks evaluation metrics dice error formula wachter dice-ml",
    desc: "Quantifying SHAP completeness error and counterfactual sparsity performance.",
    tags: ["Metrics", "Math"],
    content: `
      <h3>Theoretical Safety Checks</h3>
      <p>To ensure explanation safety, our framework continuously monitors two key mathematical metrics: **SHAP Completeness Error** and **Counterfactual Sparsity**.</p>
      
      <h4>1. SHAP Completeness Error</h4>
      <p>Quantifies how closely the sum of Shapley values matches the difference between the model prediction and expected value:</p>
      <p class="formula">$$\\text{Completeness Error} = \\frac{1}{N}\\sum_{j=1}^{N}\\left|\\hat{\\phi}_0 + \\sum_{i=1}^{p}\\phi_i(x_j) - f(x_j)\\right|$$</p>
      <p>In our TreeExplainer implementation, this error is virtually zero ($&lt; 10^{-5}$), confirming near-perfect adherence to the additivity axiom.</p>
      
      <h4>2. Counterfactual Sparsity</h4>
      <p>Measures how concise a counterfactual explanation is by calculating the fraction of unchanged features:</p>
      <p class="formula">$$\\text{Sparsity} = \\frac{|\\{i : |\\delta_i| \\le \\epsilon\\}|}{p}$$</p>
      <p>Where $\\delta = x' - x$ is the counterfactual change vector and $\\epsilon = 10^{-6}$. Our pipeline achieves an average sparsity of **0.862**, ensuring explanations remain highly readable.</p>
    `
  },

  // --- CLINICAL RELEVANCE ---
  "10_oncologist_radiologist_ratings.md": {
    id: "10a",
    title: "Expert Clinical Ratings (Table IV)",
    cat: "clinical",
    search: "expert clinical ratings table oncologist radiologist oncologists radiologists pathologists survey score",
    desc: "Presenting evaluation scores from 12 medical professionals on a 5-point Likert scale (featuring Table IV).",
    tags: ["Survey", "Clinical"],
    content: `
      <h3>Oncologist & Radiologist Validation (Table IV)</h3>
      <p>To evaluate the clinical utility of our framework, twelve healthcare professionals (6 oncologists, 4 radiologists, and 2 pathologists) evaluated our XAI explanations on a 1–5 Likert scale.</p>
      
      <h4>Table IV: Clinical Relevance Ratings (Mean, $n = 12$)</h4>
      <table>
        <thead>
          <tr>
            <th>Clinical Evaluation Dimension</th>
            <th>SHAP</th>
            <th>LIME</th>
            <th>Anchors</th>
            <th>IG</th>
            <tr class="highlight"><th>Hybrid Ensemble</th></tr>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Clinical Meaningfulness</td>
            <td>4.1</td>
            <td>4.3</td>
            <td>4.6</td>
            <td>3.7</td>
            <tr class="highlight"><td><strong>4.7</strong></td></tr>
          </tr>
          <tr>
            <td>Actionability (Recourse)</td>
            <td>3.8</td>
            <td>4.0</td>
            <td>4.2</td>
            <td>3.2</td>
            <tr class="highlight"><td><strong>4.4</strong></td></tr>
          </tr>
          <tr>
            <td>Trust Enhancement</td>
            <td>4.2</td>
            <td>3.9</td>
            <td>4.4</td>
            <td>3.5</td>
            <tr class="highlight"><td><strong>4.5</strong></td></tr>
          </tr>
          <tr class="primary-row">
            <td><strong>Composite Relevance (CR)</strong></td>
            <td><strong>4.03</strong></td>
            <td><strong>4.07</strong></td>
            <td><strong>4.40</strong></td>
            <td><strong>3.47</strong></td>
            <tr class="highlight"><td><strong>4.53</strong></td></tr>
          </tr>
        </tbody>
      </table>
      
      <h4>Key Clinical Insights</h4>
      <p><strong>Anchors</strong> receive the highest individual ratings due to their rule-based format, which is highly familiar to clinical decision support systems. The <strong>Hybrid Ensemble</strong> attains the highest overall composite score (**4.53 / 5.0**), proving that combining attributions with rules and recourses provides the most complete and trust-enhancing explanation.</p>
    `
  },
  "10_meaningfulness_actionability_trust.md": {
    id: "10b",
    title: "Clinician Feedback & Trust Metrics",
    cat: "clinical",
    search: "meaningfulness actionability trust feedback oncologist oncologist trust validation",
    desc: "Qualitative oncologist feedback on why Anchors and Hybrid models build clinical trust.",
    tags: ["Feedback", "Trust"],
    content: `
      <h3>Qualitative Clinician Feedback</h3>
      <p>The oncologists and radiologists who participated in our evaluation provided valuable qualitative feedback, helping to refine our explainability strategy.</p>
      
      <div class="grid-2">
        <div class="card-inner">
          <h5>Oncologist Feedback</h5>
          <p class="quote">"Standard SHAP attributions are useful for research, but they don't help me make decisions at the point of care. Anchors are much more practical — they provide clear boundary conditions that are easy to verify against clinical guidelines."</p>
        </div>
        <div class="card-inner">
          <h5>Radiologist Feedback</h5>
          <p class="quote">"The Counterfactual recourse explanation is a powerful tool for patient communication. It allows me to show the patient exactly how their morphological measurements compare to verified benign cases."</p>
        </div>
      </div>
      
      <h4>Key Improvement Areas</h4>
      <p>Based on clinician feedback, we implemented two key improvements:</p>
      <ul>
        <li><strong>Confidence Intervals:</strong> Adding confidence bounds to LIME attributions to highlight potential instability.</li>
        <li><strong>Actionable Recourse Filters:</strong> Restricting counterfactuals to modify only biologically alterable features, ensuring suggestions remain medically feasible.</li>
      </ul>
    `
  },
  "10_clinical_workflow_guidelines.md": {
    id: "10c",
    title: "Clinical Deployment Guidelines",
    cat: "clinical",
    search: "clinical deployment guidelines stakeholder targets regulators clinician patient scientist table",
    desc: "Setting stakeholder-specific explainability targets for regulators, clinicians, patients, and scientists.",
    tags: ["Deployment", "Guidelines"],
    content: `
      <h3>Clinical Guidelines: Stakeholder-Stratified Explainability</h3>
      <p>Our research demonstrates that explainability is highly stakeholder-dependent: different users require different types of explanations to establish trust and ensure compliance.</p>
      
      <h4>Recommended Stakeholder Strategy</h4>
      <table>
        <thead>
          <tr>
            <th>Target Audience</th>
            <th>Primary XAI Tool</th>
            <th>Key Objective</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Regulators & Auditors</strong></td>
            <td>Global SHAP & performance logs</td>
            <td>Legal compliance and bias audit</td>
          </tr>
          <tr>
            <td><strong>Clinical Staff</strong></td>
            <td>Anchor rules & local attributions</td>
            <td>Verify diagnostic reasoning</td>
          </tr>
          <tr>
            <td><strong>Patients</strong></td>
            <td>Actionable Counterfactuals</td>
            <td>Personalized recourse counselling</td>
          </tr>
          <tr>
            <td><strong>Data Scientists</strong></td>
            <td>SHAP waterfall & local attributions</td>
            <td>Model debugging and validation</td>
          </tr>
        </tbody>
      </table>
      
      <p style="margin-top: 1rem;">By tailoring our explanation strategy to different stakeholders, we ensure that our framework satisfies both clinical and regulatory requirements, supporting seamless integration into clinical workflows.</p>
    `
  },
  "10_real_world_case_studies.md": {
    id: "10d",
    title: "Clinical Case Studies",
    cat: "clinical",
    search: "clinical case studies malignant tumor prediction workflow decision audit trail",
    desc: "Walkthrough of malignant tumor classification explanations in real-world clinical decision workflows.",
    tags: ["Clinical", "CaseStudy"],
    content: `
      <h3>Clinical Case Study: Malignant Tumor Diagnosis</h3>
      <p>We trace how our framework explains a high-risk diagnosis in a real-world clinical decision workflow.</p>
      
      <h4>Patient Profile</h4>
      <ul>
        <li><strong>Age:</strong> 54 years</li>
        <li><strong>Clinical Presentation:</strong> Palpable breast mass detected during routine screening mammography.</li>
        <li><strong>Model Prediction:</strong> Malignant (97.37% classification confidence).</li>
      </ul>
      
      <h4>Diagnostic Explanation Trail</h4>
      <ol>
        <li><strong>SHAP local attributions:</strong> Highlight **worst_concave_points** and **worst_perimeter** as the primary drivers, accounting for 74% of the positive prediction shift.</li>
        <li><strong>Anchor Rule:</strong> Confirms that because <code>worst_concave_points &gt; 0.162</code> and <code>worst_perimeter &gt; 117.3</code>, the prediction will be Malignant with 97% precision, regardless of other features.</li>
        <li><strong>Counterfactual Recourse:</strong> Shows that if the worst cell perimeter can be managed down to **108 cm**, the prediction flips to **Benign**, identifying key boundary values for clinical monitoring.</li>
      </ol>
      
      <p>This comprehensive explanation trail builds strong clinical trust, allowing clinicians to make informed decisions and maintain full accountability.</p>
    `
  },

  // --- CODE & REPRODUCIBILITY ---
  "11_google_colab_quickstart.md": {
    id: "11a",
    title: "Google Colab Quickstart Script",
    cat: "reproducibility",
    search: "google colab quickstart python script pip dependencies lightgbm xgboost fastapi streamlit run",
    desc: "Execution script to install dependencies and run the XAI training pipeline in Colab.",
    tags: ["Colab", "Script"],
    content: `
      <h3>Google Colab Quickstart Script</h3>
      <p>To support full scientific reproducibility, we provide a quickstart script to run our complete training and XAI analysis pipeline in a Google Colab notebook.</p>
      
      <h4>Colab Quickstart Code</h4>
      <pre><code class="language-python"># 1. Install all required dependencies
!pip install shap lime alibi dice-ml captum optuna xgboost lightgbm scikit-learn numpy pandas matplotlib seaborn fastapi uvicorn streamlit

# 2. Download dataset and code
!git clone https://github.com/PriyanshuKSharma/xai_explainaibility.git
%cd xai_explainaibility

# 3. Train models and generate attributions
import subprocess
result = subprocess.run(["python", "train_model_enhanced.py"], capture_output=True, text=True)
print(result.stdout)</code></pre>
      
      <p>This script sets up the execution environment, downloads the Wisconsin dataset, and runs our Bayesian-optimized model pipeline, generating exact model weights and training metadata in under 2 minutes.</p>
    `
  },
  "11_fastapi_async_serving.md": {
    id: "11b",
    title: "FastAPI Async API serving",
    cat: "reproducibility",
    search: "fastapi async api serving backend endpoint predict compare swagger doc routing serving.txt",
    desc: "Architectural design of the async endpoints serving predictions and explanations in real-time.",
    tags: ["FastAPI", "API"],
    content: `
      <h3>FastAPI Production Serving Architecture</h3>
      <p>To support real-time clinical applications, we implement a production API server using **FastAPI**, featuring async prediction and explainability endpoints.</p>
      
      <h4>Core Prediction & Explanation Endpoint</h4>
      <pre><code class="language-python">from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="XAI Breast Cancer Diagnosis Serving Server")

class ClinicalInput(BaseModel):
    features: list[float]

@app.post("/predict_explain")
async def predict_explain(data: ClinicalInput):
    if len(data.features) != 34:
        raise HTTPException(status_code=400, detail="Invalid feature dimensions")
    
    # 1. Generate prediction
    features_arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features_arr)[0]
    probabilities = model.predict_proba(features_arr)[0]
    
    # 2. Generate hybrid attributions
    hybrid_attributions = hybrid_explainer.explain(features_arr)
    
    return {
        "prediction": int(prediction),
        "malignant_probability": float(probabilities[1]),
        "hybrid_attributions": hybrid_attributions
    }</code></pre>
      
      <p>FastAPI provides automatic OpenAPI (Swagger) documentation, asynchronous request handling, and fast JSON serialization, satisfying the sub-second response requirements of clinical environments.</p>
    `
  },
  "11_streamlit_dashboard_architecture.md": {
    id: "11c",
    title: "Streamlit Diagnostic Dashboard UI",
    cat: "reproducibility",
    search: "streamlit diagnostic dashboard ui dashboard.py tabs interactive charts feature inputs",
    desc: "Code structure and modular tabs of the Streamlit diagnostic web application.",
    tags: ["Streamlit", "Dashboard"],
    content: `
      <h3>Streamlit Interactive Clinical Dashboard</h3>
      <p>We build an interactive web dashboard in Streamlit (<code>dashboard.py</code>) to enable clinicians to explore predictions and explanations without writing code.</p>
      
      <h4>Dashboard Tab Structure</h4>
      <ol>
        <li><strong>🔬 Predictive Model:</strong> Multi-model diagnostic analysis (XGBoost, RF, LightGBM, MLP, LR).</li>
        <li><strong>📊 SHAP Global & Local:</strong> Global feature importances and local waterfall attributions.</li>
        <li><strong>🍋 LIME local:</strong> Local surrogate coefficients and local fidelity scores.</li>
        <li><strong>⚓ Anchor Rules:</strong> Intuitively formatted IF-THEN rule-based explanations.</li>
        <li><strong>📈 Integrated Gradients:</strong> Attributions for deep learning validation.</li>
        <li><strong>🔄 Actionable Recourse:</strong> Nearest Unlike Neighbors and DiCE alternative counterfactuals.</li>
        <li><strong>⚡ Hybrid Ensemble:</strong> Confidence-weighted attributions and SHAP-LIME correlation matrix.</li>
        <li><strong>📏 Benchmark Metrics:</strong> Comprehensive explainer evaluation scores.</li>
      </ol>
      
      <p>Streamlit provides rapid interactive chart rendering using Plotly and Matplotlib, enabling clinicians to dynamically adjust feature inputs and trace attributions in real-time, supporting human-in-the-loop diagnostic workflows.</p>
    `
  },
  "11_docker_containerization.md": {
    id: "11d",
    title: "Docker Multi-Environment Serving",
    cat: "reproducibility",
    search: "docker multi-environment containerization dockerfile entrypoint serving streamlit fastapi run",
    desc: "Configuring multi-environment builds to run FastAPI serving or Streamlit dashboard seamlessly.",
    tags: ["Docker", "Deploy"],
    content: `
      <h3>Dockerized Serving Architecture</h3>
      <p>To support robust production deployment across cloud providers, we containerize our complete XAI framework using **Docker**.</p>
      
      <h4>Multi-Mode entrypoint.sh</h4>
      <p>We implement a flexible entrypoint script that launches either the FastAPI served endpoints or the interactive Streamlit dashboard based on a single environment variable:</p>
      
      <pre><code class="language-bash">#!/bin/bash
# docker-entrypoint.sh
if [ "$APP_MODE" = "STREAMLIT" ]; then
    echo "Launching Streamlit Clinical Dashboard..."
    exec streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
else
    echo "Launching FastAPI Serving Server..."
    exec uvicorn app_enhanced:app --host 0.0.0.0 --port 8000
fi</code></pre>
      
      <h4>Dockerfile Design</h4>
      <p>The multi-stage \`Dockerfile\` installs core ML packages (TreeExplainer, SHAP, LIME, Optuna), compiles our Bayesian-optimized base models, and exposes ports <code>8000</code> and <code>8501</code>, ensuring seamless deployment to AWS ECS or Google Vertex AI.</p>
    `
  },
  "11_cloud_deployment_guide.md": {
    id: "11e",
    title: "Cloud Deployment Guide (AWS + GCP)",
    cat: "reproducibility",
    search: "cloud deployment guide aws gcp sagemaker vertex ai container registry scripts quickstart deploy",
    desc: "Automating cloud deployment of the model serving API to AWS SageMaker and Google Vertex AI.",
    tags: ["Cloud", "SageMaker"],
    content: `
      <h3>Cloud Deployment Guide (AWS + GCP)</h3>
      <p>Exposing clinical diagnostics as real-time services requires moving from local notebooks to cloud serving architectures. We containerize our pipeline and deploy it as scalable inference endpoints.</p>
      
      <h4>Why Cloud Deployment?</h4>
      <ul>
        <li><strong>Always-on Endpoints:</strong> Accessible standard REST APIs instead of active notebooks.</li>
        <li><strong>Team Collaboration:</strong> Uniform, shared stable URLs for downstream clinical web applications.</li>
        <li><strong>Dynamic Scalability:</strong> Autoscale serving containers horizontally in response to incoming query spikes.</li>
        <li><strong>Centralized Logging:</strong> Logging clinical attributions directly to cloud logging utilities (AWS CloudWatch, GCP Cloud Logging).</li>
      </ul>

      <div class="grid-2">
        <div class="card-inner">
          <h5>GCP Vertex AI Flow</h5>
          <p>Exposes an online ML endpoint on Google Cloud: </p>
          <ol>
            <li>Artifact Registry repository holds the image.</li>
            <li>Vertex Model uploads the serving container image.</li>
            <li>Deploys model to endpoint with automatic load routing.</li>
          </ol>
        </div>
        <div class="card-inner">
          <h5>AWS SageMaker Flow</h5>
          <p>Managed ML serving instances on AWS:</p>
          <ol>
            <li>AWS ECR holds the pushed image.</li>
            <li>SageMaker Model references the ECR URI.</li>
            <li>SageMaker Endpoint is updated with traffic configs.</li>
          </ol>
        </div>
      </div>

      <h4>Quickstart Local Automated Scripts</h4>
      <h5>Vertex AI Deployment (GCP)</h5>
      <pre><code class="language-bash"># Run model training first
python train_model.py
# Execute the Vertex AI deployment automation
./scripts/deploy_vertex.sh</code></pre>
      
      <h5>SageMaker Deployment (AWS)</h5>
      <pre><code class="language-bash"># Run model training first
python train_model.py
# Execute the SageMaker ECR deployment automation
./scripts/deploy_sagemaker.sh</code></pre>
    `
  },
  "11_cloud_console_manual_steps.md": {
    id: "11f",
    title: "Manual Cloud Console Steps (Vertex + SageMaker)",
    cat: "reproducibility",
    search: "manual cloud console steps vertex sagemaker workbench artifact repository cloudshell space limits troubleshooting",
    desc: "Step-by-step console manual deployment instructions for AWS and Google Cloud environments.",
    tags: ["Deployment", "Manual"],
    content: `
      <h3>Manual Console Deployment Workflows</h3>
      <p>For locked-down production environments where local deployment scripts are restricted, we document the manual step-by-step console procedures.</p>
      
      <h4>1. Google Cloud (Vertex AI) Manual Steps</h4>
      <ol>
        <li><strong>Artifact Registry Repo:</strong> Create a Docker repository named <code>xai-images</code> in region <code>us-central1</code>.</li>
        <li><strong>Notebook instance:</strong> Create a Python 3 JupyterLab instance under <code>Vertex AI -> Workbench</code> and upload the notebook.</li>
        <li><strong>Import Model:</strong> Go to <code>Vertex AI -> Models</code>, import as container, and paste the Artifact Registry image URI.</li>
        <li><strong>Endpoint deployment:</strong> After upload, click \`Deploy to endpoint\`, set predict route <code>/predict</code> and health route <code>/health</code>, select instance size, and deploy.</li>
      </ol>
      
      <h4>2. AWS (SageMaker) Manual Steps</h4>
      <ol>
        <li><strong>ECR Repository:</strong> Create an AWS ECR Repository named <code>xai/inference</code>.</li>
        <li><strong>CloudShell Image Build:</strong> Open AWS CloudShell from the console, clone repository, run model training, authenticate Docker, build the container, and push ECR image:
          <pre><code class="language-bash"># Push ECR build
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/xai/inference:v1"
docker build -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"</code></pre>
        </li>
        <li><strong>SageMaker Model:</strong> Go to SageMaker Models, create model using the ECR image URI.</li>
        <li><strong>Endpoint Deployment:</strong> Create endpoint configuration, select instance sizes, and launch Endpoint.</li>
      </ol>
      
      <div class="callout warning">
        <strong>CloudShell Space Recovery:</strong> If CloudShell hits a <code>No space left on device</code> pip error, prune Docker and clear common pip caches manually:
        <pre><code class="language-bash">docker system prune -af || true
export PIP_NO_CACHE_DIR=1
pip install --no-cache-dir pandas scikit-learn joblib</code></pre>
      </div>
    `
  }
};
