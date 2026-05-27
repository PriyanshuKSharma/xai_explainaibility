import re

files = ['index.html', 'notes.html', 'procedures.html']

emoji_map = {
    # Logo
    r'<div class="logo-icon">🧬</div>': r'<div class="logo-icon">X</div>',
    
    # Navbar
    r'🩺 Dashboard': r'Dashboard',
    r'📚 Procedures': r'Procedures',
    r'📝 Research Notes': r'Research Notes',
    
    # Section Header Emojis in index.html
    r'<span class="section-icon" style="font-size: 1.1rem;">📊</span>': r'<span class="section-icon" style="font-size: 1.1rem;"></span>',
    r'<div style="font-size: 3rem; margin-bottom: 1rem;">🩺</div>': r'<div style="font-size: 3rem; margin-bottom: 1rem; color: var(--teal);">X</div>',
    r'<span class="section-icon">📚</span>': r'<span class="section-icon"></span>',
    r'<span class="section-icon">🧬</span>': r'<span class="section-icon"></span>',
    r'<span class="section-icon">🤖</span>': r'<span class="section-icon"></span>',
    r'<span class="section-icon">📊</span>': r'<span class="section-icon"></span>',
    r'<span class="section-icon">🍋</span>': r'<span class="section-icon"></span>',
    r'<span class="section-icon">⚓</span>': r'<span class="section-icon"></span>',
    r'<span class="section-icon">📈</span>': r'<span class="section-icon"></span>',
    r'<span class="section-icon">🔄</span>': r'<span class="section-icon"></span>',
    r'<span class="section-icon">⚡</span>': r'<span class="section-icon"></span>',
    r'<span class="section-icon">📏</span>': r'<span class="section-icon"></span>',
    r'<span class="section-icon">🩺</span>': r'<span class="section-icon"></span>',
    
    # Filter Pills in index.html
    r'📚 Foundation & Concepts': r'Foundation & Concepts',
    r'🧬 Data & Features': r'Data & Features',
    r'🤖 Models & Ensemble': r'Models & Ensemble',
    r'📊 SHAP Explanations': r'SHAP Explanations',
    r'🍋 LIME Surrogate': r'LIME Surrogate',
    r'⚓ Anchor Rules': r'Anchor Rules',
    r'📈 Integrated Gradients': r'Integrated Gradients',
    r'🔄 Counterfactuals': r'Counterfactuals',
    r'⚡ Hybrid Ensemble': r'Hybrid Ensemble',
    r'📏 Evaluation Metrics': r'Evaluation Metrics',
    r'🩺 Clinical Relevance': r'Clinical Relevance',
    r'💻 Code & Colab': r'Code & Colab',
}

for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        modified = content
        for pattern, replacement in emoji_map.items():
            modified = re.sub(pattern, replacement, modified)
            
        if modified != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified)
            print(f"Scrubbed emojis from {filepath}")
        else:
            print(f"No emoji updates needed for {filepath}")
            
    except FileNotFoundError:
        print(f"File {filepath} not found.")
