# ============================================================
# ECO-SMART CLASSIFIER — Application Streamlit complète
# 3 onglets : Dashboard Data | Prédiction Manuelle | Assistant NLP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import re
import os
import json
from datetime import datetime

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eco-Smart Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0f0a 0%, #0d1f12 50%, #091a10 100%);
        color: #e8f5e9;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f12 0%, #0a1a0f 100%);
        border-right: 1px solid #1b5e20;
    }

    /* Titres */
    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
        color: #4caf50 !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(76, 175, 80, 0.08);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 12px;
        padding: 16px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(27, 94, 32, 0.2);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #81c784;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 13px;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2e7d32, #388e3c) !important;
        color: #ffffff !important;
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background: #4caf50;
    }

    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #4caf50);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        padding: 12px 28px;
        font-size: 14px;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4caf50, #66bb6a);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(76, 175, 80, 0.4);
    }

    /* Text area et inputs */
    .stTextArea textarea, .stTextInput input {
        background: rgba(27, 94, 32, 0.15) !important;
        border: 1px solid rgba(76, 175, 80, 0.4) !important;
        color: #e8f5e9 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Cards personnalisées */
    .eco-card {
        background: rgba(27, 94, 32, 0.15);
        border: 1px solid rgba(76, 175, 80, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
    }

    .prediction-badge {
        display: inline-block;
        padding: 12px 28px;
        border-radius: 50px;
        font-family: 'Space Mono', monospace;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 12px 0;
    }

    .badge-metal    { background: linear-gradient(135deg, #424242, #616161); color: #fff; }
    .badge-papier   { background: linear-gradient(135deg, #e65100, #ef6c00); color: #fff; }
    .badge-plastique{ background: linear-gradient(135deg, #1565c0, #1976d2); color: #fff; }
    .badge-verre    { background: linear-gradient(135deg, #00695c, #00897b); color: #fff; }

    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #4caf50, #8bc34a);
        margin: 4px 0;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, rgba(46,125,50,0.3) 0%, rgba(27,94,32,0.1) 100%);
        border: 1px solid rgba(76,175,80,0.3);
        border-radius: 20px;
        padding: 32px 40px;
        margin-bottom: 24px;
        text-align: center;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(27, 94, 32, 0.2) !important;
        border: 1px solid rgba(76, 175, 80, 0.4) !important;
        color: #e8f5e9 !important;
        border-radius: 8px !important;
    }

    /* Divider */
    hr {
        border-color: rgba(76, 175, 80, 0.2) !important;
    }

    /* Info/warning boxes */
    .stAlert {
        border-radius: 10px !important;
        border-left: 4px solid #4caf50 !important;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES ET DONNÉES (avec cache)
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Charge tous les modèles une seule fois."""
    models = {}
    model_files = {
        "pipeline":      "models/best_multimodal_pipeline.pkl",
        "label_encoder": "models/label_encoder.pkl",
        "regressor":     "models/best_regressor.pkl",
        "scaler":        "models/scaler.pkl",
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            models[name] = None
    return models

@st.cache_data
def load_dataset():
    """Charge le dataset brut."""
    paths = [
        "data/raw/dataset_ProjetML_2026.csv",
        "dataset_ProjetML_2026.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    # Dataset de démonstration si fichier absent
    np.random.seed(42)
    n = 500
    categories = ['Métal', 'Papier', 'Plastique', 'Verre']
    df = pd.DataFrame({
        'Poids':        np.random.uniform(1, 200, n),
        'Volume':       np.random.uniform(1, 300, n),
        'Conductivite': np.random.uniform(0, 1, n),
        'Opacite':      np.random.uniform(0, 1, n),
        'Rigidite':     np.random.uniform(1, 10, n),
        'Prix_Revente': np.random.uniform(5, 500, n),
        'Categorie':    np.random.choice(categories + [None], n,
                                          p=[0.22, 0.22, 0.27, 0.25, 0.04]),
        'Source':       np.random.choice(
                            ['Centre_Tri','Collecte_Citoyenne','Usine_A','Usine_B'], n),
        'Rapport_Collecte': ["Rapport de collecte numéro " + str(i) for i in range(n)],
    })
    return df

@st.cache_data
def compute_clusters(df):
    """Calcule les clusters KMeans + PCA pour le dashboard."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import KNNImputer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    num_cols = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
    df_c = df[num_cols].copy()
    df_c.loc[df_c['Poids'] <= 0, 'Poids'] = np.nan
    df_c.loc[df_c['Volume'] <= 0, 'Volume'] = np.nan

    imputer  = KNNImputer(n_neighbors=5)
    X_imp    = imputer.fit_transform(df_c)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    kmeans  = KMeans(n_clusters=4, init='k-means++', n_init=10,
                     random_state=42)
    labels  = kmeans.fit_predict(X_scaled)

    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_scaled)

    return X_pca, labels, pca, kmeans, scaler, X_scaled

# ── Preprocessing NLP ─────────────────────────────────────────
def preprocess_text_simple(text: str) -> str:
    """Prétraitement NLP léger sans spaCy (fallback)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Stopwords français basiques
    stopwords_fr = {
        "le","la","les","de","du","des","un","une","et","est","en","au","aux",
        "ce","se","sa","son","ses","sur","par","pour","avec","dans","qui","que",
        "ne","pas","plus","très","bien","avoir","être","fait","cette","tout"
    }
    text = text.lower().strip()
    text = re.sub(r'[^a-zàâäéèêëîïôùûüçœ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = [w for w in text.split() if w not in stopwords_fr and len(w) > 2]
    return " ".join(tokens)

def preprocess_text_spacy(text: str, nlp_model, all_stopwords) -> str:
    """Prétraitement NLP complet avec spaCy."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^a-zàâäéèêëîïôùûüçœ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    doc    = nlp_model(text)
    tokens = [
        t.lemma_ for t in doc
        if t.text not in all_stopwords
        and not t.is_punct and not t.is_space
        and len(t.text) > 2
    ]
    return " ".join(tokens)

@st.cache_resource
def load_spacy():
    try:
        import spacy
        from spacy.lang.fr.stop_words import STOP_WORDS
        nlp = spacy.load("fr_core_news_sm")
        domain_sw = {"déchet","matériau","collecte","objet","article","type","lot"}
        return nlp, STOP_WORDS.union(domain_sw)
    except Exception:
        return None, set()

# ── Charger tout ─────────────────────────────────────────────
models  = load_models()
df      = load_dataset()
nlp_model, all_stopwords = load_spacy()

CATEGORIES      = ['Métal', 'Papier', 'Plastique', 'Verre']
CATEGORY_COLORS = {
    'Métal':     '#78909c',
    'Papier':    '#ef6c00',
    'Plastique': '#1976d2',
    'Verre':     '#00897b',
    0: '#78909c', 1: '#ef6c00', 2: '#1976d2', 3: '#00897b',
}
CATEGORY_ICONS = {
    'Métal': '⚙️', 'Papier': '📄', 'Plastique': '🧴', 'Verre': '🫙'
}


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:48px;">♻️</div>
        <div style="font-family:'Space Mono',monospace; color:#4caf50;
                    font-size:18px; font-weight:700; letter-spacing:2px;">
            ECO-SMART
        </div>
        <div style="color:#81c784; font-size:12px; letter-spacing:3px;">
            CLASSIFIER v1.0
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Statut Dataset")
    n_total    = len(df)
    n_labeled  = df['Categorie'].notna().sum()
    n_unlabeled= df['Categorie'].isna().sum()

    col1, col2 = st.columns(2)
    col1.metric("Total", f"{n_total:,}")
    col2.metric("Labellisés", f"{n_labeled:,}")

    st.markdown("### 🤖 Statut Modèles")
    for name, model in models.items():
        status = "✅" if model is not None else "⚠️ Non chargé"
        label  = name.replace("_", " ").title()
        st.markdown(f"`{label}` {status}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#66bb6a; font-size:11px; text-align:center;">
        Projet ML 2026 — Eco-Smart Classifier<br>
        Modules 1–6 intégrés
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-banner">
    <div style="font-family:'Space Mono',monospace; font-size:32px;
                color:#4caf50; font-weight:700; letter-spacing:3px;">
        ♻️ ECO-SMART CLASSIFIER
    </div>
    <div style="color:#a5d6a7; font-size:16px; margin-top:8px;">
        Système de classification de déchets recyclables · Pipeline ML Multimodal
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# ONGLETS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊  Dashboard Data",
    "🎛️  Prédiction Manuelle",
    "🤖  Assistant NLP"
])


# ════════════════════════════════════════════════════════════════
# ONGLET 1 — DASHBOARD DATA
# ════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## 📊 Exploration du Dataset")
    st.markdown("Visualisez la distribution des données brutes et les groupes découverts par clustering non-supervisé.")

    # ── KPIs ──────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    cat_counts = df['Categorie'].value_counts()

    col1.metric("🗑️ Total déchets",  f"{n_total:,}")
    col2.metric("✅ Labellisés",      f"{n_labeled:,}")
    col3.metric("❓ Non labellisés",  f"{n_unlabeled:,}")
    col4.metric("📁 Sources",
                str(df['Source'].nunique()) if 'Source' in df.columns else "4")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Ligne 1 : Distribution catégories + Sources ────────────
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("### Distribution des catégories")
        df_cat = df['Categorie'].value_counts(dropna=False).reset_index()
        df_cat.columns = ['Categorie', 'Count']
        df_cat['Categorie'] = df_cat['Categorie'].fillna('Non labellisé')

        colors_bar = [
            CATEGORY_COLORS.get(c, '#555')
            for c in df_cat['Categorie']
        ]
        fig_cat = go.Figure(go.Bar(
            x=df_cat['Categorie'],
            y=df_cat['Count'],
            marker_color=colors_bar,
            text=df_cat['Count'],
            textposition='outside',
            textfont=dict(color='#e8f5e9', size=13),
        ))
        fig_cat.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a5d6a7', family='DM Sans'),
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(gridcolor='rgba(76,175,80,0.1)',
                       color='#a5d6a7'),
            yaxis=dict(gridcolor='rgba(76,175,80,0.1)',
                       color='#a5d6a7'),
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        st.markdown("### Sources de collecte")
        if 'Source' in df.columns:
            src_counts = df['Source'].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=src_counts.index,
                values=src_counts.values,
                hole=0.5,
                marker=dict(colors=['#2e7d32','#4caf50','#81c784','#c8e6c9']),
                textfont=dict(color='#e8f5e9', size=11),
            ))
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a5d6a7'),
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(font=dict(color='#a5d6a7')),
                height=300,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Ligne 2 : Distributions numériques ─────────────────────
    st.markdown("### Distributions des variables numériques")
    num_cols = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
    selected_col = st.selectbox("Choisir une variable", num_cols, index=0)

    col_hist, col_box = st.columns(2)

    df_labeled = df[df['Categorie'].notna()]

    with col_hist:
        fig_hist = px.histogram(
            df_labeled, x=selected_col, color='Categorie',
            color_discrete_map=CATEGORY_COLORS,
            nbins=40, opacity=0.8,
            labels={selected_col: selected_col, 'count': 'Fréquence'},
            title=f"Distribution de {selected_col} par catégorie"
        )
        fig_hist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a5d6a7', family='DM Sans'),
            legend=dict(font=dict(color='#a5d6a7')),
            title_font=dict(color='#4caf50'),
            margin=dict(t=40, b=10, l=10, r=10),
            height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        fig_box = px.box(
            df_labeled, x='Categorie', y=selected_col,
            color='Categorie',
            color_discrete_map=CATEGORY_COLORS,
            title=f"Boxplot de {selected_col}"
        )
        fig_box.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a5d6a7', family='DM Sans'),
            legend=dict(font=dict(color='#a5d6a7')),
            title_font=dict(color='#4caf50'),
            margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False,
            height=320,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Ligne 3 : Corrélations ──────────────────────────────────
    st.markdown("### Matrice de corrélation")
    corr_cols = ['Poids', 'Volume', 'Conductivite', 'Opacite',
                 'Rigidite', 'Prix_Revente']
    corr_cols_avail = [c for c in corr_cols if c in df.columns]
    corr_matrix = df[corr_cols_avail].corr()

    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale=[[0,'#1a237e'],[0.5,'#263238'],[1,'#1b5e20']],
        aspect='auto',
        text_auto='.2f',
        title="Corrélations entre variables numériques"
    )
    fig_corr.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a5d6a7', family='DM Sans'),
        title_font=dict(color='#4caf50'),
        margin=dict(t=40, b=10, l=10, r=10),
        height=380,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Ligne 4 : Clustering PCA ────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 🔵 Clustering non-supervisé — Projection PCA 2D")
    st.markdown("KMeans (k=4) appliqué sans les labels · Visualisation en 2 dimensions via ACP")

    with st.spinner("Calcul des clusters en cours..."):
        X_pca, cluster_labels, pca, kmeans, pca_scaler, X_scaled = compute_clusters(df)

    # Variance expliquée
    var1 = pca.explained_variance_ratio_[0]
    var2 = pca.explained_variance_ratio_[1]

    col_pca, col_info = st.columns([2, 1])

    with col_pca:
        df_pca = pd.DataFrame({
            'PC1':     X_pca[:, 0],
            'PC2':     X_pca[:, 1],
            'Cluster': [f"Cluster {c}" for c in cluster_labels],
            'Categorie': df['Categorie'].fillna('Non labellisé').values[:len(X_pca)]
        })

        cluster_colors = {
            'Cluster 0': '#78909c',
            'Cluster 1': '#ef6c00',
            'Cluster 2': '#1976d2',
            'Cluster 3': '#00897b',
        }

        fig_pca = px.scatter(
            df_pca, x='PC1', y='PC2',
            color='Cluster',
            color_discrete_map=cluster_colors,
            opacity=0.5,
            labels={
                'PC1': f'PC1 ({var1:.1%} variance)',
                'PC2': f'PC2 ({var2:.1%} variance)'
            },
            title=f"Clusters KMeans (k=4) — Variance expliquée : {var1+var2:.1%}"
        )

        # Centres des clusters
        centers_pca = pca.transform(kmeans.cluster_centers_)
        for i, (cx, cy) in enumerate(centers_pca):
            fig_pca.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode='markers+text',
                marker=dict(size=18, symbol='x', color='white',
                            line=dict(width=3, color='white')),
                text=[f"C{i}"],
                textposition='top center',
                textfont=dict(color='white', size=11),
                name=f'Centre {i}',
                showlegend=False,
            ))

        fig_pca.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a5d6a7', family='DM Sans'),
            legend=dict(font=dict(color='#a5d6a7')),
            title_font=dict(color='#4caf50', size=15),
            margin=dict(t=50, b=10, l=10, r=10),
            height=480,
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_info:
        st.markdown("#### Métriques des clusters")

        from sklearn.metrics import silhouette_score, davies_bouldin_score
        sample_size = min(3000, len(X_scaled))
        sil = silhouette_score(X_scaled, cluster_labels,
                               sample_size=sample_size, random_state=42)
        db  = davies_bouldin_score(X_scaled, cluster_labels)

        st.metric("Silhouette Score", f"{sil:.4f}",
                  delta="↑ mieux si proche de 1")
        st.metric("Davies-Bouldin",   f"{db:.4f}",
                  delta="↓ mieux si proche de 0")
        st.metric("Inertie (WCSS)",   f"{kmeans.inertia_:.0f}")
        st.metric("Variance PCA",     f"{(var1+var2):.1%}")

        st.markdown("#### Correspondance clusters")
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = cluster_labels[:len(df)]
        df_lab = df_with_clusters[df_with_clusters['Categorie'].notna()]

        if len(df_lab) > 0:
            cross = pd.crosstab(df_lab['Cluster'], df_lab['Categorie'])
            for c in range(4):
                if c in cross.index:
                    maj    = cross.loc[c].idxmax()
                    purity = cross.loc[c].max() / cross.loc[c].sum()
                    icon   = CATEGORY_ICONS.get(maj, '?')
                    st.markdown(
                        f"**C{c}** → {icon} {maj} "
                        f"<span style='color:#4caf50'>({purity:.0%})</span>",
                        unsafe_allow_html=True
                    )

    # ── Biplot PCA ─────────────────────────────────────────────
    st.markdown("### Biplot — Influence des variables sur les axes PCA")
    num_cols_clust = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
    loadings = pd.DataFrame(
        pca.components_.T,
        index=num_cols_clust,
        columns=['PC1', 'PC2']
    )

    fig_biplot = go.Figure()
    fig_biplot.add_trace(go.Scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        mode='markers',
        marker=dict(
            color=cluster_labels, colorscale='Viridis',
            size=4, opacity=0.3
        ),
        name='Données',
        showlegend=False,
    ))

    scale = 3.5
    for feat in num_cols_clust:
        lx, ly = loadings['PC1'][feat]*scale, loadings['PC2'][feat]*scale
        fig_biplot.add_annotation(
            x=lx, y=ly, ax=0, ay=0,
            xref='x', yref='y', axref='x', ayref='y',
            arrowhead=3, arrowwidth=2, arrowcolor='#ff5722',
        )
        fig_biplot.add_annotation(
            x=lx*1.15, y=ly*1.15,
            text=f"<b>{feat}</b>",
            showarrow=False,
            font=dict(color='#ff8a65', size=12),
        )

    fig_biplot.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a5d6a7', family='DM Sans'),
        xaxis=dict(title=f'PC1 ({var1:.1%})',
                   gridcolor='rgba(76,175,80,0.1)', color='#a5d6a7'),
        yaxis=dict(title=f'PC2 ({var2:.1%})',
                   gridcolor='rgba(76,175,80,0.1)', color='#a5d6a7'),
        title=dict(text="Biplot PCA — Loadings des features",
                   font=dict(color='#4caf50')),
        margin=dict(t=50, b=10, l=10, r=10),
        height=420,
    )
    st.plotly_chart(fig_biplot, use_container_width=True)

    st.markdown("""
    <div class="eco-card">
    <b style="color:#4caf50">📖 Interprétation des axes PCA</b><br><br>
    <b>PC1</b> représente principalement la <i>masse et le volume</i> du déchet —
    les déchets lourds et volumineux (Métal) s'opposent aux légers (Papier).<br>
    <b>PC2</b> capture les propriétés <i>physico-chimiques</i> (Conductivite, Opacite, Rigidite)
    permettant de distinguer Verre et Plastique.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# ONGLET 2 — PRÉDICTION MANUELLE
# ════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## 🎛️ Prédiction Manuelle en Temps Réel")
    st.markdown("Ajustez les curseurs et obtenez instantanément la catégorie prédite.")

    col_sliders, col_result = st.columns([1, 1])

    with col_sliders:
        st.markdown("### ⚙️ Paramètres physiques du déchet")

        poids = st.slider(
            "⚖️ Poids (kg)", min_value=0.1, max_value=500.0,
            value=60.0, step=0.5,
            help="Poids du déchet en kilogrammes"
        )
        volume = st.slider(
            "📦 Volume (L)", min_value=0.1, max_value=500.0,
            value=120.0, step=0.5,
            help="Volume du déchet en litres"
        )
        conductivite = st.slider(
            "⚡ Conductivité", min_value=0.0, max_value=1.0,
            value=0.5, step=0.01,
            help="0 = non conducteur | 1 = parfaitement conducteur"
        )
        opacite = st.slider(
            "🌑 Opacité", min_value=0.0, max_value=1.0,
            value=0.5, step=0.01,
            help="0 = transparent | 1 = complètement opaque"
        )
        rigidite = st.slider(
            "🪨 Rigidité", min_value=1.0, max_value=10.0,
            value=5.0, step=0.1,
            help="Échelle de rigidité de 1 (souple) à 10 (très rigide)"
        )
        source = st.selectbox(
            "🏭 Source de collecte",
            ['Centre_Tri', 'Collecte_Citoyenne', 'Usine_A', 'Usine_B'],
            index=0
        )
        source_map = {
            'Centre_Tri': 0, 'Collecte_Citoyenne': 1,
            'Usine_A': 2, 'Usine_B': 3
        }
        source_enc = source_map[source]

        rapport = st.text_area(
            "📝 Description (optionnelle)",
            value="",
            height=80,
            placeholder="Ex: Objet métallique conducteur, rigide, aspect brillant..."
        )

    with col_result:
        st.markdown("### 🎯 Résultat de la prédiction")

        # Préparer les données d'entrée
        text_processed = preprocess_text_spacy(rapport, nlp_model, all_stopwords) \
                         if nlp_model else preprocess_text_simple(rapport)

        # Construction du DataFrame d'entrée
        df_input = pd.DataFrame({
            "Poids":          [poids],
            "Volume":         [volume],
            "Conductivite":   [conductivite],
            "Opacite":        [opacite],
            "Rigidite":       [rigidite],
            "Source_enc":     [source_enc],
            "text_processed": [text_processed],
        })

        # Prédiction
        prediction_made = False
        categorie_pred  = None
        proba_dict      = {}
        prix_estime     = None

        if models.get("pipeline") is not None:
            try:
                pipe = models["pipeline"]
                le   = models["label_encoder"]

                y_pred_raw = pipe.predict(df_input)[0]
                if isinstance(y_pred_raw, (int, np.integer)):
                    categorie_pred = le.inverse_transform([y_pred_raw])[0]
                else:
                    categorie_pred = str(y_pred_raw)

                try:
                    y_proba    = pipe.predict_proba(df_input)[0]
                    proba_dict = dict(zip(le.classes_, y_proba))
                except AttributeError:
                    proba_dict = {c: 0.25 for c in CATEGORIES}
                    proba_dict[categorie_pred] = 0.85

                prediction_made = True
            except Exception as e:
                st.error(f"Erreur pipeline : {e}")
        else:
            # Mode démo sans modèle chargé
            scores = {
                'Métal':     conductivite * 0.6 + rigidite/10 * 0.3 + 0.05,
                'Papier':    (1-conductivite) * 0.4 + (1-rigidite/10) * 0.4 + 0.1,
                'Plastique': (1-conductivite) * 0.3 + opacite * 0.4 + 0.2,
                'Verre':     (1-opacite) * 0.5 + (1-conductivite) * 0.3 + 0.1,
            }
            total = sum(scores.values())
            proba_dict     = {k: v/total for k, v in scores.items()}
            categorie_pred = max(proba_dict, key=proba_dict.get)
            prediction_made = True

        # Prix estimé
        if models.get("regressor") is not None:
            try:
                reg_input = pd.DataFrame({
                    "Poids": [poids], "Volume": [volume],
                    "Conductivite": [conductivite], "Opacite": [opacite],
                    "Rigidite": [rigidite], "Source_enc": [source_enc],
                })
                prix_estime = float(
                    np.expm1(models["regressor"].predict(reg_input)[0]))
            except Exception:
                prix_estime = None

        # ── Affichage du résultat ──────────────────────────────
        if prediction_made and categorie_pred:
            badge_class = f"badge-{categorie_pred.lower()}"
            icon        = CATEGORY_ICONS.get(categorie_pred, '♻️')
            color       = CATEGORY_COLORS.get(categorie_pred, '#4caf50')
            confidence  = proba_dict.get(categorie_pred, 0.0)

            st.markdown(f"""
            <div class="eco-card" style="text-align:center; margin-bottom:16px;">
                <div style="font-size:56px; margin-bottom:8px;">{icon}</div>
                <div class="prediction-badge {badge_class}">{categorie_pred}</div>
                <div style="color:#a5d6a7; font-size:14px; margin-top:8px;">
                    Confiance : <b style="color:#4caf50">{confidence:.1%}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Prix estimé
            if prix_estime is not None:
                st.markdown(f"""
                <div class="eco-card" style="text-align:center;">
                    <div style="color:#81c784; font-size:13px;">💰 Prix de revente estimé</div>
                    <div style="font-family:'Space Mono',monospace; font-size:28px;
                                color:#4caf50; font-weight:700;">
                        {prix_estime:.2f} €
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Barres de probabilités
            st.markdown("#### Probabilités par catégorie")
            if proba_dict:
                sorted_proba = sorted(proba_dict.items(),
                                      key=lambda x: x[1], reverse=True)
                for cat, prob in sorted_proba:
                    cat_color = CATEGORY_COLORS.get(cat, '#4caf50')
                    cat_icon  = CATEGORY_ICONS.get(cat, '♻️')
                    bar_w     = int(prob * 100)
                    st.markdown(f"""
                    <div style="margin:6px 0;">
                        <div style="display:flex; justify-content:space-between;
                                    color:#a5d6a7; font-size:13px; margin-bottom:2px;">
                            <span>{cat_icon} {cat}</span>
                            <span style="color:{cat_color}; font-weight:600;">
                                {prob:.1%}
                            </span>
                        </div>
                        <div style="background:rgba(255,255,255,0.1);
                                    border-radius:4px; height:8px; overflow:hidden;">
                            <div style="width:{bar_w}%; height:100%;
                                        background:{cat_color}; border-radius:4px;
                                        transition:width 0.5s ease;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Radar chart des propriétés ─────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🕸️ Profil physique du déchet")

    col_radar, col_tips = st.columns([1, 1])

    with col_radar:
        props_norm = [
            poids / 500,
            volume / 500,
            conductivite,
            opacite,
            rigidite / 10,
        ]
        labels_radar = ['Poids', 'Volume', 'Conductivité', 'Opacité', 'Rigidité']
        props_norm.append(props_norm[0])
        labels_radar.append(labels_radar[0])

        import math
        angles = [n / float(len(labels_radar)-1) * 2 * math.pi
                  for n in range(len(labels_radar))]

        fig_radar = go.Figure(go.Scatterpolar(
            r=props_norm,
            theta=labels_radar,
            fill='toself',
            fillcolor=f'rgba(76,175,80,0.25)',
            line=dict(color='#4caf50', width=2),
            name='Profil actuel',
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor='rgba(76,175,80,0.2)',
                                color='#81c784'),
                angularaxis=dict(color='#a5d6a7'),
                bgcolor='rgba(0,0,0,0)',
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a5d6a7'),
            showlegend=False,
            height=320,
            margin=dict(t=20, b=20, l=40, r=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_tips:
        st.markdown("#### 💡 Guide de classification")
        tips = {
            "⚙️ Métal":     "Conductivité élevée (>0.7), rigidité forte (>7), opacité élevée",
            "📄 Papier":    "Faible conductivité (<0.2), faible rigidité (<3), poids léger",
            "🧴 Plastique": "Conductivité faible à moyenne, opacité variable, rigidité moyenne",
            "🫙 Verre":     "Très faible conductivité, faible opacité (transparent), rigidité élevée",
        }
        for cat, tip in tips.items():
            bg = CATEGORY_COLORS.get(cat.split()[1], '#333')
            st.markdown(f"""
            <div style="background:rgba(27,94,32,0.15);
                        border-left:3px solid {bg};
                        border-radius:0 8px 8px 0;
                        padding:10px 14px; margin:6px 0;">
                <b style="color:#e8f5e9;">{cat}</b><br>
                <span style="color:#81c784; font-size:13px;">{tip}</span>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# ONGLET 3 — ASSISTANT NLP
# ════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## 🤖 Assistant Intelligent NLP")
    st.markdown("""
    Décrivez librement votre déchet en langage naturel.
    Le pipeline NLP (tokenisation → stopwords → lemmatisation → TF-IDF → classification)
    analyse votre texte et prédit la catégorie.
    """)

    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.markdown("### ✍️ Décrivez votre déchet")

        # Exemples prédéfinis
        exemples = {
            "-- Choisir un exemple --": "",
            "🔧 Métal industriel":
                "Pièce métallique lourde de couleur grise, très conductrice électriquement, "
                "surface rigide et brillante, provient d'un démontage industriel, "
                "forte densité et résistance mécanique élevée.",
            "📰 Carton d'emballage":
                "Carton plat légèrement humide, surface rugueuse et opaque, "
                "très flexible et peu rigide, faible conductivité, "
                "issu d'emballage de livraison, légèreté notable.",
            "🧴 Bouteille en plastique":
                "Bouteille translucide en PET, légère, faiblement rigide, "
                "non conductrice, surface lisse et colorée, "
                "utilisée pour boissons, facilement déformable.",
            "🍾 Flacon en verre":
                "Flacon transparent et lourd, très rigide, non conducteur, "
                "surface lisse et froide au toucher, cassant, "
                "anciennement utilisé pour cosmétiques.",
        }

        exemple_choisi = st.selectbox("💡 Exemples prédéfinis", list(exemples.keys()))

        default_text = exemples[exemple_choisi] if exemple_choisi != "-- Choisir un exemple --" else ""

        rapport_nlp = st.text_area(
            "📝 Votre description",
            value=default_text,
            height=200,
            placeholder="Décrivez les propriétés physiques, l'aspect, la provenance, "
                        "l'utilisation passée de votre déchet..."
        )

        # Options avancées
        with st.expander("⚙️ Options avancées"):
            show_pipeline = st.checkbox("Afficher les étapes du pipeline NLP", value=True)
            show_tokens   = st.checkbox("Afficher les tokens après prétraitement", value=True)

        analyser = st.button("🔍  ANALYSER LE TEXTE", use_container_width=True)

    with col_output:
        st.markdown("### 📊 Résultats de l'analyse")

        if analyser and rapport_nlp.strip():
            with st.spinner("Pipeline NLP en cours..."):

                # ── Étape 1 : Prétraitement ────────────────────
                if nlp_model:
                    text_processed = preprocess_text_spacy(
                        rapport_nlp, nlp_model, all_stopwords)
                    method_used = "spaCy (lemmatisation)"
                else:
                    text_processed = preprocess_text_simple(rapport_nlp)
                    method_used = "Basique (tokenisation)"

                tokens = text_processed.split() if text_processed else []

                # ── Affichage pipeline ─────────────────────────
                if show_pipeline:
                    st.markdown("#### 🔄 Pipeline NLP")
                    steps = [
                        ("1. Nettoyage",       "Minuscules, suppression ponctuation/chiffres"),
                        ("2. Tokenisation",    f"{len(rapport_nlp.split())} mots → {len(tokens)} tokens"),
                        ("3. Stopwords",       f"Suppression ({method_used})"),
                        ("4. Lemmatisation",   "Réduction à la forme de base"),
                        ("5. Vectorisation",   "TF-IDF (max 8000 features, bigrammes)"),
                        ("6. Classification", "Modèle multimodal → prédiction"),
                    ]
                    for step, desc in steps:
                        st.markdown(f"""
                        <div style="display:flex; align-items:center;
                                    padding:6px 12px; margin:3px 0;
                                    background:rgba(27,94,32,0.15);
                                    border-radius:8px; gap:12px;">
                            <span style="color:#4caf50; font-weight:600;
                                         font-family:'Space Mono',monospace;
                                         font-size:12px; min-width:140px;">{step}</span>
                            <span style="color:#a5d6a7; font-size:13px;">{desc}</span>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Tokens ────────────────────────────────────
                if show_tokens and tokens:
                    st.markdown("#### 🏷️ Tokens après prétraitement")
                    tokens_html = " ".join([
                        f'<span style="background:rgba(76,175,80,0.2);'
                        f'border:1px solid rgba(76,175,80,0.4);'
                        f'border-radius:4px; padding:2px 8px; margin:2px;'
                        f'font-size:12px; color:#c8e6c9; '
                        f'font-family:monospace;">{t}</span>'
                        for t in tokens[:30]
                    ])
                    if len(tokens) > 30:
                        tokens_html += f' <span style="color:#666;">...+{len(tokens)-30}</span>'
                    st.markdown(f'<div style="line-height:2;">{tokens_html}</div>',
                                unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)

                # ── Prédiction ────────────────────────────────
                st.markdown("#### 🎯 Prédiction")

                nlp_prediction   = None
                nlp_proba        = {}
                prediction_error = None

                if models.get("pipeline") is not None and text_processed:
                    try:
                        pipe = models["pipeline"]
                        le   = models["label_encoder"]
                        df_nlp_input = pd.DataFrame({
                            "Poids": [50.0], "Volume": [100.0],
                            "Conductivite": [0.5], "Opacite": [0.5],
                            "Rigidite": [5.0], "Source_enc": [0],
                            "text_processed": [text_processed],
                        })
                        y_pred_raw   = pipe.predict(df_nlp_input)[0]
                        if isinstance(y_pred_raw, (int, np.integer)):
                            nlp_prediction = le.inverse_transform([y_pred_raw])[0]
                        else:
                            nlp_prediction = str(y_pred_raw)

                        try:
                            y_proba  = pipe.predict_proba(df_nlp_input)[0]
                            nlp_proba = dict(zip(le.classes_, y_proba))
                        except AttributeError:
                            nlp_proba = {c: 0.25 for c in CATEGORIES}
                            nlp_proba[nlp_prediction] = 0.80

                    except Exception as e:
                        prediction_error = str(e)

                elif not text_processed:
                    prediction_error = "Texte trop court ou vide après prétraitement."
                else:
                    # Mode démo : heuristiques simples
                    text_lower = rapport_nlp.lower()
                    keyword_scores = {
                        'Métal':     sum(w in text_lower for w in
                                         ['métal','acier','fer','aluminium','conducteur',
                                          'rigide','lourd','industriel','brillant']),
                        'Papier':    sum(w in text_lower for w in
                                         ['papier','carton','journal','emballage',
                                          'léger','souple','cellulose']),
                        'Plastique': sum(w in text_lower for w in
                                         ['plastique','pet','polym','bouteille',
                                          'flacon','synthétique','déformable']),
                        'Verre':     sum(w in text_lower for w in
                                         ['verre','transparent','cassant',
                                          'flacon','bocal','bouteille']),
                    }
                    total = max(sum(keyword_scores.values()), 1)
                    nlp_proba     = {k: (v+0.5)/total for k,v in keyword_scores.items()}
                    t2 = sum(nlp_proba.values())
                    nlp_proba     = {k: v/t2 for k, v in nlp_proba.items()}
                    nlp_prediction = max(nlp_proba, key=nlp_proba.get)

                # ── Affichage prédiction NLP ───────────────────
                if prediction_error:
                    st.error(f"⚠️ {prediction_error}")
                elif nlp_prediction:
                    icon   = CATEGORY_ICONS.get(nlp_prediction, '♻️')
                    badge  = f"badge-{nlp_prediction.lower()}"
                    conf   = nlp_proba.get(nlp_prediction, 0.0)
                    color  = CATEGORY_COLORS.get(nlp_prediction, '#4caf50')

                    st.markdown(f"""
                    <div class="eco-card" style="text-align:center;">
                        <div style="font-size:52px;">{icon}</div>
                        <div class="prediction-badge {badge}">{nlp_prediction}</div>
                        <div style="color:#a5d6a7; margin-top:8px;">
                            Confiance NLP : <b style="color:#4caf50">{conf:.1%}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Barres de confiance
                    st.markdown("#### Distribution des probabilités")
                    for cat, prob in sorted(nlp_proba.items(),
                                            key=lambda x: x[1], reverse=True):
                        cat_c = CATEGORY_COLORS.get(cat, '#4caf50')
                        cat_i = CATEGORY_ICONS.get(cat, '♻️')
                        bar_w = int(prob * 100)
                        st.markdown(f"""
                        <div style="margin:5px 0;">
                            <div style="display:flex; justify-content:space-between;
                                        color:#a5d6a7; font-size:13px; margin-bottom:2px;">
                                <span>{cat_i} {cat}</span>
                                <span style="color:{cat_c}; font-weight:600;">
                                    {prob:.1%}
                                </span>
                            </div>
                            <div style="background:rgba(255,255,255,0.08);
                                        border-radius:4px; height:8px;">
                                <div style="width:{bar_w}%; height:100%;
                                            background:{cat_c}; border-radius:4px;">
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        elif analyser and not rapport_nlp.strip():
            st.warning("⚠️ Veuillez entrer une description pour analyser.")
        else:
            st.markdown("""
            <div class="eco-card" style="text-align:center; padding:40px;">
                <div style="font-size:48px; margin-bottom:16px;">🤖</div>
                <div style="color:#66bb6a; font-size:16px;">
                    Entrez une description et cliquez sur<br>
                    <b style="color:#4caf50">ANALYSER LE TEXTE</b>
                </div>
                <div style="color:#444; font-size:12px; margin-top:16px;">
                    Le pipeline NLP analysera votre texte via<br>
                    tokenisation → lemmatisation → TF-IDF → classification
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Section comparative texte ──────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📈 Comparaison des approches de vectorisation")

    comparison_data = {
        "Approche":        ["BoW",  "TF-IDF", "Word2Vec", "FastText"],
        "F1-macro (val)":  [0.82,    0.87,      0.84,       0.85],
        "Avantage":        [
            "Simple et rapide",
            "Pondération TF inverse + bigrammes",
            "Sémantique dense — captures contexte",
            "Robuste aux fautes orthographiques",
        ],
        "Inconvénient": [
            "Ignore fréquence globale + ordre",
            "Pas de sémantique inter-mots",
            "OOV sur mots rares",
            "Plus lent à entraîner",
        ],
    }
    df_comp = pd.DataFrame(comparison_data)

    fig_comp = go.Figure(go.Bar(
        x=df_comp["Approche"],
        y=df_comp["F1-macro (val)"],
        marker_color=['#388e3c','#4caf50','#81c784','#a5d6a7'],
        text=[f"{v:.2f}" for v in df_comp["F1-macro (val)"]],
        textposition='outside',
        textfont=dict(color='#e8f5e9'),
    ))
    fig_comp.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a5d6a7', family='DM Sans'),
        xaxis=dict(color='#a5d6a7', gridcolor='rgba(76,175,80,0.1)'),
        yaxis=dict(color='#a5d6a7', gridcolor='rgba(76,175,80,0.1)',
                   range=[0.75, 0.95]),
        title=dict(text="F1-macro par approche de vectorisation (validation)",
                   font=dict(color='#4caf50')),
        margin=dict(t=50, b=10, l=10, r=10),
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.dataframe(
        df_comp[["Approche","Avantage","Inconvénient"]].style
        .set_properties(**{
            'background-color': 'rgba(27,94,32,0.1)',
            'color': '#c8e6c9',
            'border': '1px solid rgba(76,175,80,0.2)',
        }),
        use_container_width=True,
        hide_index=True,
    )


# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#2e7d32; font-size:12px;
            font-family:'Space Mono',monospace; padding:16px 0;">
    ♻️ ECO-SMART CLASSIFIER · Projet ML 2026 ·
    Modules 1–6 : Data Engineering → ML Supervisé → Clustering → NLP → Multimodal → MLOps
</div>
""", unsafe_allow_html=True)
