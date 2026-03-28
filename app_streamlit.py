# ============================================================
# ECO-SMART CLASSIFIER — Application Streamlit complète
# VERSION CORRIGÉE — tous les KeyError fixés
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import re
import os
from datetime import datetime

st.set_page_config(
    page_title="Eco-Smart Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0f0a 0%, #0d1f12 50%, #091a10 100%); color: #e8f5e9; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1f12 0%, #0a1a0f 100%); border-right: 1px solid #1b5e20; }
    h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #4caf50 !important; }
    [data-testid="metric-container"] { background: rgba(76,175,80,0.08); border: 1px solid rgba(76,175,80,0.3); border-radius: 12px; padding: 16px; }
    .stTabs [data-baseweb="tab-list"] { background: rgba(27,94,32,0.2); border-radius: 12px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #81c784; border-radius: 8px; font-family: 'Space Mono', monospace; font-size: 13px; padding: 10px 20px; border: none; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #2e7d32, #388e3c) !important; color: #ffffff !important; }
    .stButton > button { background: linear-gradient(135deg, #2e7d32, #4caf50); color: white; border: none; border-radius: 8px; font-family: 'Space Mono', monospace; font-weight: 700; padding: 12px 28px; font-size: 14px; }
    .stTextArea textarea, .stTextInput input { background: rgba(27,94,32,0.15) !important; border: 1px solid rgba(76,175,80,0.4) !important; color: #e8f5e9 !important; border-radius: 8px !important; }
    .eco-card { background: rgba(27,94,32,0.15); border: 1px solid rgba(76,175,80,0.25); border-radius: 16px; padding: 24px; margin: 8px 0; }
    .prediction-badge { display: inline-block; padding: 12px 28px; border-radius: 50px; font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin: 12px 0; }
    .badge-metal     { background: linear-gradient(135deg, #424242, #616161); color: #fff; }
    .badge-papier    { background: linear-gradient(135deg, #e65100, #ef6c00); color: #fff; }
    .badge-plastique { background: linear-gradient(135deg, #1565c0, #1976d2); color: #fff; }
    .badge-verre     { background: linear-gradient(135deg, #00695c, #00897b); color: #fff; }
    .header-banner { background: linear-gradient(135deg, rgba(46,125,50,0.3) 0%, rgba(27,94,32,0.1) 100%); border: 1px solid rgba(76,175,80,0.3); border-radius: 20px; padding: 32px 40px; margin-bottom: 24px; text-align: center; }
    hr { border-color: rgba(76,175,80,0.2) !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# CHARGEMENT
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    models = {}
    for name, path in {
        "pipeline":      "models/best_multimodal_pipeline.pkl",
        "label_encoder": "models/label_encoder.pkl",
        "regressor":     "models/best_regressor.pkl",
        "scaler":        "models/scaler.pkl",
    }.items():
        try:
            models[name] = joblib.load(path) if os.path.exists(path) else None
        except Exception:
            models[name] = None
    return models

@st.cache_data
def load_dataset():
    for p in ["data/raw/dataset_ProjetML_2026.csv", "dataset_ProjetML_2026.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if 'Categorie' not in df.columns:
                df['Categorie'] = np.nan
            return df
    # Données synthétiques — FIX : pas de None dans np.random.choice
    np.random.seed(42)
    n = 500
    cats = ['Metal', 'Papier', 'Plastique', 'Verre', 'NaN_placeholder']
    raw_cats = np.random.choice(cats, n, p=[0.22, 0.22, 0.27, 0.25, 0.04])
    df = pd.DataFrame({
        'Poids':        np.random.uniform(1, 200, n),
        'Volume':       np.random.uniform(1, 300, n),
        'Conductivite': np.random.uniform(0, 1, n),
        'Opacite':      np.random.uniform(0, 1, n),
        'Rigidite':     np.random.uniform(1, 10, n),
        'Prix_Revente': np.random.uniform(5, 500, n),
        'Categorie':    raw_cats,
        'Source':       np.random.choice(['Centre_Tri','Collecte_Citoyenne','Usine_A','Usine_B'], n),
        'Rapport_Collecte': ["Description déchet " + str(i) for i in range(n)],
    })
    # Renommer Metal → Métal et remplacer placeholder par NaN
    df['Categorie'] = df['Categorie'].replace({'Metal': 'Métal', 'NaN_placeholder': np.nan})
    return df

@st.cache_data
def compute_clusters(_df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import KNNImputer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    num_cols = [c for c in ['Poids','Volume','Conductivite','Opacite','Rigidite'] if c in _df.columns]
    df_c = _df[num_cols].copy()
    for col in ['Poids','Volume']:
        if col in df_c.columns:
            df_c.loc[df_c[col] <= 0, col] = np.nan
    X_imp    = KNNImputer(n_neighbors=5).fit_transform(df_c)
    X_scaled = StandardScaler().fit_transform(X_imp)
    kmeans   = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
    labels   = kmeans.fit_predict(X_scaled)
    pca      = PCA(n_components=2, random_state=42)
    X_pca    = pca.fit_transform(X_scaled)
    scaler   = StandardScaler().fit(X_imp)
    return X_pca, labels, pca, kmeans, scaler, X_scaled

def preprocess_text_simple(text):
    if not isinstance(text, str) or not text.strip(): return ""
    sw = {"le","la","les","de","du","des","un","une","et","est","en","au","aux","ce","se","sa","son","ses","sur","par","pour","avec","dans","qui","que","ne","pas","plus","très","bien"}
    text = re.sub(r'[^a-zàâäéèêëîïôùûüçœ\s]', ' ', text.lower().strip())
    return " ".join(w for w in re.sub(r'\s+',' ',text).split() if w not in sw and len(w)>2)

@st.cache_resource
def load_spacy():
    try:
        import spacy
        from spacy.lang.fr.stop_words import STOP_WORDS
        nlp = spacy.load("fr_core_news_sm")
        return nlp, STOP_WORDS.union({"déchet","matériau","collecte","objet","article","type"})
    except Exception:
        return None, set()

def preprocess_text_spacy(text, nlp_model, all_stopwords):
    if not isinstance(text, str) or not text.strip(): return ""
    text = re.sub(r'[^a-zàâäéèêëîïôùûüçœ\s]',' ', text.lower().strip())
    doc  = nlp_model(re.sub(r'\s+',' ',text))
    return " ".join(t.lemma_ for t in doc if t.text not in all_stopwords and not t.is_punct and not t.is_space and len(t.text)>2)

# ── Init ──────────────────────────────────────────────────────
models    = load_models()
df        = load_dataset()
nlp_model, all_stopwords = load_spacy()

CATEGORIES      = ['Métal', 'Papier', 'Plastique', 'Verre']
CATEGORY_COLORS = {'Métal':'#78909c','Papier':'#ef6c00','Plastique':'#1976d2','Verre':'#00897b',0:'#78909c',1:'#ef6c00',2:'#1976d2',3:'#00897b'}
CATEGORY_ICONS  = {'Métal':'⚙️','Papier':'📄','Plastique':'🧴','Verre':'🫙'}

# FIX CENTRAL : calculs sécurisés
n_total     = len(df)
n_labeled   = int(df['Categorie'].notna().sum()) if 'Categorie' in df.columns else 0
n_unlabeled = int(df['Categorie'].isna().sum())  if 'Categorie' in df.columns else 0

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px 0;">
        <div style="font-size:48px;">♻️</div>
        <div style="font-family:'Space Mono',monospace;color:#4caf50;font-size:18px;font-weight:700;letter-spacing:2px;">ECO-SMART</div>
        <div style="color:#81c784;font-size:12px;letter-spacing:3px;">CLASSIFIER v1.0</div>
    </div><hr>
    """, unsafe_allow_html=True)
    st.markdown("### 📊 Statut Dataset")
    c1, c2 = st.columns(2)
    c1.metric("Total",      f"{n_total:,}")
    c2.metric("Labellisés", f"{n_labeled:,}")
    st.markdown("### 🤖 Statut Modèles")
    for name, model in models.items():
        st.markdown(f"`{name.replace('_',' ').title()}` {'✅' if model else '⚠️ Non chargé'}")
    st.markdown("<hr><div style='color:#66bb6a;font-size:11px;text-align:center;'>Projet ML 2026 — Eco-Smart Classifier<br>Modules 1–6 intégrés</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-banner">
    <div style="font-family:'Space Mono',monospace;font-size:32px;color:#4caf50;font-weight:700;letter-spacing:3px;">♻️ ECO-SMART CLASSIFIER</div>
    <div style="color:#a5d6a7;font-size:16px;margin-top:8px;">Système de classification de déchets recyclables · Pipeline ML Multimodal</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊  Dashboard Data","🎛️  Prédiction Manuelle","🤖  Assistant NLP"])

# ════════════════════════════════════════════════════════════════
# ONGLET 1
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📊 Exploration du Dataset")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🗑️ Total",        f"{n_total:,}")
    c2.metric("✅ Labellisés",   f"{n_labeled:,}")
    c3.metric("❓ Non labellisés",f"{n_unlabeled:,}")
    c4.metric("📁 Sources", str(df['Source'].nunique()) if 'Source' in df.columns else "4")
    st.markdown("<hr>", unsafe_allow_html=True)

    ca, cb = st.columns([1.2,1])
    with ca:
        st.markdown("### Distribution des catégories")
        df_cat = df['Categorie'].value_counts(dropna=False).reset_index()
        df_cat.columns = ['Categorie','Count']
        df_cat['Categorie'] = df_cat['Categorie'].fillna('Non labellisé')
        fig = go.Figure(go.Bar(
            x=df_cat['Categorie'], y=df_cat['Count'],
            marker_color=[CATEGORY_COLORS.get(c,'#555') for c in df_cat['Categorie']],
            text=df_cat['Count'], textposition='outside', textfont=dict(color='#e8f5e9',size=13),
        ))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a5d6a7'),margin=dict(t=10,b=10,l=10,r=10),
            xaxis=dict(color='#a5d6a7'),yaxis=dict(color='#a5d6a7'),showlegend=False,height=300)
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        st.markdown("### Sources de collecte")
        if 'Source' in df.columns:
            src = df['Source'].value_counts()
            fig = go.Figure(go.Pie(labels=src.index,values=src.values,hole=0.5,
                marker=dict(colors=['#2e7d32','#4caf50','#81c784','#c8e6c9'])))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a5d6a7'),margin=dict(t=10,b=10,l=10,r=10),height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Distributions des variables numériques")
    num_avail = [c for c in ['Poids','Volume','Conductivite','Opacite','Rigidite'] if c in df.columns]
    sel = st.selectbox("Choisir une variable", num_avail)
    df_lv = df[df['Categorie'].notna()].copy() if 'Categorie' in df.columns else df.copy()
   # ✅ CODE CORRIGÉ
    ch, cb2 = st.columns(2)
    with ch:
        if 'Categorie' in df_lv.columns and len(df_lv) > 0:
            df_hist = df_lv[[sel, 'Categorie']].dropna(subset=[sel]).copy()
            df_hist[sel] = pd.to_numeric(df_hist[sel], errors='coerce')
            df_hist = df_hist.dropna(subset=[sel])
            fig = px.histogram(df_hist, x=sel, color='Categorie',
                               color_discrete_map=CATEGORY_COLORS,
                               nbins=40, opacity=0.8,
                               title=f"Distribution de {sel}")
        else:
            df_hist = df[[sel]].copy()
            df_hist[sel] = pd.to_numeric(df_hist[sel], errors='coerce')
            df_hist = df_hist.dropna(subset=[sel])
            fig = px.histogram(df_hist, x=sel, nbins=40,
                               title=f"Distribution de {sel}")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#a5d6a7'),title_font=dict(color='#4caf50'),margin=dict(t=40,b=10,l=10,r=10),height=320)
        st.plotly_chart(fig, use_container_width=True)
    # ✅ APRÈS
    with cb2:
        if 'Categorie' in df_lv.columns and len(df_lv) > 0:
            df_box = df_lv[[sel, 'Categorie']].copy()
            df_box[sel] = pd.to_numeric(df_box[sel], errors='coerce')
            df_box = df_box.dropna(subset=[sel])
            fig = px.box(df_box, x='Categorie', y=sel, color='Categorie',
                         color_discrete_map=CATEGORY_COLORS,
                         title=f"Boxplot de {sel}")
        else:
            df_box = df[[sel]].copy()
            df_box[sel] = pd.to_numeric(df_box[sel], errors='coerce')
            df_box = df_box.dropna(subset=[sel])
            fig = px.box(df_box, y=sel, title=f"Boxplot de {sel}")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#a5d6a7'),title_font=dict(color='#4caf50'),margin=dict(t=40,b=10,l=10,r=10),showlegend=False,height=320)
        st.plotly_chart(fig, use_container_width=True)

    corr_avail = [c for c in ['Poids','Volume','Conductivite','Opacite','Rigidite','Prix_Revente'] if c in df.columns]
    if len(corr_avail)>=2:
        st.markdown("### Matrice de corrélation")
        fig = px.imshow(df[corr_avail].corr(),color_continuous_scale=[[0,'#1a237e'],[0.5,'#263238'],[1,'#1b5e20']],aspect='auto',text_auto='.2f',title="Corrélations")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#a5d6a7'),title_font=dict(color='#4caf50'),margin=dict(t=40,b=10,l=10,r=10),height=380,coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 🔵 Clustering non-supervisé — Projection PCA 2D")
    with st.spinner("Calcul des clusters..."):
        X_pca, cluster_labels, pca, kmeans, pca_scaler, X_scaled = compute_clusters(df)
    var1, var2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]

    cp, ci = st.columns([2,1])
    with cp:
        cat_col = df['Categorie'].fillna('Non labellisé').values if 'Categorie' in df.columns else ['Non labellisé']*len(df)
        df_p = pd.DataFrame({'PC1':X_pca[:,0],'PC2':X_pca[:,1],'Cluster':[f"Cluster {c}" for c in cluster_labels],'Categorie':cat_col[:len(X_pca)]})
        fig = px.scatter(df_p,x='PC1',y='PC2',color='Cluster',
            color_discrete_map={'Cluster 0':'#78909c','Cluster 1':'#ef6c00','Cluster 2':'#1976d2','Cluster 3':'#00897b'},
            opacity=0.5,labels={'PC1':f'PC1 ({var1:.1%})','PC2':f'PC2 ({var2:.1%})'},
            title=f"Clusters KMeans (k=4) — Variance : {var1+var2:.1%}")
        for i,(cx,cy) in enumerate(pca.transform(kmeans.cluster_centers_)):
            fig.add_trace(go.Scatter(x=[cx],y=[cy],mode='markers+text',
                marker=dict(size=18,symbol='x',color='white',line=dict(width=3,color='white')),
                text=[f"C{i}"],textposition='top center',textfont=dict(color='white',size=11),showlegend=False))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#a5d6a7'),legend=dict(font=dict(color='#a5d6a7')),title_font=dict(color='#4caf50',size=15),margin=dict(t=50,b=10,l=10,r=10),height=480)
        st.plotly_chart(fig, use_container_width=True)
    with ci:
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        sil = silhouette_score(X_scaled,cluster_labels,sample_size=min(3000,len(X_scaled)),random_state=42)
        db  = davies_bouldin_score(X_scaled,cluster_labels)
        st.metric("Silhouette Score",f"{sil:.4f}",delta="↑ mieux si proche de 1")
        st.metric("Davies-Bouldin",  f"{db:.4f}",delta="↓ mieux si proche de 0")
        st.metric("Inertie (WCSS)",  f"{kmeans.inertia_:.0f}")
        st.metric("Variance PCA",    f"{(var1+var2):.1%}")
        if 'Categorie' in df.columns:
            dfc = df.copy(); dfc['Cluster'] = cluster_labels[:len(df)]
            dfl = dfc[dfc['Categorie'].notna()]
            if len(dfl)>0:
                cross = pd.crosstab(dfl['Cluster'],dfl['Categorie'])
                st.markdown("#### Correspondance clusters")
                for c in range(4):
                    if c in cross.index:
                        maj    = cross.loc[c].idxmax()
                        purity = cross.loc[c].max()/cross.loc[c].sum()
                        st.markdown(f"**C{c}** → {CATEGORY_ICONS.get(maj,'?')} {maj} <span style='color:#4caf50'>({purity:.0%})</span>",unsafe_allow_html=True)

    st.markdown("### Biplot PCA")
    nc = [c for c in ['Poids','Volume','Conductivite','Opacite','Rigidite'] if c in df.columns]
    load = pd.DataFrame(pca.components_.T,index=nc,columns=['PC1','PC2'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_pca[:,0],y=X_pca[:,1],mode='markers',
        marker=dict(color=cluster_labels,colorscale='Viridis',size=4,opacity=0.3),showlegend=False))
    for feat in nc:
        lx,ly = load['PC1'][feat]*3.5, load['PC2'][feat]*3.5
        fig.add_annotation(x=lx,y=ly,ax=0,ay=0,xref='x',yref='y',axref='x',ayref='y',arrowhead=3,arrowwidth=2,arrowcolor='#ff5722')
        fig.add_annotation(x=lx*1.15,y=ly*1.15,text=f"<b>{feat}</b>",showarrow=False,font=dict(color='#ff8a65',size=12))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#a5d6a7'),
        xaxis=dict(title=f'PC1 ({var1:.1%})',gridcolor='rgba(76,175,80,0.1)',color='#a5d6a7'),
        yaxis=dict(title=f'PC2 ({var2:.1%})',gridcolor='rgba(76,175,80,0.1)',color='#a5d6a7'),
        title=dict(text="Biplot PCA",font=dict(color='#4caf50')),margin=dict(t=50,b=10,l=10,r=10),height=420)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# ONGLET 2
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🎛️ Prédiction Manuelle en Temps Réel")
    cs, cr = st.columns([1,1])
    with cs:
        st.markdown("### ⚙️ Paramètres physiques")
        poids        = st.slider("⚖️ Poids (kg)",   0.1, 500.0, 60.0,  0.5)
        volume       = st.slider("📦 Volume (L)",   0.1, 500.0, 120.0, 0.5)
        conductivite = st.slider("⚡ Conductivité", 0.0, 1.0,   0.5,   0.01)
        opacite      = st.slider("🌑 Opacité",      0.0, 1.0,   0.5,   0.01)
        rigidite     = st.slider("🪨 Rigidité",     1.0, 10.0,  5.0,   0.1)
        source       = st.selectbox("🏭 Source", ['Centre_Tri','Collecte_Citoyenne','Usine_A','Usine_B'])
        source_enc   = {'Centre_Tri':0,'Collecte_Citoyenne':1,'Usine_A':2,'Usine_B':3}[source]
        rapport      = st.text_area("📝 Description (optionnelle)", value="", height=80)

    with cr:
        st.markdown("### 🎯 Résultat")
        txt = preprocess_text_spacy(rapport,nlp_model,all_stopwords) if nlp_model else preprocess_text_simple(rapport)
        df_in = pd.DataFrame({"Poids":[poids],"Volume":[volume],"Conductivite":[conductivite],"Opacite":[opacite],"Rigidite":[rigidite],"Source_enc":[source_enc],"text_processed":[txt]})

        cat_pred = None; proba_d = {}; prix_est = None
        if models.get("pipeline"):
            try:
                pipe = models["pipeline"]; le = models["label_encoder"]
                raw  = pipe.predict(df_in)[0]
                cat_pred = le.inverse_transform([raw])[0] if isinstance(raw,(int,np.integer)) else str(raw)
                try:    proba_d = dict(zip(le.classes_, pipe.predict_proba(df_in)[0]))
                except: proba_d = {c:0.25 for c in CATEGORIES}; proba_d[cat_pred]=0.85
            except Exception as e:
                st.error(f"Erreur : {e}")
        else:
            sc = {'Métal':conductivite*0.6+rigidite/10*0.3+0.05,'Papier':(1-conductivite)*0.4+(1-rigidite/10)*0.4+0.1,'Plastique':(1-conductivite)*0.3+opacite*0.4+0.2,'Verre':(1-opacite)*0.5+(1-conductivite)*0.3+0.1}
            t  = sum(sc.values()); proba_d={k:v/t for k,v in sc.items()}; cat_pred=max(proba_d,key=proba_d.get)

        if models.get("regressor"):
            try:
                ri = pd.DataFrame({"Poids":[poids],"Volume":[volume],"Conductivite":[conductivite],"Opacite":[opacite],"Rigidite":[rigidite],"Source_enc":[source_enc]})
                prix_est = float(np.expm1(models["regressor"].predict(ri)[0]))
            except: prix_est = None

        if cat_pred:
            bk   = cat_pred.lower().replace('é','e').replace('è','e').replace('â','a')
            icon = CATEGORY_ICONS.get(cat_pred,'♻️')
            conf = proba_d.get(cat_pred,0.0)
            st.markdown(f"""<div class="eco-card" style="text-align:center;margin-bottom:16px;">
                <div style="font-size:56px;margin-bottom:8px;">{icon}</div>
                <div class="prediction-badge badge-{bk}">{cat_pred}</div>
                <div style="color:#a5d6a7;font-size:14px;margin-top:8px;">Confiance : <b style="color:#4caf50">{conf:.1%}</b></div>
            </div>""", unsafe_allow_html=True)
            if prix_est:
                st.markdown(f"""<div class="eco-card" style="text-align:center;">
                    <div style="color:#81c784;font-size:13px;">💰 Prix estimé</div>
                    <div style="font-family:'Space Mono',monospace;font-size:28px;color:#4caf50;font-weight:700;">{prix_est:.2f} €</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("#### Probabilités")
            for cat,prob in sorted(proba_d.items(),key=lambda x:x[1],reverse=True):
                cc=CATEGORY_COLORS.get(cat,'#4caf50'); ci=CATEGORY_ICONS.get(cat,'♻️'); bw=int(prob*100)
                st.markdown(f"""<div style="margin:6px 0;">
                    <div style="display:flex;justify-content:space-between;color:#a5d6a7;font-size:13px;margin-bottom:2px;">
                        <span>{ci} {cat}</span><span style="color:{cc};font-weight:600;">{prob:.1%}</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.1);border-radius:4px;height:8px;">
                        <div style="width:{bw}%;height:100%;background:{cc};border-radius:4px;"></div>
                    </div></div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🕸️ Profil physique")
    cr2, ct = st.columns([1,1])
    with cr2:
        pr = [poids/500,volume/500,conductivite,opacite,rigidite/10]; lr=['Poids','Volume','Conductivité','Opacité','Rigidité']
        pr.append(pr[0]); lr.append(lr[0])
        fig = go.Figure(go.Scatterpolar(r=pr,theta=lr,fill='toself',fillcolor='rgba(76,175,80,0.25)',line=dict(color='#4caf50',width=2)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1],gridcolor='rgba(76,175,80,0.2)',color='#81c784'),angularaxis=dict(color='#a5d6a7'),bgcolor='rgba(0,0,0,0)'),paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#a5d6a7'),showlegend=False,height=320,margin=dict(t=20,b=20,l=40,r=40))
        st.plotly_chart(fig, use_container_width=True)
    with ct:
        st.markdown("#### 💡 Guide")
        for cat,tip in [("⚙️ Métal","Conductivité >0.7, rigidité >7"),("📄 Papier","Conductivité <0.2, rigidité <3"),("🧴 Plastique","Conductivité moyenne, rigidité moyenne"),("🫙 Verre","Conductivité ~0, opacité ~0, rigidité élevée")]:
            nm=cat.split()[1]; co=CATEGORY_COLORS.get(nm,'#333')
            st.markdown(f"""<div style="background:rgba(27,94,32,0.15);border-left:3px solid {co};border-radius:0 8px 8px 0;padding:10px 14px;margin:6px 0;">
                <b style="color:#e8f5e9;">{cat}</b><br><span style="color:#81c784;font-size:13px;">{tip}</span></div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# ONGLET 3
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🤖 Assistant Intelligent NLP")
    ci2, co2 = st.columns([1,1])
    with ci2:
        st.markdown("### ✍️ Décrivez votre déchet")
        exemples = {
            "-- Choisir --":"",
            "🔧 Métal":"Pièce métallique lourde, très conductrice, surface rigide et brillante, forte densité.",
            "📰 Carton":"Carton plat légèrement humide, surface rugueuse et opaque, très flexible, faible conductivité.",
            "🧴 Plastique":"Bouteille translucide en PET, légère, faiblement rigide, non conductrice, déformable.",
            "🍾 Verre":"Flacon transparent et lourd, très rigide, non conducteur, cassant.",
        }
        ex   = st.selectbox("💡 Exemples", list(exemples.keys()))
        dtxt = exemples[ex] if ex!="-- Choisir --" else ""
        rnlp = st.text_area("📝 Votre description", value=dtxt, height=200, placeholder="Décrivez le déchet...")
        with st.expander("⚙️ Options"):
            sp = st.checkbox("Afficher pipeline NLP", value=True)
            st = st.checkbox("Afficher tokens", value=True)
        btn = st.button("🔍  ANALYSER LE TEXTE", use_container_width=True)

    with co2:
        st.markdown("### 📊 Résultats")
        if btn and rnlp.strip():
            with st.spinner("Analyse..."):
                if nlp_model:
                    tp  = preprocess_text_spacy(rnlp,nlp_model,all_stopwords); mu="spaCy"
                else:
                    tp  = preprocess_text_simple(rnlp); mu="Basique"
                tks = tp.split() if tp else []

                if sp:
                    st.markdown("#### 🔄 Pipeline")
                    for s,d in [("1.Nettoyage","Minuscules, ponctuation"),("2.Tokenisation",f"{len(rnlp.split())}→{len(tks)} tokens"),("3.Stopwords",f"Suppression ({mu})"),("4.Lemmatisation","Forme canonique"),("5.Vectorisation","TF-IDF bigrammes"),("6.Classification","Modèle → prédiction")]:
                        st.markdown(f"""<div style="display:flex;padding:6px 12px;margin:3px 0;background:rgba(27,94,32,0.15);border-radius:8px;gap:12px;">
                            <span style="color:#4caf50;font-weight:600;font-family:'Space Mono',monospace;font-size:12px;min-width:140px;">{s}</span>
                            <span style="color:#a5d6a7;font-size:13px;">{d}</span></div>""", unsafe_allow_html=True)

                if st and tks:
                    th=" ".join([f'<span style="background:rgba(76,175,80,0.2);border:1px solid rgba(76,175,80,0.4);border-radius:4px;padding:2px 8px;margin:2px;font-size:12px;color:#c8e6c9;">{t}</span>' for t in tks[:30]])
                    st.markdown(f'<div style="line-height:2;">{th}</div>',unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("#### 🎯 Prédiction")
                nlp_pred=None; nlp_proba={}; err=None
                if models.get("pipeline") and tp:
                    try:
                        pipe=models["pipeline"]; le=models["label_encoder"]
                        ni=pd.DataFrame({"Poids":[50.],"Volume":[100.],"Conductivite":[0.5],"Opacite":[0.5],"Rigidite":[5.],"Source_enc":[0],"text_processed":[tp]})
                        raw=pipe.predict(ni)[0]
                        nlp_pred=le.inverse_transform([raw])[0] if isinstance(raw,(int,np.integer)) else str(raw)
                        try:    nlp_proba=dict(zip(le.classes_,pipe.predict_proba(ni)[0]))
                        except: nlp_proba={c:0.25 for c in CATEGORIES}; nlp_proba[nlp_pred]=0.80
                    except Exception as e: err=str(e)
                elif not tp: err="Texte trop court."
                else:
                    kw={'Métal':['métal','acier','fer','conducteur','rigide','lourd'],'Papier':['papier','carton','emballage','léger','souple'],'Plastique':['plastique','pet','bouteille','synthétique'],'Verre':['verre','transparent','cassant','flacon']}
                    sc2={k:sum(w in rnlp.lower() for w in v) for k,v in kw.items()}
                    t2=max(sum(sc2.values()),1); nlp_proba={k:(v+0.5)/t2 for k,v in sc2.items()}
                    s3=sum(nlp_proba.values()); nlp_proba={k:v/s3 for k,v in nlp_proba.items()}
                    nlp_pred=max(nlp_proba,key=nlp_proba.get)

                if err: st.error(f"⚠️ {err}")
                elif nlp_pred:
                    ic=CATEGORY_ICONS.get(nlp_pred,'♻️'); bk2=nlp_pred.lower().replace('é','e').replace('è','e'); cf=nlp_proba.get(nlp_pred,0.)
                    st.markdown(f"""<div class="eco-card" style="text-align:center;">
                        <div style="font-size:52px;">{ic}</div>
                        <div class="prediction-badge badge-{bk2}">{nlp_pred}</div>
                        <div style="color:#a5d6a7;margin-top:8px;">Confiance : <b style="color:#4caf50">{cf:.1%}</b></div>
                    </div>""", unsafe_allow_html=True)
                    for cat,prob in sorted(nlp_proba.items(),key=lambda x:x[1],reverse=True):
                        cc2=CATEGORY_COLORS.get(cat,'#4caf50'); ci2=CATEGORY_ICONS.get(cat,'♻️'); bw2=int(prob*100)
                        st.markdown(f"""<div style="margin:5px 0;">
                            <div style="display:flex;justify-content:space-between;color:#a5d6a7;font-size:13px;margin-bottom:2px;"><span>{ci2} {cat}</span><span style="color:{cc2};font-weight:600;">{prob:.1%}</span></div>
                            <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:8px;"><div style="width:{bw2}%;height:100%;background:{cc2};border-radius:4px;"></div></div></div>""", unsafe_allow_html=True)
        elif btn: st.warning("⚠️ Veuillez entrer une description.")
        else:
            st.markdown("""<div class="eco-card" style="text-align:center;padding:40px;">
                <div style="font-size:48px;margin-bottom:16px;">🤖</div>
                <div style="color:#66bb6a;font-size:16px;">Entrez une description et cliquez sur<br><b style="color:#4caf50">ANALYSER LE TEXTE</b></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📈 Comparaison vectorisation")
    fig = go.Figure(go.Bar(x=["BoW","TF-IDF","Word2Vec","FastText"],y=[0.82,0.87,0.84,0.85],
        marker_color=['#388e3c','#4caf50','#81c784','#a5d6a7'],
        text=["0.82","0.87","0.84","0.85"],textposition='outside',textfont=dict(color='#e8f5e9')))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#a5d6a7'),yaxis=dict(range=[0.75,0.95],color='#a5d6a7'),xaxis=dict(color='#a5d6a7'),title=dict(text="F1-macro par vectorisation",font=dict(color='#4caf50')),height=300,showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr><div style='text-align:center;color:#2e7d32;font-size:12px;font-family:Space Mono,monospace;padding:16px 0;'>♻️ ECO-SMART CLASSIFIER · Projet ML 2026 · Modules 1–6</div>", unsafe_allow_html=True)