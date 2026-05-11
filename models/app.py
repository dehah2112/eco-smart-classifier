"""
app.py - Application Web Eco-Smart Classifier
Streamlit - 3 onglets : Dashboard · Prédiction Manuelle · Assistant NLP
"""
import json
import os
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
for r in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(r, quiet=True)

# ── CONFIG PAGE ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eco-Smart Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path("models")
DATA_PATH  = Path("data/raw/dataset_ProjetML_2026.csv")

# ── THEME CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2e7d32;
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .predict-result {
        background: linear-gradient(135deg, #e3f2fd, #e8f5e9);
        border: 2px solid #1976d2;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a237e;
        margin: 1rem 0;
    }
    .nlp-result {
        background: linear-gradient(135deg, #fce4ec, #f3e5f5);
        border: 2px solid #c2185b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        color: #880e4f;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── NLP helpers ───────────────────────────────────────────────────────────────
try:
    _STOP_FR = set(stopwords.words("french"))
except Exception:
    _STOP_FR = set()

_STOP_DOM = {
    "dechet", "dechets", "collecte", "rapport", "collecteur", "materiau",
    "echantillon", "analyse", "type", "lot", "code", "date", "heure",
    "kg", "litre", "cm", "mm", "unite", "resultat", "observation", "note",
    "site", "zone",
}
STOPWORDS_ALL = _STOP_FR | _STOP_DOM
_stemmer = SnowballStemmer("french")


def pretraiter_texte(texte: str) -> str:
    t = str(texte).lower()
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = [_stemmer.stem(tk) for tk in t.split()
              if tk not in STOPWORDS_ALL and len(tk) > 2]
    return " ".join(tokens)


# ── Chargement des modèles (avec cache) ───────────────────────────────────────
@st.cache_resource(show_spinner="Chargement des modèles...")
def charger_modeles():
    def _load(name):
        p = MODELS_DIR / name
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    artefacts = {
        "classifier":    _load("classifier.pkl"),
        "regressor":     _load("regressor.pkl"),
        "scaler":        _load("scaler.pkl"),
        "tfidf":         _load("tfidf.pkl"),
        "label_encoder": _load("label_encoder.pkl"),
        "kmeans":        _load("kmeans.pkl"),
        "scaler_clust":  _load("scaler_clust.pkl"),
        "pca_2d":        _load("pca_2d.pkl"),
        "nlp_clf":       _load("nlp_classifier.pkl"),
    }

    # Charger num_features si disponible
    feat_path = MODELS_DIR / "num_features.json"
    if feat_path.exists():
        with open(feat_path) as f:
            artefacts["num_features"] = json.load(f).get("num_all", [])
    else:
        artefacts["num_features"] = [
            "Poids", "Volume", "Conductivite", "Opacite", "Rigidite",
            "Densite", "Cond_Rig_Ratio",
        ]
    return artefacts


@st.cache_data(show_spinner="Chargement des données...")
def charger_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">♻️ Eco-Smart Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Classification des déchets · Estimation du prix de revente · Assistant NLP</div>',
    unsafe_allow_html=True
)

# ── Chargement ────────────────────────────────────────────────────────────────
artefacts = charger_modeles()
df_raw    = charger_dataset()

models_ok = artefacts["classifier"] is not None
le        = artefacts["label_encoder"]
categories = list(le.classes_) if le is not None else ["Inconnu"]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/recycling.png", width=80)
    st.title("Navigation")
    st.markdown("---")
    st.markdown("**Statut des modèles**")

    status_items = [
        ("Classifier",     artefacts["classifier"]),
        ("Régresseur",     artefacts["regressor"]),
        ("TF-IDF",         artefacts["tfidf"]),
        ("K-Means",        artefacts["kmeans"]),
        ("LabelEncoder",   artefacts["label_encoder"]),
    ]
    for name, obj in status_items:
        icon = "✅" if obj is not None else "❌"
        st.markdown(f"{icon} {name}")

    st.markdown("---")
    st.markdown("**Catégories**")
    for cat in categories:
        st.markdown(f"• {cat}")

    st.markdown("---")
    st.caption("Projet ML 2026 – Eco-Smart Classifier")

# ── ONGLETS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard Data",
    "🎛️ Prédiction Manuelle",
    "💬 Assistant NLP",
])

# ═══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 – DASHBOARD DATA
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Dashboard – Exploration des Données")

    if df_raw is None:
        st.error("Dataset non trouvé. Vérifier : data/raw/dataset_ProjetML_2026.csv")
    else:
        # ── KPIs ────────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Total échantillons", f"{len(df_raw):,}")
        with col2:
            n_labeled = df_raw["Categorie"].notna().sum()
            st.metric("🏷️ Avec label", f"{n_labeled:,}")
        with col3:
            n_missing_cat = df_raw["Categorie"].isna().sum()
            st.metric("❓ Labels manquants", f"{n_missing_cat:,}")
        with col4:
            n_classes = df_raw["Categorie"].nunique()
            st.metric("🗂️ Catégories", n_classes)

        st.markdown("---")

        # ── Row 1 : Distribution catégories + Poids ──────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Distribution des Catégories")
            fig, ax = plt.subplots(figsize=(6, 4))
            counts = df_raw["Categorie"].value_counts(dropna=True)
            colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
            ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
            ax.set_xlabel("Catégorie")
            ax.set_ylabel("Nombre d'échantillons")
            ax.set_title("Distribution des catégories de déchets")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.subheader("Valeurs Manquantes")
            missing = df_raw.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=True)
            if len(missing) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(missing.index, missing.values / len(df_raw) * 100,
                        color="coral", alpha=0.8)
                ax.set_xlabel("% de valeurs manquantes")
                ax.set_title("Taux de valeurs manquantes par colonne")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Aucune valeur manquante détectée.")

        # ── Row 2 : Distributions numériques ─────────────────────────────
        st.markdown("---")
        st.subheader("Distributions des Variables Numériques")
        num_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
        num_cols = [c for c in num_cols if c in df_raw.columns]

        fig, axes = plt.subplots(1, len(num_cols), figsize=(15, 3))
        for ax, col in zip(axes, num_cols):
            data = df_raw[col].dropna()
            ax.hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Row 3 : Corrélations ──────────────────────────────────────────
        st.markdown("---")
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Matrice de Corrélation")
            corr_cols = num_cols + (["Prix_Revente"] if "Prix_Revente" in df_raw.columns else [])
            fig, ax = plt.subplots(figsize=(6, 5))
            corr = df_raw[corr_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                        mask=mask, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
            ax.set_title("Corrélations des features numériques")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_d:
            st.subheader("Prix de Revente par Catégorie")
            if "Prix_Revente" in df_raw.columns and "Categorie" in df_raw.columns:
                df_box = df_raw[["Categorie", "Prix_Revente"]].dropna()
                fig, ax = plt.subplots(figsize=(6, 5))
                df_box.boxplot(column="Prix_Revente", by="Categorie", ax=ax)
                ax.set_xlabel("Catégorie")
                ax.set_ylabel("Prix (€)")
                ax.set_title("Distribution prix par catégorie")
                plt.suptitle("")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # ── Row 4 : Clusters PCA ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("Visualisation des Clusters (PCA 2D)")

        if artefacts["kmeans"] is not None and artefacts["pca_2d"] is not None:
            st.info("Clusters K-Means projetés en 2D via PCA.")

            # Reconstruire les données pour la visualisation
            cluster_csv = Path("data/processed/dataset_with_clusters.csv")
            if cluster_csv.exists():
                df_clust = pd.read_csv(cluster_csv)
                n_clusters = df_clust["Cluster_KMeans"].nunique()

                # PCA 2D via les colonnes numériques
                num_c = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
                num_c = [c for c in num_c if c in df_clust.columns]
                df_viz = df_clust[num_c + ["Cluster_KMeans"]].dropna()

                pca_viz = PCA(n_components=2, random_state=42)
                X_2d = pca_viz.fit_transform(df_viz[num_c])

                fig, ax = plt.subplots(figsize=(8, 6))
                palette = plt.cm.Set1(np.linspace(0, 0.9, n_clusters))
                for cid in range(n_clusters):
                    mask = df_viz["Cluster_KMeans"].values == cid
                    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                               c=[palette[cid]], label=f"Cluster {cid}",
                               alpha=0.6, s=20, edgecolors="none")
                ax.set_xlabel(f"PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}%)")
                ax.set_title(f"K-Means (k={n_clusters}) – PCA 2D")
                ax.legend(title="Clusters", bbox_to_anchor=(1.01, 1), loc="upper left")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Lancer cluster.py pour générer les données de clusters.")
        else:
            st.warning("Modèle K-Means non disponible. Lancer cluster.py d'abord.")

        # ── Aperçu données brutes ────────────────────────────────────────
        st.markdown("---")
        with st.expander("📋 Aperçu du dataset brut"):
            st.dataframe(df_raw.head(20), use_container_width=True)
            st.caption(f"Shape : {df_raw.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 – PRÉDICTION MANUELLE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🎛️ Prédiction Manuelle")
    st.markdown("Ajustez les curseurs pour prédire **en temps réel** la catégorie et le prix de revente.")

    if not models_ok:
        st.error("⚠️ Modèles non disponibles. Lancer multimodal.py d'abord.")
    else:
        col_sliders, col_result = st.columns([1, 1])

        with col_sliders:
            st.subheader("🔧 Paramètres physiques")

            poids = st.slider(
                "⚖️ Poids (kg)", min_value=0.01, max_value=50.0, value=1.5, step=0.1
            )
            volume = st.slider(
                "📦 Volume (L)", min_value=0.01, max_value=100.0, value=5.0, step=0.5
            )
            conductivite = st.slider(
                "⚡ Conductivité", min_value=0.0, max_value=1.0, value=0.3, step=0.01
            )
            opacite = st.slider(
                "🔲 Opacité", min_value=0.0, max_value=1.0, value=0.5, step=0.01
            )
            rigidite = st.slider(
                "🔩 Rigidité", min_value=0.0, max_value=1.0, value=0.6, step=0.01
            )

            st.subheader("📝 Description textuelle")
            rapport = st.text_area(
                "Rapport de collecte",
                value="Matériau plastique rigide transparent, collecté en tri sélectif, léger et recyclable.",
                height=100,
                help="Décrivez le matériau collecté pour améliorer la prédiction.",
            )

            source_opts = ["collecte_selective", "dechetterie", "industriel", "menager", "autre"]
            source = st.selectbox("🏭 Source de collecte", source_opts)

        with col_result:
            st.subheader("📊 Résultats de la prédiction")

            # Construire le vecteur features
            features_num = {
                "Poids":         poids,
                "Volume":        volume,
                "Conductivite":  conductivite,
                "Opacite":       opacite,
                "Rigidite":      rigidite,
                "Densite":       poids / (volume + 1e-9),
                "Cond_Rig_Ratio": conductivite / (rigidite + 1e-9),
            }
            for src in source_opts:
                features_num[f"Source_{src}"] = 1 if source == src else 0

            num_all = artefacts.get("num_features", list(features_num.keys()))

            try:
                row = pd.DataFrame([features_num])
                row = row.reindex(columns=num_all, fill_value=0)

                X_num  = csr_matrix(artefacts["scaler"].transform(row))
                X_nlp  = artefacts["tfidf"].transform([pretraiter_texte(rapport)])
                X_full = hstack([X_num, X_nlp])

                pred_enc  = artefacts["classifier"].predict(X_full)[0]
                categorie = le.classes_[pred_enc]

                confiance = None
                if hasattr(artefacts["classifier"], "predict_proba"):
                    proba = artefacts["classifier"].predict_proba(X_full)[0]
                    confiance = proba.max()

                # Résultat catégorie
                st.markdown(
                    f'<div class="predict-result">🏷️ Catégorie prédite<br>'
                    f'<span style="font-size:1.8rem">{categorie}</span></div>',
                    unsafe_allow_html=True
                )

                if confiance is not None:
                    st.progress(confiance)
                    st.caption(f"Confiance : {confiance:.1%}")

                # Prix estimé
                prix_estime = None
                if artefacts["regressor"] is not None:
                    try:
                        prix_estime = float(artefacts["regressor"].predict(row.values)[0])
                        st.metric("💶 Prix de revente estimé", f"{prix_estime:.2f} €",
                                  delta=f"+{max(0, prix_estime - 10):.2f} € vs seuil")
                    except Exception:
                        pass

                # Features dérivées
                st.markdown("---")
                st.subheader("📐 Features calculées")
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.metric("Densité", f"{poids/(volume+1e-9):.3f}")
                with col_f2:
                    st.metric("Cond/Rigidité", f"{conductivite/(rigidite+1e-9):.3f}")

                # Probabilités par classe (si dispo)
                if hasattr(artefacts["classifier"], "predict_proba") and confiance is not None:
                    st.markdown("---")
                    st.subheader("📊 Probabilités par catégorie")
                    proba_df = pd.DataFrame({
                        "Catégorie": le.classes_,
                        "Probabilité": artefacts["classifier"].predict_proba(X_full)[0]
                    }).sort_values("Probabilité", ascending=True)

                    fig, ax = plt.subplots(figsize=(6, 3))
                    colors_bar = ["#2e7d32" if c == categorie else "#90a4ae"
                                  for c in proba_df["Catégorie"]]
                    ax.barh(proba_df["Catégorie"], proba_df["Probabilité"],
                            color=colors_bar, alpha=0.85)
                    ax.set_xlabel("Probabilité")
                    ax.set_xlim(0, 1)
                    ax.set_title("Distribution des probabilités")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")
                st.info("Vérifier que les modèles sont bien entraînés et sauvegardés.")

# ═══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 – ASSISTANT NLP
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("💬 Assistant Intelligent NLP")
    st.markdown(
        "Décrivez le déchet en **langage naturel** et l'assistant prédit automatiquement "
        "sa catégorie grâce au pipeline NLP."
    )

    if artefacts["tfidf"] is None or le is None:
        st.error("⚠️ Pipeline NLP non disponible. Lancer nlp_pipeline.py d'abord.")
    else:
        # Choisir le classificateur NLP
        clf_nlp = artefacts.get("nlp_clf") or artefacts.get("classifier")

        col_nlp_in, col_nlp_out = st.columns([1, 1])

        with col_nlp_in:
            st.subheader("✏️ Décrivez votre déchet")

            exemples = [
                "Sélectionner un exemple...",
                "Bouteilles en plastique PET transparentes, légères, issues du tri sélectif, recyclables",
                "Ferraille rouillée conductrice, lourde, rigidité élevée, provenant de débris industriels",
                "Verre cassé opaque, lourd, non conducteur, fragile, provenant de bouteilles",
                "Carton ondulé légèrement humide, pliable, récupéré en déchetterie",
                "Câbles électriques en cuivre avec isolant, haute conductivité, mélange plastique métal",
            ]
            exemple_selec = st.selectbox("💡 Exemples prédéfinis", exemples)

            if exemple_selec != "Sélectionner un exemple...":
                valeur_defaut = exemple_selec
            else:
                valeur_defaut = ""

            texte_nlp = st.text_area(
                "Description du déchet",
                value=valeur_defaut,
                height=150,
                placeholder="Ex: Plastique rigide transparent, léger, recyclable, issu du tri sélectif...",
            )

            st.markdown("---")
            st.subheader("⚙️ Options d'analyse")
            show_tokens   = st.checkbox("Afficher les tokens après prétraitement", value=True)
            show_topwords = st.checkbox("Afficher les mots-clés importants", value=True)

        with col_nlp_out:
            st.subheader("🔍 Analyse & Résultat")

            if texte_nlp and texte_nlp.strip():
                # Prétraitement
                texte_clean = pretraiter_texte(texte_nlp)

                if show_tokens:
                    tokens = texte_clean.split()
                    st.markdown("**Tokens après prétraitement :**")
                    tokens_html = " ".join(
                        [f'<span style="background:#e8f5e9;border-radius:4px;padding:2px 6px;'
                         f'margin:2px;font-size:0.85rem">{t}</span>' for t in tokens]
                    )
                    st.markdown(tokens_html, unsafe_allow_html=True)
                    st.caption(f"{len(tokens)} tokens conservés")
                    st.markdown("---")

                # Prédiction
                try:
                    X_vec  = artefacts["tfidf"].transform([texte_clean])
                    pred   = clf_nlp.predict(X_vec)[0]
                    categ  = le.classes_[pred]

                    confiance_nlp = None
                    if hasattr(clf_nlp, "predict_proba"):
                        proba_nlp    = clf_nlp.predict_proba(X_vec)[0]
                        confiance_nlp = proba_nlp.max()

                    # Résultat principal
                    st.markdown(
                        f'<div class="nlp-result">🤖 Catégorie identifiée<br>'
                        f'<span style="font-size:1.8rem">{categ}</span></div>',
                        unsafe_allow_html=True
                    )

                    if confiance_nlp is not None:
                        st.progress(confiance_nlp)
                        col_conf = "🟢" if confiance_nlp > 0.8 else ("🟡" if confiance_nlp > 0.5 else "🔴")
                        st.caption(f"{col_conf} Confiance : {confiance_nlp:.1%}")

                    # Top mots-clés
                    if show_topwords:
                        st.markdown("---")
                        st.subheader("🔑 Mots-clés importants")
                        try:
                            vocab = artefacts["tfidf"].vocabulary_
                            tokens_in_vocab = [(t, vocab.get(t, -1)) for t in texte_clean.split()]
                            tokens_in_vocab = [(t, idx) for t, idx in tokens_in_vocab if idx >= 0]

                            if tokens_in_vocab and X_vec.nnz > 0:
                                scores = [(t, X_vec[0, idx]) for t, idx in tokens_in_vocab]
                                scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]

                                if scores:
                                    words_kw, vals_kw = zip(*scores)
                                    fig, ax = plt.subplots(figsize=(6, 3))
                                    ax.barh(list(words_kw)[::-1], list(vals_kw)[::-1],
                                            color="#c2185b", alpha=0.8)
                                    ax.set_xlabel("Score TF-IDF")
                                    ax.set_title("Top mots-clés (score TF-IDF)")
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                        except Exception:
                            pass

                    # Probabilités par classe
                    if hasattr(clf_nlp, "predict_proba") and confiance_nlp is not None:
                        st.markdown("---")
                        proba_df2 = pd.DataFrame({
                            "Catégorie":  le.classes_,
                            "Probabilité": clf_nlp.predict_proba(X_vec)[0]
                        }).sort_values("Probabilité", ascending=True)

                        fig, ax = plt.subplots(figsize=(6, 3))
                        colors_p = ["#c2185b" if c == categ else "#bcaaa4"
                                    for c in proba_df2["Catégorie"]]
                        ax.barh(proba_df2["Catégorie"], proba_df2["Probabilité"],
                                color=colors_p, alpha=0.85)
                        ax.set_xlabel("Probabilité")
                        ax.set_xlim(0, 1)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                except Exception as e:
                    st.error(f"Erreur NLP : {e}")

            else:
                st.info("👆 Entrez une description dans le panneau de gauche pour commencer l'analyse.")

                # Afficher les catégories disponibles
                st.markdown("---")
                st.subheader("📋 Catégories disponibles")
                for cat in categories:
                    st.markdown(f"• **{cat}**")

        # ── Section Comparaison Multi-Textes ─────────────────────────────
        st.markdown("---")
        st.subheader("🆚 Comparer plusieurs descriptions")
        with st.expander("Ouvrir le comparateur"):
            n_comp = st.number_input("Nombre de textes à comparer", 2, 5, 3)
            textes_comp = []
            for i in range(int(n_comp)):
                t = st.text_input(f"Texte {i+1}", key=f"comp_{i}",
                                  placeholder=f"Description du déchet {i+1}...")
                textes_comp.append(t)

            if st.button("Comparer", type="primary") and clf_nlp is not None:
                results = []
                for t in textes_comp:
                    if t.strip():
                        try:
                            tc  = pretraiter_texte(t)
                            Xv  = artefacts["tfidf"].transform([tc])
                            p   = clf_nlp.predict(Xv)[0]
                            cat = le.classes_[p]
                            conf_v = None
                            if hasattr(clf_nlp, "predict_proba"):
                                conf_v = clf_nlp.predict_proba(Xv)[0].max()
                            results.append({
                                "Texte": t[:60] + "...",
                                "Catégorie": cat,
                                "Confiance": f"{conf_v:.1%}" if conf_v else "N/A"
                            })
                        except Exception:
                            results.append({"Texte": t[:60] + "...", "Catégorie": "Erreur", "Confiance": "—"})

                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True)