"""
predict.py - Module d'inférence unifié
Eco-Smart Classifier
"""
import pickle
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from scipy.sparse import csr_matrix, hstack

warnings.filterwarnings("ignore")

for r in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(r, quiet=True)

MODELS_DIR = Path("models")

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
    """Prétraitement NLP : nettoyage → stopwords → stemming."""
    t = str(texte).lower()
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = [_stemmer.stem(tk) for tk in t.split()
              if tk not in STOPWORDS_ALL and len(tk) > 2]
    return " ".join(tokens)


# ── Chargement des artefacts ──────────────────────────────────────────────────
def _load(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def charger_artefacts() -> dict:
    """Charge tous les artefacts nécessaires à l'inférence."""
    return {
        "classifier":    _load("classifier.pkl"),
        "regressor":     _load("regressor.pkl"),
        "scaler":        _load("scaler.pkl"),
        "tfidf":         _load("tfidf.pkl"),
        "label_encoder": _load("label_encoder.pkl"),
        "kmeans":        _load("kmeans.pkl"),
        "scaler_clust":  _load("scaler_clust.pkl"),
        "pca_2d":        _load("pca_2d.pkl"),
    }


# ── Fonctions d'inférence ─────────────────────────────────────────────────────
def predire_categorie(
    features_num: dict,
    texte: str,
    artefacts: dict,
    num_feature_names: list,
) -> Tuple[str, Optional[float]]:
    """
    Prédit la catégorie à partir des features numériques + texte.

    Parameters
    ----------
    features_num : dict  – ex: {"Poids": 0.5, "Volume": 1.2, ...}
    texte        : str   – Rapport_Collecte brut
    artefacts    : dict  – résultat de charger_artefacts()
    num_feature_names : list – ordre des features numériques

    Returns
    -------
    (categorie, confiance)
    """
    clf = artefacts.get("classifier")
    sc  = artefacts.get("scaler")
    tfidf = artefacts.get("tfidf")
    le  = artefacts.get("label_encoder")

    if clf is None or sc is None or tfidf is None or le is None:
        raise RuntimeError("Artefacts manquants. Exécuter multimodal.py d'abord.")

    import pandas as pd
    row = pd.DataFrame([features_num])[num_feature_names]
    X_num = csr_matrix(sc.transform(row))
    X_nlp = tfidf.transform([pretraiter_texte(texte)])
    X_full = hstack([X_num, X_nlp])

    pred_enc  = clf.predict(X_full)[0]
    categorie = le.classes_[pred_enc]

    confiance = None
    if hasattr(clf, "predict_proba"):
        confiance = float(clf.predict_proba(X_full)[0].max())

    return categorie, confiance


def predire_prix(
    features_num: dict,
    artefacts: dict,
    num_feature_names: list,
) -> Optional[float]:
    """
    Prédit le prix de revente à partir des features numériques.
    """
    reg = artefacts.get("regressor")
    sc  = artefacts.get("scaler")

    if reg is None or sc is None:
        return None

    import pandas as pd
    row = pd.DataFrame([features_num])[num_feature_names]
    X_num = sc.transform(row)
    return float(reg.predict(X_num)[0])


def predire_cluster(
    features_num: dict,
    artefacts: dict,
    num_feature_names_clust: list,
) -> Optional[int]:
    """
    Attribue un cluster K-Means à un échantillon.
    """
    km    = artefacts.get("kmeans")
    sc_c  = artefacts.get("scaler_clust")

    if km is None or sc_c is None:
        return None

    import pandas as pd
    row = pd.DataFrame([features_num])
    # Utiliser uniquement les colonnes disponibles
    cols_ok = [c for c in num_feature_names_clust if c in row.columns]
    row = row[cols_ok].reindex(columns=num_feature_names_clust, fill_value=0)
    X_scaled = sc_c.transform(row)
    return int(km.predict(X_scaled)[0])


def predire_nlp_seul(texte: str, artefacts: dict) -> Tuple[str, Optional[float]]:
    """
    Prédit uniquement via la colonne texte (NLP only).
    Utilisé par l'Assistant Intelligent de l'application web.
    """
    tfidf = artefacts.get("tfidf")
    le    = artefacts.get("label_encoder")

    # Essayer d'abord nlp_classifier, sinon classifier
    clf = _load("nlp_classifier.pkl") or artefacts.get("classifier")

    if tfidf is None or le is None or clf is None:
        raise RuntimeError("Artefacts NLP manquants.")

    texte_clean = pretraiter_texte(texte)
    X_nlp       = tfidf.transform([texte_clean])
    pred_enc    = clf.predict(X_nlp)[0]
    categorie   = le.classes_[pred_enc]

    confiance = None
    if hasattr(clf, "predict_proba"):
        confiance = float(clf.predict_proba(X_nlp)[0].max())

    return categorie, confiance


# ── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    artefacts = charger_artefacts()
    print("Artefacts chargés :")
    for k, v in artefacts.items():
        print(f"  {k:20s} : {'OK' if v is not None else 'MANQUANT'}")

    # Test NLP seul
    texte_test = "Plastique rigide transparent récupéré en tri sélectif"
    try:
        cat, conf = predire_nlp_seul(texte_test, artefacts)
        print(f"\nTest NLP : '{texte_test[:50]}...'")
        print(f"  → Catégorie : {cat} | Confiance : {conf}")
    except Exception as e:
        print(f"  [SKIP] {e}")