"""
main.py - API REST FastAPI
Eco-Smart Classifier
"""
import json
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.sparse import csr_matrix, hstack

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

for r in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(r, quiet=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Eco-Smart Classifier API",
    description="Classifie les déchets et estime leur prix de revente.",
    version="1.0.0",
)

# ── Chemins ───────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")


# ── Chargement des modèles ────────────────────────────────────────────────────
def load_model(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


classifier    = load_model("classifier.pkl")
regressor     = load_model("regressor.pkl")
scaler        = load_model("scaler.pkl")
tfidf         = load_model("tfidf.pkl")
label_encoder = load_model("label_encoder.pkl")

# Charger les noms de features numériques
_feat_path = MODELS_DIR / "num_features.json"
if _feat_path.exists():
    with open(_feat_path) as f:
        NUM_FEATURES: list = json.load(f).get("num_all", [])
else:
    NUM_FEATURES = [
        "Poids", "Volume", "Conductivite", "Opacite", "Rigidite",
        "Densite", "Cond_Rig_Ratio",
    ]

# ── NLP helpers ───────────────────────────────────────────────────────────────
_stemmer = SnowballStemmer("french")

try:
    _STOP_FR = set(stopwords.words("french"))
except Exception:
    _STOP_FR = set()

_STOP_DOM = {
    "dechet", "dechets", "collecte", "rapport", "collecteur", "materiau",
    "materiaux", "echantillon", "analyse", "type", "lot", "reference",
    "code", "numero", "date", "heure", "kg", "litre", "cm", "mm",
    "metre", "unite", "valeur", "mesure", "resultat", "observation",
    "note", "commentaire", "traitement", "site", "zone", "point",
}
STOPWORDS_ALL = _STOP_FR | _STOP_DOM


def pretraiter(texte: str) -> str:
    """Nettoyage → tokenisation → suppression stopwords → stemming."""
    t = str(texte).lower()
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = [
        _stemmer.stem(tk)
        for tk in t.split()
        if tk not in STOPWORDS_ALL and len(tk) > 2
    ]
    return " ".join(tokens)


# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    Poids:            float = Field(..., ge=0,   description="Poids en kg")
    Volume:           float = Field(..., ge=0,   description="Volume en litres")
    Conductivite:     float = Field(..., ge=0, le=1, description="Conductivité [0-1]")
    Opacite:          float = Field(..., ge=0, le=1, description="Opacité [0-1]")
    Rigidite:         float = Field(..., ge=0, le=1, description="Rigidité [0-1]")
    Source:           str   = Field(..., description="Source de collecte")
    Rapport_Collecte: str   = Field(..., description="Description textuelle du déchet")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Poids": 0.5,
                "Volume": 1.2,
                "Conductivite": 0.3,
                "Opacite": 0.8,
                "Rigidite": 0.6,
                "Source": "collecte_selective",
                "Rapport_Collecte": "Bouteilles en plastique PET transparentes recyclables",
            }
        }
    }


class PredictResponse(BaseModel):
    categorie:      str
    confiance:      Optional[float] = None
    prix_estime:    Optional[float] = None
    modele_version: str = "1.0.0"


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {"message": "Eco-Smart Classifier API v1.0", "docs": "/docs"}


@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok",
        "classifier_loaded": classifier is not None,
        "regressor_loaded":  regressor  is not None,
        "tfidf_loaded":      tfidf      is not None,
    }


@app.get("/classes", tags=["Info"])
def get_classes():
    if label_encoder is None:
        return {"classes": []}
    return {"classes": list(label_encoder.classes_)}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Prédit la catégorie du déchet et estime son prix de revente.
    Combine les features numériques + la description textuelle (multimodal).
    """
    if classifier is None or scaler is None or tfidf is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Modèles non chargés. Lancer multimodal.py d'abord.",
        )

    # ── Construire le vecteur numérique ───────────────────────────────────────
    source_feats = [c for c in NUM_FEATURES if c.startswith("Source_")]
    num_data = {
        "Poids":         req.Poids,
        "Volume":        req.Volume,
        "Conductivite":  req.Conductivite,
        "Opacite":       req.Opacite,
        "Rigidite":      req.Rigidite,
        "Densite":       req.Poids / (req.Volume + 1e-9),
        "Cond_Rig_Ratio": req.Conductivite / (req.Rigidite + 1e-9),
    }
    for feat in source_feats:
        cat_name = feat.replace("Source_", "")
        num_data[feat] = 1 if req.Source == cat_name else 0

    # Ordonner selon NUM_FEATURES
    df_in = pd.DataFrame([num_data]).reindex(columns=NUM_FEATURES, fill_value=0)

    try:
        X_num  = csr_matrix(scaler.transform(df_in))
        X_nlp  = tfidf.transform([pretraiter(req.Rapport_Collecte)])
        X_full = hstack([X_num, X_nlp])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de transformation : {e}")

    # ── Prédiction catégorie ──────────────────────────────────────────────────
    try:
        pred_enc  = classifier.predict(X_full)[0]
        categorie = label_encoder.classes_[pred_enc]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    confiance = None
    if hasattr(classifier, "predict_proba"):
        try:
            confiance = float(classifier.predict_proba(X_full)[0].max())
        except Exception:
            pass

    # ── Prédiction prix ───────────────────────────────────────────────────────
    prix_estime = None
    if regressor is not None:
        try:
            prix_estime = float(regressor.predict(df_in.values)[0])
        except Exception:
            pass

    return PredictResponse(
        categorie=categorie,
        confiance=round(confiance, 4) if confiance is not None else None,
        prix_estime=round(prix_estime, 2) if prix_estime is not None else None,
    )