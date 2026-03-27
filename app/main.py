# app/main.py — VERSION FINALE CORRIGÉE
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import logging
import json
import time
import re
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as SPACY_STOPWORDS
from datetime import datetime

# ── Logging JSON ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eco-smart-api")

app = FastAPI(
    title="Eco-Smart Classifier API",
    description="Classification et estimation de prix de déchets recyclables",
    version="1.0.0"
)

# ── Variables globales ────────────────────────────────────────────────────────
pipeline          = None
label_encoder     = None
prix_pipeline     = None
nlp_model         = None
prediction_counter = 0
confidence_sum     = 0.0
start_time         = time.time()

DOMAIN_STOPWORDS = {"déchet","matériau","collecte","objet","article","type"}
ALL_STOPWORDS    = SPACY_STOPWORDS.union(DOMAIN_STOPWORDS)

# ── Prétraitement NLP (cohérent Module 4) ────────────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^a-zàâäéèêëîïôùûüçœ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    doc  = nlp_model(text)
    tokens = [
        t.lemma_ for t in doc
        if t.text not in ALL_STOPWORDS
        and not t.is_punct and not t.is_space
        and len(t.text) > 2
    ]
    return " ".join(tokens)

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_models():
    global pipeline, label_encoder, prix_pipeline, nlp_model
    try:
        pipeline      = joblib.load("models/best_multimodal_pipeline.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        prix_pipeline = joblib.load("models/best_regressor.pkl")
        nlp_model     = spacy.load("fr_core_news_sm")
        logger.info("✅ Tous les modèles chargés")
    except FileNotFoundError as e:
        logger.error(f"Modèle manquant : {e}")
        raise

# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    Poids:            float = Field(..., gt=0, le=500)
    Volume:           float = Field(..., gt=0, le=500)
    Conductivite:     float = Field(..., ge=0, le=1)
    Opacite:          float = Field(..., ge=0, le=1)
    Rigidite:         float = Field(..., ge=1, le=10)
    Rapport_Collecte: Optional[str] = Field(None)

class PredictionResponse(BaseModel):
    categorie:          str
    probabilites:       dict
    prix_estime:        float
    confidence:         float
    timestamp:          str
    processing_time_ms: float

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status":       "healthy",
        "model_loaded": pipeline is not None,
        "timestamp":    datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global prediction_counter, confidence_sum, avg_confidence
    start = time.time()

    try:
        # Prétraitement NLP
        text_processed = preprocess_text(request.Rapport_Collecte or "")

        # DataFrame dans le format attendu par le pipeline (Module 5)
        df_input = pd.DataFrame({
            "Poids":          [request.Poids],
            "Volume":         [request.Volume],
            "Conductivite":   [request.Conductivite],
            "Opacite":        [request.Opacite],
            "Rigidite":       [request.Rigidite],
            "text_processed": [text_processed]
        })

        # Prédiction classification
        y_pred_raw = pipeline.predict(df_input)[0]
        if isinstance(y_pred_raw, (int, np.integer)):
            categorie = label_encoder.inverse_transform([y_pred_raw])[0]
        else:
            categorie = str(y_pred_raw)

        # Probabilités (si disponible)
        try:
            y_proba    = pipeline.predict_proba(df_input)[0]
            confidence = float(np.max(y_proba))
            proba_dict = {
                str(cls): float(prob)
                for cls, prob in zip(label_encoder.classes_, y_proba)
            }
        except AttributeError:
            confidence = 1.0
            proba_dict = {str(cls): 0.0 for cls in label_encoder.classes_}
            proba_dict[categorie] = 1.0

        # Prix estimé (régression Module 2)
        reg_input = df_input[
            ["Poids","Volume","Conductivite","Opacite","Rigidite"]
        ].copy()
        reg_input["Source_enc"] = 0  # valeur par défaut
        try:
            prix_estime = float(
                np.expm1(prix_pipeline.predict(reg_input)[0]))
        except Exception:
            prix_estime = 0.0

        # Compteurs monitoring
        prediction_counter += 1
        confidence_sum     += confidence

        # Log JSON structuré
        logger.info(json.dumps({
            "timestamp":  datetime.utcnow().isoformat(),
            "prediction": categorie,
            "confidence": confidence,
            "prix":       prix_estime,
        }))

        elapsed_ms = (time.time() - start) * 1000
        return PredictionResponse(
            categorie=categorie,
            probabilites=proba_dict,
            prix_estime=round(prix_estime, 2),
            confidence=round(confidence, 4),
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 2)
        )

    except Exception as e:
        logger.error(f"Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    avg_conf = confidence_sum / prediction_counter if prediction_counter > 0 else 0.0
    return {
        "model_version":      "1.0.0",
        "total_predictions":  prediction_counter,
        "avg_confidence":     round(avg_conf, 4),
        "uptime_seconds":     round(time.time() - start_time, 1)
    }