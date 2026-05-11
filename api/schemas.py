"""
schemas.py - Schémas Pydantic supplémentaires
Eco-Smart Classifier API
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    Poids:            float = Field(..., ge=0,   description="Poids en kg")
    Volume:           float = Field(..., ge=0,   description="Volume en litres")
    Conductivite:     float = Field(..., ge=0, le=1)
    Opacite:          float = Field(..., ge=0, le=1)
    Rigidite:         float = Field(..., ge=0, le=1)
    Source:           str   = Field(..., description="Source de collecte")
    Rapport_Collecte: str   = Field(..., description="Description textuelle")


class PredictResponse(BaseModel):
    categorie:      str
    confiance:      Optional[float] = None
    prix_estime:    Optional[float] = None
    modele_version: str = "1.0.0"


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest] = Field(..., description="Liste d'échantillons")


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    n_items: int


class HealthResponse(BaseModel):
    status:             str
    classifier_loaded:  bool
    regressor_loaded:   bool
    tfidf_loaded:       bool


class ClassesResponse(BaseModel):
    classes: List[str]