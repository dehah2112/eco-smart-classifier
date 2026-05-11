"""
preprocess.py - Module 1 : Nettoyage & Préparation des données
Eco-Smart Classifier
"""
import pickle
import re
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
 
warnings.filterwarnings("ignore")
 
# ── Chemins ──────────────────────────────────────────────────────────────────
DATA_RAW       = Path("data/raw/dataset_ProjetML_2026.csv")
DATA_PROCESSED = Path("data/processed")
MODELS_DIR     = Path("models")
 
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
 
# ── Constantes ────────────────────────────────────────────────────────────────
NUM_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
SEED     = 42
 
 
# ── Fonctions ─────────────────────────────────────────────────────────────────
def clip_iqr(series: pd.Series, factor: float = 3.0) -> pd.Series:
    """Clipping des outliers par méthode IQR × factor."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(q1 - factor * iqr, q3 + factor * iqr)
 
 
def traiter_outliers(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """Applique le clipping IQR sur toutes les colonnes numériques."""
    df = df.copy()
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = clip_iqr(df[col], factor)
    return df
 
 
def imputer_knn(df: pd.DataFrame, n_neighbors: int = 5):
    """Imputation KNN sur les colonnes numériques. Retourne (df, imputer)."""
    df = df.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[NUM_COLS] = imputer.fit_transform(df[NUM_COLS])
    return df, imputer
 
 
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features dérivées : Densite, Cond_Rig_Ratio."""
    df = df.copy()
    df["Densite"]        = df["Poids"] / (df["Volume"] + 1e-9)
    df["Cond_Rig_Ratio"] = df["Conductivite"] / (df["Rigidite"] + 1e-9)
    return df
 
 
def encoder_source(df: pd.DataFrame) -> pd.DataFrame:
    """One-Hot Encoding de la variable catégorielle Source."""
    return pd.get_dummies(df, columns=["Source"], drop_first=False, prefix="Source")
 
 
def standardiser(df: pd.DataFrame, feature_cols: list):
    """StandardScaler sur les features numériques. Retourne (df, scaler)."""
    df = df.copy()
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler
 
 
def pipeline_complet(df_raw: pd.DataFrame):
    """
    Pipeline complet Module 1 :
    outliers → imputation KNN → feature engineering → OHE → scaling
    Retourne : df_clean, artefacts (imputer, scaler, label_encoder)
    """
    print("[preprocess] Début du pipeline...")
 
    # 1. Outliers
    df = traiter_outliers(df_raw)
    print(f"  [1/5] Outliers traités (IQR×3)")
 
    # 2. Imputation KNN
    df, imputer = imputer_knn(df)
    print(f"  [2/5] Imputation KNN terminée | NaN restants : {df[NUM_COLS].isna().sum().sum()}")
 
    # 3. Feature Engineering
    df = feature_engineering(df)
    print(f"  [3/5] Features dérivées créées : Densite, Cond_Rig_Ratio")
 
    # 4. Encodage Source
    df = encoder_source(df)
    print(f"  [4/5] OHE Source appliqué")
 
    # 5. Standardisation
    num_feats = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite",
                 "Densite", "Cond_Rig_Ratio"]
    df, scaler = standardiser(df, num_feats)
    print(f"  [5/5] Standardisation appliquée")
 
    # Encodage de la cible
    label_encoder = None
    if "Categorie" in df.columns:
        le = LabelEncoder()
        df_labeled = df[df["Categorie"].notna()].copy()
        df_labeled["Categorie_enc"] = le.fit_transform(df_labeled["Categorie"])
        df.loc[df_labeled.index, "Categorie_enc"] = df_labeled["Categorie_enc"]
        label_encoder = le
        print(f"  [+] LabelEncoder : {list(le.classes_)}")
 
    artefacts = {
        "imputer":       imputer,
        "scaler":        scaler,
        "label_encoder": label_encoder,
    }
    print(f"[preprocess] Terminé | Shape : {df.shape}")
    return df, artefacts
 
 
def run():
    """Point d'entrée DVC."""
    print("=" * 55)
    print("MODULE 1 – Préprocessing")
    print("=" * 55)
 
    df_raw = pd.read_csv(DATA_RAW)
    print(f"Données chargées : {df_raw.shape}")
 
    df_clean, artefacts = pipeline_complet(df_raw)
 
    # Sauvegarde
    df_clean.to_csv(DATA_PROCESSED / "dataset_clean.csv", index=False)
    with open(DATA_PROCESSED / "dataset_clean.pkl", "wb") as f:
        pickle.dump(df_clean, f)
 
    for name, obj in artefacts.items():
        if obj is not None:
            with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
                pickle.dump(obj, f)
            print(f"  Artefact sauvegardé : models/{name}.pkl")
 
    print(f"\nOK dataset_clean.pkl sauvegardé dans {DATA_PROCESSED}")
 
 
if __name__ == "__main__":
    run()