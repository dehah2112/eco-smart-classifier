"""
multimodal.py - Module 5 : Pipeline Multimodal (NLP + Numérique)
Eco-Smart Classifier
"""
import json
import pickle
import re
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.impute import KNNImputer
 
warnings.filterwarnings("ignore")
 
for r in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(r, quiet=True)
 
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False
 
DATA_RAW    = Path("data/raw/dataset_ProjetML_2026.csv")
MODELS_DIR  = Path("models")
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
 
NUM_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
 
# ── NLP helpers ──────────────────────────────────────────────────────────────
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
 
 
def pretraiter_nlp(texte: str) -> str:
    t = str(texte).lower()
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = t.split()
    tokens = [_stemmer.stem(tk) for tk in tokens
              if tk not in STOPWORDS_ALL and len(tk) > 2]
    return " ".join(tokens)
 
 
def preparer_dataset():
    """Charge, nettoie et prépare le dataset multimodal."""
    df = pd.read_csv(DATA_RAW)
 
    # Nettoyage Module 1
    def clip_iqr(s, f=3.0):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return s.clip(q1 - f * (q3 - q1), q3 + f * (q3 - q1))
 
    for col in NUM_COLS:
        df[col] = clip_iqr(df[col])
 
    knn = KNNImputer(n_neighbors=5)
    df[NUM_COLS] = knn.fit_transform(df[NUM_COLS])
    df["Densite"]        = df["Poids"] / (df["Volume"] + 1e-9)
    df["Cond_Rig_Ratio"] = df["Conductivite"] / (df["Rigidite"] + 1e-9)
    df = pd.get_dummies(df, columns=["Source"], drop_first=False, prefix="Source")
    df["texte_clean"] = df["Rapport_Collecte"].fillna("").apply(pretraiter_nlp)
 
    # Garder lignes avec label + texte
    df_mm = df[df["Categorie"].notna() & (df["texte_clean"].str.strip() != "")].copy()
    df_mm = df_mm.reset_index(drop=True)
    return df_mm
 
 
def construire_pipeline_sklearn(num_feats: list) -> Pipeline:
    """
    Pipeline sklearn reproductible via ColumnTransformer.
    Garantit la compatibilité avec dvc repro.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            ("nlp", TfidfVectorizer(
                max_features=10000, min_df=2, max_df=0.95,
                ngram_range=(1, 2), sublinear_tf=True,
            ), "texte_clean"),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("prep", preprocessor),
        ("clf",  LogisticRegression(max_iter=1000, C=5.0, random_state=SEED)),
    ])
 
 
def run():
    print("=" * 55)
    print("MODULE 5 – Pipeline Multimodal")
    print("=" * 55)
 
    df_mm = preparer_dataset()
    print(f"Dataset multimodal : {df_mm.shape}")
 
    num_feats  = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite",
                  "Densite", "Cond_Rig_Ratio"]
    src_feats  = [c for c in df_mm.columns if c.startswith("Source_")]
    num_all    = num_feats + src_feats
 
    le = LabelEncoder()
    y  = le.fit_transform(df_mm["Categorie"])
 
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=SEED)
    y_tr, y_val, y_te = y[idx_tr], y[idx_val], y[idx_te]
 
    # Matrices séparées
    scaler = StandardScaler()
    X_num_tr  = csr_matrix(scaler.fit_transform(df_mm[num_all].iloc[idx_tr]))
    X_num_val = csr_matrix(scaler.transform(df_mm[num_all].iloc[idx_val]))
    X_num_te  = csr_matrix(scaler.transform(df_mm[num_all].iloc[idx_te]))
 
    tfidf = TfidfVectorizer(max_features=10000, min_df=2, ngram_range=(1, 2), sublinear_tf=True)
    X_nlp_tr  = tfidf.fit_transform(df_mm["texte_clean"].iloc[idx_tr])
    X_nlp_val = tfidf.transform(df_mm["texte_clean"].iloc[idx_val])
    X_nlp_te  = tfidf.transform(df_mm["texte_clean"].iloc[idx_te])
 
    X_mm_tr  = hstack([X_num_tr,  X_nlp_tr])
    X_mm_val = hstack([X_num_val, X_nlp_val])
    X_mm_te  = hstack([X_num_te,  X_nlp_te])
 
    print(f"Matrice multimodale : {X_mm_tr.shape}")
 
    # Comparaison des modèles
    configs = {
        "Num_seul_SVC":  (X_num_tr, X_num_val, LinearSVC(max_iter=3000, random_state=SEED)),
        "NLP_seul_SVC":  (X_nlp_tr, X_nlp_val, LinearSVC(max_iter=3000, random_state=SEED)),
        "MM_SVC":        (X_mm_tr,  X_mm_val,  LinearSVC(max_iter=3000, random_state=SEED)),
        "MM_LogReg":     (X_mm_tr,  X_mm_val,  LogisticRegression(max_iter=500, C=5.0, random_state=SEED)),
    }
 
    if MLFLOW_OK:
        mlflow.set_experiment("eco_smart_multimodal")
 
    best_acc, best_clf, best_name = 0, None, ""
 
    for name, (Xtr, Xval, clf) in configs.items():
        with (mlflow.start_run(run_name=name) if MLFLOW_OK else __import__("contextlib").nullcontext()):
            clf.fit(Xtr, y_tr)
            yp  = clf.predict(Xval)
            acc = accuracy_score(y_val, yp)
            f1  = f1_score(y_val, yp, average="weighted")
            print(f"  {name:22s} | Val Acc={acc:.4f} | F1={f1:.4f}")
            if MLFLOW_OK:
                mlflow.log_metric("val_accuracy", acc)
                mlflow.log_metric("val_f1", f1)
                mlflow.log_param("config", name)
            if acc > best_acc and "MM" in name:
                best_acc, best_clf, best_name = acc, clf, name
 
    # Évaluation finale
    if best_clf is None:
        best_clf  = LogisticRegression(max_iter=500, C=5.0, random_state=SEED)
        best_name = "MM_LogReg"
 
    best_clf.fit(X_mm_tr, y_tr)
    yp_te    = best_clf.predict(X_mm_te)
    test_acc = accuracy_score(y_te, yp_te)
    test_f1  = f1_score(y_te, yp_te, average="weighted")
 
    print(f"\nMeilleur multimodal : {best_name}")
    print(f"Test Accuracy : {test_acc:.4f}  {'OK' if test_acc >= 0.70 else 'FAIL'}")
    print(f"Test F1 (wtd) : {test_f1:.4f}")
    print(classification_report(y_te, yp_te, target_names=le.classes_))
 
    # Sauvegarde
    artefacts = {
        "classifier.pkl":    best_clf,
        "scaler.pkl":        scaler,
        "tfidf.pkl":         tfidf,
        "label_encoder.pkl": le,
    }
    for fname, obj in artefacts.items():
        with open(MODELS_DIR / fname, "wb") as f:
            pickle.dump(obj, f)
        print(f"  OK models/{fname}")
 
    # Sauvegarder aussi num_all pour l'API
    with open(MODELS_DIR / "num_features.json", "w") as f:
        json.dump({"num_all": num_all}, f)
 
    metrics = {
        "test_accuracy_mm":    round(test_acc, 4),
        "test_f1_weighted_mm": round(test_f1, 4),
        "best_mm_model":       best_name,
    }
    with open(METRICS_DIR / "mm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
 
    print("\nOK Pipeline multimodal sauvegardé")
 
 
if __name__ == "__main__":
    run()