"""
nlp_pipeline.py - Module 4 : Pipeline NLP (TF-IDF + Classification texte)
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
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
 
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
 
# ── Stopwords ─────────────────────────────────────────────────────────────────
try:
    _STOP_FR = set(stopwords.words("french"))
except Exception:
    _STOP_FR = set()
 
_STOP_DOMAIN = {
    "dechet", "dechets", "collecte", "rapport", "collecteur", "materiau",
    "materiaux", "echantillon", "analyse", "type", "lot", "reference",
    "code", "numero", "date", "heure", "kg", "litre", "cm", "mm",
    "metre", "unite", "valeur", "mesure", "resultat", "observation",
    "note", "commentaire", "traitement", "site", "zone", "point",
}
STOPWORDS_ALL = _STOP_FR | _STOP_DOMAIN
_stemmer = SnowballStemmer("french")
 
 
# ── Fonctions prétraitement ───────────────────────────────────────────────────
def nettoyer(texte: str) -> str:
    t = str(texte).lower()
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
 
 
def tokeniser(texte: str) -> list:
    try:
        return word_tokenize(texte, language="french")
    except Exception:
        return texte.split()
 
 
def pretraiter(texte: str, stemming: bool = True) -> str:
    """Pipeline complet : nettoyage → tokenisation → stopwords → stemming."""
    t      = nettoyer(texte)
    tokens = tokeniser(t)
    tokens = [tk for tk in tokens if tk not in STOPWORDS_ALL and len(tk) > 2]
    if stemming:
        tokens = [_stemmer.stem(tk) for tk in tokens]
    return " ".join(tokens)
 
 
# ── Pipeline NLP ──────────────────────────────────────────────────────────────
def construire_tfidf(max_features: int = 10000, ngram_range: tuple = (1, 2)):
    return TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        ngram_range=ngram_range,
        sublinear_tf=True,
        norm="l2",
    )
 
 
def run():
    print("=" * 55)
    print("MODULE 4 – Pipeline NLP")
    print("=" * 55)
 
    df = pd.read_csv(DATA_RAW)
    df_nlp = df[["Rapport_Collecte", "Categorie"]].dropna().copy()
    df_nlp = df_nlp[df_nlp["Rapport_Collecte"].str.strip() != ""].reset_index(drop=True)
    print(f"Textes utilisables : {len(df_nlp)}")
 
    # Prétraitement
    df_nlp["texte_clean"] = df_nlp["Rapport_Collecte"].apply(pretraiter)
 
    le = LabelEncoder()
    y  = le.fit_transform(df_nlp["Categorie"])
 
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=SEED)
 
    X_tr  = df_nlp["texte_clean"].iloc[idx_tr].values
    X_val = df_nlp["texte_clean"].iloc[idx_val].values
    X_te  = df_nlp["texte_clean"].iloc[idx_te].values
    y_tr, y_val, y_te = y[idx_tr], y[idx_val], y[idx_te]
 
    # Vectorisation TF-IDF
    tfidf    = construire_tfidf()
    X_tfidf_tr  = tfidf.fit_transform(X_tr)
    X_tfidf_val = tfidf.transform(X_val)
    X_tfidf_te  = tfidf.transform(X_te)
    print(f"TF-IDF : {X_tfidf_tr.shape}")
 
    classifiers = {
        "NaiveBayes":  ComplementNB(alpha=0.1),
        "LogReg":      LogisticRegression(max_iter=1000, C=5.0, random_state=SEED),
        "LinearSVC":   LinearSVC(max_iter=2000, C=1.0, random_state=SEED),
    }
 
    if MLFLOW_OK:
        mlflow.set_experiment("eco_smart_nlp")
 
    best_acc, best_clf, best_name = 0, None, ""
 
    for name, clf in classifiers.items():
        with (mlflow.start_run(run_name=f"TFIDF_{name}") if MLFLOW_OK else __import__("contextlib").nullcontext()):
            clf.fit(X_tfidf_tr, y_tr)
            yp  = clf.predict(X_tfidf_val)
            acc = accuracy_score(y_val, yp)
            f1  = f1_score(y_val, yp, average="weighted")
            print(f"  TFIDF+{name:12s} | Val Acc={acc:.4f} | F1={f1:.4f}")
            if MLFLOW_OK:
                mlflow.log_metric("val_accuracy", acc)
                mlflow.log_metric("val_f1_weighted", f1)
            if acc > best_acc:
                best_acc, best_clf, best_name = acc, clf, name
 
    # Évaluation finale
    best_clf.fit(X_tfidf_tr, y_tr)
    yp_te    = best_clf.predict(X_tfidf_te)
    test_acc = accuracy_score(y_te, yp_te)
    test_f1  = f1_score(y_te, yp_te, average="weighted")
 
    print(f"\nMeilleur NLP : TFIDF+{best_name} | Test Acc={test_acc:.4f}")
    print(classification_report(y_te, yp_te, target_names=le.classes_))
 
    # Sauvegarde
    for name, obj in [
        ("tfidf.pkl",         tfidf),
        ("nlp_classifier.pkl", best_clf),
        ("label_encoder.pkl", le),
    ]:
        with open(MODELS_DIR / name, "wb") as f:
            pickle.dump(obj, f)
 
    metrics = {
        "test_accuracy_nlp":    round(test_acc, 4),
        "test_f1_weighted_nlp": round(test_f1, 4),
        "best_nlp_model":       f"TFIDF+{best_name}",
    }
    with open(METRICS_DIR / "nlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
 
    print("OK tfidf.pkl, nlp_classifier.pkl, label_encoder.pkl sauvegardés")
 
 
if __name__ == "__main__":
    run()
 