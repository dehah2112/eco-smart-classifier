# === MODULE 5 — PIPELINE MULTIMODAL (version finale corrigée) ===
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import warnings
from mlflow.tracking import MlflowClient
from spacy.lang.fr.stop_words import STOP_WORDS as SPACY_STOPWORDS
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import (f1_score, classification_report,
                             ConfusionMatrixDisplay)
warnings.filterwarnings('ignore')

# ====================== STOPWORDS (cohérent avec Module 4) ======================
DOMAIN_STOPWORDS = {
    "déchet", "matériau", "collecte", "objet", "article",
    "type", "matière", "élément", "produit", "item", "lot"
}
ALL_STOPWORDS = SPACY_STOPWORDS.union(DOMAIN_STOPWORDS)

# ====================== CHARGEMENT ======================
X_train_num = pd.read_csv('data/processed/X_train.csv')
X_val_num   = pd.read_csv('data/processed/X_val.csv')
X_test_num  = pd.read_csv('data/processed/X_test.csv')

y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_val   = pd.read_csv('data/processed/y_val.csv').squeeze()
y_test  = pd.read_csv('data/processed/y_test.csv').squeeze()

# Indices sauvegardés depuis Module 1
train_idx = pd.read_csv('data/processed/train_idx.csv').squeeze().values
val_idx   = pd.read_csv('data/processed/val_idx.csv').squeeze().values
test_idx  = pd.read_csv('data/processed/test_idx.csv').squeeze().values

# Texte lemmatisé depuis Module 4
df_nlp = pd.read_csv('data/processed/dataset_with_nlp.csv')
df_nlp_labeled = df_nlp[df_nlp['Categorie'].notna()].reset_index(drop=True)

X_train_text = df_nlp_labeled.loc[train_idx, 'text_processed'].reset_index(drop=True)
X_val_text   = df_nlp_labeled.loc[val_idx,   'text_processed'].reset_index(drop=True)
X_test_text  = df_nlp_labeled.loc[test_idx,  'text_processed'].reset_index(drop=True)

print(f"Train: {len(X_train_num)} | Val: {len(X_val_num)} | Test: {len(X_test_num)}")
assert len(X_train_text) == len(X_train_num), "Désalignement train text/num !"
assert len(X_val_text)   == len(X_val_num),   "Désalignement val text/num !"
assert len(X_test_text)  == len(X_test_num),  "Désalignement test text/num !"

# ====================== CONSTRUCTION DATAFRAMES COMBINÉS ======================
NUM_COLS = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
TEXT_COL = 'text_processed'

def build_combined(X_num, X_text):
    """Fusionne features numériques et textuelles en un seul DataFrame"""
    df = X_num[NUM_COLS].copy().reset_index(drop=True)
    df[TEXT_COL] = X_text.values
    return df

X_train_combined = build_combined(X_train_num, X_train_text)
X_val_combined   = build_combined(X_val_num,   X_val_text)
X_test_combined  = build_combined(X_test_num,  X_test_text)

# ====================== COLUMNTRANSFORMER ======================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler(with_mean=False))
        ]), NUM_COLS),
        ('text', TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.90,
            sublinear_tf=True,
            stop_words=list(ALL_STOPWORDS)
        ), TEXT_COL)
    ],
    remainder='drop',
    n_jobs=-1
)

# ====================== PIPELINES MULTIMODAUX ======================
multimodal_pipelines = {
    "MM_RandomForest": Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1))
    ]),
    "MM_XGBoost": Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='mlogloss', verbosity=0))
    ]),
    "MM_LinearSVC": Pipeline([
        ('prep', preprocessor),
        ('clf', LinearSVC(C=1.0, max_iter=3000, random_state=42))
    ]),
}

# ====================== ENTRAÎNEMENT + VALIDATION ======================
mlflow.set_experiment("eco-smart-multimodal")

mm_results = {}
for name, pipe in multimodal_pipelines.items():
    with mlflow.start_run(run_name=name):
        pipe.fit(X_train_combined, y_train)
        y_pred = pipe.predict(X_val_combined)
        f1     = f1_score(y_val, y_pred, average='macro')
        mm_results[name] = f1

        mlflow.log_param("model_type",        name)
        mlflow.log_param("num_features",      len(NUM_COLS))
        mlflow.log_param("tfidf_max_features", 8000)
        mlflow.log_metric("val_f1_macro",     f1)
        mlflow.sklearn.log_model(pipe, "pipeline")
        print(f"{name:20s} → F1-macro = {f1:.4f}")

# ====================== ANALYSE COMPARATIVE ======================
# ⚠️ Remplacer par vos scores réels des modules précédents
best_num_f1 = 0.91   # XGBoost tuné — Module 2
best_nlp_f1 = 0.87   # meilleur NLP — Module 4

print("\n=== ANALYSE COMPARATIVE : Numérique vs NLP vs Multimodal ===")
comparison = {
    "Numérique seul (XGBoost tuné)": best_num_f1,
    "NLP seul (meilleur M4)":        best_nlp_f1,
    **mm_results
}
for model, score in sorted(comparison.items(),
                            key=lambda x: x[1], reverse=True):
    delta = score - best_num_f1
    sign  = "+" if delta >= 0 else ""
    print(f"  {model:35s} → {score:.4f}  ({sign}{delta:.4f} vs num seul)")

# ====================== MEILLEUR MODÈLE — TEST SET ======================
best_name = max(mm_results, key=mm_results.get)
best_pipe = multimodal_pipelines[best_name]

y_test_pred = best_pipe.predict(X_test_combined)
f1_test     = f1_score(y_test, y_test_pred, average='macro')

print(f"\n=== Évaluation FINALE — {best_name} (test set) ===")
print(classification_report(y_test, y_test_pred,
      target_names=['Métal', 'Papier', 'Plastique', 'Verre']))

# Matrice de confusion
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred,
    display_labels=['Métal', 'Papier', 'Plastique', 'Verre'],
    cmap='Blues', ax=ax)
ax.set_title(f"Matrice de confusion — {best_name} (test set)")
plt.tight_layout()
plt.savefig("reports/multimodal_confusion_matrix.png", dpi=150)
plt.close()

# ====================== MLFLOW REGISTRY ======================
with mlflow.start_run(run_name=f"{best_name}_FINAL_TEST"):
    mlflow.log_param("model_type", best_name)
    mlflow.log_metric("test_f1_macro", f1_test)
    mlflow.sklearn.log_model(
        best_pipe, "pipeline",
        registered_model_name="eco-smart-multimodal-pipeline"
    )
    mlflow.set_tag("production_candidate", "true")

client = MlflowClient()
client.transition_model_version_stage(
    name="eco-smart-multimodal-pipeline",
    version=1,
    stage="Production"
)

# ====================== SAUVEGARDE ======================
joblib.dump(best_pipe, "models/best_multimodal_pipeline.pkl")
print(f"\n✅ Meilleur pipeline ({best_name}) sauvegardé")
print("✅ Module 5 terminé — artefacts dans reports/, models/ et MLflow Registry")