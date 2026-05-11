"""
train_clf.py - Module 2 : Classification (Categorie)
Eco-Smart Classifier
"""
import json
import pickle
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC
 
warnings.filterwarnings("ignore")
 
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False
 
DATA_PROCESSED = Path("data/processed")
MODELS_DIR     = Path("models")
METRICS_DIR    = Path("metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
 
 
def charger_donnees():
    with open(DATA_PROCESSED / "dataset_clean.pkl", "rb") as f:
        df = pickle.load(f)
    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return df, le
 
 
def preparer_xy(df, le):
    exclude = ["Categorie", "Categorie_enc", "Prix_Revente", "Rapport_Collecte"]
    if "texte_clean" in df.columns:
        exclude.append("texte_clean")
    feat_cols = [c for c in df.columns if c not in exclude]
 
    df_labeled = df[df["Categorie"].notna()].copy()
    df_labeled["Categorie_enc"] = le.transform(df_labeled["Categorie"])
 
    X = df_labeled[feat_cols].values
    y = df_labeled["Categorie_enc"].values
    return X, y, feat_cols
 
 
def run():
    print("=" * 55)
    print("MODULE 2 – Classification")
    print("=" * 55)
 
    df, le = charger_donnees()
    X, y, feat_cols = preparer_xy(df, le)
 
    # Split 70/15/15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    print(f"Train={len(y_tr)} | Val={len(y_val)} | Test={len(y_te)}")
 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
 
    # Baseline
    models = {
        "LogReg":        LogisticRegression(max_iter=500, random_state=SEED),
        "RandomForest":  RandomForestClassifier(n_estimators=100, random_state=SEED),
        "GradientBoost": GradientBoostingClassifier(random_state=SEED),
        "LinearSVC":     LinearSVC(max_iter=2000, random_state=SEED),
    }
 
    if MLFLOW_OK:
        mlflow.set_experiment("eco_smart_classification")
 
    best_acc, best_model, best_name = 0, None, ""
 
    for name, model in models.items():
        with (mlflow.start_run(run_name=name) if MLFLOW_OK else __import__("contextlib").nullcontext()):
            model.fit(X_tr, y_tr)
            yp_val = model.predict(X_val)
            acc = accuracy_score(y_val, yp_val)
            f1  = f1_score(y_val, yp_val, average="weighted")
            print(f"  {name:20s} | Val Acc={acc:.4f} | F1={f1:.4f}")
            if MLFLOW_OK:
                mlflow.log_params(model.get_params())
                mlflow.log_metric("val_accuracy", acc)
                mlflow.log_metric("val_f1_weighted", f1)
                try:
                    mlflow.sklearn.log_model(model, artifact_path="model")
                except Exception:
                    pass
            if acc > best_acc:
                best_acc, best_model, best_name = acc, model, name
 
    # Tuning RF
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=SEED),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_tr, y_tr)
    yp_val_tuned = grid.best_estimator_.predict(X_val)
    acc_tuned = accuracy_score(y_val, yp_val_tuned)
    print(f"  RF_tuned              | Val Acc={acc_tuned:.4f} | Params={grid.best_params_}")
 
    if acc_tuned > best_acc:
        best_acc   = acc_tuned
        best_model = grid.best_estimator_
        best_name  = "RF_tuned"
 
    # Évaluation finale
    best_model.fit(X_tr, y_tr)
    yp_te = best_model.predict(X_te)
    test_acc = accuracy_score(y_te, yp_te)
    test_f1  = f1_score(y_te, yp_te, average="weighted")
 
    print(f"\nMeilleur modèle : {best_name}")
    print(f"Test Accuracy   : {test_acc:.4f}  {'OK' if test_acc >= 0.70 else 'FAIL'}")
    print(f"Test F1 (wtd)   : {test_f1:.4f}")
    print(classification_report(y_te, yp_te, target_names=le.classes_))
 
    # Sauvegarde
    with open(MODELS_DIR / "classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)
 
    metrics = {
        "test_accuracy":    round(test_acc, 4),
        "test_f1_weighted": round(test_f1, 4),
        "best_model":       best_name,
        "n_classes":        len(le.classes_),
    }
    with open(METRICS_DIR / "clf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
 
    print(f"\nOK models/classifier.pkl sauvegardé")
    print(f"OK metrics/clf_metrics.json sauvegardé")
 
 
if __name__ == "__main__":
    run()