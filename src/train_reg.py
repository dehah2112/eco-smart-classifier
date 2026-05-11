"""
train_reg.py - Module 2 : Régression (Prix_Revente)
Eco-Smart Classifier
"""
import json
import pickle
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
 
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
    return df
 
 
def preparer_xy_reg(df):
    exclude = ["Categorie", "Categorie_enc", "Prix_Revente", "Rapport_Collecte"]
    if "texte_clean" in df.columns:
        exclude.append("texte_clean")
    feat_cols = [c for c in df.columns if c not in exclude]
 
    df_reg = df[df["Prix_Revente"].notna()].copy()
    X = df_reg[feat_cols].values
    y = df_reg["Prix_Revente"].values
    return X, y, feat_cols
 
 
def run():
    print("=" * 55)
    print("MODULE 2 – Régression (Prix_Revente)")
    print("=" * 55)
 
    df = charger_donnees()
    X, y, feat_cols = preparer_xy_reg(df)
    print(f"Samples disponibles : {len(y)}")
 
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED
    )
    print(f"Train={len(y_tr)} | Val={len(y_val)} | Test={len(y_te)}")
 
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
 
    models = {
        "LinearReg":     LinearRegression(),
        "Ridge":         Ridge(alpha=1.0),
        "Lasso":         Lasso(alpha=0.1, max_iter=5000),
        "RandomForest":  RandomForestRegressor(n_estimators=100, random_state=SEED),
        "GradientBoost": GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    }
 
    if MLFLOW_OK:
        mlflow.set_experiment("eco_smart_regression")
 
    best_r2, best_model, best_name = -np.inf, None, ""
 
    for name, model in models.items():
        with (mlflow.start_run(run_name=name) if MLFLOW_OK else __import__("contextlib").nullcontext()):
            model.fit(X_tr, y_tr)
            yp_val = model.predict(X_val)
            mae = mean_absolute_error(y_val, yp_val)
            r2  = r2_score(y_val, yp_val)
            print(f"  {name:20s} | Val R²={r2:.4f} | MAE={mae:.2f}")
            if MLFLOW_OK:
                mlflow.log_metric("val_r2",  r2)
                mlflow.log_metric("val_mae", mae)
                try:
                    mlflow.sklearn.log_model(model, artifact_path="model")
                except Exception:
                    pass
            if r2 > best_r2:
                best_r2, best_model, best_name = r2, model, name
 
    # Tuning RF Regressor
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [None, 10, 20],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=SEED),
        param_grid, cv=cv, scoring="r2", n_jobs=-1
    )
    grid.fit(X_tr, y_tr)
    yp_tuned = grid.best_estimator_.predict(X_val)
    r2_tuned = r2_score(y_val, yp_tuned)
    print(f"  RF_tuned              | Val R²={r2_tuned:.4f} | Params={grid.best_params_}")
 
    if r2_tuned > best_r2:
        best_r2   = r2_tuned
        best_model = grid.best_estimator_
        best_name  = "RF_tuned"
 
    # Évaluation finale
    best_model.fit(X_tr, y_tr)
    yp_te  = best_model.predict(X_te)
    test_r2   = r2_score(y_te, yp_te)
    test_mae  = mean_absolute_error(y_te, yp_te)
    test_rmse = np.sqrt(mean_squared_error(y_te, yp_te))
 
    print(f"\nMeilleur modèle : {best_name}")
    print(f"Test R²   : {test_r2:.4f}")
    print(f"Test MAE  : {test_mae:.2f} €")
    print(f"Test RMSE : {test_rmse:.2f} €")
 
    # Sauvegarde
    with open(MODELS_DIR / "regressor.pkl", "wb") as f:
        pickle.dump(best_model, f)
 
    metrics = {
        "test_r2":   round(test_r2, 4),
        "test_mae":  round(test_mae, 2),
        "test_rmse": round(test_rmse, 2),
        "best_model": best_name,
    }
    with open(METRICS_DIR / "reg_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
 
    print(f"\nOK models/regressor.pkl sauvegardé")
 
 
if __name__ == "__main__":
    run()