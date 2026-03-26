# === MODULE 2 — ML SUPERVISÉ (version finale corrigée) ===
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import shap
import optuna
import joblib
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    ConfusionMatrixDisplay, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# ====================== CHARGEMENT ======================
X_train = pd.read_csv('data/processed/X_train.csv')
X_val   = pd.read_csv('data/processed/X_val.csv')
X_test  = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_val   = pd.read_csv('data/processed/y_val.csv').squeeze()
y_test  = pd.read_csv('data/processed/y_test.csv').squeeze()

# Features explicitement séparées
clf_features = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite', 'Source_enc']
reg_features = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite', 'Source_enc']

# Targets régression
y_prix_train = X_train['Prix_Revente'].copy()
y_prix_val   = X_val['Prix_Revente'].copy()
y_prix_test  = X_test['Prix_Revente'].copy()
y_train_log  = np.log1p(y_prix_train)
y_val_log    = np.log1p(y_prix_val)

assert 'Prix_Revente' not in reg_features, "Data leakage détecté !"

# ====================== PARTIE A — CLASSIFICATION ======================
mlflow.set_experiment("eco-smart-classification")

models = {
    "Dummy (baseline)":   DummyClassifier(strategy='most_frequent'),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost":            XGBClassifier(n_estimators=200, random_state=42,
                                        eval_metric='mlogloss', verbosity=0),
    "SVC":                SVC(kernel='rbf', probability=True, random_state=42),
}

results = {}
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train[clf_features], y_train)
        y_pred = model.predict(X_val[clf_features])
        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average='macro')
        results[name] = {'accuracy': acc, 'f1_macro': f1}
        mlflow.log_param("model_type", name)
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_macro", f1)
        mlflow.sklearn.log_model(model, "model")
        print(f"{name:25s} | acc={acc:.4f} | f1={f1:.4f}")

print("\n", pd.DataFrame(results).T.sort_values('f1_macro', ascending=False))

# ====================== TUNING OPTUNA ======================
def objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
        'eval_metric': 'mlogloss', 'verbosity': 0, 'random_state': 42,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train[clf_features], y_train,
                             cv=5, scoring='f1_macro', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

best_params = study.best_params
print("Meilleurs paramètres:", best_params)
print("Meilleur F1-macro CV:", study.best_value)

best_xgb = XGBClassifier(**best_params, eval_metric='mlogloss', verbosity=0)
best_xgb.fit(X_train[clf_features], y_train)
joblib.dump(best_xgb, 'models/best_classifier.pkl')

y_test_pred = best_xgb.predict(X_test[clf_features])
print("\n=== Évaluation FINALE classification (test set) ===")
print(classification_report(y_test, y_test_pred,
      target_names=['Métal', 'Papier', 'Plastique', 'Verre']))

# Matrice de confusion
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred,
    display_labels=['Métal', 'Papier', 'Plastique', 'Verre'],
    cmap='Blues', ax=ax)
ax.set_title("Matrice de confusion — XGBoost tuné (test set)")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()

# MLflow Registry
with mlflow.start_run(run_name="XGBoost_BEST_REGISTRY"):
    mlflow.log_params(best_params)
    mlflow.log_metric("test_f1_macro",
                      f1_score(y_test, y_test_pred, average='macro'))
    mlflow.sklearn.log_model(
        best_xgb, "model",
        registered_model_name="eco-smart-classifier"
    )

client = MlflowClient()
client.transition_model_version_stage(
    name="eco-smart-classifier", version=1, stage="Production")
print("✅ Modèle enregistré au MLflow Model Registry — Production")

# ====================== SHAP ======================
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_val[clf_features])

plt.figure()
shap.summary_plot(shap_values, X_val[clf_features], plot_type="bar",
                  class_names=['Métal','Papier','Plastique','Verre'], show=False)
plt.savefig("reports/shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_val[clf_features], plot_type="dot",
                  class_names=['Métal','Papier','Plastique','Verre'], show=False)
plt.savefig("reports/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()

# ====================== PARTIE B — RÉGRESSION ======================
mlflow.set_experiment("eco-smart-regression")

reg_models = {
    "Ridge":            Ridge(alpha=1.0),
    "Lasso":            Lasso(alpha=0.1),
    "SVR (RBF)":        SVR(kernel='rbf', C=10, epsilon=0.1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                   learning_rate=0.05, random_state=42),
    "RandomForest Reg": RandomForestRegressor(n_estimators=200, random_state=42),
}

reg_results = {}
trained_reg_models = {}

for name, model in reg_models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train[reg_features], y_train_log)
        y_pred_log  = model.predict(X_val[reg_features])
        y_pred_orig = np.expm1(y_pred_log)
        y_val_orig  = np.expm1(y_val_log)

        rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        mae  = mean_absolute_error(y_val_orig, y_pred_orig)
        r2   = r2_score(y_val_log, y_pred_log)

        reg_results[name] = {'RMSE': rmse, 'MAE': mae, 'R2_log': r2}
        trained_reg_models[name] = model

        mlflow.log_param("model_type", name)
        mlflow.log_param("log_transform", True)
        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_r2_log", r2)
        mlflow.sklearn.log_model(model, "model")
        print(f"{name:22s} | RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.4f}")

print("\n", pd.DataFrame(reg_results).T.sort_values('R2_log', ascending=False))

# ====================== MEILLEUR MODÈLE — TEST SET ======================
best_reg_name = max(reg_results, key=lambda k: reg_results[k]['R2_log'])
best_reg = trained_reg_models[best_reg_name]

y_test_log_pred  = best_reg.predict(X_test[reg_features])
y_test_orig_pred = np.expm1(y_test_log_pred)
y_test_orig      = np.expm1(np.log1p(y_prix_test))

rmse_test = np.sqrt(mean_squared_error(y_test_orig, y_test_orig_pred))
mae_test  = mean_absolute_error(y_test_orig, y_test_orig_pred)
r2_test   = r2_score(np.log1p(y_prix_test), y_test_log_pred)

print(f"\n=== Évaluation FINALE régression — {best_reg_name} (test set) ===")
print(f"RMSE={rmse_test:.2f} | MAE={mae_test:.2f} | R²={r2_test:.4f}")

joblib.dump(best_reg, 'models/best_regressor.pkl')

with mlflow.start_run(run_name=f"{best_reg_name}_FINAL_TEST"):
    mlflow.log_metric("test_rmse", rmse_test)
    mlflow.log_metric("test_mae",  mae_test)
    mlflow.log_metric("test_r2",   r2_test)
    mlflow.sklearn.log_model(best_reg, "model",
                             registered_model_name="eco-smart-regressor")

# ====================== ANALYSE RÉSIDUS ======================
y_pred_log_val = best_reg.predict(X_val[reg_features])
y_val_orig     = np.expm1(y_val_log)
y_pred_orig    = np.expm1(y_pred_log_val)
residuals      = y_val_log - y_pred_log_val

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(y_val_log, y_pred_log_val, alpha=0.3, s=10)
axes[0].plot([y_val_log.min(), y_val_log.max()],
             [y_val_log.min(), y_val_log.max()], 'r--')
axes[0].set_title("Predicted vs Actual (log)")

axes[1].hist(residuals, bins=50)
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title("Distribution des résidus")

axes[2].scatter(y_pred_log_val, residuals, alpha=0.3, s=10)
axes[2].axhline(0, color='red', linestyle='--')
axes[2].set_title("Résidus vs Prédictions")

plt.tight_layout()
plt.savefig("reports/residuals_analysis.png", dpi=150)
plt.close()

# ====================== TUNING RIDGE (GridSearch) ======================
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
gs = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2', n_jobs=-1)
gs.fit(X_train[reg_features], y_train_log)
print("Meilleur alpha Ridge:", gs.best_params_)
print("Meilleur R² CV:", gs.best_score_)

with mlflow.start_run(run_name="Ridge_GridSearch"):
    mlflow.log_params(gs.best_params_)
    mlflow.log_metric("cv_r2", gs.best_score_)
    mlflow.sklearn.log_model(gs.best_estimator_, "model")

print("\n✅ Module 2 terminé — modèles sauvegardés dans models/")