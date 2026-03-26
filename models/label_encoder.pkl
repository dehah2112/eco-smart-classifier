# === MODULE 1 — EDA & NETTOYAGE (version finale corrigée) ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import joblib
import os
import warnings
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

warnings.filterwarnings('ignore')
os.makedirs('data/processed', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ====================== ÉTAPE 1 — CHARGEMENT & EDA ======================
num_cols = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite', 'Prix_Revente']

df = pd.read_csv("data/raw/dataset_ProjetML_2026.csv")
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df['Categorie'].value_counts(dropna=False))
print(df.describe())

# Distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), num_cols):
    ax.hist(df[col].dropna(), bins=50, color='steelblue', alpha=0.7)
    ax.set_title(col)
plt.suptitle("Distributions des variables numériques")
plt.tight_layout()
plt.savefig("reports/distributions.png", dpi=150)

# Boxplots par catégorie
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for ax, col in zip(axes, ['Poids','Volume','Conductivite','Opacite','Rigidite']):
    df.boxplot(column=col, by='Categorie', ax=ax)
    ax.set_title(col)
plt.suptitle("Boxplots par catégorie")
plt.tight_layout()
plt.savefig("reports/boxplots_by_category.png", dpi=150)

# Valeurs manquantes
msno.matrix(df)
plt.savefig("reports/missing_matrix.png", dpi=150)

# Corrélations
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png", dpi=150)

# ====================== ÉTAPE 2 — NETTOYAGE ABERRATIONS ======================
df_clean = df.copy()

df_clean.loc[df_clean['Poids'] <= 0, 'Poids'] = np.nan
df_clean.loc[df_clean['Volume'] <= 0, 'Volume'] = np.nan
df_clean.loc[df_clean['Prix_Revente'] <= 0, 'Prix_Revente'] = np.nan
df_clean.loc[df_clean['Poids'] > 500, 'Poids'] = np.nan
df_clean.loc[df_clean['Prix_Revente'] > 1000, 'Prix_Revente'] = np.nan

print("NaN après nettoyage physique:\n", df_clean.isnull().sum())

# ====================== ÉTAPE 3 — DIAGNOSTIC MCAR/MAR/MNAR ======================
missing_mask = df_clean[num_cols].isnull().astype(int)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(missing_mask.corr(), annot=True, fmt=".2f", ax=ax)
ax.set_title("Corrélation entre patterns de valeurs manquantes")
plt.tight_layout()
plt.savefig("reports/missing_correlation.png", dpi=150)

# Test de Little (MCAR)
try:
    from pyampute.exploration.mcar_statistical_tests import MCARTest
    mcar = MCARTest(method="little")
    p_value = mcar.little_mcar_test(df_clean[num_cols])
    print(f"Little MCAR test p-value: {p_value:.4f}")
    print("→ MCAR rejeté" if p_value < 0.05 else "→ Compatible avec MCAR")
except ImportError:
    print("pyampute non installé — diagnostic visuel uniquement")

# MAR : NaN de Poids liés à Source ?
print("\nTaux de NaN par Source:")
print(df_clean.groupby('Source')['Poids'].apply(lambda x: x.isnull().mean()).round(3))

# ====================== ÉTAPE 4 — COMPARAISON STRATÉGIES IMPUTATION ======================
df_labeled_raw = df_clean[df_clean['Categorie'].notna()].copy()
X_eval = df_labeled_raw[num_cols].copy()

le_eval = LabelEncoder()
y_eval = le_eval.fit_transform(df_labeled_raw['Categorie'])

imputers = {
    'Médiane':          SimpleImputer(strategy='median'),
    'KNN (k=5)':        KNNImputer(n_neighbors=5),
    'IterativeImputer': IterativeImputer(max_iter=10, random_state=42),
}

imp_results = {}
for name, imp in imputers.items():
    X_imp = pd.DataFrame(imp.fit_transform(X_eval), columns=num_cols)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    scores = cross_val_score(clf, X_imp, y_eval, cv=5, scoring='accuracy')
    imp_results[name] = scores.mean()
    print(f"{name:20s}: accuracy = {scores.mean():.4f} ± {scores.std():.4f}")

best_imp_name = max(imp_results, key=imp_results.get)
print(f"\n✅ Meilleure stratégie : {best_imp_name}")

# Visualisation distributions après imputation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, imp) in zip(axes, imputers.items()):
    X_imp = pd.DataFrame(imp.fit_transform(X_eval), columns=num_cols)
    ax.hist(X_imp['Poids'], bins=50, alpha=0.7)
    ax.set_title(name)
plt.suptitle("Distribution de Poids après imputation")
plt.tight_layout()
plt.savefig("reports/imputation_comparison.png", dpi=150)

# ====================== IMPUTATION FINALE SUR df_clean ======================
# Appliquer KNN (meilleure stratégie) sur tout df_clean
imp_final = KNNImputer(n_neighbors=5)
df_clean[num_cols] = imp_final.fit_transform(df_clean[num_cols])
print("NaN restants après imputation finale:", df_clean[num_cols].isnull().sum().sum())

# ====================== ÉTAPE 5 — CAPPING OUTLIERS IQR ======================
def cap_outliers_iqr(df, col, factor=3.0):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
    n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"{col}: {n_capped} valeurs cappées → [{lower:.2f}, {upper:.2f}]")
    return df

for col in ['Poids', 'Volume', 'Prix_Revente']:
    df_clean = cap_outliers_iqr(df_clean, col, factor=3.0)

# ====================== ÉTAPE 6 — ENCODAGE ======================
# Encodage Source
source_categories = sorted(df_clean['Source'].dropna().unique().tolist())
oe = OrdinalEncoder(categories=[source_categories],
                    handle_unknown='use_encoded_value', unknown_value=-1)
df_clean['Source_enc'] = oe.fit_transform(df_clean[['Source']])

# Encodage cible : fit UNIQUEMENT sur les 4 classes réelles
le = LabelEncoder()
le.fit(df_clean['Categorie'].dropna())
df_clean.loc[df_clean['Categorie'].notna(), 'Categorie_enc'] = \
    le.transform(df_clean.loc[df_clean['Categorie'].notna(), 'Categorie'])

print("Classes LabelEncoder:", le.classes_)  # ['Métal','Papier','Plastique','Verre']

# Sauvegarde encoders (PAS le scaler — il sera fitté après le split)
joblib.dump(oe, 'models/ordinal_encoder.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

# ====================== ÉTAPE 7 — SPLIT 70/15/15 ======================
df_labeled   = df_clean[df_clean['Categorie'].notna()].copy()
df_unlabeled = df_clean[df_clean['Categorie'].isna()].copy()

# Colonnes features (sans target, sans texte brut, sans Source string)
feature_cols = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite',
                'Prix_Revente', 'Source_enc']

X = df_labeled[feature_cols].copy()
y = df_labeled['Categorie_enc'].astype(int).copy()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Standardisation APRÈS le split (pas de data leakage)
scaler = StandardScaler()
X_train[num_cols[:-1]] = scaler.fit_transform(X_train[num_cols[:-1]])  # sans Prix_Revente
X_val[num_cols[:-1]]   = scaler.transform(X_val[num_cols[:-1]])
X_test[num_cols[:-1]]  = scaler.transform(X_test[num_cols[:-1]])

joblib.dump(scaler, 'models/scaler.pkl')

# Vérification stratification
for name, y_s in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    print(f"\n{name}:", pd.Series(y_s).value_counts(normalize=True).round(3).to_dict())

# Sauvegarde
X_train.to_csv('data/processed/X_train.csv', index=False)
X_val.to_csv('data/processed/X_val.csv',     index=False)
X_test.to_csv('data/processed/X_test.csv',   index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_val.to_csv('data/processed/y_val.csv',     index=False)
y_test.to_csv('data/processed/y_test.csv',   index=False)
df_unlabeled.to_csv('data/processed/X_unlabeled.csv', index=False)

print("\n✅ Module 1 terminé — tous les fichiers sauvegardés dans data/processed/ et models/")