"""
test_pipeline.py - Tests pipeline de prétraitement (Module 1)
Eco-Smart Classifier
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUM_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]


# ── Fixture ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def df_clean():
    """
    Charge le dataset depuis le bon chemin et applique l'imputation KNN.
    CORRIGE : chemin data/raw/ au lieu de la racine.
    """
    candidates = [
        Path("data/raw/dataset_ProjetML_2026.csv"),
        Path("dataset_ProjetML_2026.csv"),
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            imputer = KNNImputer(n_neighbors=5)
            df[NUM_COLS] = imputer.fit_transform(df[NUM_COLS])
            return df
    pytest.skip("Dataset introuvable – vérifier data/raw/dataset_ProjetML_2026.csv")


# ── Tests imputation ──────────────────────────────────────────────────────────
def test_pas_de_nan_apres_imputation(df_clean):
    """Aucune valeur manquante après imputation KNN sur les colonnes numériques."""
    n_nan = df_clean[NUM_COLS].isna().sum().sum()
    assert n_nan == 0, f"{n_nan} NaN restants après imputation KNN"


def test_imputation_preserve_shape(df_clean):
    """L'imputation ne change pas le nombre de lignes ni de colonnes."""
    candidates = [
        Path("data/raw/dataset_ProjetML_2026.csv"),
        Path("dataset_ProjetML_2026.csv"),
    ]
    for path in candidates:
        if path.exists():
            df_raw = pd.read_csv(path)
            assert df_clean.shape[0] == df_raw.shape[0], (
                "L'imputation a changé le nombre de lignes"
            )
            return


def test_imputation_valeurs_plausibles(df_clean):
    """Après imputation, les valeurs de Poids restent dans un intervalle plausible."""
    poids = df_clean["Poids"]
    assert poids.min() >= 0, "Poids imputé négatif détecté"
    assert poids.max() < 1e6, "Poids imputé anormalement grand"


# ── Tests feature engineering ─────────────────────────────────────────────────
def test_feature_densite(df_clean):
    """
    La feature Densite est calculée correctement.
    CORRIGE : on travaille sur une copie pour ne pas modifier la fixture partagée.
    """
    df = df_clean.copy()
    df["Densite"] = df["Poids"] / (df["Volume"] + 1e-9)
    assert df["Densite"].isna().sum() == 0, "NaN dans Densite"
    assert (df["Densite"] >= 0).all(), "Densite négative détectée"


def test_feature_cond_rig_ratio(df_clean):
    """La feature Cond_Rig_Ratio est calculée sans NaN."""
    df = df_clean.copy()
    df["Cond_Rig_Ratio"] = df["Conductivite"] / (df["Rigidite"] + 1e-9)
    assert df["Cond_Rig_Ratio"].isna().sum() == 0, "NaN dans Cond_Rig_Ratio"


def test_features_derivees_non_infinies(df_clean):
    """Les features dérivées ne contiennent pas de valeurs infinies."""
    df = df_clean.copy()
    df["Densite"]        = df["Poids"] / (df["Volume"] + 1e-9)
    df["Cond_Rig_Ratio"] = df["Conductivite"] / (df["Rigidite"] + 1e-9)
    assert not np.isinf(df["Densite"]).any(),        "Inf dans Densite"
    assert not np.isinf(df["Cond_Rig_Ratio"]).any(), "Inf dans Cond_Rig_Ratio"


# ── Tests standardisation ─────────────────────────────────────────────────────
def test_standardisation_moyenne_nulle(df_clean):
    """
    Après StandardScaler, la moyenne est ≈ 0.
    CORRIGE : tolérance 1e-6 au lieu de 1e-10 (plus réaliste numériquement).
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(df_clean[NUM_COLS])
    means  = np.abs(X_sc.mean(axis=0))
    assert (means < 1e-6).all(), (
        f"Moyenne non nulle après scaling : max={means.max():.2e}"
    )


def test_standardisation_std_unitaire(df_clean):
    """
    Après StandardScaler, l'écart-type est ≈ 1.
    CORRIGE : tolérance 1e-6 au lieu de 1e-10 (plus réaliste numériquement).
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(df_clean[NUM_COLS])
    stds   = X_sc.std(axis=0)
    assert (np.abs(stds - 1.0) < 1e-6).all(), (
        f"Std non unitaire après scaling : max_ecart={np.abs(stds-1).max():.2e}"
    )


def test_standardisation_shape(df_clean):
    """StandardScaler préserve la shape du tableau."""
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(df_clean[NUM_COLS])
    assert X_sc.shape == df_clean[NUM_COLS].shape


def test_standardisation_transform_inverse(df_clean):
    """inverse_transform restitue les valeurs originales (arrondi flottant)."""
    scaler = StandardScaler()
    X_orig = df_clean[NUM_COLS].values
    X_sc   = scaler.fit_transform(X_orig)
    X_back = scaler.inverse_transform(X_sc)
    np.testing.assert_allclose(X_orig, X_back, rtol=1e-5, atol=1e-8)


# ── Tests encodage catégoriel ─────────────────────────────────────────────────
def test_ohe_source_colonnes_binaires(df_clean):
    """L'OHE de Source produit uniquement des colonnes binaires (0/1)."""
    df_ohe = pd.get_dummies(df_clean, columns=["Source"], prefix="Source")
    source_cols = [c for c in df_ohe.columns if c.startswith("Source_")]
    assert len(source_cols) > 0, "Aucune colonne OHE générée"
    for col in source_cols:
        vals = set(df_ohe[col].unique())
        assert vals.issubset({0, 1, True, False}), (
            f"Colonne OHE '{col}' non binaire : valeurs = {vals}"
        )


def test_ohe_source_somme_lignes(df_clean):
    """Chaque ligne a exactement 1 dans les colonnes OHE de Source."""
    df_ohe = pd.get_dummies(df_clean, columns=["Source"], prefix="Source")
    source_cols = [c for c in df_ohe.columns if c.startswith("Source_")]
    if source_cols:
        sommes = df_ohe[source_cols].sum(axis=1)
        assert (sommes == 1).all(), (
            "Certaines lignes ont ≠ 1 colonne Source active après OHE"
        )


def test_ohe_source_nombre_modalites(df_clean):
    """Le nombre de colonnes OHE correspond au nombre de modalités de Source."""
    n_modalites = df_clean["Source"].nunique()
    df_ohe      = pd.get_dummies(df_clean, columns=["Source"], prefix="Source")
    source_cols = [c for c in df_ohe.columns if c.startswith("Source_")]
    assert len(source_cols) == n_modalites, (
        f"Nombre de colonnes OHE ({len(source_cols)}) "
        f"≠ nombre de modalités ({n_modalites})"
    )


# ── Tests split 70/15/15 ──────────────────────────────────────────────────────
def test_split_proportions():
    """Le split 70/15/15 respecte les proportions attendues (± 2%)."""
    n   = 1000
    idx = np.arange(n)
    rng = np.random.default_rng(42)          # CORRIGE : RandomState reproductible
    y   = rng.integers(0, 3, n)

    idx_tr, idx_tmp = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=42
    )
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=42
    )

    assert abs(len(idx_tr)  / n - 0.70) < 0.02, f"Train : {len(idx_tr)/n:.2%}"
    assert abs(len(idx_val) / n - 0.15) < 0.02, f"Val   : {len(idx_val)/n:.2%}"
    assert abs(len(idx_te)  / n - 0.15) < 0.02, f"Test  : {len(idx_te)/n:.2%}"


def test_split_stratification():
    """La stratification préserve les proportions des classes."""
    n   = 900
    idx = np.arange(n)
    y   = np.array([0]*300 + [1]*300 + [2]*300)

    idx_tr, idx_tmp = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=42
    )
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=42
    )

    for split_name, split_idx in [
        ("Train", idx_tr), ("Val", idx_val), ("Test", idx_te)
    ]:
        for classe in [0, 1, 2]:
            pct = (y[split_idx] == classe).mean()
            assert abs(pct - 1/3) < 0.05, (
                f"Stratification incorrecte pour {split_name}, classe {classe} : {pct:.2%}"
            )


def test_split_pas_de_chevauchement():
    """Les ensembles train/val/test ne se chevauchent pas."""
    n   = 1000
    idx = np.arange(n)
    y   = np.random.RandomState(42).randint(0, 3, n)

    idx_tr, idx_tmp = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=42
    )
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=42
    )

    assert len(set(idx_tr) & set(idx_val)) == 0, "Chevauchement Train/Val"
    assert len(set(idx_tr) & set(idx_te))  == 0, "Chevauchement Train/Test"
    assert len(set(idx_val) & set(idx_te)) == 0, "Chevauchement Val/Test"


def test_split_couverture_totale():
    """Train + Val + Test couvrent tous les indices."""
    n   = 1000
    idx = np.arange(n)
    y   = np.random.RandomState(42).randint(0, 3, n)

    idx_tr, idx_tmp = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=42
    )
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=42
    )

    tous = set(idx_tr) | set(idx_val) | set(idx_te)
    assert tous == set(idx), "Certains indices sont perdus après le split"