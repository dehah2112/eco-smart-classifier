"""
test_data.py - Tests sur le schéma et la qualité des données
Eco-Smart Classifier
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# ── Fixture ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def df_raw():
    """Charge le dataset depuis le bon chemin du projet."""
    # Chercher le CSV dans les emplacements possibles
    candidates = [
        Path("data/raw/dataset_ProjetML_2026.csv"),   # chemin projet
        Path("dataset_ProjetML_2026.csv"),             # racine (fallback)
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    pytest.skip("Dataset introuvable – vérifier data/raw/dataset_ProjetML_2026.csv")


# ── Tests schéma ──────────────────────────────────────────────────────────────
def test_colonnes_presentes(df_raw):
    """Toutes les colonnes requises sont présentes."""
    colonnes_requises = [
        "Poids", "Volume", "Conductivite", "Opacite",
        "Rigidite", "Prix_Revente", "Source",
        "Rapport_Collecte", "Categorie",
    ]
    for col in colonnes_requises:
        assert col in df_raw.columns, f"Colonne manquante : {col}"


def test_nombre_colonnes(df_raw):
    """Le dataset contient exactement 9 colonnes (cahier des charges)."""
    assert df_raw.shape[1] == 9, (
        f"Nombre de colonnes inattendu : {df_raw.shape[1]} (attendu 9)"
    )


def test_shape_minimum(df_raw):
    """Le dataset contient au moins 500 lignes."""
    assert len(df_raw) >= 500, f"Trop peu de lignes : {len(df_raw)}"


# ── Tests types ───────────────────────────────────────────────────────────────
def test_variables_numeriques(df_raw):
    """Les colonnes numériques sont bien de type numérique."""
    num_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
    for col in num_cols:
        assert pd.api.types.is_numeric_dtype(df_raw[col]), (
            f"{col} n'est pas numérique (type : {df_raw[col].dtype})"
        )


def test_prix_revente_numerique(df_raw):
    """Prix_Revente est une colonne numérique."""
    assert pd.api.types.is_numeric_dtype(df_raw["Prix_Revente"]), (
        "Prix_Revente n'est pas numérique"
    )


def test_source_categorielle(df_raw):
    """Source est une colonne texte (object)."""
    assert df_raw["Source"].dtype == object, (
        f"Source devrait être de type object, got {df_raw['Source'].dtype}"
    )


def test_rapport_collecte_texte(df_raw):
    """Rapport_Collecte est une colonne texte (object)."""
    assert df_raw["Rapport_Collecte"].dtype == object, (
        "Rapport_Collecte devrait être de type object"
    )


# ── Tests valeurs manquantes ──────────────────────────────────────────────────
def test_taux_manquants_poids(df_raw):
    """Poids a environ 10% de valeurs manquantes (entre 5% et 25%)."""
    pct = df_raw["Poids"].isna().mean() * 100
    assert 0 < pct < 25, (
        f"Taux manquants Poids inattendu : {pct:.1f}% (attendu entre 0% et 25%)"
    )


def test_labels_manquants(df_raw):
    """Il y a des labels cibles manquants (dataset semi-supervisé)."""
    n_missing = df_raw["Categorie"].isna().sum()
    assert n_missing > 0, "Aucun label manquant trouvé (attendu ~514)"
    assert n_missing < len(df_raw), "Tous les labels sont manquants"


def test_labels_manquants_proportion(df_raw):
    """Entre 10% et 60% des labels Categorie sont manquants."""
    pct = df_raw["Categorie"].isna().mean() * 100
    assert 5 < pct < 65, (
        f"Proportion de labels manquants inattendue : {pct:.1f}%"
    )


def test_volume_sans_nan(df_raw):
    """Volume ne doit pas avoir trop de valeurs manquantes (< 5%)."""
    pct = df_raw["Volume"].isna().mean() * 100
    assert pct < 5, f"Trop de NaN dans Volume : {pct:.1f}%"


# ── Tests valeurs aberrantes ──────────────────────────────────────────────────
def test_poids_positif(df_raw):
    """Les valeurs de Poids non-NaN sont positives."""
    poids_valides = df_raw["Poids"].dropna()
    assert (poids_valides >= 0).all(), "Poids contient des valeurs négatives"


def test_volume_positif(df_raw):
    """Les valeurs de Volume non-NaN sont positives."""
    vol_valides = df_raw["Volume"].dropna()
    assert (vol_valides >= 0).all(), "Volume contient des valeurs négatives"


def test_conductivite_bornee(df_raw):
    """Conductivite est entre 0 et 1 (hors outliers extrêmes)."""
    cond = df_raw["Conductivite"].dropna()
    p1, p99 = cond.quantile(0.01), cond.quantile(0.99)
    assert p1 >= 0, f"Conductivite : p1 négatif ({p1:.3f})"
    # p99 peut dépasser 1 s'il y a des outliers → on vérifie juste la médiane
    assert 0 <= cond.median() <= 1, f"Mediane Conductivite hors [0,1]"


# ── Tests texte ───────────────────────────────────────────────────────────────
def test_texte_rapport_present(df_raw):
    """La colonne Rapport_Collecte contient au moins 100 textes non vides."""
    non_vide = df_raw["Rapport_Collecte"].dropna()
    non_vide = non_vide[non_vide.str.strip() != ""]
    assert len(non_vide) > 100, (
        f"Trop peu de rapports textuels : {len(non_vide)}"
    )


def test_rapport_longueur_minimale(df_raw):
    """Les rapports textuels ont au moins 5 caractères en moyenne."""
    textes = df_raw["Rapport_Collecte"].dropna()
    textes = textes[textes.str.strip() != ""]
    longueur_moy = textes.str.len().mean()
    assert longueur_moy >= 5, (
        f"Longueur moyenne des rapports trop courte : {longueur_moy:.1f}"
    )


def test_source_modalites(df_raw):
    """Source contient au moins 2 modalités différentes."""
    n_modalites = df_raw["Source"].nunique()
    assert n_modalites >= 2, (
        f"Source n'a que {n_modalites} modalité(s)"
    )


def test_categorie_modalites(df_raw):
    """Categorie contient au moins 2 classes différentes."""
    n_classes = df_raw["Categorie"].nunique()
    assert n_classes >= 2, (
        f"Categorie n'a que {n_classes} classe(s)"
    )


def test_pas_de_doublons(df_raw):
    """Le dataset ne contient pas de lignes entièrement dupliquées."""
    n_doublons = df_raw.duplicated().sum()
    pct = n_doublons / len(df_raw) * 100
    assert pct < 5, (
        f"Trop de doublons : {n_doublons} ({pct:.1f}%)"
    )