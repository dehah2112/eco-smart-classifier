# tests/test_data_schema.py (zone ROUGE — sans IA)
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def raw_df():
    return pd.read_csv("data/raw/dataset_ProjetML_2026.csv")

@pytest.fixture
def processed_df():
    return pd.read_csv("data/processed/X_train.csv")

def test_raw_shape(raw_df):
    assert raw_df.shape == (10500, 9)

def test_required_columns(raw_df):
    expected = ['Poids','Volume','Conductivite','Opacite','Rigidite',
                'Prix_Revente','Categorie','Source','Rapport_Collecte']
    assert list(raw_df.columns) == expected

def test_categorie_values(raw_df):
    valid  = {'Métal', 'Papier', 'Plastique', 'Verre'}
    actual = set(raw_df['Categorie'].dropna().unique())
    assert actual == valid

def test_categorie_missing_count(raw_df):
    """514 labels manquants selon le cahier des charges."""
    nan_count = raw_df['Categorie'].isnull().sum()
    assert 500 <= nan_count <= 530, \
        f"Nombre de NaN inattendu dans Categorie : {nan_count}"

def test_no_negative_after_processing(processed_df):
    num_cols = ['Poids','Volume','Conductivite','Opacite','Rigidite']
    for col in num_cols:
        if col in processed_df.columns:
            assert (processed_df[col] >= 0).all(), \
                f"{col} contient des valeurs négatives"

def test_no_nan_after_imputation(processed_df):
    assert processed_df.isnull().sum().sum() == 0

def test_rapport_collecte_exists(raw_df):
    assert 'Rapport_Collecte' in raw_df.columns

def test_rapport_collecte_nan_rate(raw_df):
    nan_rate = raw_df['Rapport_Collecte'].isnull().mean()
    assert nan_rate < 0.20, \
        f"Taux de NaN trop élevé dans Rapport_Collecte : {nan_rate:.1%}"

def test_split_sizes():
    """Vérifier les proportions 70/15/15."""
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val   = pd.read_csv('data/processed/X_val.csv')
    X_test  = pd.read_csv('data/processed/X_test.csv')
    total   = len(X_train) + len(X_val) + len(X_test)
    assert abs(len(X_train)/total - 0.70) < 0.02
    assert abs(len(X_val)/total   - 0.15) < 0.02
    assert abs(len(X_test)/total  - 0.15) < 0.02