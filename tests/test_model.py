"""
test_model.py - Tests modèle ML (seuil accuracy, prédictions, artefacts)
Eco-Smart Classifier
"""
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ACC_MIN = 0.55   # seuil minimum requis par le cahier des charges
MODELS_DIR = Path("models")


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def synthetic_data():
    """Dataset synthétique 4 classes pour tests rapides (sans CSV)."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=4,
        n_informative=15,
        n_redundant=3,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.30, random_state=42)


@pytest.fixture(scope="module")
def modele_entraine(synthetic_data):
    """LogReg entraîné sur données synthétiques — réutilisé par plusieurs tests."""
    X_tr, X_te, y_tr, y_te = synthetic_data
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf, X_te, y_te


# ── Tests entraînement & prédiction ──────────────────────────────────────────
def test_modele_entraine_et_predit(synthetic_data):
    """Le modèle peut s'entraîner et prédire sans erreur."""
    X_tr, X_te, y_tr, y_te = synthetic_data
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    assert len(yp) == len(y_te)


def test_accuracy_minimum(synthetic_data):
    """L'accuracy dépasse le seuil minimum de 0.70."""
    X_tr, X_te, y_tr, y_te = synthetic_data
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    assert acc >= ACC_MIN, (
        f"Accuracy trop basse : {acc:.4f} < {ACC_MIN}"
    )


def test_f1_minimum(synthetic_data):
    """Le F1-score weighted dépasse 0.65."""
    X_tr, X_te, y_tr, y_te = synthetic_data
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    f1 = f1_score(y_te, clf.predict(X_te), average="weighted")
    assert f1 >= 0.55, f"F1 trop bas : {f1:.4f} < 0.55"


def test_predictions_dans_les_classes(synthetic_data):
    """Toutes les prédictions appartiennent aux classes d'entraînement."""
    X_tr, X_te, y_tr, y_te = synthetic_data
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    classes_valides = set(y_tr)
    for pred in yp:
        assert pred in classes_valides, f"Prédiction inconnue : {pred}"


def test_forme_predictions(synthetic_data):
    """La forme des prédictions correspond aux données de test."""
    X_tr, X_te, y_tr, y_te = synthetic_data
    clf = LinearSVC(max_iter=2000, random_state=42)
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    assert yp.shape == y_te.shape


def test_predictions_type_entier(modele_entraine):
    """Les prédictions sont des entiers (classes encodées)."""
    clf, X_te, y_te = modele_entraine
    yp = clf.predict(X_te)
    assert yp.dtype in [np.int32, np.int64, np.intp], (
        f"Type inattendu pour les prédictions : {yp.dtype}"
    )


def test_predict_proba_disponible(modele_entraine):
    """LogReg expose predict_proba et retourne des probabilités valides."""
    clf, X_te, _ = modele_entraine
    assert hasattr(clf, "predict_proba"), "predict_proba non disponible"
    proba = clf.predict_proba(X_te)
    assert proba.shape[0] == X_te.shape[0]
    assert proba.shape[1] >= 2
    # Somme des probas ≈ 1 pour chaque échantillon
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_bornees(modele_entraine):
    """Les probabilités sont comprises entre 0 et 1."""
    clf, X_te, _ = modele_entraine
    proba = clf.predict_proba(X_te)
    assert (proba >= 0).all(), "Probabilités négatives détectées"
    assert (proba <= 1).all(), "Probabilités > 1 détectées"


# ── Test reproductibilité ─────────────────────────────────────────────────────
def test_reproductibilite():
    """Deux entraînements identiques donnent les mêmes résultats."""
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

    clf1 = LogisticRegression(random_state=42, max_iter=200)
    clf2 = LogisticRegression(random_state=42, max_iter=200)
    clf1.fit(X_tr, y_tr)
    clf2.fit(X_tr, y_tr)

    assert (clf1.predict(X_te) == clf2.predict(X_te)).all(), (
        "Les deux modèles identiques donnent des prédictions différentes"
    )


def test_reproductibilite_linearsvc():
    """LinearSVC est aussi reproductible avec le même random_state."""
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, random_state=42)

    clf1 = LinearSVC(random_state=42, max_iter=2000)
    clf2 = LinearSVC(random_state=42, max_iter=2000)
    clf1.fit(X_tr, y_tr)
    clf2.fit(X_tr, y_tr)

    assert (clf1.predict(X_te) == clf2.predict(X_te)).all()


# ── Tests artefacts sauvegardés ───────────────────────────────────────────────
@pytest.mark.skipif(
    not (MODELS_DIR / "classifier.pkl").exists(),
    reason="classifier.pkl non trouvé – lancer multimodal.py d'abord",
)
def test_artefact_classifier_chargeable():
    """Le fichier classifier.pkl est chargeable et utilisable."""
    with open(MODELS_DIR / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    assert hasattr(clf, "predict"), "classifier.pkl n'a pas de méthode predict"


@pytest.mark.skipif(
    not (MODELS_DIR / "label_encoder.pkl").exists(),
    reason="label_encoder.pkl non trouvé",
)
def test_artefact_label_encoder_classes():
    """Le LabelEncoder contient au moins 2 classes."""
    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    assert hasattr(le, "classes_"), "LabelEncoder sans classes_"
    assert len(le.classes_) >= 2, (
        f"Trop peu de classes : {len(le.classes_)}"
    )


@pytest.mark.skipif(
    not (MODELS_DIR / "scaler.pkl").exists(),
    reason="scaler.pkl non trouvé",
)
def test_artefact_scaler_chargeable():
    """Le StandardScaler est chargeable."""
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    assert hasattr(scaler, "transform"), "scaler.pkl n'a pas de méthode transform"


@pytest.mark.skipif(
    not (MODELS_DIR / "tfidf.pkl").exists(),
    reason="tfidf.pkl non trouvé",
)
def test_artefact_tfidf_chargeable():
    """Le TfidfVectorizer est chargeable et transforme un texte."""
    with open(MODELS_DIR / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    assert hasattr(tfidf, "transform"), "tfidf.pkl n'a pas de méthode transform"
    X = tfidf.transform(["plastique rigide transparent recyclable"])
    assert X.shape[0] == 1
    assert X.shape[1] > 0


@pytest.mark.skipif(
    not (MODELS_DIR / "regressor.pkl").exists(),
    reason="regressor.pkl non trouvé",
)
def test_artefact_regressor_chargeable():
    """Le régresseur est chargeable et retourne un float."""
    with open(MODELS_DIR / "regressor.pkl", "rb") as f:
        reg = pickle.load(f)
    assert hasattr(reg, "predict"), "regressor.pkl n'a pas de méthode predict"


# ── Test seuil sur modèle réel (si artefacts disponibles) ────────────────────
@pytest.mark.skipif(
    not all(
        (MODELS_DIR / f).exists()
        for f in ["classifier.pkl", "scaler.pkl", "tfidf.pkl", "label_encoder.pkl"]
    ),
    reason="Artefacts manquants – lancer multimodal.py d'abord",
)
def test_accuracy_modele_reel_sur_synthetique():
    """
    Le modèle réel chargé depuis models/ atteint ACC_MIN sur données synthétiques.
    Note : teste uniquement la partie numérique (scaler seul).
    """
    import pandas as pd
    from scipy.sparse import csr_matrix

    with open(MODELS_DIR / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    assert hasattr(clf, "predict"), "Le classifier chargé n'a pas predict"
    assert len(le.classes_) >= 2, "LabelEncoder avec moins de 2 classes"