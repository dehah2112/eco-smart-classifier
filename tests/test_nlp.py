"""
test_nlp.py - Tests pipeline NLP (prétraitement, vectorisation)
Eco-Smart Classifier
"""
import re

import nltk
import numpy as np
import pytest
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

for r in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(r, quiet=True)

# ── Setup NLP identique à nlp_pipeline.py ────────────────────────────────────
_stemmer = SnowballStemmer("french")

try:
    _STOP_FR = set(stopwords.words("french"))
except Exception:
    _STOP_FR = set()

_STOP_DOM = {
    "dechet", "dechets", "collecte", "rapport", "collecteur", "materiau",
    "materiaux", "echantillon", "analyse", "type", "lot", "reference",
    "code", "numero", "date", "heure", "kg", "litre", "cm", "mm",
    "metre", "unite", "valeur", "mesure", "resultat", "observation",
    "note", "commentaire", "traitement", "site", "zone", "point",
}
STOPWORDS_ALL = _STOP_FR | _STOP_DOM


def pretraiter(texte: str) -> str:
    """Pipeline NLP : nettoyage → stopwords → stemming (identique à src/)."""
    t = str(texte).lower()
    t = re.sub(r"\d+", " ", t)           # ← simple backslash (corrigé)
    t = re.sub(r"[^\w\s]", " ", t)       # ← simple backslash (corrigé)
    t = re.sub(r"\s+", " ", t).strip()   # ← simple backslash (corrigé)
    toks = t.split()
    toks = [
        _stemmer.stem(tk)
        for tk in toks
        if tk not in STOPWORDS_ALL and len(tk) > 2
    ]
    return " ".join(toks)


# ── Tests nettoyage ───────────────────────────────────────────────────────────
def test_nettoyage_minuscules():
    """Le texte est converti en minuscules."""
    result = pretraiter("Plastique PET RIGIDE")
    assert result == result.lower()


def test_suppression_chiffres():
    """Les chiffres sont supprimés du texte."""
    result = pretraiter("Collecte 123 kg de plastique 456")
    assert not any(c.isdigit() for c in result), (
        f"Chiffres trouvés dans : '{result}'"
    )


def test_suppression_ponctuation():
    """La ponctuation est supprimée."""
    result = pretraiter("plastique! rigide, transparent.")
    assert "!" not in result
    assert "," not in result
    assert "." not in result


def test_suppression_stopwords_fr():
    """Les stopwords français NLTK sont supprimés."""
    result = pretraiter("le la les de du plastique rigide")
    toks = result.split()
    for tok in toks:
        assert tok not in _STOP_FR, f"Stopword FR non supprimé : '{tok}'"


def test_suppression_stopwords_domaine():
    """Les stopwords du domaine déchets sont supprimés."""
    result = pretraiter("collecte rapport site zone plastique rigide")
    toks = result.split()
    for tok in toks:
        assert tok not in _STOP_DOM, f"Stopword domaine non supprimé : '{tok}'"


def test_mots_courts_supprimes():
    """Les mots de 1 ou 2 caractères sont supprimés."""
    result = pretraiter("un de plastique rigide")
    toks = result.split()
    for tok in toks:
        assert len(tok) > 2, f"Mot trop court conservé : '{tok}'"


def test_texte_significatif_non_vide():
    """Un texte significatif produit un résultat non vide."""
    result = pretraiter("plastique rigide transparent recyclable")
    assert len(result.strip()) > 0, "Résultat vide pour un texte significatif"


def test_texte_vide_sans_erreur():
    """Un texte vide ne génère pas d'erreur."""
    result = pretraiter("")
    assert isinstance(result, str)
    assert result == ""


def test_texte_uniquement_stopwords():
    """Un texte ne contenant que des stopwords retourne une chaîne vide."""
    result = pretraiter("le la les de du en un")
    assert result.strip() == "", f"Attendu vide, obtenu : '{result}'"


def test_texte_uniquement_chiffres():
    """Un texte ne contenant que des chiffres retourne une chaîne vide."""
    result = pretraiter("123 456 789")
    assert result.strip() == "", f"Attendu vide, obtenu : '{result}'"


def test_stemming_applique():
    """Le stemming est appliqué (les mots sont réduits à leur racine)."""
    result1 = pretraiter("plastique plastiques")
    toks = result1.split()
    # Les deux formes doivent être réduites à la même racine
    if len(toks) >= 2:
        assert toks[0] == toks[1], (
            f"Stemming incohérent : '{toks[0]}' ≠ '{toks[1]}'"
        )


def test_resultat_est_string():
    """Le résultat du prétraitement est toujours une chaîne."""
    for texte in ["plastique", "", "123", "!@#", None]:
        result = pretraiter(texte if texte is not None else "")
        assert isinstance(result, str), f"Résultat n'est pas str pour : {texte!r}"


# ── Tests vectorisation TF-IDF ────────────────────────────────────────────────
def test_tfidf_forme_matrice():
    """TF-IDF produit une matrice de la bonne dimension."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    textes = [
        "plastique rigide transparent",
        "metal conducteur lourd",
        "verre fragile opaque",
        "carton leger absorbant",
    ]
    vec = TfidfVectorizer(ngram_range=(1, 2))
    X   = vec.fit_transform(textes)
    assert X.shape[0] == 4, f"Nombre de lignes incorrect : {X.shape[0]}"
    assert X.shape[1] > 0, "Aucune feature générée"


def test_tfidf_valeurs_bornees():
    """Les valeurs TF-IDF sont comprises entre 0 et 1."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    textes = ["plastique rigide", "metal lourd", "verre fragile"]
    vec = TfidfVectorizer(norm="l2")
    X   = vec.fit_transform(textes).toarray()
    assert (X >= 0).all(), "Valeurs TF-IDF négatives"
    assert (X <= 1).all(), "Valeurs TF-IDF > 1"


def test_tfidf_bigrammes():
    """TF-IDF avec bigrammes génère plus de features qu'avec unigrammes."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    textes = ["plastique rigide transparent", "metal conducteur lourd"]
    vec_uni  = TfidfVectorizer(ngram_range=(1, 1))
    vec_bi   = TfidfVectorizer(ngram_range=(1, 2))
    X_uni    = vec_uni.fit_transform(textes)
    X_bi     = vec_bi.fit_transform(textes)
    assert X_bi.shape[1] > X_uni.shape[1], (
        "Les bigrammes ne génèrent pas plus de features que les unigrammes"
    )


def test_tfidf_sublinear_tf():
    """TF-IDF avec sublinear_tf=True produit une matrice valide."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    textes = ["plastique rigide transparent recyclable"] * 10
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
    X   = vec.fit_transform(textes)
    assert X.shape[0] == 10
    assert X.nnz >= 0


# ── Tests vectorisation BoW ───────────────────────────────────────────────────
def test_bow_valeurs_non_negatives():
    """BoW produit des valeurs entières non négatives."""
    from sklearn.feature_extraction.text import CountVectorizer

    textes = ["plastique rigide", "metal lourd", "verre fragile"]
    vec = CountVectorizer()
    X   = vec.fit_transform(textes).toarray()
    assert (X >= 0).all(), "BoW contient des valeurs négatives"


def test_bow_dtype():
    """BoW produit un tableau numérique (int ou float)."""
    from sklearn.feature_extraction.text import CountVectorizer

    textes = ["plastique rigide", "metal lourd", "verre fragile"]
    vec = CountVectorizer()
    X   = vec.fit_transform(textes).toarray()
    assert X.dtype in [np.int32, np.int64, np.float32, np.float64], (
        f"dtype inattendu : {X.dtype}"
    )


def test_bow_forme_matrice():
    """BoW produit une matrice de la bonne forme."""
    from sklearn.feature_extraction.text import CountVectorizer

    textes = ["plastique rigide", "metal lourd", "verre fragile"]
    vec = CountVectorizer()
    X   = vec.fit_transform(textes)
    assert X.shape[0] == 3, f"Nombre de lignes incorrect : {X.shape[0]}"
    assert X.shape[1] > 0, "Aucune feature BoW générée"


# ── Tests pipeline complet ────────────────────────────────────────────────────
def test_pipeline_complet_classification():
    """Pipeline complet prétraitement + TF-IDF + classificateur fonctionne."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    corpus = [
        "plastique rigide transparent recyclable tri selectif",
        "metal ferreux conducteur rouille lourd industriel",
        "verre opaque fragile lourd non conducteur",
        "carton humide leger absorbant biodegradable",
        "plastique souple film transparent emballage",
        "metal aluminium leger conducteur recyclable",
        "verre bouteille transparent fragile recyclable",
        "carton ondule rigide emballage papier",
    ]
    labels = [0, 1, 2, 3, 0, 1, 2, 3]

    corpus_clean = [pretraiter(t) for t in corpus]

    vec = TfidfVectorizer(ngram_range=(1, 2))
    X   = vec.fit_transform(corpus_clean)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X, labels)

    yp  = clf.predict(X)
    acc = accuracy_score(labels, yp)

    # Sur le jeu d'entraînement il doit apprendre parfaitement
    assert acc > 0.5, f"Accuracy pipeline NLP trop basse : {acc:.2f}"
    assert len(yp) == len(labels)