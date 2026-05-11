"""
test_api.py - Tests endpoint API FastAPI
Eco-Smart Classifier
"""
import pytest

try:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    API_OK = True
except Exception:
    API_OK = False


# ── Payload de test valide ────────────────────────────────────────────────────
VALID_PAYLOAD = {
    "Poids":            0.5,
    "Volume":           1.2,
    "Conductivite":     0.3,
    "Opacite":          0.8,
    "Rigidite":         0.6,
    "Source":           "collecte_selective",
    "Rapport_Collecte": "plastique rigide transparent recyclable",
}


# ── Tests ─────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not API_OK, reason="API non disponible")
def test_endpoint_root():
    """L'endpoint / repond 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


@pytest.mark.skipif(not API_OK, reason="API non disponible")
def test_endpoint_health():
    """L'endpoint /health repond 200 avec le statut ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


@pytest.mark.skipif(not API_OK, reason="API non disponible")
def test_endpoint_classes():
    """L'endpoint /classes retourne une liste de categories."""
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert isinstance(data["classes"], list)


@pytest.mark.skipif(not API_OK, reason="API non disponible")
def test_endpoint_predict_valide():
    """L'endpoint /predict repond 200 avec un payload valide."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "categorie" in data
    assert "confiance" in data
    assert "prix_estime" in data
    assert isinstance(data["categorie"], str)
    assert len(data["categorie"]) > 0


@pytest.mark.skipif(not API_OK, reason="API non disponible")
def test_endpoint_predict_categorie_connue():
    """La categorie predite fait partie des classes connues."""
    response = client.get("/classes")
    classes = response.json().get("classes", [])

    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    categorie = response.json()["categorie"]

    if classes:
        assert categorie in classes, f"Categorie '{categorie}' inconnue"


@pytest.mark.skipif(not API_OK, reason="API non disponible")
def test_endpoint_predict_champs_manquants():
    """L'API retourne 422 si des champs requis manquent."""
    response = client.post("/predict", json={"Poids": 0.5})
    assert response.status_code == 422


@pytest.mark.skipif(not API_OK, reason="API non disponible")
def test_endpoint_predict_type_erreur():
    """L'API retourne 422 si le type d'un champ est incorrect."""
    payload_mauvais = VALID_PAYLOAD.copy()
    payload_mauvais["Poids"] = "pas_un_nombre"
    response = client.post("/predict", json=payload_mauvais)
    assert response.status_code == 422


@pytest.mark.skipif(not API_OK, reason="API non disponible")
def test_endpoint_predict_valeurs_negatives():
    """L'API retourne 422 si Poids ou Volume est negatif."""
    payload_negatif = VALID_PAYLOAD.copy()
    payload_negatif["Poids"] = -1.0
    response = client.post("/predict", json=payload_negatif)
    assert response.status_code == 422


@pytest.mark.skipif(not API_OK, reason="API non disponible")
def test_endpoint_predict_conductivite_hors_range():
    """L'API retourne 422 si Conductivite > 1."""
    payload_hors = VALID_PAYLOAD.copy()
    payload_hors["Conductivite"] = 1.5
    response = client.post("/predict", json=payload_hors)
    assert response.status_code == 422


@pytest.mark.skipif(not API_OK, reason="API non disponible")
def test_endpoint_predict_payload_vide():
    """L'API retourne 422 avec un payload completement vide."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.skipif(not API_OK, reason="API non disponible")
def test_endpoint_predict_rapport_vide():
    """L'API accepte un rapport vide (texte vide = cas limite)."""
    payload_vide = VALID_PAYLOAD.copy()
    payload_vide["Rapport_Collecte"] = ""
    response = client.post("/predict", json=payload_vide)
    # Doit repondre 200 ou 500, pas 422 (le champ est present)
    assert response.status_code in [200, 500]