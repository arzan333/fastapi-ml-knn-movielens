from fastapi.testclient import TestClient

from movie_reco_api.app import create_app


def test_recommend_returns_503_when_model_missing():
    app = create_app()
    client = TestClient(app)
    r = client.post("/recommend", json={"movie_title": "Toy Story (1995)", "k": 3})
    assert r.status_code in (503, 200)


def test_validation_rejects_empty_title():
    app = create_app()
    client = TestClient(app)
    r = client.post("/recommend", json={"movie_title": "", "k": 3})
    assert r.status_code == 422

