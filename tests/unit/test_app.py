from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_completions():
    response = client.get("/v1/completions")
    assert response.status_code == 200
    assert response.json() == {}
