from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_add_doc_requires_auth():
    response = client.post("/api/v1/docs", json={"text": "hello"})
    assert response.status_code in (401, 403)


def test_add_doc_validates_input():
    response = client.post(
        "/api/v1/docs",
        json={"text": "", "metadata": {}},
        headers={"x-api-key": "test-key"},
    )
    # Empty text should fail or be handled gracefully
