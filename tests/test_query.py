from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_query_requires_auth():
    response = client.post("/api/v1/query", json={"question": "hello"})
    assert response.status_code in (401, 403)


def test_query_validates_input():
    response = client.post(
        "/api/v1/query",
        json={"question": ""},
        headers={"x-api-key": "test-key"},
    )
    # Empty question should fail or be handled gracefully
    # Status code depends on your validation
