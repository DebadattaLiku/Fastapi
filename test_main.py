from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_valid_question():
    response = client.post(
        "/ask",
        json={"question": "What is RAG?", "temperature": 0.5}
    )
    assert response.status_code == 200
    assert len(response.json()["answer"]) > 0

def test_invalid_question():
    response = client.post(
        "/ask",
        json={"question": "", "temperature": 5}
    )
    assert response.status_code == 422

def test_temperature_is_used():
    response = client.post(
        "/ask",
        json={"question": "What is RAG?", "temperature": 0.2}
    )

    assert response.status_code == 200
    assert len(response.json()["answer"]) > 0   