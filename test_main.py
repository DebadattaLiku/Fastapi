import os

os.environ["API_ACCESS_KEY"] = "test-api-key"
from fastapi.testclient import TestClient
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app,headers={"X-API-Key": "test-api-key"})

def test_valid_question():
    with patch("main.simple_agent", return_value="Mock answer"):
        response = client.post(
            "/ask",
            json={"question": "What is RAG?", "temperature": 0.5}
        )
    assert response.status_code == 200
    assert response.json()["answer"] == "Mock answer"

def test_invalid_question():
    with patch("main.simple_agent") as mock_agent:
        response = client.post(
            "/ask",
            json={"question": "", "temperature": 5}
        )
    assert response.status_code == 422
    mock_agent.assert_not_called()

def test_temperature_is_used():
    with patch("main.simple_agent", return_value="Mock answer") as mock_agent:
        response = client.post(
            "/ask",
            json={"question": "What is RAG?", "temperature": 0.2}
        )

    assert response.status_code == 200
    mock_agent.assert_called_once_with("What is RAG?", 0.2)  