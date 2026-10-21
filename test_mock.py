
import os

os.environ["API_ACCESS_KEY"] = "test-api-key"
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app,headers={"X-API-Key": "test-api-key"})


def test_mocked_llm_response():
    with patch("main.simple_agent", return_value="Mock answer"):
        response = client.post(
            "/ask",
            json={"question": "What is RAG?", "temperature": 0.5}
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Mock answer"


def test_temperature_passed_correctly():
    with patch("main.simple_agent", return_value="Mock answer") as mock_agent:
        client.post(
            "/ask",
            json={"question": "What is RAG?", "temperature": 0.2}
        )

    mock_agent.assert_called_once_with("What is RAG?", 0.2)


def test_llm_failure():
    with patch(
        "main.simple_agent",
        side_effect=Exception("Gemini unavailable")
    ):
        response = client.post(
            "/ask",
            json={"question": "What is RAG?", "temperature": 0.5}
        )

    assert response.status_code == 502
    assert response.json()["detail"].rstrip(".") == "The LLM service failed. Please try again"

def test_invalid_input_does_not_call_agent():
    with patch("main.simple_agent") as mock_agent:
        response = client.post(
            "/ask",
            json={"question": "", "temperature": 5}
        )

    assert response.status_code == 422
    mock_agent.assert_not_called()

def test_rate_limit():
    responses = [
        client.get("/rate-test")
        for _ in range(6)
    ]

    status_codes = [r.status_code for r in responses]

    assert status_codes[:5] == [200, 200, 200, 200, 200]
    assert status_codes[5] == 429    