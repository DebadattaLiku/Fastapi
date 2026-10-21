
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

def test_oversized_request_body():
    response = client.post(
        "/ask",
        json={
            "question": "A" * (65 * 1024),
            "temperature": 0.5,
        },
    )

    assert response.status_code == 422
    assert any(
        error["type"] == "string_too_long"
        for error in response.json()["detail"]
    )

# def test_oversized_streamed_request_body():
#     test_app = TestClient(app)

#     def body_stream():
#         yield b"A" * (32 * 1024)
#         yield b"B" * (33 * 1024)

#     response = test_app.post(
#         "/ask",
#         content=body_stream(),
#         headers={"X-API-Key": "test-api-key"},
#     )

#     assert response.status_code == 413
#     assert response.json()["detail"] == "Request body too large"
