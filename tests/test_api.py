from __future__ import annotations

from fastapi.testclient import TestClient

from app.space_app import app


client = TestClient(app)


def test_root_returns_200() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "billing_refund_easy" in payload["task_ids"]


def test_reset_endpoint_returns_observation() -> None:
    response = client.post("/reset", json={"task_id": "billing_refund_easy"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["done"] is False
    assert payload["observation"]["task_id"] == "billing_refund_easy"


def test_state_endpoint_auto_initializes() -> None:
    response = client.get("/state")
    assert response.status_code == 200
    payload = response.json()
    assert "task_id" in payload
    assert "observation" in payload


def test_step_endpoint_accepts_structured_action() -> None:
    client.post("/reset", json={"task_id": "billing_refund_easy"})
    response = client.post(
        "/step",
        json={
            "action": {
                "action_type": "classify_ticket",
                "category": "billing_charge_dispute",
            }
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["reward"]["value"] >= 0.0
    assert payload["observation"]["completion_flags"]["classified"] is True
