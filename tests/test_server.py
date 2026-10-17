from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_dashboard_and_health():
    page = client.get("/")
    assert page.status_code == 200 and "Email Threat Triage" in page.text
    assert client.get("/health").json()["agents"][:2] == ["baseline", "investigator"]


def test_reset_then_step():
    state = client.post("/reset").json()["state"]
    assert state["id"] == "E01" and "label" not in state

    body = client.post("/step", json={"action": "check_sender"}).json()
    assert body["reward"] == -0.1 and body["done"] is False
    assert body["state"]["evidence"][0]["tool"] == "check_sender"

    body = client.post("/step", json={"action": "allow"}).json()
    assert body["reward"] == 1.0 and body["info"]["outcome"] == "correct"
    assert body["state"]["id"] == "E02"


def test_run_and_compare():
    run = client.get("/run", params={"agent": "baseline"}).json()
    assert run["steps"] == len(run["logs"]) == len(run["emails"])       # baseline never investigates
    assert 0 <= run["score"] <= 1

    rows = client.get("/compare").json()
    assert [r["kind"] for r in rows][:2] == ["baseline", "investigator"]
    assert rows[1]["metrics"]["total_reward"] > rows[0]["metrics"]["total_reward"]

    assert client.get("/run", params={"agent": "nope"}).status_code == 400
