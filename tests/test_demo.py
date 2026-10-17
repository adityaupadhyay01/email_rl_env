"""The live demo: triage one email, execute the action, let a human resolve it."""

from fastapi.testclient import TestClient

from server.app import app, mailbox

client = TestClient(app)
PASTED = ("From: SBI Alerts <alerts@sbi-kyc-verify.example>\nSubject: KYC pending\n\n"
          "Aapka KYC pending hai. Turant verify karein aur OTP bhejein: http://sbi-kyc-verify.example/login")


def test_triage_executes_the_decision_on_the_demo_inbox():
    client.post("/api/mailbox/reset")
    inbox = client.get("/api/mailbox").json()["folders"]["inbox"]
    assert len(inbox) >= 6
    result = client.post("/api/triage", json={"file": "01-E15.eml", "agent": "investigator"}).json()
    assert result["decision"] == "quarantine" and 0 < result["confidence"] <= 1
    assert result["steps"][0]["action"] == "check_content" and result["steps"][-1]["kind"] == "decision"
    assert "INCIDENT REPORT" in result["report"]
    assert result["action"]["folder"] == "quarantine"
    folders = client.get("/api/mailbox").json()["folders"]
    assert any(m["file"] == "01-E15.eml" for m in folders["quarantine"])
    assert not any(m["file"] == "01-E15.eml" for m in folders["inbox"])


def test_pasted_email_and_human_review():
    client.post("/api/mailbox/reset")
    learned = client.post("/api/triage", json={"raw": PASTED, "agent": "pg"}).json()
    assert learned["decision"] in ("quarantine", "escalate", "allow") and learned["action"]
    result = client.post("/api/triage", json={"raw": PASTED, "agent": "investigator"}).json()
    assert result["unavailable"]                       # lookups are off in tests, and it says so
    assert "NOT treated as safe" in result["report"]
    box = mailbox()
    name = box.add(PASTED)
    box.apply_decision(name, {**result, "decision": "escalate"})
    assert client.post("/api/review", json={"file": name, "verdict": "release"}).json()["folder"] == "delivered"
    log = client.get("/api/mailbox").json()["log"]
    assert log[0]["action"] == "human_release"


def test_demo_rejects_path_tricks():
    assert client.post("/api/triage", json={"file": "../../etc/passwd"}).status_code == 404
    assert client.get("/api/message", params={"folder": "inbox", "file": "../x.eml"}).status_code == 404
