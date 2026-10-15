"""
FastAPI server.

    GET  /          dashboard (baseline vs agent, per-email investigation trace)
    GET  /health    liveness check
    POST /reset     start a new episode, returns the first state
    POST /step      {"action": "..."} -> state, reward, done, info
    GET  /run       run one agent over the whole inbox  (?agent=investigator)
    GET  /compare   metrics for every available agent

    Live demo (controlled demo inbox, see server/mailbox.py):
    GET  /demo                  the live triage page
    GET  /api/mailbox           folders (inbox, quarantine, review, delivered) + audit log
    GET  /api/message           one message with its stored triage result
    POST /api/triage            {"file": "01-E15.eml"} or {"raw": "<full email>"}, "agent"
                                -> evidence, decision, confidence, report, executed action
    POST /api/review            {"file": "...", "verdict": "release" | "quarantine"}
    POST /api/mailbox/reset     restore the demo inbox
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from agent.triage import triage
from env.eml import parse_eml
from env.tools import InvestigationTools
from server.mailbox import DemoMailbox

from email_env import ACTIONS, EmailEnv
from evaluator.grader import run_episode
from inference import make_agent
from utils.config import HF_TOKEN

DASHBOARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
DEMO_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.html")
MAX_RAW_BYTES = 512 * 1024

app = FastAPI(title="Email Threat-Triage Environment")

env = None
_results = {}     # finished runs, so the dashboard does not re-run an LLM on every refresh
_mailbox = None
_tools = None
_agents = {}


def available_agents():
    return ["baseline", "investigator", "pg"] + (["llm"] if HF_TOKEN else [])


def mailbox():
    global _mailbox
    if _mailbox is None:
        _mailbox = DemoMailbox()
    return _mailbox


def live_tools():
    global _tools
    if _tools is None:
        _tools = InvestigationTools()
    return _tools


def demo_agent(kind):
    if kind not in ("investigator", "pg", "llm"):
        raise HTTPException(status_code=400, detail=f"Unknown agent '{kind}'")
    if kind == "llm" and not HF_TOKEN:
        raise HTTPException(status_code=400, detail="The LLM agent needs HF_TOKEN on the server")
    if kind not in _agents:
        _agents[kind] = make_agent(kind)
    return _agents[kind]


def get_result(kind, fresh=False):
    if kind not in ("baseline", "investigator", "pg", "llm"):
        raise HTTPException(status_code=400, detail=f"Unknown agent '{kind}'")
    if fresh or kind not in _results:
        _results[kind] = run_episode(make_agent(kind))
    return _results[kind]


@app.get("/")
def dashboard():
    return FileResponse(DASHBOARD, media_type="text/html")


@app.get("/health")
def health():
    return {"message": "Email RL Agent Running 🚀", "agents": available_agents()}


@app.post("/reset")
def reset():
    global env
    env = EmailEnv()
    return {"state": env.reset(), "actions": list(ACTIONS)}


@app.post("/step")
def step(action: dict):
    if env is None or env.done:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")
    state, reward, done, info = env.step(action.get("action"))
    return {"state": state, "reward": reward, "done": done, "info": info}


@app.get("/run")
def run_agent(agent: str = "investigator", fresh: bool = False):
    result = get_result(agent, fresh)
    # `steps`, `score` and `logs` keep the shape of the original /run response.
    return {**result, "steps": len(result["log"]), "score": result["metrics"]["score"],
            "logs": result["log"]}


@app.get("/compare")
def compare(fresh: bool = False):
    results = [get_result(kind, fresh) for kind in available_agents()]
    return [{"kind": kind, "agent": r["agent"], "metrics": r["metrics"]}
            for kind, r in zip(available_agents(), results)]


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _read_result(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/results")
def api_results():
    """Saved held-out test results, learning curves and the false-positive cost sweep."""
    test = _read_result("eval_test.json")
    if test:
        for data in test["agents"].values():
            data.pop("errors", None)
    return {"test": test, "curves": _read_result("pg_curves.json"),
            "fp_sweep": _read_result("fp_sweep.json")}


@app.get("/demo")
def demo_page():
    return FileResponse(DEMO_PAGE, media_type="text/html")


@app.get("/api/mailbox")
def api_mailbox():
    return {**mailbox().listing(), "agents": [a for a in available_agents() if a != "baseline"]}


@app.get("/api/message")
def api_message(folder: str, file: str):
    try:
        raw = mailbox().read(folder, file)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    parsed = parse_eml(raw)
    return {"folder": folder, "file": file,
            "email": {k: parsed[k] for k in ("sender_name", "sender", "subject", "date",
                                             "text", "urls", "attachments")},
            "auth_headers": parsed["headers"].get("Authentication-Results", []),
            "triage": mailbox().triage_record(folder, file)}


@app.post("/api/triage")
def api_triage(body: dict):
    box = mailbox()
    kind = body.get("agent") or "investigator"
    agent = demo_agent(kind)
    raw = body.get("raw")
    try:
        if raw:
            if len(raw.encode("utf-8", errors="replace")) > MAX_RAW_BYTES:
                raise HTTPException(status_code=413, detail="Email is larger than 512 KB")
            name = box.add(raw)
        else:
            name = body.get("file") or ""
            raw = box.read("inbox", name)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    row = parse_eml(raw)
    row["raw"] = raw
    row["id"] = name[:-4]
    result = triage(row, agent, live_tools())
    result["file"] = name
    result["action"] = box.apply_decision(name, result) if body.get("execute", True) else None
    if hasattr(live_tools().intel, "save"):
        live_tools().intel.save()
    return result


@app.post("/api/review")
def api_review(body: dict):
    try:
        return mailbox().human_review(body.get("file") or "", body.get("verdict"),
                                      analyst=(body.get("analyst") or "analyst")[:40])
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/mailbox/reset")
def api_reset():
    mailbox().reset()
    return {"ok": True}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
