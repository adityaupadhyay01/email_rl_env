"""
Triage one email end to end: investigate, decide, explain.

Used by the live demo. The email goes through the same EmailEnv and tools as in
training and evaluation, so what the demo shows is exactly what was measured.
"""

import time

from agent.confidence import decision_confidence
from agent.report import build_report
from email_env import DECISIONS, EmailEnv


def triage(row, agent, tools):
    """`row` is a parsed email (env.eml.parse_eml output, plus an optional id)."""
    row = dict(row)
    row.setdefault("label", "unknown")
    env = EmailEnv(emails=[row], tools=tools)
    state, done = env.reset(), False
    steps = []
    started = time.perf_counter()
    while not done:
        t0 = time.perf_counter()
        action = agent.act(state)
        think_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        state, reward, done, info = env.step(action)
        tool_ms = (time.perf_counter() - t1) * 1000
        step = {"action": info["action"], "kind": info["kind"],
                "agent_ms": round(think_ms, 1), "tool_ms": round(tool_ms, 1),
                "decided_by": getattr(agent, "last_source", None) or getattr(agent, "name", "")}
        if info["kind"] == "investigate":
            step["evidence"] = info["evidence"]
        elif info["kind"] == "invalid":
            step["error"] = info.get("error")
        steps.append(step)

    decision = info["action"]
    chosen = action in DECISIONS
    record = {
        "id": env.data[0]["id"], "sender_name": env.data[0]["sender_name"],
        "sender": env.data[0]["sender"], "subject": env.data[0]["subject"],
        "text": env.data[0]["text"], "label": env.data[0]["label"],
        "decision": decision, "timed_out": info["timed_out"],
        "reason": getattr(agent, "reason", "") if chosen else "Ran out of investigation steps",
        "steps": [{"action": s["action"], "kind": s["kind"],
                   "reward": -tools_cost(s["action"]) if s["kind"] == "investigate" else 0.0,
                   "risk": s.get("evidence", {}).get("risk"),
                   "finding": s.get("evidence", {}).get("finding", "")} for s in steps],
    }
    evidence = info["evidence"]
    record["confidence"] = decision_confidence(decision, evidence)
    record["unavailable"] = [u for e in evidence for u in e.get("unavailable", [])]
    record["report"] = build_report(record)
    return {"email": {k: record[k] for k in ("id", "sender_name", "sender", "subject")},
            "agent": getattr(agent, "name", type(agent).__name__),
            "steps": steps, "decision": decision, "confidence": record["confidence"],
            "reason": record["reason"], "unavailable": record["unavailable"],
            "report": record["report"],
            "total_ms": round((time.perf_counter() - started) * 1000, 1)}


def tools_cost(action):
    from email_env import INVESTIGATE_COST
    return INVESTIGATE_COST.get(action, 0.0)
