"""Runs an agent through the environment and measures how it did."""

from agent.confidence import decision_confidence
from agent.report import build_report
from email_env import DECISIONS, LEGACY_ACTIONS, EmailEnv


def run_episode(agent, env=None):
    """
    Play one full pass over the inbox.

    Returns {"agent", "metrics", "emails", "log"} where `emails` has one record per
    email (steps taken, decision, outcome, incident report) and `log` has one
    entry per environment step.
    """
    env = env or EmailEnv()
    state = env.reset()
    emails, log, done = [], [], False
    record = _new_record(env.current)

    while not done:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        log.append({"step": len(log) + 1, "action": action, "reward": reward, "done": done})

        evidence = info.get("evidence") if info["kind"] == "investigate" else None
        record["steps"].append({
            "action": info["action"],
            "kind": info["kind"],
            "reward": reward,
            "risk": evidence["risk"] if evidence else None,
            "finding": evidence["finding"] if evidence else info.get("error", ""),
            "unavailable": evidence.get("unavailable", []) if evidence else [],
            "coverage": evidence.get("coverage") if evidence else None,
            "latency_ms": getattr(agent, "last_latency_ms", None),
            "source": getattr(agent, "last_source", None),
        })
        record["reward"] += reward

        if info["kind"] == "decision":
            record["decision"] = info["action"]
            record["outcome"] = info["outcome"]
            record["timed_out"] = info["timed_out"]
            # A forced escalation was the environment's choice, not the agent's.
            chosen = LEGACY_ACTIONS.get(action, action) in DECISIONS
            record["reason"] = getattr(agent, "reason", "") if chosen else "Ran out of steps"
            record["tool_calls"] = sum(1 for s in record["steps"] if s["kind"] == "investigate")
            record["confidence"] = decision_confidence(info["action"], info["evidence"])
            record["unavailable"] = [u for e in info["evidence"] for u in e.get("unavailable", [])]
            record["reward"] = round(record["reward"], 2) + 0.0   # + 0.0 turns -0.0 into 0.0
            record["report"] = build_report(record)
            emails.append(record)
            if not done:
                record = _new_record(env.current)

    return {
        "agent": getattr(agent, "name", type(agent).__name__),
        "metrics": summarise(emails),
        "emails": emails,
        "log": log,
    }


def evaluate(agent, env=None):
    """Just the metrics for one agent."""
    return run_episode(agent, env)["metrics"]


def classification(emails):
    """
    Threat detection metrics. Positive class = threat (phishing or spam).

    quarantine = predicted threat, allow = predicted legit, escalate = sent to a human.
    precision, recall, F1 and FPR count only automatic decisions: an escalated threat
    is not a true positive and an escalated real email is not a false positive.
    phishing_miss_rate counts phishing that was DELIVERED (allowed); escalated phishing
    reaches an analyst, not the inbox.
    """
    threat = lambda e: e["label"] in ("phishing", "spam")
    tp = sum(1 for e in emails if threat(e) and e["decision"] == "quarantine")
    fp = sum(1 for e in emails if not threat(e) and e["decision"] == "quarantine")
    fn = sum(1 for e in emails if threat(e) and e["decision"] != "quarantine")
    threats = sum(1 for e in emails if threat(e))
    legit = len(emails) - threats
    phishing = [e for e in emails if e["label"] == "phishing"]
    flagged = sum(1 for e in emails if threat(e) and e["decision"] in ("quarantine", "escalate"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positive_rate": round(fp / legit, 3) if legit else 0.0,
        "phishing_miss_rate": round(sum(1 for e in phishing if e["decision"] == "allow")
                                    / len(phishing), 3) if phishing else 0.0,
        "threats_quarantined_or_escalated": round(flagged / threats, 3) if threats else 0.0,
        "escalation_rate": round(sum(1 for e in emails if e["decision"] == "escalate")
                                 / len(emails), 3) if emails else 0.0,
    }


def summarise(emails):
    total = len(emails)
    total_reward = sum(e["reward"] for e in emails)
    average = total_reward / total if total else 0.0
    count = lambda outcome: sum(1 for e in emails if e["outcome"] == outcome)
    return {
        "emails": total,
        "correct": count("correct"),
        "accuracy": round(count("correct") / total, 3) if total else 0.0,
        "false_positives": count("false_positive"),      # real emails that got blocked
        "missed_threats": count("missed_threat"),        # phishing or spam that got through
        "missed_phishing": sum(1 for e in emails
                               if e["outcome"] == "missed_threat" and e["label"] == "phishing"),
        "escalated": count("escalated"),
        "tool_calls": sum(e["tool_calls"] for e in emails),
        "avg_tool_calls": round(sum(e["tool_calls"] for e in emails) / total, 2) if total else 0.0,
        "total_reward": round(total_reward, 2),
        "avg_reward": round(average, 3),
        # 0..1 score: 0.5 means break-even, 1.0 means every email right with no tool cost.
        "score": round(max(0.0, min(1.0, (average + 1) / 2)), 3),
        **classification(emails),
    }


def _new_record(email):
    return {
        "id": email["id"],
        "sender_name": email["sender_name"],
        "sender": email["sender"],
        "subject": email["subject"],
        "text": email["text"],
        "label": email["label"],
        "steps": [],
        "reward": 0.0,
    }
