"""Builds the short incident report that is written after every decision."""

VERDICT = {
    "quarantine": "QUARANTINED (blocked as a threat)",
    "allow": "ALLOWED (delivered)",
    "escalate": "ESCALATED (sent to a human analyst)",
}


def build_report(record):
    """`record` is one email entry produced by evaluator.grader.run_episode."""
    sender = f"{record['sender_name']} <{record['sender']}>".strip() if record["sender"] else "unknown"
    lines = [
        f"INCIDENT REPORT  {record['id']}",
        f"Verdict  : {VERDICT.get(record['decision'], record['decision'])}",
        f"Email    : \"{record['subject'] or record['text'][:60]}\" from {sender}",
    ]
    checks = [step for step in record["steps"] if step["kind"] == "investigate"]
    if checks:
        lines.append("Evidence :")
        for step in checks:
            risk = "n/a" if step["risk"] is None else f"{step['risk']:.2f}"
            lines.append(f"  - {step['action']}: {step['finding']} (risk {risk})")
    else:
        lines.append("Evidence : none collected")
    if record.get("unavailable"):
        lines.append("Not verified (lookups unavailable, NOT treated as safe):")
        for gap in record["unavailable"][:6]:
            lines.append(f"  - {gap}")
    if record["reason"]:
        lines.append(f"Reason   : {record['reason']}")
    if record.get("confidence") is not None:
        lines.append(f"Confidence: {record['confidence']:.0%} (evidence strength x coverage)")
    cost = -sum(step["reward"] for step in checks)
    lines.append(f"Cost     : {len(checks)} tool call(s), {cost:.2f} reward spent")
    return "\n".join(lines)
