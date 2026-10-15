"""Turns raw dataset rows into what the environment stores and what the agent sees."""

from env.tools import extract_urls

# Old rows use spam / not_spam. New rows use phishing / spam / legit.
LABELS = {
    "phishing": "phishing",
    "spam": "spam",
    "legit": "legit",
    "not_spam": "legit",
    "ham": "legit",
    "unknown": "unknown",     # live demo: a pasted email has no ground truth
}

# The only fields an agent is allowed to see. The label is deliberately not here.
VISIBLE_FIELDS = ("id", "sender_name", "sender", "subject", "text", "urls", "attachments")

# Hand-written demo rows carry a simple "auth" field. It is turned into the header a
# receiving mail server would have added, so the sender check reads it the same way
# as a real Authentication-Results header.
DEMO_AUTH_HEADER = {
    "pass": "demo-inbox.local; spf=pass smtp.mailfrom={d}; dkim=pass header.d={d}; dmarc=pass header.from={d}",
    "fail": "demo-inbox.local; spf=fail smtp.mailfrom={d}; dkim=none; dmarc=fail header.from={d}",
}


def normalise_email(row, position):
    """Fill in missing fields so old two-column rows (text, label) still work."""
    label = LABELS.get(str(row.get("label", "")).strip().lower())
    if label is None:
        raise ValueError(f"Email {position + 1} has an unknown label: {row.get('label')!r}")
    text = row.get("text") or ""
    sender = (row.get("sender") or "").lower()
    headers = dict(row.get("headers") or {})
    auth = row.get("auth") or "none"
    if not headers and auth in DEMO_AUTH_HEADER and "@" in sender:
        headers["Authentication-Results"] = [DEMO_AUTH_HEADER[auth].format(d=sender.split("@")[1])]
    return {
        "id": str(row.get("id") or f"E{position + 1:02d}"),
        "sender_name": row.get("sender_name") or "",
        "sender": row.get("sender") or "",
        "auth": auth,
        "reply_to": row.get("reply_to") or "",
        "return_path": row.get("return_path") or "",
        "date": row.get("date"),
        "attachments": row.get("attachments") or [],
        "headers": headers,
        "raw": row.get("raw"),
        "subject": row.get("subject") or "",
        "text": text,
        "urls": row.get("urls") or extract_urls(text),
        "label": label,
        "category": row.get("category") or "",
        "source": row.get("source") or "",
    }


def process_state(email, evidence, steps_left, index, total):
    """Build the observation for the agent: the email, what it has found, what is left."""
    state = {field: email[field] for field in VISIBLE_FIELDS}
    state["evidence"] = [dict(item) for item in evidence]
    state["steps_left"] = steps_left
    state["position"] = index + 1
    state["total"] = total
    return state
