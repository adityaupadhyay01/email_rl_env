"""Load the evaluation sets."""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTH_KEYS = ("Authentication-Results", "ARC-Authentication-Results", "Received-SPF",
             "DKIM-Signature")


def load_split(name, auth="strip"):
    """
    name: "train" or "test" (public corpora) or "demo" (the 34 hand-written emails).

    auth="strip" removes the authentication headers from every email. The public
    legit emails are from 2002 and never carry them, while the phishing emails are
    recent and almost always do, so leaving them in would let an agent spot
    "has the header" instead of "failed the check". Stripping keeps the comparison
    fair; use auth="keep" only for phishing-only analysis.
    """
    if name == "demo":
        from db.fetch_data import load_fallback
        return load_fallback()
    with open(os.path.join(ROOT, "data", "public", f"{name}.json"), encoding="utf-8") as f:
        rows = json.load(f)
    if auth == "strip":
        for row in rows:
            row["headers"] = {k: v for k, v in (row.get("headers") or {}).items()
                              if k not in AUTH_KEYS}
    return rows
