"""
Build the controlled demo inbox (data/demo_inbox/seed/*.eml).

Six hand-written scenarios from data/fallback.json (fictional brands, reserved
.example domains) plus a few real public samples: phishing from phishing_pot
(CC BY-NC 4.0) and a genuine mailing-list email from the SpamAssassin corpus.

    python scripts/make_demo_inbox.py PHISHING_POT_DIR SPAMASSASSIN_DATA_DIR
"""

import json
import os
import re
import sys
from email.message import EmailMessage
from email.utils import formataddr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = os.path.join(ROOT, "data", "demo_inbox", "seed")
SCENARIOS = ["E15", "E21", "E33", "E01", "E03", "E28"]
AUTH = {
    "pass": "mx.demo-inbox.local; spf=pass smtp.mailfrom={d}; dkim=pass header.d={d}; "
            "dmarc=pass header.from={d}",
    "fail": "mx.demo-inbox.local; spf=fail smtp.mailfrom={d}; dkim=none; dmarc=fail header.from={d}",
    "none": "mx.demo-inbox.local; spf=none smtp.mailfrom={d}; dkim=none; dmarc=none header.from={d}",
}
REAL = [("phishing_pot", "email/sample-850.eml"), ("phishing_pot", "email/sample-673.eml"),
        ("spamassassin", "easy-ham-1/00012.48a387bc38d1316a6f6b49e8c2e43a03.txt")]


def main(pot_dir, sa_dir):
    os.makedirs(SEED, exist_ok=True)
    rows = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "data", "fallback.json")))}
    for n, key in enumerate(SCENARIOS, 1):
        row = rows[key]
        domain = row["sender"].split("@")[1]
        msg = EmailMessage()
        msg["Authentication-Results"] = AUTH[row["auth"]].format(d=domain)
        msg["From"] = formataddr((row["sender_name"], row["sender"]))
        msg["To"] = "aditya@demo-inbox.local"
        msg["Subject"] = row["subject"]
        msg["Date"] = f"Wed, 14 Oct 2026 {9 + n:02d}:{10 + n * 7:02d}:00 +0530"
        msg["X-Demo-Scenario"] = f"{key} (hand-written, label {row['label']})"
        msg.set_content(row["text"])
        with open(os.path.join(SEED, f"{n:02d}-{key}.eml"), "wb") as f:
            f.write(bytes(msg))

    for n, (corpus, rel) in enumerate(REAL, len(SCENARIOS) + 1):
        base = pot_dir if corpus == "phishing_pot" else sa_dir
        with open(os.path.join(base, rel), "rb") as f:
            raw = f.read()
        raw = re.sub(rb"phishing@pot", b"aditya@demo-inbox.local", raw, flags=re.IGNORECASE)
        name = f"{n:02d}-real-{corpus}-{os.path.basename(rel).split('.')[0]}.eml"
        with open(os.path.join(SEED, name), "wb") as f:
            f.write(raw)
    print(f"{len(os.listdir(SEED))} emails in {SEED}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
