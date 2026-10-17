"""
Build the labelled public test set from two public corpora.

    phishing  rf-peixoto/phishing_pot   real phishing emails, 2022-2026, CC BY-NC 4.0
              https://github.com/rf-peixoto/phishing_pot
    legit     SpamAssassin public corpus, easy_ham + hard_ham (2002), via the npm
    spam      package @stdlib/datasets-spam-assassin (corpus: Apache SpamAssassin project)

    python scripts/build_public_dataset.py PHISHING_POT_DIR SPAMASSASSIN_DATA_DIR

Writes data/public/train.json and data/public/test.json (stratified 50/50 split,
fixed seed). The train half is the only data used for tuning thresholds and for
training the policy-gradient agent; every reported number comes from the test half.
"""

import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.eml import parse_eml   # noqa: E402

SEED = 2026
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "public")
# Redaction artefacts that would give the label away to a text-reading agent:
# phishing_pot replaces the victim address with "phishing@pot", SpamAssassin
# rewrote local addresses to yyyy@/zzzz@...spamassassin.taint.org.
SCRUB = [
    (re.compile(r"phishing\s*@\s*pot|phishing[ _]pot", re.IGNORECASE), "you@example.com"),
    (re.compile(r"\b[yz]{3,}(@[\w.-]*spamassassin\.taint\.org)?", re.IGNORECASE), "you"),
    (re.compile(r"[\w.-]*spamassassin\.taint\.org", re.IGNORECASE), "example.com"),
]
SIZES = {"phishing": 700, "easy-ham": 550, "hard-ham": 150, "spam": 300}


def load(paths, label, source, limit, prefix):
    rows, seen = [], set()
    for path in paths:
        if len(rows) >= limit:
            break
        with open(path, "rb") as f:
            raw = f.read()
        try:
            row = parse_eml(raw)
        except Exception:
            continue
        if not (row["text"] or row["subject"]) or "@" not in row["sender"]:
            continue
        for pattern, replacement in SCRUB:
            for field in ("subject", "text", "sender_name", "sender", "reply_to", "return_path"):
                row[field] = pattern.sub(replacement, row[field])
            row["urls"] = [pattern.sub(replacement, u) for u in row["urls"]]
            row["headers"] = {k: [pattern.sub(replacement, v) for v in values]
                              for k, values in row["headers"].items()}
        key = (row["sender"], row["subject"], row["text"][:200])
        if key in seen:
            continue
        seen.add(key)
        row.update(id=f"{prefix}{len(rows) + 1:04d}", label=label, source=source,
                   source_file=os.path.basename(path))
        rows.append(row)
    return rows


def main(pot_dir, sa_dir):
    rng = random.Random(SEED)
    pick = lambda pattern: sorted(glob.glob(pattern))
    shuffle = lambda items: rng.sample(items, len(items))

    phishing = load(shuffle(pick(os.path.join(pot_dir, "email", "*.eml"))), "phishing",
                    "phishing_pot", SIZES["phishing"], "P")
    easy = load(shuffle(pick(os.path.join(sa_dir, "easy-ham-*", "*.txt"))), "legit",
                "spamassassin/easy_ham", SIZES["easy-ham"], "H")
    hard = load(shuffle(pick(os.path.join(sa_dir, "hard-ham-*", "*.txt"))), "legit",
                "spamassassin/hard_ham", SIZES["hard-ham"], "HH")
    spam = load(shuffle(pick(os.path.join(sa_dir, "spam-*", "*.txt"))), "spam",
                "spamassassin/spam", SIZES["spam"], "S")

    train, test = [], []
    for group in (phishing, easy, hard, spam):
        group = shuffle(group)
        half = len(group) // 2
        train += group[:half]
        test += group[half:]
    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("train", shuffle(train)), ("test", shuffle(test))):
        with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=0)
        counts = {label: sum(r["label"] == label for r in rows)
                  for label in ("phishing", "spam", "legit")}
        print(f"{name}: {len(rows)} emails {counts}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
