import json
import os
import sys

from utils.config import EMAIL_SOURCE

FALLBACK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "fallback.json")


def load_fallback():
    with open(FALLBACK_PATH, encoding="utf-8") as f:
        return json.load(f)


def fetch_emails(source=None):
    """
    Return the list of emails for the environment.

    source="local"    -> data/fallback.json (default, works offline)
    source="supabase" -> the `emails` table; falls back to the local file if
                         Supabase is not configured, unreachable or empty.
    """
    source = (source or EMAIL_SOURCE).lower()
    if source == "supabase":
        try:
            from db.supabase_client import get_client
            rows = get_client().table("emails").select("*").execute().data
            if rows:
                return rows
            print("[WARN] Supabase returned no emails, using data/fallback.json", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Supabase unavailable ({e}), using data/fallback.json", file=sys.stderr)
    return load_fallback()


if __name__ == "__main__":
    print(f"{len(fetch_emails())} emails loaded")
