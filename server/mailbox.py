"""
A controlled demo mailbox on disk, so the live demo can execute real actions.

    data/demo_inbox/seed/        the starting emails (.eml), checked in
    data/demo_inbox/live/        working copy the demo changes
        inbox/  quarantine/  review/  delivered/
        actions.jsonl            audit log: every move, by the agent or a human

Triage moves a message out of inbox/: quarantine -> quarantine/, escalate ->
review/ (the human review queue), allow -> delivered/. A human can then release
or quarantine what is waiting in review/. Next to each message a .json file keeps
the agent's evidence, decision and incident report.

To connect a real mailbox instead, swap this class for one that moves messages
over IMAP (UID MOVE) or the Gmail / Microsoft Graph APIs; the agent side does not
change.
"""

import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone

from env.eml import parse_eml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_DIR = os.path.join(ROOT, "data", "demo_inbox", "seed")
LIVE_DIR = os.getenv("DEMO_INBOX_DIR", os.path.join(ROOT, "data", "demo_inbox", "live"))
FOLDERS = ("inbox", "quarantine", "review", "delivered")
DESTINATION = {"quarantine": "quarantine", "escalate": "review", "allow": "delivered"}
NAME_RE = re.compile(r"^[\w.-]+\.eml$")


class DemoMailbox:
    def __init__(self, live_dir=LIVE_DIR, seed_dir=SEED_DIR):
        self.live, self.seed = live_dir, seed_dir
        self.lock = threading.Lock()
        if not os.path.isdir(os.path.join(self.live, "inbox")):
            self.reset()

    # ---- state -------------------------------------------------------------

    def reset(self):
        with self.lock:
            if os.path.isdir(self.live):
                shutil.rmtree(self.live)
            for folder in FOLDERS:
                os.makedirs(os.path.join(self.live, folder))
            for name in sorted(os.listdir(self.seed)):
                if name.endswith(".eml"):
                    shutil.copy(os.path.join(self.seed, name), os.path.join(self.live, "inbox", name))
            self._log("system", "reset", None, "inbox", "Demo inbox restored from seed")

    def listing(self):
        out = {}
        for folder in FOLDERS:
            items = []
            for name in sorted(os.listdir(os.path.join(self.live, folder))):
                if not name.endswith(".eml"):
                    continue
                parsed = parse_eml(self.read(folder, name))
                item = {"file": name, "folder": folder, "from": parsed["sender"],
                        "from_name": parsed["sender_name"], "subject": parsed["subject"],
                        "date": parsed["date"]}
                meta = self._meta_path(folder, name)
                if os.path.exists(meta):
                    with open(meta, encoding="utf-8") as f:
                        triage = json.load(f)
                    item["decision"] = triage.get("decision")
                    item["confidence"] = triage.get("confidence")
                    item["human"] = triage.get("human")
                items.append(item)
            out[folder] = items
        return {"folders": out, "log": self.log(25)}

    def read(self, folder, name):
        with open(self._path(folder, name), "rb") as f:
            return f.read()

    def triage_record(self, folder, name):
        meta = self._meta_path(folder, name)
        if not os.path.exists(meta):
            return None
        with open(meta, encoding="utf-8") as f:
            return json.load(f)

    # ---- actions -----------------------------------------------------------

    def add(self, raw, name=None):
        """Drop a pasted email into the inbox."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        name = name if name and NAME_RE.match(name) else f"pasted-{stamp}.eml"
        with open(self._path("inbox", name), "wb") as f:
            f.write(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
        self._log("user", "add", None, "inbox", f"Pasted email saved as {name}")
        return name

    def apply_decision(self, name, result):
        """Execute the agent's decision on a message that is in the inbox."""
        target = DESTINATION[result["decision"]]
        self._move("inbox", target, name)
        with open(self._meta_path(target, name), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
        what = {"quarantine": "Quarantined", "review": "Sent to human review",
                "delivered": "Delivered to the user"}[target]
        self._log(result.get("agent", "agent"), result["decision"], "inbox", target,
                  f"{what}: {name} (confidence {result['confidence']:.0%})")
        return {"folder": target, "message": f"{what}: moved to {target}/"}

    def human_review(self, name, verdict, analyst="analyst"):
        """A human resolves an escalated message: release it or quarantine it."""
        if verdict not in ("release", "quarantine"):
            raise ValueError("verdict must be release or quarantine")
        target = "delivered" if verdict == "release" else "quarantine"
        self._move("review", target, name)
        meta = self.triage_record(target, name) or {}
        meta["human"] = {"analyst": analyst, "verdict": verdict,
                         "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with open(self._meta_path(target, name), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        self._log(analyst, f"human_{verdict}", "review", target, f"Analyst chose {verdict}: {name}")
        return {"folder": target}

    def log(self, limit=25):
        path = os.path.join(self.live, "actions.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        return [json.loads(line) for line in reversed(lines)]

    # ---- internals ---------------------------------------------------------

    def _path(self, folder, name):
        if folder not in FOLDERS or not NAME_RE.match(name or ""):
            raise ValueError(f"bad mailbox path {folder}/{name}")
        return os.path.join(self.live, folder, name)

    def _meta_path(self, folder, name):
        return self._path(folder, name)[:-4] + ".json"

    def _move(self, source, target, name):
        with self.lock:
            src = self._path(source, name)
            if not os.path.exists(src):
                raise FileNotFoundError(f"{name} is not in {source}/")
            shutil.move(src, self._path(target, name))
            meta = self._meta_path(source, name)
            if os.path.exists(meta):
                shutil.move(meta, self._meta_path(target, name))

    def _log(self, actor, action, source, target, message):
        entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "actor": actor, "action": action, "from": source, "to": target,
                 "message": message}
        with open(os.path.join(self.live, "actions.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
