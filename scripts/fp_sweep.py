"""
False-positive cost sweep: how the trained policy's behaviour moves when blocking a
genuine email costs more.

    python scripts/train_pg.py --fp-cost 2 --seed 0 --init results/pg_seeds/seed0.npz \\
        --epochs 200 --out results/pg_fp/fp2_seed0.npz        # for each cost and seed
    python scripts/fp_sweep.py                                # test-set table

Policies for cost 1 are results/pg_seeds/seed*.npz; others are results/pg_fp/fp<C>_seed*.npz
(fine-tuned from the cost-1 policies; trained from scratch, every seed collapsed to
"always escalate"). Everything is scored on the held-out test split. Writes
results/fp_sweep.json.
"""

import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.pg_agent import PolicyGradientAgent, PolicyNetwork  # noqa: E402
from agent.policy import InvestigatorAgent  # noqa: E402
from email_env import EmailEnv  # noqa: E402
from env.intel import ThreatIntel  # noqa: E402
from env.tools import InvestigationTools  # noqa: E402
from evaluator.datasets import load_split  # noqa: E402
from evaluator.grader import run_episode  # noqa: E402
from scripts.train_pg import CachedTools  # noqa: E402

KEYS = ["false_positive_rate", "phishing_miss_rate", "escalation_rate", "precision", "recall",
        "f1", "avg_tool_calls", "avg_reward"]


def main():
    rows = load_split("test")
    tools = CachedTools(InvestigationTools(intel=ThreatIntel(mode=os.getenv("INTEL_MODE", "cache"))))
    groups = {"fp1-esc0.3": sorted(glob.glob(os.path.join(ROOT, "results", "pg_seeds", "seed*.npz")))}
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "pg_fp", "fp*_seed*.npz"))):
        name = os.path.basename(path).split("_seed")[0]
        groups.setdefault(name if "-esc" in name else name + "-esc0.3", []).append(path)

    out = {"split": "test", "emails": len(rows), "costs": {}}
    rules = run_episode(InvestigatorAgent(), EmailEnv(emails=rows, tools=tools))["metrics"]
    out["rules"] = {k: rules[k] for k in KEYS}
    for cost, paths in groups.items():
        runs = []
        for path in paths:
            agent = PolicyGradientAgent(PolicyNetwork.load(path))
            m = run_episode(agent, EmailEnv(emails=rows, tools=tools))["metrics"]
            runs.append({"weights": os.path.relpath(path, ROOT), **{k: m[k] for k in KEYS}})
        out["costs"][cost] = {
            "seeds": runs,
            "mean": {k: round(statistics.mean(r[k] for r in runs), 3) for k in KEYS},
            "std": {k: round(statistics.pstdev(r[k] for r in runs), 3) for k in KEYS},
        }
    with open(os.path.join(ROOT, "results", "fp_sweep.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("| Policy | " + " | ".join(KEYS) + " |")
    print("|---|" + "---|" * len(KEYS))
    print("| rules | " + " | ".join(str(out["rules"][k]) for k in KEYS) + " |")
    for cost, data in out["costs"].items():
        for r in data["seeds"]:
            print(f"| {cost}, {os.path.basename(r['weights'])} | "
                  + " | ".join(str(r[k]) for k in KEYS) + " |")
        print(f"| **{cost}, mean** | "
              + " | ".join(f"{data['mean'][k]} ± {data['std'][k]}" for k in KEYS) + " |")


if __name__ == "__main__":
    main()
