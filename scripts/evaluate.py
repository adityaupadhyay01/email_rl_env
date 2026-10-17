"""
Evaluate every agent on the held-out test set with the same reward function.

    python scripts/evaluate.py                 # test split, INTEL_MODE=cache
    INTEL_MODE=live python scripts/evaluate.py # real lookups (fills data/intel_cache.json)
    python scripts/evaluate.py --split demo    # the 34 hand-written scenarios

Writes results/eval_<split>.json and prints a markdown table.
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.pg_agent import PolicyGradientAgent, PolicyNetwork  # noqa: E402
from agent.policy import InvestigatorAgent  # noqa: E402
from baseline_agent import BaselineAgent  # noqa: E402
from email_env import EmailEnv  # noqa: E402
from env.intel import ThreatIntel, parse_authentication_results  # noqa: E402
from env.tools import InvestigationTools  # noqa: E402
from evaluator.datasets import load_split  # noqa: E402
from evaluator.grader import run_episode  # noqa: E402
from scripts.train_pg import CachedTools  # noqa: E402

COLUMNS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
           ("false_positive_rate", "FPR"), ("phishing_miss_rate", "Phishing miss"),
           ("escalation_rate", "Escalated"), ("avg_tool_calls", "Checks/email"),
           ("avg_reward", "Avg reward")]


def agents():
    baseline = BaselineAgent()
    baseline.name = "keyword baseline"
    out = [baseline, InvestigatorAgent()]
    seeds = sorted(glob.glob(os.path.join(ROOT, "results", "pg_seeds", "seed*.npz")))
    if seeds:
        for path in seeds:
            agent = PolicyGradientAgent(PolicyNetwork.load(path))
            agent.name = f"policy gradient ({os.path.basename(path)[:-4]})"
            out.append(agent)
    elif os.path.exists(os.path.join(ROOT, "agent", "pg_policy.npz")):
        out.append(PolicyGradientAgent())
    return out


def auth_profile(rows):
    """How often real phishing fails SPF / DKIM / DMARC (headers kept)."""
    counts = {m: Counter() for m in ("spf", "dkim", "dmarc")}
    with_header = 0
    for row in rows:
        parsed = parse_authentication_results(row.get("headers") or {})
        if not parsed:
            continue
        with_header += 1
        for m in counts:
            counts[m][parsed.get(m) or "missing"] += 1
    return {"emails": len(rows), "with_auth_header": with_header,
            **{m: dict(c.most_common()) for m, c in counts.items()}}


def main(split):
    rows = load_split(split)
    intel = ThreatIntel(mode=os.getenv("INTEL_MODE", "cache"))
    tools = CachedTools(InvestigationTools(intel=intel))
    results = {"split": split, "emails": len(rows), "intel_mode": intel.mode,
               "labels": dict(Counter(r["label"] for r in rows)), "agents": {}}

    for agent in agents():
        run = run_episode(agent, EmailEnv(emails=rows, tools=tools))
        results["agents"][agent.name] = {
            "metrics": run["metrics"],
            "confusion": {f"{t}->{d}": n for (t, d), n in
                          sorted(Counter((e["label"], e["decision"]) for e in run["emails"]).items())},
        }
        if split != "demo":
            results["agents"][agent.name]["errors"] = [
                {k: e[k] for k in ("id", "label", "decision", "subject", "reason")}
                for e in run["emails"] if e["outcome"] in ("false_positive", "missed_threat")][:40]

    pg = [v["metrics"] for k, v in results["agents"].items() if k.startswith("policy gradient (")]
    if pg:
        import statistics
        results["policy_gradient_mean"] = {
            key: {"mean": round(statistics.mean(m[key] for m in pg), 3),
                  "std": round(statistics.pstdev(m[key] for m in pg), 3)}
            for key, _ in COLUMNS}

    if split != "demo":
        phishing = [r for r in load_split(split, auth="keep") if r["label"] == "phishing"]
        results["phishing_auth_profile"] = auth_profile(phishing)
        intel.save()

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"eval_{split}.json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    print(f"{split}: {results['emails']} emails {results['labels']}  intel={intel.mode}\n")
    print("| Agent | " + " | ".join(label for _, label in COLUMNS) + " |")
    print("|---|" + "---|" * len(COLUMNS))
    for name, data in results["agents"].items():
        m = data["metrics"]
        print(f"| {name} | " + " | ".join(f"{m[k]:.3f}" if isinstance(m[k], float) else str(m[k])
                                         for k, _ in COLUMNS) + " |")
    if pg:
        mean = results["policy_gradient_mean"]
        print("| policy gradient, mean of seeds | " +
              " | ".join(f"{mean[k]['mean']:.3f} ± {mean[k]['std']:.3f}" for k, _ in COLUMNS) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test", "demo"])
    main(parser.parse_args().split)
