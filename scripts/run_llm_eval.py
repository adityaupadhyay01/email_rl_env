"""
Run the LLM agent for real and record what it did.

    export HF_TOKEN=hf_...                      # or API_BASE_URL + HF_TOKEN for any
    python scripts/run_llm_eval.py --n 60       # OpenAI-compatible endpoint
    python scripts/run_llm_eval.py --split demo # the 34 hand-written scenarios

For a stratified sample of the held-out test set it records, per API call:
latency, tokens, the tool the model chose, and every failure (timeout, HTTP error,
invalid action, plain-text answer). The circuit breaker is disabled, so a failure
on one email does not hide the model's behaviour on the next; a failed call still
falls back to the rule policy for that step, and those emails are reported apart.

The same emails are also run through the rule-based investigator and the
policy-gradient agent, with the same tools and rewards, for a side-by-side table.

Writes results/llm_run_<model>_<time>.json and prints a markdown summary.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.llm_agent import LLMAgent  # noqa: E402
from agent.policy import InvestigatorAgent  # noqa: E402
from email_env import EmailEnv  # noqa: E402
from env.intel import ThreatIntel  # noqa: E402
from env.tools import InvestigationTools  # noqa: E402
from evaluator.datasets import load_split  # noqa: E402
from evaluator.grader import run_episode, summarise  # noqa: E402
from scripts.train_pg import CachedTools  # noqa: E402
from utils.config import API_BASE_URL, HF_TOKEN, MODEL_NAME  # noqa: E402

KEYS = ["precision", "recall", "f1", "false_positive_rate", "phishing_miss_rate",
        "escalation_rate", "avg_tool_calls", "avg_reward"]


def sample(rows, n, seed=7):
    """Stratified: keep the label mix of the test set."""
    rng = random.Random(seed)
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    out = []
    for label, group in sorted(by_label.items()):
        k = max(1, round(n * len(group) / len(rows)))
        out += rng.sample(group, min(k, len(group)))
    rng.shuffle(out)
    return out


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, max(0, round(q / 100 * (len(values) - 1))))
    return values[index]


def main(split, n, model):
    if not HF_TOKEN:
        sys.exit("HF_TOKEN is not set. Export your Hugging Face token (or another "
                 "OpenAI-compatible key with API_BASE_URL) and run again.")
    rows = load_split(split)
    if split != "demo":
        rows = sample(rows, n)
    tools = CachedTools(InvestigationTools(intel=ThreatIntel(mode=os.getenv("INTEL_MODE", "cache"))))

    agent = LLMAgent(model=model, max_failures=10 ** 9)
    print(f"Running {agent.name} on {len(rows)} emails from '{split}' via {API_BASE_URL} ...",
          file=sys.stderr)
    started = time.time()
    run = run_episode(agent, EmailEnv(emails=rows, tools=tools))
    wall = time.time() - started

    calls = agent.calls
    ok = [c for c in calls if c["ok"]]
    latencies = [c["latency_ms"] for c in calls if c.get("latency_ms") is not None]
    failures = {}
    for c in calls:
        if not c["ok"]:
            kind = c.get("error", "unknown").split(":")[0]
            failures[kind] = failures.get(kind, 0) + 1

    pure = [e for e in run["emails"] if all(s.get("source") == "llm" for s in e["steps"]
                                            if s["kind"] != "invalid")]
    comparison = {"llm (all emails, fallback steps included)": run["metrics"],
                  "llm (only emails decided fully by the model)": summarise(pure) if pure else None}
    for other in (InvestigatorAgent(),):
        comparison[other.name] = run_episode(other, EmailEnv(emails=rows, tools=tools))["metrics"]
    try:
        from agent.pg_agent import PolicyGradientAgent
        comparison["policy_gradient"] = run_episode(PolicyGradientAgent(),
                                                    EmailEnv(emails=rows, tools=tools))["metrics"]
    except FileNotFoundError:
        pass

    summary = {
        "model": agent.model, "endpoint": API_BASE_URL, "split": split, "emails": len(rows),
        "wall_seconds": round(wall, 1),
        "api_calls": len(calls), "successful_calls": len(ok),
        "failed_calls": len(calls) - len(ok), "failures_by_type": failures,
        "answer_modes": {m: sum(1 for c in ok if c.get("mode") == m)
                         for m in ("tool_call", "plain_text")},
        "fallback_steps": agent.fallback_steps, "llm_steps": agent.llm_steps,
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95),
                       "mean": round(statistics.mean(latencies), 1) if latencies else None,
                       "max": max(latencies) if latencies else None},
        "tokens": {"prompt": sum(c.get("prompt_tokens") or 0 for c in calls),
                   "completion": sum(c.get("completion_tokens") or 0 for c in calls)},
        "tool_choice_counts": {a: sum(1 for c in ok if c.get("action") == a)
                               for a in ("check_content", "check_sender", "check_url",
                                         "quarantine", "allow", "escalate")},
        "emails_fully_decided_by_model": len(pure),
        "comparison": comparison,
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = agent.model.replace("/", "_")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"llm_run_{safe_model}_{stamp}.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "calls": calls,
                   "emails": [{k: e.get(k) for k in ("id", "label", "decision", "outcome",
                                                     "reason", "tool_calls", "steps", "report")}
                              for e in run["emails"]]}, f, indent=1, ensure_ascii=False)

    print(f"\n## LLM agent run: {agent.model}\n")
    print(f"- {len(rows)} emails, {len(calls)} API calls in {wall:.0f} s; "
          f"{len(ok)} succeeded, {len(calls) - len(ok)} failed {failures or ''}")
    print(f"- latency per call: p50 {summary['latency_ms']['p50']} ms, "
          f"p95 {summary['latency_ms']['p95']} ms")
    print(f"- fallback steps: {agent.fallback_steps} of {agent.fallback_steps + agent.llm_steps}; "
          f"emails decided fully by the model: {len(pure)}")
    print(f"- tool choices: {summary['tool_choice_counts']}\n")
    print("| Agent | " + " | ".join(KEYS) + " |")
    print("|---|" + "---|" * len(KEYS))
    for name, m in comparison.items():
        if m:
            print(f"| {name} | " + " | ".join(str(m[k]) for k in KEYS) + " |")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["test", "demo"])
    parser.add_argument("--n", type=int, default=60, help="emails sampled from the test set")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    main(args.split, args.n, args.model)
