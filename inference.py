"""
Run an agent through the email environment from the command line.

    python inference.py                      # best available agent, step-by-step log
    python inference.py --agent baseline     # the original keyword agent
    python inference.py --agent investigator # tool-using agent, no API key needed
    python inference.py --agent llm          # LLM agent (needs HF_TOKEN)
    python inference.py --agent pg           # trained policy-gradient agent
    python inference.py --compare            # table: every agent side by side
    python inference.py --reports            # also print the incident reports
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # so Rs/Hinglish text prints on Windows too

from agent.llm_agent import LLMAgent
from agent.policy import InvestigatorAgent
from baseline_agent import BaselineAgent
from evaluator.grader import run_episode
from utils.config import HF_TOKEN


def make_agent(kind="auto"):
    if kind == "auto":
        kind = "llm" if HF_TOKEN else "investigator"
    if kind == "baseline":
        agent = BaselineAgent()
        agent.name = "baseline"
        return agent
    if kind == "investigator":
        return InvestigatorAgent()
    if kind == "llm":
        return LLMAgent()
    if kind in ("pg", "policy_gradient"):
        from agent.pg_agent import PolicyGradientAgent
        return PolicyGradientAgent()
    raise ValueError(f"Unknown agent: {kind}")


def run(kind="auto", show_reports=False):
    agent = make_agent(kind)
    name = getattr(agent, "name", kind)
    print(f"[START] task=email env=email_rl model={name}")

    result = run_episode(agent)
    for entry in result["log"]:
        print(f"[STEP] step={entry['step']} action={entry['action']} "
              f"reward={entry['reward']:.2f} done={str(entry['done']).lower()} error=null")

    metrics = result["metrics"]
    rewards = ",".join(f"{entry['reward']:.2f}" for entry in result["log"])
    print(f"[END] success={str(metrics['score'] > 0.5).lower()} steps={len(result['log'])} "
          f"score={metrics['score']:.2f} rewards={rewards}")

    if isinstance(agent, LLMAgent) and agent.fallback_steps:
        print(f"[NOTE] {agent.fallback_steps} of {agent.fallback_steps + agent.llm_steps} steps "
              f"used the fallback policy, not the LLM", file=sys.stderr)
    if show_reports:
        for email in result["emails"]:
            print("\n" + email["report"])
    return result


def compare(kinds=None):
    kinds = kinds or (["baseline", "investigator", "pg"] + (["llm"] if HF_TOKEN else []))
    results = [run_episode(make_agent(kind)) for kind in kinds]

    rows = [
        ("Emails", "emails"), ("Correct decisions", "correct"), ("Accuracy", "accuracy"),
        ("Real emails blocked", "false_positives"), ("Threats missed", "missed_threats"),
        ("  of which phishing", "missed_phishing"), ("Sent to a human", "escalated"),
        ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
        ("False-positive rate", "false_positive_rate"), ("Phishing miss rate", "phishing_miss_rate"),
        ("Avg tool calls / email", "avg_tool_calls"), ("Total reward", "total_reward"),
        ("Score (0-1)", "score"),
    ]
    width = max(14, *(len(r["agent"]) + 2 for r in results))
    print(f"{'':<24}" + "".join(f"{r['agent']:>{width}}" for r in results))
    for label, key in rows:
        cells = ""
        for r in results:
            value = r["metrics"][key]
            text = f"{value:.0%}" if key == "accuracy" else str(value)
            cells += f"{text:>{width}}"
        print(f"{label:<24}{cells}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email threat-triage environment runner")
    parser.add_argument("--agent", default="auto",
                        choices=["auto", "baseline", "investigator", "pg", "llm"])
    parser.add_argument("--compare", action="store_true", help="compare all agents in a table")
    parser.add_argument("--reports", action="store_true", help="print incident reports")
    args = parser.parse_args()

    if args.compare:
        compare()
    else:
        run(args.agent, args.reports)
