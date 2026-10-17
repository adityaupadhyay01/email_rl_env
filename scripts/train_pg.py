"""
Train the policy-gradient agent (REINFORCE with a baseline) on EmailEnv.

    python scripts/train_pg.py [--epochs 60] [--seed 0]

Data: the train half of the public set (data/public/train.json), authentication
headers stripped like in evaluation, threat-intel lookups in INTEL_MODE (default
"cache", so training is reproducible). 20% of the train half is held out as a
validation set to pick the best checkpoint. The test half is never touched here.

Writes agent/pg_policy.npz and results/pg_training_seed<N>.json (learning curve).
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.pg_agent import ACTIONS, PolicyGradientAgent, PolicyNetwork, action_mask, features  # noqa: E402
from email_env import EmailEnv  # noqa: E402
from env.intel import ThreatIntel  # noqa: E402
from env.tools import InvestigationTools  # noqa: E402
from evaluator.datasets import load_split  # noqa: E402
from evaluator.grader import run_episode  # noqa: E402


class CachedTools:
    """Tool results are deterministic for a fixed intel cache, so compute each once."""

    def __init__(self, tools):
        self.tools, self.memo = tools, {}

    def run(self, tool, email):
        key = (tool, email["id"])
        if key not in self.memo:
            self.memo[key] = self.tools.run(tool, email)
        return self.memo[key]


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        """Gradient ASCENT step."""
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p += self.lr * mh / (np.sqrt(vh) + self.eps)


def rollout(net, env, rng):
    """One sampled pass over env's emails. Returns per-email trajectories."""
    state, done = env.reset(), False
    episodes, current = [], {"steps": [], "return": 0.0}
    while not done:
        x, mask = features(state), action_mask(state)
        p, _ = net.forward(x, mask)
        a = int(rng.choice(len(ACTIONS), p=p))
        state, reward, done, info = env.step(ACTIONS[a])
        current["steps"].append((x, mask, a))
        current["return"] += reward
        if info["kind"] == "decision":
            episodes.append(current)
            current = {"steps": [], "return": 0.0}
    return episodes


def train(epochs=300, seed=0, out=None, fp_cost=1.0, init=None, esc=0.3, batch=32, lr=0.003, entropy_start=0.1, hidden=32):
    rng = np.random.default_rng(seed)
    rows = load_split("train")
    random.Random(seed).shuffle(rows)
    cut = int(len(rows) * 0.8)
    fit_rows, val_rows = rows[:cut], rows[cut:]

    tools = CachedTools(InvestigationTools(intel=ThreatIntel(mode=os.getenv("INTEL_MODE", "cache"))))
    val_env = EmailEnv(emails=val_rows, tools=tools, false_positive_cost=fp_cost, escalate_reward=esc)

    net = PolicyNetwork.load(init) if init else PolicyNetwork(hidden=hidden, seed=seed)
    if init:
        entropy_start = min(entropy_start, 0.02)   # fine-tuning: keep what it already knows
    opt = Adam(net.params(), lr=lr)
    baseline = 0.0
    curve, best = [], (-1e9, None, -1)
    started = time.time()

    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(fit_rows)
        env = EmailEnv(emails=fit_rows, tools=tools, false_positive_cost=fp_cost, escalate_reward=esc)
        episodes = rollout(net, env, rng)
        entropy_coef = max(0.005, entropy_start * (1 - epoch / (0.7 * epochs)))
        returns = []
        for start in range(0, len(episodes), batch):
            chunk = episodes[start:start + batch]
            G = np.array([ep["return"] for ep in chunk])
            adv = G - baseline            # not normalised: a flat batch must not be amplified
            grads = [np.zeros_like(p) for p in net.params()]
            for ep, advantage in zip(chunk, adv):
                for x, mask, a in ep["steps"]:
                    glog, gent, _ = net.grad_log_prob(x, mask, a)
                    for i in range(4):
                        grads[i] += advantage * glog[i] + entropy_coef * gent[i]
            opt.step([g / len(chunk) for g in grads])
            baseline = 0.9 * baseline + 0.1 * G.mean()
            returns += list(G)

        val = run_episode(PolicyGradientAgent(net, greedy=True), val_env)["metrics"]
        point = {"epoch": epoch, "train_avg_return": round(float(np.mean(returns)), 4),
                 "val_avg_reward": val["avg_reward"], "val_f1": val["f1"],
                 "val_escalation_rate": val["escalation_rate"],
                 "val_avg_tool_calls": val["avg_tool_calls"]}
        curve.append(point)
        if val["avg_reward"] > best[0]:
            best = (val["avg_reward"], [p.copy() for p in net.params()], epoch)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train return {point['train_avg_return']:+.3f}  "
                  f"val reward {val['avg_reward']:+.3f}  val F1 {val['f1']:.3f}  "
                  f"escalate {val['escalation_rate']:.2f}  tools {val['avg_tool_calls']:.2f}")

    net.W1, net.b1, net.W2, net.b2 = best[1]
    net.save(out or os.path.join(ROOT, "agent", "pg_policy.npz"), best_epoch=best[2], seed=seed, fp_cost=fp_cost, escalate_reward=esc,
             best_val_reward=best[0])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    tag = f"seed{seed}" if (fp_cost, esc) == (1.0, 0.3) else f"fp{fp_cost:g}_esc{esc:g}_seed{seed}"
    with open(os.path.join(ROOT, "results", f"pg_training_{tag}.json"), "w") as f:
        json.dump({"algorithm": "REINFORCE with moving-average baseline, entropy bonus",
                   "initialised_from": init,
                   "network": f"{len(features({'evidence': []}))}-{hidden}-6 tanh MLP, action masking",
                   "fit_emails": len(fit_rows), "val_emails": len(val_rows),
                   "epochs": epochs, "seed": seed, "false_positive_cost": fp_cost, "escalate_reward": esc, "best_epoch": best[2],
                   "seconds": round(time.time() - started, 1), "curve": curve}, f, indent=1)
    print(f"best validation reward {best[0]:+.3f} at epoch {best[2]}; saved agent/pg_policy.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="weights file (default agent/pg_policy.npz)")
    parser.add_argument("--fp-cost", type=float, default=1.0,
                        help="reward cost of quarantining a genuine email (default 1.0)")
    parser.add_argument("--init", default=None,
                        help="start from these weights (fine-tune a trained policy)")
    parser.add_argument("--escalate-reward", type=float, default=0.3,
                        help="reward for handing an email to a person (default 0.3)")
    args = parser.parse_args()
    train(args.epochs, args.seed, args.out, args.fp_cost, args.init, args.escalate_reward)
