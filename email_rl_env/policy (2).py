"""
PolicyGradientAgent: a policy learned with REINFORCE on EmailEnv.

The policy only sees what the environment exposes: which checks have been run,
the risk each returned, how much of each check's evidence was unavailable, whether
the email has links or attachments, and how many steps are left. It never sees the
label. It is a small two-layer network (numpy only) that outputs a probability for
each of the six actions; checks that are already done, or a URL check on an email
without links, are masked out.

Train it with:  python scripts/train_pg.py      (weights -> agent/pg_policy.npz)
"""

import os

import numpy as np

ACTIONS = ("check_content", "check_sender", "check_url", "quarantine", "allow", "escalate")
TOOLS = ACTIONS[:3]
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pg_policy.npz")


def features(state):
    """Fixed-length numeric view of a state."""
    evidence = {item["tool"]: item for item in state.get("evidence", [])}
    out = []
    risks = []
    for tool in TOOLS:
        item = evidence.get(tool)
        checked = item is not None
        risk = item.get("risk") if checked else None
        if risk is not None:
            risks.append(risk)
        out += [
            1.0 if checked else 0.0,
            risk if risk is not None else 0.0,
            1.0 if checked and risk is None else 0.0,
            (1.0 - item.get("coverage", 1.0)) if checked else 0.0,   # share unavailable
            1.0 if checked and risk is not None and risk >= 0.8 else 0.0,
        ]
    out += [
        max(risks) if risks else 0.0,
        sum(risks) / len(risks) if risks else 0.0,
        1.0 if state.get("urls") else 0.0,
        1.0 if state.get("attachments") else 0.0,
        state.get("steps_left", 4) / 4.0,
        1.0,                                                          # bias
    ]
    return np.array(out, dtype=np.float64)


N_FEATURES = len(features({"evidence": [], "urls": [], "steps_left": 4}))


def action_mask(state):
    done = {item["tool"] for item in state.get("evidence", [])}
    mask = np.ones(len(ACTIONS), dtype=bool)
    for i, tool in enumerate(TOOLS):
        if tool in done:
            mask[i] = False
    if not state.get("urls"):
        mask[ACTIONS.index("check_url")] = False
    return mask


class PolicyNetwork:
    def __init__(self, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.3, (N_FEATURES, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, (hidden, len(ACTIONS)))
        self.b2 = np.zeros(len(ACTIONS))

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x, mask):
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits = np.where(mask, logits, -1e9)
        logits = logits - logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return p, h

    def grad_log_prob(self, x, mask, action):
        """Gradients of log pi(action | x) and of the policy entropy."""
        p, h = self.forward(x, mask)
        dlogits = -p.copy()
        dlogits[action] += 1.0
        dlogits[~mask] = 0.0
        # entropy H = -sum p log p ; dH/dlogits = -p (log p + H)
        logp = np.log(np.clip(p, 1e-12, 1))
        entropy = -(p * logp).sum()
        dent = -p * (logp + entropy)
        dent[~mask] = 0.0
        grads = []
        for d in (dlogits, dent):
            dW2 = np.outer(h, d)
            db2 = d
            dh = (self.W2 @ d) * (1 - h ** 2)
            grads.append([np.outer(x, dh), dh, dW2, db2])
        return grads[0], grads[1], entropy

    def save(self, path=WEIGHTS_PATH, **meta):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 **{k: np.array(v) for k, v in meta.items()})

    @classmethod
    def load(cls, path=WEIGHTS_PATH):
        data = np.load(path)
        net = cls(hidden=data["W1"].shape[1])
        net.W1, net.b1, net.W2, net.b2 = data["W1"], data["b1"], data["W2"], data["b2"]
        return net


class PolicyGradientAgent:
    name = "policy_gradient"

    def __init__(self, network=None, greedy=True, seed=0):
        if network is None:
            if not os.path.exists(WEIGHTS_PATH):
                raise FileNotFoundError("No trained policy found. Run: python scripts/train_pg.py")
            network = PolicyNetwork.load()
        self.net = network
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)
        self.reason = ""

    def act(self, state):
        x, mask = features(state), action_mask(state)
        p, _ = self.net.forward(x, mask)
        index = int(np.argmax(p)) if self.greedy else int(self.rng.choice(len(ACTIONS), p=p))
        action = ACTIONS[index]
        if action in ("quarantine", "allow", "escalate"):
            scored = [e for e in state.get("evidence", []) if e.get("risk") is not None]
            top = max(scored, key=lambda e: e["risk"], default=None)
            self.reason = (f"Learned policy: {p[index]:.0%} for {action}"
                           + (f"; strongest evidence: {top['finding']}" if top else ""))
        return action
