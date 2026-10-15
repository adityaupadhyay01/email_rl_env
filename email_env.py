"""
EmailEnv: a small reinforcement-learning style environment for email threat triage.

For every email the agent can first investigate, then must decide:

    investigate (costs a little)        decide (ends the email)
    ----------------------------        -----------------------
    check_content   -0.05               quarantine
    check_sender    -0.10               allow
    check_url       -0.15               escalate   (hand it to a human)

Reward for the decision:

    truth      quarantine   allow   escalate
    phishing      +1.0      -2.0      +0.3
    spam          +1.0      -0.5      +0.3
    legit         -1.0      +1.0      +0.3

So a missed phishing email is the worst outcome, blocking a real email is bad,
and asking a human is safe but earns little. The agent has to learn how much
checking is worth paying for before it acts.

The old actions `spam` and `not_spam` still work: they mean quarantine and allow.
"""

from db.fetch_data import fetch_emails
from env.state_processor import normalise_email, process_state
from env.tools import InvestigationTools

INVESTIGATE_COST = {"check_content": 0.05, "check_sender": 0.10, "check_url": 0.15}
DECISIONS = ("quarantine", "allow", "escalate")
LEGACY_ACTIONS = {"spam": "quarantine", "not_spam": "allow"}
ACTIONS = tuple(INVESTIGATE_COST) + DECISIONS

DECISION_REWARD = {
    ("phishing", "quarantine"): 1.0,
    ("phishing", "allow"): -2.0,
    ("spam", "quarantine"): 1.0,
    ("spam", "allow"): -0.5,
    ("legit", "quarantine"): -1.0,
    ("legit", "allow"): 1.0,
}
ESCALATE_REWARD = 0.3
INVALID_ACTION_PENALTY = 0.5
TIMEOUT_PENALTY = 0.5
MAX_STEPS_BEFORE_DECISION = 4   # after this many non-decision steps the email is escalated


class EmailEnv:
    def __init__(self, emails=None, tools=None, false_positive_cost=1.0,
                 escalate_reward=ESCALATE_REWARD):
        """
        false_positive_cost: what quarantining a genuine email costs (default 1.0, as in
        the table above). Raising it trades a few more escalations or missed threats for
        fewer blocked genuine emails; scripts/train_pg.py --fp-cost trains under it.
        escalate_reward: what handing an email to a person earns (default +0.3). Lower it
        to charge for analyst time, so a policy cannot hide behind escalation.
        """
        self.false_positive_cost = float(false_positive_cost)
        self.escalate_reward = float(escalate_reward)
        rows = emails if emails is not None else fetch_emails()
        if not rows:
            raise ValueError("EmailEnv needs at least one email")
        self.data = [normalise_email(row, i) for i, row in enumerate(rows)]
        self.tools = tools or InvestigationTools()
        self.reset()

    def reset(self):
        self.index = 0
        self.done = False
        self._start_email()
        return self._get_state()

    def step(self, action):
        """Apply one action. Returns (state, reward, done, info)."""
        if self.done:
            raise RuntimeError("Episode is finished. Call reset() to start again.")
        action = LEGACY_ACTIONS.get(action, action)

        if action in DECISIONS:
            return self._decide(action)

        # Anything that is not a decision uses up one of the limited steps.
        if self.steps_used >= MAX_STEPS_BEFORE_DECISION:
            return self._decide("escalate", timed_out=True)
        self.steps_used += 1

        if action not in INVESTIGATE_COST:
            info = {"email_id": self.current["id"], "kind": "invalid", "action": str(action),
                    "error": f"Unknown action. Valid actions: {', '.join(ACTIONS)}"}
            return self._get_state(), -INVALID_ACTION_PENALTY, False, info

        evidence = self.tools.run(action, self.current)
        already_checked = any(item["tool"] == action for item in self.evidence)
        if not already_checked:
            self.evidence.append(evidence)
        info = {"email_id": self.current["id"], "kind": "investigate", "action": action,
                "evidence": evidence, "repeated": already_checked}
        return self._get_state(), -INVESTIGATE_COST[action], False, info

    # ---- internals -------------------------------------------------------

    def _decide(self, action, timed_out=False):
        email = self.current
        truth = email["label"]

        if truth == "unknown":
            # Live triage of an unlabelled email: there is nothing to score against.
            reward, outcome = 0.0, "unlabelled"
        elif action == "escalate":
            reward, outcome = self.escalate_reward, "escalated"
            if timed_out:
                reward -= TIMEOUT_PENALTY
        else:
            reward = DECISION_REWARD[(truth, action)]
            if truth == "legit" and action == "quarantine":
                reward = -self.false_positive_cost
            if reward > 0:
                outcome = "correct"
            elif truth == "legit":
                outcome = "false_positive"
            else:
                outcome = "missed_threat"

        info = {"email_id": email["id"], "kind": "decision", "action": action,
                "truth": truth, "outcome": outcome, "timed_out": timed_out,
                "evidence": list(self.evidence), "steps_used": self.steps_used}

        self.index += 1
        self.done = self.index >= len(self.data)
        if not self.done:
            self._start_email()
        return self._get_state(), reward, self.done, info

    def _start_email(self):
        self.current = self.data[self.index]
        self.evidence = []
        self.steps_used = 0

    def _get_state(self):
        """What the agent sees. The true label is never part of it."""
        if self.done:
            return None
        steps_left = MAX_STEPS_BEFORE_DECISION - self.steps_used
        return process_state(self.current, self.evidence, steps_left, self.index, len(self.data))
