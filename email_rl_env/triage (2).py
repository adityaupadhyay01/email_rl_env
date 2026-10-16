"""
InvestigatorAgent: a tool-using agent with a hand-written, cost-aware policy.

It does not need an LLM or an API key. It starts with the cheapest check, stops
as soon as the evidence is strong, and only pays for the expensive URL check
when the cheaper checks leave a doubt.
"""

from env.tools import host_of, same_site

STRONG_RISK = 0.80      # one signal this strong is enough to quarantine
QUARANTINE_AT = 0.45    # average risk at or above this -> quarantine
ALLOW_AT = 0.25         # average risk at or below this -> allow; in between -> escalate
UNVERIFIED_DOUBT = 0.40 # would allow, but lookups failed and one risk is at least this -> escalate


class InvestigatorAgent:
    name = "investigator"

    def __init__(self, strong=STRONG_RISK, quarantine_at=QUARANTINE_AT, allow_at=ALLOW_AT,
                 unverified_doubt=UNVERIFIED_DOUBT):
        self.reason = ""
        self.strong, self.quarantine_at, self.allow_at = strong, quarantine_at, allow_at
        self.unverified_doubt = unverified_doubt

    def act(self, state):
        evidence = {item["tool"]: item for item in state.get("evidence", [])}
        risks = {tool: item["risk"] for tool, item in evidence.items() if item["risk"] is not None}

        if "check_content" not in evidence:
            return "check_content"
        if "check_sender" not in evidence:
            return "check_sender"

        if max(risks.values(), default=0) >= self.strong:
            return self._decide("quarantine", evidence)

        if state.get("urls") and "check_url" not in evidence:
            trusted = (risks.get("check_sender", 1) <= 0.10
                       and not evidence["check_sender"].get("unavailable")
                       and risks.get("check_content", 1) <= 0.20)
            if not (trusted and _links_stay_with_sender(state)):
                return "check_url"

        average = sum(risks.values()) / len(risks) if risks else 0.5
        if average >= self.quarantine_at:
            return self._decide("quarantine", evidence, average)
        if average <= self.allow_at:
            # Failed lookups are not evidence of safety. If something already looks
            # off and part of the evidence could not be collected, ask a human.
            gaps = [u for item in evidence.values() for u in item.get("unavailable", [])]
            if gaps and max(risks.values(), default=0) >= self.unverified_doubt:
                self.reason = (f"Average risk {average:.2f} looks low, but {len(gaps)} lookup(s) "
                               f"failed and one signal is {max(risks.values()):.2f}; needs a human")
                return "escalate"
            return self._decide("allow", evidence, average)
        return self._decide("escalate", evidence, average)

    def _decide(self, action, evidence, average=None):
        scored = [item for item in evidence.values() if item["risk"] is not None]
        strongest = max(scored, key=lambda item: item["risk"], default=None)
        if action == "quarantine" and strongest:
            self.reason = strongest["finding"]
        elif action == "allow":
            self.reason = f"No strong warning sign (average risk {average:.2f})"
        else:
            self.reason = f"Mixed signals (average risk {average:.2f}), needs a human look"
        return action


def _links_stay_with_sender(state):
    """Free check: do all links point back to the sender's own domain?"""
    sender = state.get("sender", "")
    domain = sender.rsplit("@", 1)[1] if "@" in sender else ""
    return all(same_site(host_of(url), domain) for url in state.get("urls", []))
