"""
A simple, explainable confidence number for a decision.

It is not a calibrated probability. It combines two things the analyst can see:
how far the evidence is from the decision boundary, and how much of the planned
evidence could actually be collected (failed lookups lower it).

    score       strongest risk if it is >= 0.80, else the average risk
    quarantine  score
    allow       1 - score
    escalate    reported as low confidence by construction
    then        x (0.5 + 0.5 * coverage)
"""


def decision_confidence(action, evidence):
    scored = [e for e in evidence if e.get("risk") is not None]
    if not scored:
        return 0.0
    risks = [e["risk"] for e in scored]
    strongest = max(risks)
    score = strongest if strongest >= 0.80 else sum(risks) / len(risks)
    if action == "quarantine":
        base = score
    elif action == "allow":
        base = 1 - score
    else:
        base = min(score, 1 - score)
    coverage = sum(e.get("coverage", 1.0) for e in evidence) / len(evidence)
    return round(base * (0.5 + 0.5 * coverage), 2)
