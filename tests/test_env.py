"""Run with:  python -m pytest"""

import json
from types import SimpleNamespace

import pytest

from agent.llm_agent import LLMAgent, VALID_ACTIONS
from agent.policy import InvestigatorAgent
from baseline_agent import BaselineAgent
from db.fetch_data import fetch_emails, load_fallback
from email_env import (ACTIONS, ESCALATE_REWARD, INVESTIGATE_COST, MAX_STEPS_BEFORE_DECISION,
                       EmailEnv)
from env.tools import InvestigationTools, extract_urls
from evaluator.grader import evaluate, run_episode

PHISH = {"id": "P1", "sender_name": "Bharat Bank KYC", "sender": "kyc@bharatbank-secure.example",
         "auth": "fail", "subject": "KYC", "label": "phishing",
         "text": "Aapka KYC expire ho gaya hai. Turant update karein: http://bharatbank-secure.example/kyc"}
LEGIT = {"id": "L1", "sender_name": "Riya Sharma", "sender": "riya.sharma@technova.example",
         "auth": "pass", "subject": "Meeting", "label": "legit", "text": "Meeting at 5 PM"}
SPAM = {"id": "S1", "text": "Win a free iPhone!!!", "label": "spam"}


# ---- environment ---------------------------------------------------------

def test_state_never_contains_the_label():
    env = EmailEnv([PHISH, LEGIT])
    state = env.reset()
    assert "label" not in state and "auth" not in state and "category" not in state
    state, _, _, _ = env.step("check_sender")
    assert "label" not in json.dumps(state)


@pytest.mark.parametrize("email,action,expected", [
    (PHISH, "quarantine", 1.0), (PHISH, "allow", -2.0),
    (SPAM, "quarantine", 1.0), (SPAM, "allow", -0.5),
    (LEGIT, "allow", 1.0), (LEGIT, "quarantine", -1.0),
    (PHISH, "escalate", ESCALATE_REWARD), (LEGIT, "escalate", ESCALATE_REWARD),
])
def test_decision_rewards(email, action, expected):
    _, reward, done, info = EmailEnv([email]).step(action)
    assert reward == expected and done and info["kind"] == "decision"


def test_legacy_actions_still_work():
    assert EmailEnv([SPAM]).step("spam")[1] == 1.0
    assert EmailEnv([LEGIT]).step("not_spam")[1] == 1.0
    old_row = {"text": "Meeting at 5 PM", "label": "not_spam"}       # original two-column format
    assert EmailEnv([old_row]).step("not_spam")[1] == 1.0


def test_investigation_costs_and_adds_evidence():
    env = EmailEnv([PHISH])
    state, reward, done, info = env.step("check_sender")
    assert reward == -INVESTIGATE_COST["check_sender"] and not done
    assert state["evidence"][0]["tool"] == "check_sender" and state["evidence"][0]["risk"] >= 0.9
    assert state["steps_left"] == MAX_STEPS_BEFORE_DECISION - 1
    # repeating a tool costs again but adds nothing new
    state, reward, _, info = env.step("check_sender")
    assert reward == -INVESTIGATE_COST["check_sender"] and info["repeated"]
    assert len(state["evidence"]) == 1


def test_evidence_resets_for_the_next_email():
    env = EmailEnv([PHISH, LEGIT])
    env.step("check_content")
    state, _, done, _ = env.step("quarantine")
    assert not done and state["id"] == "L1" and state["evidence"] == []


def test_endless_investigation_is_escalated():
    env = EmailEnv([PHISH])
    for _ in range(MAX_STEPS_BEFORE_DECISION):
        _, _, done, _ = env.step("check_content")
        assert not done
    _, reward, done, info = env.step("check_content")
    assert done and info["action"] == "escalate" and info["timed_out"]
    assert reward == pytest.approx(ESCALATE_REWARD - 0.5)


def test_invalid_action_is_penalised_not_fatal():
    env = EmailEnv([PHISH])
    _, reward, done, info = env.step("delete_everything")
    assert reward == -0.5 and not done and info["kind"] == "invalid"


def test_step_after_done_raises():
    env = EmailEnv([LEGIT])
    env.step("allow")
    with pytest.raises(RuntimeError):
        env.step("allow")


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        EmailEnv([{"text": "hi", "label": "maybe"}])


# ---- data ----------------------------------------------------------------

def test_fallback_dataset_is_well_formed():
    rows = load_fallback()
    assert len(rows) >= 30 and len({r["id"] for r in rows}) == len(rows)
    assert {r["label"] for r in rows} == {"phishing", "spam", "legit"}
    for row in rows:                       # every address and link stays on the reserved TLD
        assert row["sender"].endswith(".example")
        assert row["urls"] == extract_urls(row["text"])
        for url in row["urls"]:
            assert ".example" in url or "203.0.113." in url


def test_supabase_source_falls_back_to_local(capsys):
    assert fetch_emails("supabase") == load_fallback()     # no credentials in the test run
    assert "fallback" in capsys.readouterr().err


# ---- tools ---------------------------------------------------------------

def test_brand_impersonation_detection():
    tools = InvestigationTools()
    assert tools.impersonated_brand("bharatbank-secure.example") == "Bharat Bank"
    assert tools.impersonated_brand("vidyutsevaa-bill.example") == "VidyutSeva"      # typo
    assert tools.impersonated_brand("mailbox.example", "JobSetu Careers") == "JobSetu"
    assert tools.impersonated_brand("netbanking.bharatbank.example") is None         # official
    assert tools.impersonated_brand("citycollege.example", "Prof. Anil Verma") is None
    assert tools.impersonated_brand("paypal-secure-login.com") == "PayPal"           # real brands
    assert tools.impersonated_brand("www.paypal.com") is None
    assert tools.impersonated_brand("sbi-kyc.example") == "SBI"
    assert tools.impersonated_brand("groups.example", "Groups team") is None         # 'ups' is whole-word


def test_url_tool_local_analysis():
    tools = InvestigationTools()
    assert tools.check_url({"urls": ["http://203.0.113.45/login"]})["risk"] == 0.85
    assert tools.check_url({"urls": [], "text": "no links"})["risk"] is None
    lookalike = tools.check_url({"urls": ["https://paypal-account-verify.top/x"]})
    assert lookalike["risk"] == 0.9 and "PayPal" in lookalike["finding"]
    hosted = tools.check_url({"urls": ["https://x.a.run.app/login"]})
    assert hosted["risk"] >= 0.55 and "free platform" in " ".join(hosted["signals"])


def test_short_link_that_cannot_be_expanded_is_not_trusted():
    item = InvestigationTools().check_url({"urls": ["http://bit.example/sal-sheet"]})
    assert item["risk"] >= 0.5 and "hides its real destination" in item["finding"]
    assert any("short-link expansion" in gap for gap in item["unavailable"])


def test_failed_lookups_are_reported_and_never_make_an_email_look_safer():
    tools = InvestigationTools()                     # INTEL_MODE=off: every lookup fails
    email = {"sender": "news@perkel.com", "sender_name": "Marc", "text": "hi",
             "urls": ["https://perkel.com/a"]}
    sender, url = tools.check_sender(email), tools.check_url(email)
    for item in (sender, url):
        assert item["unavailable"] and item["coverage"] < 1
        assert "Not fully verified" in item["finding"]
        assert item["risk"] > 0.05                   # a fully verified clean result scores 0.05
    assert any("SPF/DKIM/DMARC" in gap for gap in sender["unavailable"])
    assert any("domain age" in gap for gap in sender["unavailable"])


def test_content_tool_reads_hinglish():
    risk = InvestigationTools().check_content(PHISH)["risk"]
    assert risk >= 0.5
    assert InvestigationTools().check_content(LEGIT)["risk"] == 0.05


def test_risky_attachment_raises_content_risk():
    item = InvestigationTools().check_content({"text": "see attached", "attachments": ["invoice.html"]})
    assert item["risk"] >= 0.3 and "attachment" in item["finding"]


# ---- agents --------------------------------------------------------------

def test_investigator_handles_clear_cases():
    result = run_episode(InvestigatorAgent(), EmailEnv([PHISH, LEGIT]))
    assert [e["decision"] for e in result["emails"]] == ["quarantine", "allow"]
    assert all(e["outcome"] == "correct" and e["report"] for e in result["emails"])
    assert result["emails"][0]["tool_calls"] == 2       # stopped early, never paid for check_url
    assert "Not verified" in result["emails"][1]["report"]   # the domain-age gap is reported


def test_investigator_escalates_instead_of_allowing_when_evidence_is_missing():
    agent = InvestigatorAgent()
    state = {"urls": [], "evidence": [
        {"tool": "check_content", "risk": 0.05, "unavailable": [], "coverage": 1.0},
        {"tool": "check_sender", "risk": 0.45, "unavailable": ["domain age: timeout"], "coverage": 0.67},
    ]}
    assert agent.act(state) == "escalate"
    state["evidence"][1].update(unavailable=[], coverage=1.0, risk=0.10)
    assert agent.act(state) == "allow"


def test_investigator_beats_baseline_on_the_demo_set():
    baseline, agent = evaluate(BaselineAgent()), evaluate(InvestigatorAgent())
    assert agent["emails"] == baseline["emails"] == len(load_fallback())
    assert agent["total_reward"] > baseline["total_reward"]
    assert agent["missed_phishing"] < baseline["missed_phishing"]
    for m in (baseline, agent):
        assert m["correct"] + m["false_positives"] + m["missed_threats"] + m["escalated"] == m["emails"]


def test_classification_metrics():
    from evaluator.grader import classification
    emails = [{"label": "phishing", "decision": "quarantine"}, {"label": "phishing", "decision": "allow"},
              {"label": "spam", "decision": "escalate"}, {"label": "legit", "decision": "quarantine"},
              {"label": "legit", "decision": "allow"}]
    m = classification(emails)
    assert m["precision"] == 0.5 and m["recall"] == round(1 / 3, 3)
    assert m["false_positive_rate"] == 0.5 and m["phishing_miss_rate"] == 0.5
    assert m["escalation_rate"] == 0.2 and m["threats_quarantined_or_escalated"] == round(2 / 3, 3)


def test_policy_gradient_agent_loads_and_only_sees_the_state():
    from agent.pg_agent import PolicyGradientAgent, action_mask, features
    env = EmailEnv([PHISH, LEGIT])
    state = env.reset()
    assert features(state).shape == features({"evidence": []}).shape
    assert not action_mask({"evidence": [], "urls": []})[2]      # no links -> no URL check
    result = run_episode(PolicyGradientAgent(), env)
    assert len(result["emails"]) == 2 and all(e["decision"] for e in result["emails"])


class FakeClient:
    """Stands in for the OpenAI client: replies with a scripted list of tool calls."""

    def __init__(self, script):
        self.script, self.seen = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.seen.append(kwargs)
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        name, args, content = reply
        calls = [SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(args)))] if name else None
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls, content=content))])


def test_llm_agent_uses_tool_calls_as_actions():
    client = FakeClient([("check_sender", {}, None),
                         ("quarantine", {"reason": "look-alike bank domain"}, None)])
    agent = LLMAgent(client=client, model="fake-model")
    result = run_episode(agent, EmailEnv([PHISH]))
    email = result["emails"][0]
    assert email["decision"] == "quarantine" and email["reason"] == "look-alike bank domain"
    assert agent.llm_steps == 2 and agent.fallback_steps == 0
    # the second prompt must contain the evidence from the first tool call, and never the label
    second_prompt = client.seen[1]["messages"][1]["content"]
    assert "check_sender" in second_prompt and "phishing" not in second_prompt
    assert [t["function"]["name"] for t in client.seen[0]["tools"]] == VALID_ACTIONS
    assert set(VALID_ACTIONS) == set(ACTIONS)


def test_llm_agent_falls_back_when_the_api_fails():
    client = FakeClient([RuntimeError("network down")] * 10)
    agent = LLMAgent(client=client, model="fake-model")
    result = run_episode(agent, EmailEnv([PHISH]))
    assert result["emails"][0]["decision"] == "quarantine"
    assert agent.llm_steps == 0 and agent.fallback_steps == 3
    assert len(client.seen) == 3                 # then the circuit breaker opens:
    agent.act(EmailEnv([PHISH]).reset())
    assert len(client.seen) == 3                 # no further API calls


def test_llm_agent_reads_plain_text_answers():
    agent = LLMAgent(client=FakeClient([(None, {}, "I would allow this email.")]), model="fake-model")
    assert agent.act(EmailEnv([LEGIT]).reset()) == "allow"


def test_llm_agent_without_token_uses_fallback(monkeypatch):
    monkeypatch.setattr("agent.llm_agent.HF_TOKEN", None)
    agent = LLMAgent()
    assert agent.client is None and "fallback" in agent.name
    assert agent.act(EmailEnv([LEGIT]).reset()) == "check_content"
