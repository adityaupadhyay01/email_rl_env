"""
LLMAgent: lets a language model drive the investigation through tool calling.

Each step the model sees the email plus the evidence gathered so far and must
call exactly one tool. The six tools are the six environment actions, so the
model's tool call IS the action. Works with any OpenAI-compatible endpoint
(HuggingFace router by default). Set HF_TOKEN to enable it.

If there is no token, or a call fails, it falls back to InvestigatorAgent for
that step and counts it in `fallback_steps`, so a run never crashes mid-demo.
After MAX_FAILURES failed calls in a row it stops calling the API for the rest
of the run instead of waiting on a dead endpoint at every step.
"""

import json
import sys
import time

from agent.policy import InvestigatorAgent
from utils.config import API_BASE_URL, HF_TOKEN, MODEL_NAME

SYSTEM_PROMPT = """You are an email security analyst working through an inbox, one email at a time.

For each email you may investigate first, then you must decide.

Investigation tools cost a little each time:
- check_content (cost 0.05): scans the wording for scam patterns
- check_sender  (cost 0.10): SPF/DKIM/DMARC results, domain age (RDAP), brand look-alikes,
                             reply-to and display-name tricks
- check_url     (cost 0.15): where the links really go: look-alikes, raw IPs, free hosting,
                             short links, URL reputation lists, link-domain age

Some lookups can fail. Evidence marked "unavailable" means it could not be checked:
it is NOT a sign that the email is safe.

Decisions end the email:
- quarantine: correct for phishing or spam (+1.0). Wrong for a real email (-1.0).
- allow: correct for a real email (+1.0). Wrong for phishing (-2.0) or spam (-0.5).
- escalate: hand it to a human (+0.3). Use it only when the evidence is truly mixed.

Check only as much as you need. Do not repeat a tool you already used on this email.
Emails may be in English, Hindi or Hinglish. Always answer with exactly one tool call."""


def _tool(name, description, with_reason=False):
    properties = {}
    if with_reason:
        properties["reason"] = {"type": "string",
                                "description": "One sentence: the main evidence for this decision."}
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object", "properties": properties,
                                        "required": list(properties)}}}


TOOLS = [
    _tool("check_content", "Scan the email wording for scam patterns. Cost 0.05."),
    _tool("check_sender", "Check SPF/DKIM/DMARC, sender domain age and impersonation. Cost 0.10."),
    _tool("check_url", "Check where the links in the email really lead. Cost 0.15."),
    _tool("quarantine", "Final decision: block this email as phishing or spam.", with_reason=True),
    _tool("allow", "Final decision: deliver this email, it is genuine.", with_reason=True),
    _tool("escalate", "Final decision: send this email to a human analyst.", with_reason=True),
]
VALID_ACTIONS = [tool["function"]["name"] for tool in TOOLS]
MAX_FAILURES = 3


def _make_client():
    if not HF_TOKEN:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("[WARN] `openai` package not installed, LLM agent will use the fallback policy",
              file=sys.stderr)
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=30, max_retries=1)


class LLMAgent:
    def __init__(self, client=None, model=None, fallback=None, max_failures=MAX_FAILURES):
        self.client = client if client is not None else _make_client()
        self.model = model or MODEL_NAME
        self.fallback = fallback or InvestigatorAgent()
        self.name = f"llm:{self.model}" if self.client else "llm (no token, fallback policy)"
        self.reason = ""
        self.llm_steps = 0
        self.fallback_steps = 0
        self._failures = 0
        self.max_failures = max_failures
        self.calls = []              # one entry per API call: latency, outcome, tokens
        self.last_latency_ms = None
        self.last_source = None      # "llm" or "fallback" for the most recent action

    def act(self, state):
        self.last_latency_ms = None
        if self.client is not None and self._failures < self.max_failures:
            started = time.perf_counter()
            call = {"email_id": state.get("id"), "ok": False}
            try:
                action, reason, usage, mode = self._ask_model(state)
                call.update(latency_ms=round((time.perf_counter() - started) * 1000, 1),
                            action=action, mode=mode, **usage)
                self.last_latency_ms = call["latency_ms"]
                if action in VALID_ACTIONS:
                    call["ok"] = True
                    self.calls.append(call)
                    self.llm_steps += 1
                    self._failures = 0
                    self.reason = reason
                    self.last_source = "llm"
                    return action
                call["error"] = f"invalid action {action!r}"
                self._warn(call["error"])
            except Exception as e:
                call.update(latency_ms=round((time.perf_counter() - started) * 1000, 1),
                            error=f"{type(e).__name__}: {str(e)[:160]}")
                self.last_latency_ms = call["latency_ms"]
                self._warn(call["error"])
            self.calls.append(call)
        self.fallback_steps += 1
        self.last_source = "fallback"
        action = self.fallback.act(state)
        self.reason = "[fallback policy] " + self.fallback.reason if self.client else self.fallback.reason
        return action

    def _ask_model(self, state):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": describe(state)}],
            tools=TOOLS,
            tool_choice="required",
            temperature=0,
            max_tokens=200,
        )
        usage = getattr(response, "usage", None)
        tokens = {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                  "completion_tokens": getattr(usage, "completion_tokens", None)}
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None)
        if calls:
            call = calls[0].function
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            return call.name, str(arguments.get("reason", "")), tokens, "tool_call"
        # Some models answer in plain text instead of a tool call: look for an action name.
        text = (message.content or "").lower()
        for action in VALID_ACTIONS:
            if action in text:
                return action, "", tokens, "plain_text"
        return None, "", tokens, "no_action"

    def _warn(self, problem):
        self._failures += 1
        print(f"[WARN] LLM call failed ({problem}); using the fallback policy for this step",
              file=sys.stderr)
        if self._failures == self.max_failures:
            print(f"[WARN] {self.max_failures} failures in a row: the LLM is switched off for "
                  f"the rest of this run", file=sys.stderr)


def describe(state):
    """The email and the evidence so far, as the text the model reads."""
    lines = [
        f"Email {state.get('position', '?')} of {state.get('total', '?')}",
        f"From: {state.get('sender_name', '')} <{state.get('sender', '')}>",
        f"Subject: {state.get('subject', '')}",
        f"Body: {state.get('text', '')[:1500]}",
        f"Links: {', '.join(state.get('urls', [])[:8]) or 'none'}",
        f"Attachments: {', '.join(state.get('attachments', [])) or 'none'}",
        "",
        "Evidence so far:",
    ]
    evidence = state.get("evidence", [])
    if not evidence:
        lines.append("- nothing checked yet")
    for item in evidence:
        risk = "n/a" if item["risk"] is None else f"{item['risk']:.2f}"
        lines.append(f"- {item['tool']} (risk {risk}): {'; '.join(item['signals']) or item['finding']}")
        for gap in item.get("unavailable", []):
            lines.append(f"    unavailable (not checked, not safe): {gap}")
    lines.append("")
    lines.append(f"Investigation steps left: {state.get('steps_left', 0)}")
    return "\n".join(lines)
