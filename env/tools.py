"""
Investigation tools.

These are the "look deeper" actions an agent can take inside EmailEnv before it
decides what to do with an email. Each tool returns one piece of evidence:

    {"tool": "check_sender", "risk": 0.9, "finding": "...", "signals": [...],
     "unavailable": ["domain age: RDAP timeout after 4s"], "coverage": 0.67}

risk         0 (looks safe) .. 1 (looks malicious); None when there was nothing to
             check (no sender, no links)
unavailable  lookups that failed or could not run. These are reported, never
             treated as "safe"
coverage     share of the planned checks that actually produced an answer

What each tool uses:
    check_content  local text analysis (English and Hinglish patterns, risky attachments)
    check_sender   SPF/DKIM/DMARC results, domain age (RDAP), brand look-alikes,
                   reply-to and display-name tricks, free-mail and free-tenant senders
    check_url      local URL analysis (IPs, punycode, look-alikes, shared hosting),
                   short-link expansion, URL reputation (Safe Browsing, URLhaus,
                   OpenPhish) and link-domain age

The external lookups live in env/intel.py.
"""

import ipaddress
import json
import os
import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

BRANDS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "brands.json")

URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

# (name, weight, patterns). English and Hinglish, matched on subject + body.
CONTENT_SIGNALS = [
    ("asks for credentials", 0.30, [
        r"\botp\b", r"\bpin\b", r"password", r"\bcvv\b", r"\bkyc\b", r"card number",
        r"bank details", r"pan card", r"aadhaar", r"verify your (account|identity|email|details)",
        r"confirm your (account|identity|details|password|email)", r"seed phrase",
        r"update your (payment|billing|account) (details|information|method)",
        r"(sign|log)[ -]?in to (verify|confirm|restore|unlock)",
    ]),
    ("asks for payment", 0.30, [
        r"gift cards?", r"registration fee", r"processing fee", r"\bpay rs\b",
        r"send money", r"transfer rs", r"wire transfer", r"\bbitcoin\b", r"\bbtc\b",
        r"customs (fee|charge)", r"delivery fee",
    ]),
    ("money lure", 0.25, [
        r"\bwon\b", r"\bwin\b", r"lottery", r"prize", r"refund", r"cashback",
        r"selected for", r"claim your", r"earn rs", r"\binaam\b", r"congratulations",
        r"free spins", r"\bbonus\b", r"\breward", r"inheritance", r"\$\s?\d{1,3}(,\d{3})+",
    ]),
    ("urgency or threat", 0.20, [
        r"urgent", r"immediately", r"within 24 hours", r"24 ghante", r"\bturant\b",
        r"\bjaldi\b", r"today only", r"tonight", r"suspen", r"disconnect",
        r"band ho", r"expire", r"blocked", r"final notice", r"action required",
        r"unusual (sign-?in|activity|login)", r"(locked|limited|deactivated|closed) account",
        r"account (will be|has been|is) (locked|limited|closed|deactivated|on hold)",
        r"last (chance|warning|reminder)",
    ]),
    # Common phishing wording in the other languages seen in real campaigns
    # (Portuguese, Spanish, French, German, Dutch, Italian).
    ("non-English account or payment lure", 0.30, [
        r"\bsenha\b", r"\bcontraseña\b", r"mot de passe", r"\bpasswort\b", r"wachtwoord",
        r"clique aqui", r"haga clic", r"cliquez ici", r"klicken sie", r"klik hier",
        r"\bbloquead[oa]\b", r"\bgesperrt\b", r"\bbloqué", r"\bgeblokkeerd\b",
        r"\bverifi(que|car|ez|zieren)\b", r"\bconfirm(e|ar|ez)\s+(sua|su|votre)\b",
        r"\bparab[ée]ns\b", r"\bfelicidades\b", r"\bf[ée]licitations\b", r"herzlichen glückwunsch",
        r"\bpr[êe]mio\b", r"\bgratuit[oa]?\b", r"\bgr[áa]tis\b", r"\bgewinn", r"\bgagn",
        r"\breembolso\b", r"\bremboursement\b", r"rückerstattung", r"\bterugbetaling\b",
        r"\becono?mi[sz]e\b", r"\bdesconto\b", r"\bdescuento\b", r"\bsconto\b",
        r"\bcolis\b", r"\bencomenda\b", r"\bpaket\b", r"\bpakket\b", r"\btaxa\b",
        r"\bclaim\b", r"\burgente\b", r"\bdringend\b", r"\bimm[ée]diatement\b",
    ]),
    ("asks for secrecy", 0.20, [
        r"confidential", r"don'?t tell", r"keep this between",
    ]),
    ("bulk-spam wording", 0.20, [
        r"click here", r"to be removed", r"remove me", r"opt[- ]?out", r"\b100% (free|guaranteed)",
        r"risk[- ]free", r"order now", r"act now", r"money back", r"mortgage", r"viagra",
        r"weight loss", r"lose weight", r"make money", r"work from home", r"no obligation",
        r"\$\$+", r"!!!+", r"credit (card )?debt", r"lowest (price|rate)",
    ]),
    ("promotional wording", 0.10, [
        r"\bfree\b", r"discount", r"\boffer\b", r"buy now", r"\bsale\b", r"limited seats",
        r"unsubscribe",
    ]),
]

RISKY_ATTACHMENT_RE = re.compile(
    r"\.(html?|shtml|svg|exe|scr|js|vbs|jar|iso|img|lnk|hta|bat|cmd|ps1|docm|xlsm|pptm|"
    r"zip|rar|7z|gz)$", re.IGNORECASE)

AUTHORITY_TITLE_RE = re.compile(r"\b(ceo|cfo|cto|director|chairman|md|principal|manager)\b",
                                re.IGNORECASE)
VOWEL_RE = re.compile(r"[aeiou]")
CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxz]{5,}")


def random_looking(domain):
    """Machine-generated names such as 1pweq8mahfwtqdfck8tccpyo8itt.com."""
    from env.intel import registrable_domain
    label = registrable_domain(domain).split(".")[0]
    if len(label) < 10:
        return False
    digits = sum(c.isdigit() for c in label)
    vowels = len(VOWEL_RE.findall(label))
    return (3 <= digits < len(label)) or bool(CONSONANT_RUN_RE.search(label)) \
        or vowels / len(label) < 0.2


EMAIL_IN_NAME_RE = re.compile(r"[\w.+-]+@([\w-]+\.[\w.-]+)")


def extract_urls(text):
    """Find links in plain text, without trailing punctuation."""
    return [u.rstrip(".,;:!?)") for u in URL_RE.findall(text or "")]


def host_of(url):
    """Hostname of a URL, lower-cased. Works with or without a scheme."""
    if "://" not in url:
        url = "http://" + url
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def same_site(host, domain):
    """True when `host` is `domain` or one of its subdomains."""
    host, domain = host.lower(), domain.lower()
    return bool(domain) and (host == domain or host.endswith("." + domain))


def _evidence(tool, signals, unavailable, coverage, clean_finding, clean_risk):
    """Turn (risk, text) signals plus the lookups that failed into one evidence dict."""
    item = {"tool": tool, "unavailable": unavailable, "coverage": round(coverage, 2)}
    if signals:
        signals.sort(key=lambda s: s[0], reverse=True)
        item.update(risk=round(signals[0][0], 2), finding=signals[0][1],
                    signals=[text for _, text in signals])
        return item
    if unavailable:
        # No warning sign, but some lookups failed: say so, and do not call it safe.
        clean_risk = round(clean_risk + 0.30 * (1 - coverage), 2)
        clean_finding = (f"{clean_finding}. Not fully verified: {len(unavailable)} "
                         f"lookup(s) unavailable")
    item.update(risk=clean_risk, finding=clean_finding, signals=[])
    return item


class InvestigationTools:
    def __init__(self, intel=None, brands=None):
        if intel is None:
            from env.intel import ThreatIntel
            intel = ThreatIntel()
        self.intel = intel
        if brands is None:
            with open(BRANDS_PATH, encoding="utf-8") as f:
                brands = json.load(f)
        self.brands = dict(brands.get("brands", {}))
        self.brands.update(brands.get("scenario_brands", {}))
        self.freemail = set(brands.get("freemail", []))
        self.abused_tlds = set(brands.get("abused_tlds", []))

    # ---- the three tools -------------------------------------------------

    def run(self, tool, email):
        return getattr(self, tool)(email)

    def check_content(self, email):
        """Cheap text scan for the usual phishing and spam wording."""
        text = f"{email.get('subject', '')} {email.get('text', '')}".lower()
        risk, hits = 0.05, []
        for name, weight, patterns in CONTENT_SIGNALS:
            matched = [m.group(0) for p in patterns if (m := re.search(p, text))]
            if matched:
                risk += weight
                hits.append(f"{name} ({', '.join(matched[:3])})")
        risky = [a for a in email.get("attachments") or [] if RISKY_ATTACHMENT_RE.search(a)]
        if risky:
            risk += 0.25
            hits.append(f"risky attachment type ({', '.join(risky[:2])})")
        risk = round(min(risk, 0.75), 2)   # wording alone is never treated as proof
        finding = "Wording: " + "; ".join(hits) if hits else "Wording looks normal"
        return {"tool": "check_content", "risk": risk, "finding": finding, "signals": hits,
                "unavailable": [], "coverage": 1.0}

    def check_sender(self, email):
        """Who really sent this? Authentication, domain age, impersonation tricks."""
        address = (email.get("sender") or "").lower()
        if "@" not in address:
            return {"tool": "check_sender", "risk": None, "finding": "No sender address",
                    "signals": [], "unavailable": ["sender: the email has no From address"],
                    "coverage": 0.0}

        domain = address.rsplit("@", 1)[1]
        name = email.get("sender_name") or ""
        signals, unavailable, verified = [], [], []
        answered = 1                       # the local checks below always run

        # 1. SPF / DKIM / DMARC
        auth = self.intel.auth(email)
        if auth.get("status") == "ok":
            answered += 1
            spf, dkim, dmarc = auth.get("spf"), auth.get("dkim"), auth.get("dmarc")
            if dmarc == "fail":
                signals.append((0.85, f"DMARC failed for {domain} ({auth['source']})"))
            if spf in ("fail", "softfail"):
                signals.append((0.60, f"SPF {spf}: the sending server is not allowed to send "
                                      f"for this domain"))
            if dkim == "fail":
                signals.append((0.60, "DKIM signature failed verification"))
            if dmarc == "pass" or (spf == "pass" and dkim == "pass"):
                verified.append("authentication passed")
            elif not any(v in ("fail", "softfail") for v in (spf, dkim, dmarc)):
                verified.append(f"authentication spf={spf} dkim={dkim} dmarc={dmarc}")
        else:
            unavailable.append(f"SPF/DKIM/DMARC: {auth.get('error')}")

        # 2. Domain age at the time the email was sent
        age = self.intel.domain_age(domain, at=email.get("date"))
        if age.get("status") == "ok":
            answered += 1
            days = age["age_days"]
            if days < 30:
                signals.append((0.75, f"Sender domain {age['domain']} was registered only "
                                      f"{days} days before this email"))
            elif days < 180:
                signals.append((0.45, f"Sender domain {age['domain']} is new ({days} days old)"))
            else:
                verified.append(f"domain registered {days // 365} year(s) earlier")
        elif age.get("status") == "not_applicable":
            answered += 1
            signals.append((0.60, f"Sent from a free subdomain on {age['platform']}, "
                                  f"not from an organisation's own domain"))
        else:
            unavailable.append(f"domain age: {age.get('error')}")

        # 3. Local checks: impersonation and header tricks
        if random_looking(domain):
            signals.append((0.65, f"Sender domain {domain} looks machine-generated"))
        brand = self.impersonated_brand(domain, name)
        if brand:
            signals.append((0.90, f"Claims to be '{brand}' but {domain} is not an official "
                                  f"{brand} domain"))
        shown = EMAIL_IN_NAME_RE.search(name)
        if shown and not same_site(domain, shown.group(1).lower()) \
                and not same_site(shown.group(1).lower(), domain):
            signals.append((0.60, f"Display name shows {shown.group(0)} but the real address "
                                  f"is {address}"))
        freemail = domain in self.freemail
        if freemail and AUTHORITY_TITLE_RE.search(name):
            signals.append((0.70, f"Uses an authority title ('{name}') on a free-mail address"))
        reply = (email.get("reply_to") or "").lower()
        if "@" in reply:
            reply_domain = reply.rsplit("@", 1)[1]
            from env.intel import registrable_domain
            if registrable_domain(reply_domain) != registrable_domain(domain):
                signals.append((0.50 if not freemail else 0.40,
                                f"Replies go to a different domain ({reply_domain})"))
        if freemail and not signals:
            signals.append((0.30, "Personal free-mail address"))

        coverage = answered / 3
        clean = f"No warning signs for {domain}" + (f" ({', '.join(verified)})" if verified else "")
        clean_risk = 0.05 if auth.get("status") == "ok" and "authentication passed" in verified \
            else 0.10
        return _evidence("check_sender", signals, unavailable, coverage, clean, clean_risk)

    def check_url(self, email, max_lookups=3):
        """Where do the links really go? The most expensive tool."""
        urls = email.get("urls") or extract_urls(email.get("text", ""))
        if not urls:
            return {"tool": "check_url", "risk": None, "finding": "No links in this email",
                    "signals": [], "unavailable": [], "coverage": 1.0}

        from env.intel import SHORTENERS, registrable_domain, shared_platform
        sender = email.get("sender") or ""
        sender_site = registrable_domain(sender.rsplit("@", 1)[1]) if "@" in sender else ""
        signals, unavailable, checked = [], [], []
        attempted = answered = 0
        looked_up = set()

        for url in urls:
            host = host_of(url)
            if not host:
                continue
            note = ""
            if host in SHORTENERS:
                attempted += 1
                expanded = self.intel.expand_short_link(url)
                if expanded.get("status") == "ok":
                    answered += 1
                    note = f"Short link {host} expands to {host_of(expanded['target'])}: "
                    url, host = expanded["target"], host_of(expanded["target"])
                else:
                    signals.append((0.50, f"Short link {host} hides its real destination"))
                    unavailable.append(f"short-link expansion ({host}): {expanded.get('error')}")
                    continue
            checked.append(host)

            # Local analysis: always available
            if _is_ip(host):
                signals.append((0.85, f"{note}Link points to a raw IP address ({host})"))
                continue
            if random_looking(host):
                signals.append((0.55, f"{note}Link domain {host} looks machine-generated"))
            if "xn--" in host:
                signals.append((0.60, f"{note}Link uses a punycode (look-alike letters) "
                                      f"domain: {host}"))
            brand = self.impersonated_brand(host)
            if brand:
                signals.append((0.90, f"{note}Link domain {host} imitates '{brand}'"))
            platform = shared_platform(host)
            if platform:
                signals.append((0.55, f"{note}Link is hosted on a free platform ({platform})"))
            if "@" in (urlparse(url).netloc or ""):
                signals.append((0.60, f"{note}Link hides its real host behind an '@'"))
            tld = host.rsplit(".", 1)[-1]
            if tld in self.abused_tlds:
                signals.append((0.35, f"{note}Link uses a TLD often abused for phishing (.{tld})"))
            if url.lower().startswith("http://"):
                signals.append((0.30, f"Link to {host} does not use HTTPS"))

            # External lookups, capped per email to keep the tool affordable
            site = registrable_domain(host)
            if site in looked_up or len(looked_up) >= max_lookups:
                continue
            looked_up.add(site)

            attempted += 1
            reputation = self.intel.url_reputation(url)
            if reputation.get("status") == "ok":
                answered += 1
                if reputation["verdict"] == "malicious":
                    signals.append((0.95, f"{note}Link {host} is listed as "
                                          f"{reputation.get('threat') or 'malicious'} by "
                                          f"{', '.join(reputation['listed_by'])}"))
            else:
                unavailable.append(f"URL reputation ({host}): {reputation.get('error')}")

            if site != sender_site and not platform:
                attempted += 1
                age = self.intel.domain_age(host, at=email.get("date"))
                if age.get("status") == "ok":
                    answered += 1
                    if age["age_days"] < 30:
                        signals.append((0.65, f"{note}Link domain {age['domain']} was registered "
                                              f"{age['age_days']} days before this email"))
                elif age.get("status") != "not_applicable":
                    unavailable.append(f"link domain age ({site}): {age.get('error')}")

        coverage = (1 + answered) / (1 + attempted)
        clean = "Links look normal: " + ", ".join(sorted(set(checked))[:4])
        return _evidence("check_url", signals, unavailable, coverage, clean, 0.05)

    # ---- helpers ---------------------------------------------------------

    def impersonated_brand(self, domain, display_name=""):
        """
        Return a brand name when `domain` (or the display name) pretends to be that
        brand without being one of its official domains. Catches look-alikes such
        as paypal-secure-login.com and typos such as hdfcbnak.com.
        """
        domain = domain.lower()
        for official_list in self.brands.values():
            if any(same_site(domain, d) for d in official_list):
                return None                      # a real brand domain
        if domain in self.freemail and not display_name:
            return None

        labels = [p for p in re.split(r"[.\-_]", domain) if p]
        name = display_name.lower()
        for brand in self.brands:
            words = re.findall(r"[a-z0-9]+", brand.lower())
            token = "".join(words)
            if len(token) < 3:
                continue
            # Display name: the brand as whole words ("Bharat Bank KYC", "BharatBank"),
            # so "Groups team" does not match "Steam".
            if re.search(r"(?<![a-z0-9])" + r"[\s._-]*".join(map(re.escape, words)) + r"(?![a-z0-9])",
                         name):
                return brand
            short = len(token) <= 4             # "ups", "dhl", "sbi": whole labels only
            for label in labels:
                if short:
                    if label == token:
                        return brand
                elif token in label or (len(label) >= 5 and
                                        SequenceMatcher(None, token, label).ratio() >= 0.85):
                    return brand
        return None


def _is_ip(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False
