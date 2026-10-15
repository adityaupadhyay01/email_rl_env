"""
Real threat-intelligence lookups for the investigation tools.

    domain_age(domain, at)   RDAP (the IANA-backed successor of WHOIS), via rdap.org
    url_reputation(url)      Google Safe Browsing, URLhaus and the OpenPhish feed
    expand_short_link(url)   one HTTP HEAD request, redirect not followed
    auth(email)              SPF / DKIM / DMARC: the receiving server's
                             Authentication-Results header first, live DNS checks second

Every lookup returns a dict with a "status" key:

    {"status": "ok", ...data..., "source": "rdap.org"}
    {"status": "unavailable", "error": "timeout after 4s", "source": "rdap.org"}

A failed lookup is NEVER turned into "safe". The tools report it as unavailable
evidence, and the agent treats it as uncertainty.

Modes (env var INTEL_MODE):
    live   call the services, cache successful answers in data/intel_cache.json (default)
    cache  only use answers already in the cache, never touch the network
           (makes evaluation runs reproducible)
    off    no external lookups at all; everything external is unavailable

Optional API keys (env vars): GOOGLE_SAFE_BROWSING_KEY, URLHAUS_AUTH_KEY.
OpenPhish's free community feed and rdap.org need no key.
Live SPF/DKIM/DMARC checks need `dnspython`, `dkimpy` and `pyspf`; without them
those checks are reported as unavailable.
"""

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(ROOT, "data", "intel_cache.json")

RDAP_URL = "https://rdap.org/domain/{domain}"
GSB_URL = "https://safebrowsing.googleapis.com/v4/threatMatches:find?key={key}"
URLHAUS_URL = "https://urlhaus-api.abuse.ch/v1/url/"
OPENPHISH_URL = os.getenv("OPENPHISH_FEED", "https://openphish.com/feed.txt")

# Two-label public suffixes we see in practice. Enough to find the registrable
# domain without pulling in the full Public Suffix List.
TWO_LEVEL_SUFFIXES = {
    "co.in", "gov.in", "org.in", "net.in", "ac.in", "edu.in", "nic.in", "res.in",
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "com.br",
    "com.mx", "co.jp", "ne.jp", "co.kr", "com.cn", "com.tw", "com.sg", "co.za",
    "com.tr", "com.ar", "co.nz", "com.my", "co.id", "com.pk", "com.ng", "com.ua",
    "gov.br", "org.br", "net.br", "com.co", "com.pe", "com.ve", "com.vn", "com.ph",
    "co.th", "com.eg", "com.sa", "org.za", "gob.mx", "com.hk", "or.jp", "ac.jp",
}

# Platforms that hand out free subdomains. Their registration date says nothing
# about the tenant, so domain age is not meaningful there.
SHARED_PLATFORMS = {
    "onmicrosoft.com", "sharepoint.com", "gappssmtp.com", "firebaseapp.com", "web.app",
    "github.io", "herokuapp.com", "blogspot.com", "netlify.app", "vercel.app",
    "pages.dev", "workers.dev", "r2.dev", "wixsite.com", "weebly.com", "godaddysites.com",
    "glitch.me", "azurewebsites.net", "cloudfront.net", "amazonaws.com", "appspot.com",
    "ipfs.io", "dweb.link", "webflow.io", "framer.app", "square.site", "mystrikingly.com",
    "000webhostapp.com", "replit.app", "surge.sh", "translate.goog", "ngrok.io",
    "ngrok-free.app", "trycloudflare.com", "backblazeb2.com", "storage.googleapis.com",
    "googleusercontent.com", "run.app", "cloudfunctions.net", "blob.core.windows.net",
    "web.core.windows.net", "onrender.com", "fly.dev", "beehiiv.com", "notion.site",
    "typeform.com", "forms.gle", "jotform.com", "canva.site", "linktr.ee",
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rb.gy",
    "shorturl.at", "tiny.cc", "rebrand.ly", "buff.ly", "t.ly", "s.id", "lnkd.in",
    "bitly.com", "short.gy", "urlz.fr", "v.gd", "qrco.de", "bl.ink",
    "tiny.example", "bit.example",          # the fictional shorteners in the demo scenarios
}


def registrable_domain(host):
    """mail.alerts.hdfcbank.com -> hdfcbank.com, x.sbi.co.in -> sbi.co.in."""
    host = (host or "").lower().strip(".")
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    if ".".join(parts[-2:]) in TWO_LEVEL_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def shared_platform(host):
    """The free-hosting platform a host lives on, or None."""
    host = (host or "").lower()
    for platform in SHARED_PLATFORMS:
        if host == platform or host.endswith("." + platform):
            return platform
    return None


def _unavailable(source, error):
    return {"status": "unavailable", "source": source, "error": error}


def _parse_time(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class IntelCache:
    """Successful lookups, stored on disk so evaluation can be re-run offline."""

    def __init__(self, path=CACHE_PATH):
        self.path = path
        self.data = {}
        self.dirty = False
        self.lock = threading.Lock()
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except (OSError, ValueError):
                self.data = {}

    def get(self, key):
        entry = self.data.get(key)
        return dict(entry["result"]) if entry else None

    def put(self, key, result):
        with self.lock:
            self.data[key] = {"result": result,
                              "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            self.dirty = True

    def save(self):
        if not (self.path and self.dirty):
            return
        with self.lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
            self.dirty = False


class ThreatIntel:
    def __init__(self, mode=None, cache=None, http=None, timeout=4.0):
        self.mode = (mode or os.getenv("INTEL_MODE", "live")).strip().lower()
        if self.mode not in ("live", "cache", "off"):
            raise ValueError(f"INTEL_MODE must be live, cache or off, not {self.mode!r}")
        self.cache = cache if cache is not None else IntelCache()
        self.http = http                      # injectable for tests; requests.Session otherwise
        self.timeout = timeout
        self.gsb_key = os.getenv("GOOGLE_SAFE_BROWSING_KEY")
        self.urlhaus_key = os.getenv("URLHAUS_AUTH_KEY")
        self._failures = {}                   # provider -> consecutive network failures
        self._openphish = None                # (fetched_at, set of urls)
        self.calls = 0                        # real network calls made

    # ---- plumbing ---------------------------------------------------------

    def _session(self):
        if self.http is None:
            import requests
            self.http = requests.Session()
            self.http.headers["User-Agent"] = "email-threat-triage/2.0 (research)"
        return self.http

    def _down(self, provider):
        return self._failures.get(provider, 0) >= 2

    def _request(self, provider, method, url, **kwargs):
        """One HTTP call. Raises on network trouble and counts it for the breaker."""
        if self._down(provider):
            raise ConnectionError(f"{urlparse(url).hostname or provider} unreachable, "
                                  f"skipped after repeated failures")
        self.calls += 1
        try:
            response = getattr(self._session(), method)(url, timeout=self.timeout, **kwargs)
        except Exception as e:
            self._failures[provider] = self._failures.get(provider, 0) + 1
            host = urlparse(url).hostname or provider
            name = type(e).__name__
            if "Timeout" in name or "timed out" in str(e).lower():
                raise ConnectionError(f"{host} did not answer within {self.timeout:g}s") from None
            if name in ("ProxyError", "ConnectionError", "SSLError", "ConnectTimeout") \
                    or "connect" in str(e).lower():
                raise ConnectionError(f"could not reach {host} (network blocked or down)") from None
            raise ConnectionError(f"{host}: {name}") from None
        self._failures[provider] = 0
        return response

    def _cached(self, key, source, fetch):
        """Cache first; then the network if the mode allows it."""
        hit = self.cache.get(key)
        if hit is not None:
            hit["cached"] = True
            return hit
        if self.mode == "off":
            return _unavailable(source, "external lookups are switched off (INTEL_MODE=off)")
        if self.mode == "cache":
            return _unavailable(source, "not in the offline cache (INTEL_MODE=cache)")
        try:
            result = fetch()
        except ConnectionError as e:
            return _unavailable(source, str(e))
        except Exception as e:                       # bad JSON, unexpected answer
            return _unavailable(source, f"unexpected response: {type(e).__name__}")
        if result.get("status") == "ok":
            self.cache.put(key, result)
        return result

    # ---- domain age (RDAP) -----------------------------------------------

    def domain_age(self, host, at=None):
        """Age of the registrable domain in days, measured at `at` (the email's date)."""
        domain = registrable_domain(host)
        platform = shared_platform(host)
        if platform:
            return {"status": "not_applicable", "source": "rdap.org", "domain": domain,
                    "platform": platform,
                    "error": f"{host} is a free subdomain on {platform}; its age says nothing"}
        if not domain or "." not in domain or re.fullmatch(r"[\d.]+", domain):
            return _unavailable("rdap.org", f"not a domain name: {host!r}")
        if domain.endswith((".example", ".test", ".invalid", ".localhost")):
            return _unavailable("rdap.org", f"{domain} is a reserved test domain, no registry")

        result = self._cached(f"rdap:{domain}", "rdap.org", lambda: self._rdap(domain))
        if result.get("status") != "ok":
            return result

        registered = _parse_time(result.get("registered"))
        when = _parse_time(at) or datetime.now(timezone.utc)
        out = dict(result, domain=domain)
        if registered is None:
            return _unavailable("rdap.org", f"RDAP record for {domain} has no registration date")
        age = (when - registered).days
        if age < 0:
            return _unavailable("rdap.org", f"{domain} was registered after this email was sent "
                                            f"(re-registered since), so its age then is unknown")
        out["age_days"] = age
        return out

    def _rdap(self, domain):
        response = self._request("rdap", "get", RDAP_URL.format(domain=domain),
                                 headers={"Accept": "application/rdap+json"})
        if response.status_code == 404:
            return {"status": "ok", "source": "rdap.org", "registered": None,
                    "note": "domain not found in its registry (unregistered or expired)"}
        if response.status_code != 200:
            return _unavailable("rdap.org", f"HTTP {response.status_code}")
        events = response.json().get("events", [])
        registered = next((e.get("eventDate") for e in events
                           if e.get("eventAction") == "registration"), None)
        return {"status": "ok", "source": "rdap.org", "registered": registered}

    # ---- URL reputation ---------------------------------------------------

    def url_reputation(self, url):
        """
        Ask every configured reputation source. "ok" when at least one answered.
        verdict: "malicious" if any source lists it, else "not_listed".
        Not being listed is weak evidence: new phishing URLs are often not listed yet.
        """
        answers, errors = [], []
        providers = [("openphish", self._openphish_check)]
        if self.gsb_key:
            providers.append(("google_safe_browsing", self._gsb_check))
        if self.urlhaus_key:
            providers.append(("urlhaus", self._urlhaus_check))

        for name, check in providers:
            result = self._cached(f"{name}:{url}", name, lambda check=check: check(url))
            if result.get("status") == "ok":
                answers.append(dict(result, source=name))
            else:
                errors.append(f"{name}: {result.get('error')}")

        if not answers:
            return {"status": "unavailable", "source": "url reputation",
                    "error": "; ".join(errors) or "no reputation source configured"}
        listed = [a for a in answers if a.get("listed")]
        return {"status": "ok", "verdict": "malicious" if listed else "not_listed",
                "sources": [a["source"] for a in answers],
                "listed_by": [a["source"] for a in listed],
                "threat": next((a.get("threat") for a in listed if a.get("threat")), None),
                "missing": errors}

    def _openphish_check(self, url):
        now = time.time()
        if self._openphish is None or now - self._openphish[0] > 6 * 3600:
            response = self._request("openphish", "get", OPENPHISH_URL)
            if response.status_code != 200:
                raise ConnectionError(f"OpenPhish feed HTTP {response.status_code}")
            feed = {line.strip().rstrip("/") for line in response.text.splitlines() if line.strip()}
            self._openphish = (now, feed)
        listed = url.rstrip("/") in self._openphish[1]
        return {"status": "ok", "listed": listed, "threat": "phishing" if listed else None}

    def _gsb_check(self, url):
        body = {"client": {"clientId": "email-threat-triage", "clientVersion": "2.0"},
                "threatInfo": {"threatTypes": ["MALWARE", "SOCIAL_ENGINEERING",
                                               "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
                               "platformTypes": ["ANY_PLATFORM"],
                               "threatEntryTypes": ["URL"],
                               "threatEntries": [{"url": url}]}}
        response = self._request("google_safe_browsing", "post",
                                 GSB_URL.format(key=self.gsb_key), json=body)
        if response.status_code != 200:
            raise ConnectionError(f"Safe Browsing HTTP {response.status_code}")
        matches = response.json().get("matches", [])
        return {"status": "ok", "listed": bool(matches),
                "threat": matches[0].get("threatType") if matches else None}

    def _urlhaus_check(self, url):
        response = self._request("urlhaus", "post", URLHAUS_URL, data={"url": url},
                                 headers={"Auth-Key": self.urlhaus_key})
        if response.status_code != 200:
            raise ConnectionError(f"URLhaus HTTP {response.status_code}")
        data = response.json()
        status = data.get("query_status")
        if status not in ("ok", "no_results"):
            raise ConnectionError(f"URLhaus said {status}")
        return {"status": "ok", "listed": status == "ok",
                "threat": data.get("threat") if status == "ok" else None}

    # ---- short links ------------------------------------------------------

    def expand_short_link(self, url):
        def fetch():
            response = self._request("shortlink", "head", url, allow_redirects=False)
            target = response.headers.get("Location")
            if response.status_code in (301, 302, 303, 307, 308) and target:
                return {"status": "ok", "source": "HTTP HEAD", "target": target}
            return _unavailable("HTTP HEAD", f"no redirect (HTTP {response.status_code})")
        return self._cached(f"short:{url}", "HTTP HEAD", fetch)

    # ---- SPF / DKIM / DMARC ----------------------------------------------

    def auth(self, email):
        """
        Authentication results for an email.

        1. The receiving server's Authentication-Results header (RFC 8601). Only the
           top-most header is trusted (it is the one our own server adds; anything
           lower could have been written by the sender). Set TRUSTED_AUTHSERV_ID to
           only accept headers from your own mail server.
        2. Otherwise, with the raw message and INTEL_MODE=live: verify DKIM with
           dkimpy, check SPF with pyspf against the first external Received hop,
           and look up the DMARC policy in DNS.
        """
        headers = email.get("headers") or {}
        parsed = parse_authentication_results(headers)
        if parsed:
            return parsed
        if not email.get("raw"):
            return _unavailable("Authentication-Results",
                                "no Authentication-Results header in this email")
        if self.mode != "live":
            return _unavailable("live DNS", f"live SPF/DKIM/DMARC checks need INTEL_MODE=live")
        return self._live_auth(email)

    def _live_auth(self, email):
        try:
            import dkim
            import dns.resolver
            import spf
        except ImportError as e:
            return _unavailable("live DNS", f"{e.name} not installed "
                                            f"(pip install dnspython dkimpy pyspf)")
        raw = email["raw"].encode() if isinstance(email["raw"], str) else email["raw"]
        sender_domain = (email.get("sender") or "").rsplit("@", 1)[-1]
        result = {"status": "ok", "source": "live DNS checks"}
        problems = []

        # A resolver that is down, filtered or sinkholed often answers NXDOMAIN for
        # everything, which would look like "no SPF / no DMARC record". Check a
        # record that must exist before believing any negative answer.
        healthy, why = self._dns_healthy(dns.resolver)
        if not healthy:
            return _unavailable("live DNS", f"DNS resolver failed the canary check ({why}); "
                                            f"negative DNS answers cannot be trusted")

        try:
            result["dkim"] = "pass" if dkim.verify(raw, timeout=int(self.timeout)) else (
                "fail" if b"dkim-signature" in raw.lower() else "none")
        except Exception as e:
            result["dkim"] = None
            problems.append(f"DKIM: {type(e).__name__}")

        ip, helo = _first_external_hop(email.get("headers", {}).get("Received", []))
        mail_from = email.get("return_path") or email.get("sender") or ""
        if ip:
            try:
                result["spf"] = spf.check2(i=ip, s=mail_from, h=helo or sender_domain,
                                           timeout=self.timeout)[0]
            except Exception as e:
                result["spf"] = None
                problems.append(f"SPF: {type(e).__name__}")
        else:
            result["spf"] = None
            problems.append("SPF: no sending IP in the Received headers")

        try:
            answers = dns.resolver.resolve(f"_dmarc.{registrable_domain(sender_domain)}", "TXT",
                                           lifetime=self.timeout)
            record = " ".join(b"".join(a.strings).decode() for a in answers)
            policy = re.search(r"\bp=(\w+)", record)
            result["dmarc_policy"] = policy.group(1) if policy else None
            aligned = (result.get("dkim") == "pass" or
                       (result.get("spf") == "pass" and
                        registrable_domain(mail_from.rsplit("@", 1)[-1]) ==
                        registrable_domain(sender_domain)))
            result["dmarc"] = "pass" if aligned else "fail"
        except Exception as e:
            name = type(e).__name__
            result["dmarc"] = "none" if name in ("NXDOMAIN", "NoAnswer") else None
            if result["dmarc"] is None:
                problems.append(f"DMARC: {name}")

        if all(result.get(k) is None for k in ("spf", "dkim", "dmarc")):
            return _unavailable("live DNS", "; ".join(problems))
        result["missing"] = problems
        return result

    def _dns_healthy(self, resolver):
        if getattr(self, "_dns_state", None) is None:
            try:
                answers = resolver.resolve("_dmarc.google.com", "TXT", lifetime=self.timeout)
                ok = any(b"v=DMARC1" in b"".join(a.strings) for a in answers)
                self._dns_state = (ok, "" if ok else "_dmarc.google.com has no DMARC record")
            except Exception as e:
                self._dns_state = (False, f"_dmarc.google.com: {type(e).__name__}")
        return self._dns_state

    def save(self):
        self.cache.save()


AR_RESULT_RE = re.compile(r"\b(spf|dkim|dmarc)\s*=\s*([a-z]+)", re.IGNORECASE)


def parse_authentication_results(headers):
    """Read SPF/DKIM/DMARC from the top-most trusted Authentication-Results header."""
    trusted_id = os.getenv("TRUSTED_AUTHSERV_ID", "").strip().lower()
    candidates = headers.get("Authentication-Results") or []
    chosen = None
    for value in candidates:
        if trusted_id and not value.strip().lower().startswith(trusted_id):
            continue
        chosen = value
        break
    result = {}
    if chosen:
        for method, verdict in AR_RESULT_RE.findall(chosen):
            result.setdefault(method.lower(), verdict.lower())   # first verdict per method
    if "spf" not in result:
        for value in headers.get("Received-SPF") or []:
            match = re.match(r"\s*(pass|fail|softfail|neutral|none|temperror|permerror)",
                             value, re.IGNORECASE)
            if match:
                result["spf"] = match.group(1).lower()
                break
    if not result:
        return None
    for method in ("spf", "dkim", "dmarc"):
        result.setdefault(method, None)
    result["status"] = "ok"
    result["source"] = "Authentication-Results header" if chosen else "Received-SPF header"
    return result


def _first_external_hop(received):
    """(ip, helo) of the first Received hop that came from a public address."""
    for line in received:
        match = re.search(r"from\s+(\S+).*?\[(\d{1,3}(?:\.\d{1,3}){3})\]", line, re.DOTALL)
        if match:
            ip = match.group(2)
            if not ip.startswith(("10.", "127.", "192.168.", "172.16.", "172.17.", "172.18.")):
                return ip, match.group(1)
    return None, None


def host_of_url(url):
    if "://" not in url:
        url = "http://" + url
    return (urlparse(url).hostname or "").lower()
