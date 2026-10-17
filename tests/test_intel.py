"""Threat-intel lookups, with a fake HTTP client instead of the network."""

from types import SimpleNamespace

import pytest

from env.eml import parse_eml
from env.intel import IntelCache, ThreatIntel, parse_authentication_results, registrable_domain
from env.tools import InvestigationTools


class FakeHTTP:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _reply(self, method, url, **kwargs):
        self.calls.append((method, url))
        for prefix, reply in self.routes.items():
            if url.startswith(prefix):
                if isinstance(reply, Exception):
                    raise reply
                status, body = reply
                return SimpleNamespace(status_code=status, json=lambda body=body: body,
                                       text=body if isinstance(body, str) else "",
                                       headers=body if isinstance(body, dict) else {})
        raise ConnectionError("no route")

    def get(self, url, **kw):
        return self._reply("get", url, **kw)

    def post(self, url, **kw):
        return self._reply("post", url, **kw)

    def head(self, url, **kw):
        return self._reply("head", url, **kw)


def intel(routes, mode="live"):
    return ThreatIntel(mode=mode, cache=IntelCache(path=None), http=FakeHTTP(routes))


RDAP = {"events": [{"eventAction": "registration", "eventDate": "2026-10-01T00:00:00Z"}]}


def test_domain_age_is_measured_at_the_email_date():
    ti = intel({"https://rdap.org/domain/fresh-shop.com": (200, RDAP)})
    age = ti.domain_age("login.fresh-shop.com", at="2026-10-11T10:00:00+00:00")
    assert age["status"] == "ok" and age["age_days"] == 10 and age["domain"] == "fresh-shop.com"


def test_domain_registered_after_the_email_is_unavailable_not_old():
    ti = intel({"https://rdap.org/domain/": (200, RDAP)})
    age = ti.domain_age("fresh-shop.com", at="2002-08-01T00:00:00+00:00")
    assert age["status"] == "unavailable" and "after this email" in age["error"]


def test_network_failure_is_unavailable_and_the_breaker_stops_retrying():
    ti = intel({"https://rdap.org/": TimeoutError("timed out")})
    for domain in ("a.com", "b.com", "c.com", "d.com"):
        assert ti.domain_age(domain)["status"] == "unavailable"
    assert len(ti.http.calls) == 2                       # then skipped, not hammered


def test_cache_and_off_modes_never_touch_the_network():
    for mode in ("cache", "off"):
        ti = intel({"https://rdap.org/": (200, RDAP)}, mode=mode)
        assert ti.domain_age("x.com")["status"] == "unavailable"
        assert ti.http.calls == []


def test_successful_lookups_are_cached_for_offline_replay(tmp_path):
    cache = IntelCache(path=str(tmp_path / "cache.json"))
    live = ThreatIntel(mode="live", cache=cache, http=FakeHTTP({"https://rdap.org/": (200, RDAP)}))
    assert live.domain_age("x.com", at="2026-10-11T00:00:00Z")["age_days"] == 10
    live.save()
    replay = ThreatIntel(mode="cache", cache=IntelCache(path=str(tmp_path / "cache.json")), http=FakeHTTP({}))
    assert replay.domain_age("x.com", at="2026-10-11T00:00:00Z")["age_days"] == 10


def test_free_platform_subdomains_have_no_meaningful_age():
    ti = intel({})
    age = ti.domain_age("evil.onmicrosoft.com")
    assert age["status"] == "not_applicable" and age["platform"] == "onmicrosoft.com"


def test_url_reputation_listed_and_unreachable():
    ti = intel({"https://openphish.com/feed.txt": (200, "https://bad.example/login\n")})
    assert ti.url_reputation("https://bad.example/login")["verdict"] == "malicious"
    assert ti.url_reputation("https://good.example/")["verdict"] == "not_listed"
    down = intel({"https://openphish.com/": ConnectionError("refused")})
    assert down.url_reputation("https://bad.example/login")["status"] == "unavailable"


def test_listed_url_drives_the_url_check_risk():
    ti = intel({"https://openphish.com/feed.txt": (200, "https://pay-now.example/x\n"),
                "https://rdap.org/": (404, {})})
    item = InvestigationTools(intel=ti).check_url({"urls": ["https://pay-now.example/x"]})
    assert item["risk"] == 0.95 and "openphish" in item["finding"]


def test_short_link_expansion():
    ti = intel({"https://bit.ly/abc": (301, {"Location": "https://paypal-verify.top/login"})})
    item = InvestigationTools(intel=ti).check_url({"urls": ["https://bit.ly/abc"]})
    assert "expands to paypal-verify.top" in item["finding"] and item["risk"] == 0.9


AR = ("Authentication-Results: mx.google.com; spf=fail smtp.mailfrom=evil.com; "
      "dkim=none; dmarc=fail header.from=paypal.com\n"
      "Authentication-Results: attacker.example; spf=pass; dkim=pass; dmarc=pass\n"
      "From: PayPal <service@paypal.com>\nSubject: hi\n\nbody")


def test_only_the_top_most_authentication_results_header_is_trusted():
    parsed = parse_authentication_results(parse_eml(AR)["headers"])
    assert parsed["spf"] == "fail" and parsed["dmarc"] == "fail"


def test_trusted_authserv_id(monkeypatch):
    monkeypatch.setenv("TRUSTED_AUTHSERV_ID", "attacker.example")
    assert parse_authentication_results(parse_eml(AR)["headers"])["dmarc"] == "pass"


def test_dmarc_failure_is_strong_sender_evidence():
    row = parse_eml(AR)
    item = InvestigationTools(intel=intel({})).check_sender(row)
    assert item["risk"] >= 0.85 and "DMARC failed" in item["finding"]


def test_missing_auth_header_is_unavailable_not_pass():
    result = intel({}).auth({"sender": "a@b.com", "headers": {}})
    assert result["status"] == "unavailable"


def test_lying_dns_resolver_is_not_believed(monkeypatch):
    pytest.importorskip("dns.resolver")
    pytest.importorskip("dkim")
    pytest.importorskip("spf")
    import dns.resolver

    def nxdomain(*a, **k):
        raise dns.resolver.NXDOMAIN()
    monkeypatch.setattr(dns.resolver, "resolve", nxdomain)
    raw = "From: a@example.org\nSubject: x\n\nbody"
    result = intel({}).auth({"sender": "a@example.org", "headers": {}, "raw": raw})
    assert result["status"] == "unavailable" and "canary" in result["error"]


def test_registrable_domain():
    assert registrable_domain("mail.alerts.hdfcbank.com") == "hdfcbank.com"
    assert registrable_domain("x.sbi.co.in") == "sbi.co.in"
