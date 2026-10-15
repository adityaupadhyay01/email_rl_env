"""
Parse a raw RFC 5322 email (.eml) into the fields the environment uses.

Used by the dataset builder (public phishing / ham corpora) and by the live demo,
where a user pastes or uploads a raw email. Only the standard library is needed.
"""

import email
import html
import re
from email import policy
from email.header import decode_header, make_header
from email.utils import getaddresses, parsedate_to_datetime

from env.tools import extract_urls

HREF_RE = re.compile(r"""href\s*=\s*["']?(https?://[^"'\s>]+)""", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
SPACE_RE = re.compile(r"\s+")

# Headers kept so the sender check can read authentication results later.
AUTH_HEADERS = ("Authentication-Results", "ARC-Authentication-Results", "Received-SPF")


def _decode(value):
    if value is None:
        return ""
    try:
        return str(make_header(decode_header(str(value))))
    except Exception:
        return str(value)


def _html_to_text(markup):
    markup = SCRIPT_RE.sub(" ", markup)
    markup = re.sub(r"<br\s*/?>|</p>|</div>|</tr>", "\n", markup, flags=re.IGNORECASE)
    return html.unescape(TAG_RE.sub(" ", markup))


def _part_text(part):
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def parse_eml(raw, max_text=2000, max_urls=20):
    """`raw` is bytes or str. Returns a dict ready for EmailEnv (without a label)."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="replace")
    msg = email.message_from_bytes(raw, policy=policy.compat32)

    plain, markup, attachments = [], [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        disposition = (part.get("Content-Disposition") or "").lower()
        filename = part.get_filename()
        if "attachment" in disposition or filename:
            attachments.append(_decode(filename) or part.get_content_type())
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain.append(_part_text(part))
        elif ctype == "text/html":
            markup.append(_part_text(part))

    html_text = "\n".join(markup)
    body = "\n".join(plain).strip() or _html_to_text(html_text)
    body = SPACE_RE.sub(" ", body).strip()

    urls = []
    for url in HREF_RE.findall(html_text) + extract_urls(body):
        url = html.unescape(url).rstrip(".,;:!?)")
        if url not in urls:
            urls.append(url)

    name, address = (getaddresses([_decode(msg.get("From"))]) or [("", "")])[0]
    reply_to = (getaddresses([_decode(msg.get("Reply-To"))]) or [("", "")])[0][1]
    return_path = (getaddresses([_decode(msg.get("Return-Path"))]) or [("", "")])[0][1]

    date = None
    try:
        parsed = parsedate_to_datetime(msg.get("Date"))
        date = parsed.isoformat() if parsed else None
    except (TypeError, ValueError, IndexError):
        pass

    headers = {}
    for key in AUTH_HEADERS:
        values = msg.get_all(key)
        if values:
            headers[key] = [SPACE_RE.sub(" ", str(v)) for v in values]
    dkim = msg.get_all("DKIM-Signature") or []
    if dkim:
        headers["DKIM-Signature"] = [SPACE_RE.sub(" ", str(v)) for v in dkim]
    received = msg.get_all("Received") or []
    if received:
        headers["Received"] = [SPACE_RE.sub(" ", str(v)) for v in received[:6]]

    return {
        "sender_name": _decode(name).strip().strip('"'),
        "sender": (address or "").strip().lower(),
        "reply_to": (reply_to or "").strip().lower(),
        "return_path": (return_path or "").strip().lower(),
        "subject": SPACE_RE.sub(" ", _decode(msg.get("Subject"))).strip(),
        "text": body[:max_text],
        "urls": urls[:max_urls],
        "date": date,
        "attachments": attachments[:10],
        "headers": headers,
    }
