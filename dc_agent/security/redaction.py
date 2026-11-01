"""Mask PII in string cells of result rows. Numbers are never touched, so metrics are safe.

Patterns are deliberately conservative: a masked identifier is better than a leaked one, but
false positives on ordinary values (dates, timestamps, facility codes) are avoided.
"""
from __future__ import annotations

import re
from typing import Callable

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Number patterns must not start or end inside a longer digit group ("1234 5678 9012 3456" is not an
# Aadhaar number followed by noise), hence the paired look-arounds for a digit with or without a separator.
_NB, _NA = r"(?<!\d)(?<!\d[ -])", r"(?!\d)(?![ -]\d)"
_SSN = re.compile(_NB + r"\d{3}-\d{2}-\d{4}" + _NA)
_AADHAAR = re.compile(_NB + r"[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}" + _NA)
_PAN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Za-z0-9])")
_IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
_CARD = re.compile(_NB + r"(?:\d[ -]?){12,18}\d" + _NA)
_PHONE = re.compile(r"(?<![\w+(])(?:\+|\()?\d[\d ().-]{8,18}\d(?!\w)")
_DATEISH = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[card]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)


def _phone(m: re.Match) -> str:
    text = m.group(0)
    if _DATEISH.search(text):
        return text
    return "[phone]" if 10 <= sum(c.isdigit() for c in text) <= 15 else text


# Order matters: specific formats first, so a card number is not masked as a phone number.
_RULES: dict = {
    "email": lambda s: _EMAIL.sub("[email]", s),
    "card": lambda s: _CARD.sub(_card, s),
    "ssn": lambda s: _SSN.sub("[ssn]", s),
    "aadhaar": lambda s: _AADHAAR.sub("[aadhaar]", s),
    "pan": lambda s: _PAN.sub("[pan]", s),
    "ipv4": lambda s: _IPV4.sub("[ip]", s),
    "phone": lambda s: _PHONE.sub(_phone, s),
}


def make_redactor(patterns=tuple(_RULES)) -> Callable[[str], str]:
    rules = [_RULES[p] for p in _RULES if p in set(patterns)]

    def redact(text: str) -> str:
        for rule in rules:
            text = rule(text)
        return text

    return redact


_DEFAULT = make_redactor()


def redact_rows(rows: list, redact: Callable[[str], str] = _DEFAULT) -> list:
    """Mask PII in string cells. Extend _RULES with the patterns your data needs."""
    return [[redact(v) if isinstance(v, str) else v for v in row] for row in rows]
