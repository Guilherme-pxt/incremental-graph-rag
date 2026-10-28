"""Parsing and normalizing values: money, months, small integers, names.

Shared by the extractor (which normalizes what the LLM returned), the graph
(which compares values to decide whether a fact changed) and the scorer
(which finds values in an answer). Deterministic and dependency free.
"""

from __future__ import annotations

import re

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
_MONTH_IDX = {m: i + 1 for i, m in enumerate(MONTHS)} | {m[:3]: i + 1 for i, m in enumerate(MONTHS)} | {"sept": 9}
NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

_MONEY_RE = re.compile(r"(\$\s*)?(\d[\d,]*(?:\.\d+)?)\s*(k\b|thousand\b|m\b|mm\b|million\b)?", re.IGNORECASE)
_MONTH_NAME_RE = re.compile(r"\b(" + "|".join(sorted(_MONTH_IDX, key=len, reverse=True)) + r")\.?,?\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?(20\d\d)\b",
                            re.IGNORECASE)
_ISO_MONTH_RE = re.compile(r"\b(20\d\d)-(\d{1,2})(?:-\d{1,2})?\b")
_SLASH_MONTH_RE = re.compile(r"\b(\d{1,2})/(20\d\d)\b")


def parse_money(x) -> int | None:
    """950000, "950000", "$950,000", "950k", "$0.95M", "950 thousand" -> 950000. None if no amount."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return int(round(x)) if x >= 1000 else None
    amounts = money_amounts(str(x))
    return amounts[0] if amounts else None


def money_amounts(text: str) -> list[int]:
    """Every dollar amount in ``text`` (a figure of at least 1,000 once its k/M suffix is applied)."""
    out = []
    for m in _MONEY_RE.finditer(text):
        num, suf = m.group(2).replace(",", ""), (m.group(3) or "").lower()
        try:
            v = float(num)
        except ValueError:
            continue
        if suf in ("k", "thousand"):
            v *= 1_000
        elif suf in ("m", "mm", "million"):
            v *= 1_000_000
        if v >= 1000 and not (1900 < v < 2100 and not suf and not m.group(1) and "," not in m.group(2)):
            out.append(int(round(v)))
    return out


def parse_month(x) -> str | None:
    """'June 2027', 'Jun 2027', '2027-06', '06/2027', 'June 15, 2027' -> '2027-06'."""
    months = months_in(str(x))
    return months[0] if months else None


def months_in(text: str) -> list[str]:
    out = []
    for m in _MONTH_NAME_RE.finditer(text):
        out.append((m.start(), f"{m.group(2)}-{_MONTH_IDX[m.group(1).lower()]:02d}"))
    for m in _ISO_MONTH_RE.finditer(text):
        if 1 <= int(m.group(2)) <= 12:
            out.append((m.start(), f"{m.group(1)}-{int(m.group(2)):02d}"))
    for m in _SLASH_MONTH_RE.finditer(text):
        if 1 <= int(m.group(1)) <= 12:
            out.append((m.start(), f"{m.group(2)}-{int(m.group(1)):02d}"))
    return [v for _, v in sorted(out)]


def parse_int(x) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    vals = small_ints(str(x))
    return vals[0] if vals else None


def small_ints(text: str) -> list[int]:
    """Standalone integers below 100 and number words; digits inside dates, amounts or ids do not count."""
    t = text.lower()
    t = _ISO_MONTH_RE.sub(" ", t)
    t = _SLASH_MONTH_RE.sub(" ", t)
    t = re.sub(r"\[[^\]]*\]", " ", t)
    out = []
    for m in re.finditer(r"(?<![\w$.,])(\d{1,2})(?![\d,.]*\s*(?:k\b|thousand|million))(?![\w])|\b(" +
                         "|".join(NUM_WORDS) + r")\b", t):
        out.append(int(m.group(1)) if m.group(1) else NUM_WORDS[m.group(2)])
    return out


_HONORIFIC = re.compile(r"^(mr|mrs|ms|miss|dr)\.?\s+")


def norm_name(name: str) -> str:
    """Lowercase, punctuation dropped (initials keep their letter), leading 'the' and honorifics dropped."""
    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _HONORIFIC.sub("", s)
    if s.startswith("the "):
        s = s[4:]
    return s
