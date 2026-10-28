"""Tokenizer and "I don't know" detector, from my hybrid-search-rag project (same author).

Deterministic and dependency free, so a reader can predict every
tokenization by hand.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how
i if in into is it its itself just me more most my no nor not now of off on once only or other our ours out over own
same she should so some such than that the their theirs them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your yours
get gets got per much many s t
""".split())

# A token is a run of letters/digits, optionally joined by - . / (so "qv-pu200",
# "ing-5031", "7.3" and "0.35" stay whole). Compound tokens also emit their parts.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def stem(word: str) -> str:
    """Tiny suffix stripper: enough to match 'days'/'day', 'policies'/'policy'."""
    if any(c.isdigit() for c in word):
        return word
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, keep compound codes whole and also emit their parts, drop stopwords, stem."""
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text.lower()):
        tok = m.group(0)
        parts = re.split(r"[-./]", tok)
        if len(parts) > 1:
            out.append(tok)
            out.extend(p for p in parts if p and p not in STOPWORDS)
        elif tok not in STOPWORDS:
            out.append(stem(tok))
    return out


_IDK_RE = re.compile(
    r"\bi (?:do not|don't|dont) know\b|\b(?:not|no information) (?:\w+ ){0,3}(?:in|from) the (?:provided |given )?"
    r"(?:documents|passages|context)\b|\bdocuments do not (?:contain|mention|say|specify)\b|"
    r"\bpassages do not (?:contain|mention|say|specify)\b",
    re.IGNORECASE,
)


def is_idk(text: str) -> bool:
    return bool(_IDK_RE.search(text or ""))
