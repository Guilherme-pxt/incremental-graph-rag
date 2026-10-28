"""Scoring answers against the gold atoms, and paired bootstrap intervals.

An answer is scored on its answer line only (the text after "ANSWER:"):

* correct      kind ``idk``: the line declines. Otherwise: it does not decline,
               every required atom is in it, and (lists, "which client")
               no forbidden entity is in it.
* stale        not correct, and it states a value the asked slot held at
               another time (a superseded budget, a dropped building). For
               "now" questions that is the stale-fact rate; for "as of"
               questions it is the wrong-period rate.
* leak         an isolation probe whose answer (whole text, not only the line)
               states the other broker's value for the probed client.
* confusion    an isolation probe answered with the value of this broker's
               client who shares the first name: not a leak, but a wrong entity.
* supported    correct, and it cites at least one episode that states the gold fact.
"""

from __future__ import annotations

import re

import numpy as np

from .answer import answer_line, citations, is_idk
from .values import money_amounts, months_in, small_ints


def _words(s: str) -> str:
    """Lowercase words separated by single spaces, punctuation dropped."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def atom_in(atom: dict, line: str) -> bool:
    t = atom["t"]
    if t == "money":
        return atom["v"] in money_amounts(line)
    if t == "month":
        return atom["v"] in months_in(line)
    if t == "int":
        return atom["v"] in small_ints(line)
    if t == "entity":
        padded = f" {_words(line)} "
        return any(n and f" {n} " in padded for n in (_words(x).removeprefix("the ") for x in atom["names"]))
    if t == "word":
        return atom["v"].lower() in line.lower()
    raise ValueError(f"unknown atom type {t}")


def score(q: dict, text: str) -> dict:
    line = answer_line(text)
    declined = is_idk(line)
    if q["kind"] == "idk":
        correct = declined
    else:
        correct = (not declined and all(atom_in(a, line) for a in q["required"])
                   and not any(atom_in(a, line) for a in q["forbidden"]))
    stale = (not correct and not declined and bool(q["stale"]) and any(atom_in(a, line) for a in q["stale"]))
    leak = any(atom_in(a, text) for a in q.get("leak", []))
    confusion = any(atom_in(a, text) for a in q.get("confusion", []))
    cites = citations(text)
    return {"line": line, "correct": bool(correct), "declined": bool(declined), "stale": bool(stale),
            "leak": bool(leak), "confusion": bool(confusion), "cites": cites,
            "supported": bool(correct and q["evidence"] and set(cites) & set(q["evidence"]))}


def bootstrap_ci(values: list[float], n: int = 10_000, seed: int = 42) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap(a: list[float], b: list[float], n: int = 10_000, seed: int = 42) -> dict:
    """Delta mean(b) - mean(a) over the same questions, with a 95% percentile interval (hybrid-search-rag)."""
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if xa.shape != xb.shape or len(xa) == 0:
        raise ValueError("paired bootstrap needs two equal-length, non-empty samples")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xa), size=(n, len(xa)))
    d = (xb[idx] - xa[idx]).mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(xb.mean() - xa.mean()), "ci95": [float(lo), float(hi)], "n": int(len(xa)),
            "excludes_zero": bool(lo > 0 or hi < 0)}
