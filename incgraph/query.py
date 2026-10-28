"""Deterministic graph queries for the multi-hop question forms of the gold set (incremental graph only).

The 7B answerer does not reliably join four conditions in one pass (bedrooms,
budget against price, delivery year, station), even with every fact in its
context. When a question matches one of the forms below, code runs the query
on the graph's current edges and the model receives the result with the facts
that prove it, and only has to phrase it.

This is **not** a natural-language query parser. It covers exactly the four
multi-hop forms the question generator writes (``questions.py``):

* who develops the building <client> is interested in / focused on now?
* which metro station is near the building <client> is interested in / focused on now?
* which of my clients are currently interested in a building near <station> station?
* which of my clients looking for <n> bedrooms can afford a building delivered in <year> near <station> station?

Third run: the aggregation questions get the same treatment. The graph side
is general (``value_at``, ``count_where`` and ``extreme`` work for any entity
type and any single-valued predicate, at any date, respecting
``valid_from``/``valid_to``); the recognizer is not. It covers the forms the
generator writes, with a little slack (above/below/over/under, highest/lowest,
"right now" or "on <date>"):

* how many of my clients want <n> bedrooms right now?
* how many of my clients have a budget above $<amount> right now?
* which of my clients has the highest budget right now?

A question that matches no form, or whose client or station cannot be linked
to exactly one node, takes the ordinary retrieval path.
"""

from __future__ import annotations

import re

from .values import money_amounts, months_in, norm_name

_CHAIN = re.compile(r"^(who develops|which metro station is near) the building (.+?) is "
                    r"(?:interested in|focused on) now\??$", re.IGNORECASE)
_NEAR = re.compile(r"^which of my clients are currently interested in a building near (.+?) station\??$",
                   re.IGNORECASE)
_FIT = re.compile(r"^which of my clients looking for (\d+) bedrooms? can afford a building delivered in "
                  r"(20\d\d) near (.+?) station\??$", re.IGNORECASE)


_WHEN = r"(?:\s+(right now|now|today|on ([A-Za-z]+ \d{1,2}, 20\d\d|20\d\d-\d\d-\d\d)))?"
_COUNT_BEDS = re.compile(r"^how many of my clients (?:want|need|are looking for) (\d+) bedrooms?" + _WHEN + r"\??$",
                         re.IGNORECASE)
_COUNT_BUDGET = re.compile(r"^how many of my clients have a budget (above|over|below|under) (\$?[\d,.]+\s*k?)"
                           + _WHEN + r"\??$", re.IGNORECASE)
_EXTREME = re.compile(r"^which of my clients has the (highest|largest|biggest|lowest|smallest) budget" + _WHEN
                      + r"\??$", re.IGNORECASE)
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]


def _date(when: str | None) -> str | None:
    """'March 21, 2026' or '2026-03-21' -> '2026-03-21'; None (now) for 'right now' or no date."""
    if not when:
        return None
    m = re.match(r"^(20\d\d-\d\d-\d\d)$", when)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Za-z]+) (\d{1,2}), (20\d\d)$", when)
    if m and m.group(1).lower() in _MONTHS:
        return f"{m.group(3)}-{_MONTHS.index(m.group(1).lower()) + 1:02d}-{int(m.group(2)):02d}"
    return None


class Query:
    def __init__(self, graph):
        self.g = graph

    def current(self, nid: str, pred: str) -> list[dict]:
        return [e for e in self.g.edges_of(nid) if e["src"] == nid and e["pred"] == pred and e["valid_to"] is None]

    def one(self, nid: str, pred: str) -> dict | None:
        es = self.current(nid, pred)
        return es[0] if len(es) == 1 else None

    # ---------------------------------------------------- general graph operations

    def value_at(self, nid: str, pred: str, at: str | None = None) -> dict | None:
        """The one edge of a single-valued predicate that holds at date ``at`` (None = now: the open edge)."""
        es = [e for e in self.g.edges_of(nid) if e["src"] == nid and e["pred"] == pred and (
            e["valid_to"] is None if at is None else e["valid_from"] <= at and (e["valid_to"] is None or at < e["valid_to"]))]
        return es[0] if len(es) == 1 else None

    def count_where(self, etype: str, pred: str, test, at: str | None = None) -> tuple[list[str], list[dict]]:
        """Nodes of ``etype`` whose value of ``pred`` at ``at`` passes ``test``; with the edges that decided it."""
        hits, ev = [], []
        for nid in self.g.of_type(etype):
            e = self.value_at(nid, pred, at)
            if e is not None and test(e["value"] if e["dst"] is None else e["dst"]):
                hits.append(nid)
                ev.append(e)
        return hits, ev

    def extreme(self, etype: str, pred: str, highest: bool = True, at: str | None = None) -> tuple[list[str], list[dict]]:
        """Nodes of ``etype`` with the highest (or lowest) numeric ``pred`` at ``at``; ties all returned."""
        vals = [(nid, e) for nid in self.g.of_type(etype) for e in [self.value_at(nid, pred, at)]
                if e is not None and isinstance(e["value"], (int, float))]
        if not vals:
            return [], []
        best = (max if highest else min)(e["value"] for _, e in vals)
        top = [(n, e) for n, e in vals if e["value"] == best]
        return [n for n, _ in top], [e for _, e in top]

    def aggregate(self, q: str) -> tuple[str, list[dict], str] | None:
        names = lambda ids: ", ".join(self.g.nodes[n]["name"] for n in ids)
        m = _COUNT_BEDS.match(q)
        if m:
            n, at = int(m.group(1)), _date(m.group(3))
            hits, ev = self.count_where("client", "bedrooms", lambda v: v == n, at)
            return (f"count of clients wanting {n} bedrooms {('on ' + at) if at else 'now'}", ev,
                    f"{len(hits)} ({names(hits)})" if hits else "0")
        m = _COUNT_BUDGET.match(q)
        if m:
            amounts = money_amounts(m.group(2))
            if not amounts:
                return None
            cut, above, at = amounts[0], m.group(1).lower() in ("above", "over"), _date(m.group(4))
            hits, ev = self.count_where("client", "budget", (lambda v: v > cut) if above else (lambda v: v < cut), at)
            return (f"count of clients with a budget {'above' if above else 'below'} ${cut:,} "
                    f"{('on ' + at) if at else 'now'}", ev, f"{len(hits)} ({names(hits)})" if hits else "0")
        m = _EXTREME.match(q)
        if m:
            highest, at = m.group(1).lower() in ("highest", "largest", "biggest"), _date(m.group(3))
            top, ev = self.extreme("client", "budget", highest, at)
            if not top:
                return None
            return (f"client with the {'highest' if highest else 'lowest'} budget {('on ' + at) if at else 'now'}",
                    ev, names(top))
        return None

    def node_by_name(self, name: str, etype: str) -> str | None:
        from .retrieve import link
        if etype == "station":
            target = norm_name(name).removesuffix(" station")
            hits = [n for n in self.g.of_type("station")
                    if any(norm_name(x).removesuffix(" station") == target for x in self.g.names(n))]
        else:
            hits = [n for n in link(self.g, name) if self.g.nodes[n]["type"] == etype]
        return hits[0] if len(hits) == 1 else None

    def run(self, question: str) -> tuple[str, list[dict], str] | None:
        """(question form, evidence edges, result text), or None to fall back to retrieval."""
        q = question.strip()
        agg = self.aggregate(q)
        if agg is not None:
            return agg
        m = _CHAIN.match(q)
        if m:
            client = self.node_by_name(m.group(2), "client")
            want = "developer" if m.group(1).lower().startswith("who") else "near_station"
            b = self.one(client, "interested_in") if client else None
            if not b:
                return None
            target = self.one(b["dst"], want)
            if not target:
                return None
            form = ("developer" if want == "developer" else "station") + " of the building a client is focused on now"
            return form, [b, target], self.g.obj_text(target)
        m = _NEAR.match(q)
        if m:
            stn = self.node_by_name(m.group(1), "station")
            if not stn:
                return None
            ev, names = [], []
            for c in self.g.of_type("client"):
                b = self.one(c, "interested_in")
                s = self.one(b["dst"], "near_station") if b else None
                if s and s["dst"] == stn:
                    ev += [b, s]
                    names.append(self.g.nodes[c]["name"])
            return "clients whose current building is near a station", ev, ", ".join(names) or "none of your clients"
        m = _FIT.match(q)
        if m:
            n, year, stn = int(m.group(1)), m.group(2), self.node_by_name(m.group(3), "station")
            if not stn:
                return None
            ev, blds = [], []
            for b in self.g.of_type("building"):
                s, d, p = self.one(b, "near_station"), self.one(b, "delivery"), self.one(b, "price")
                units = [e for e in self.current(b, "unit_bedrooms") if e["value"] == n]
                if s and d and p and units and s["dst"] == stn and str(d["value"]).startswith(year):
                    blds.append((b, p["value"], [s, d, p, units[0]]))
            names = []
            for c in self.g.of_type("client"):
                beds, budget = self.one(c, "bedrooms"), self.one(c, "budget")
                if not (beds and budget and beds["value"] == n):
                    continue
                ok = [x for x in blds if x[1] <= budget["value"]]
                if ok:
                    names.append(self.g.nodes[c]["name"])
                    ev += [beds, budget] + ok[0][2]
            form = "clients wanting N bedrooms who can afford a building near a station delivered in a year"
            return form, ev, ", ".join(names) or "none of your clients"
        return None

    def context(self, question: str) -> tuple[str, list[str], dict] | None:
        out = self.run(question)
        if out is None:
            return None
        form, ev, result = out
        seen, lines = set(), []
        for e in ev:
            if e["id"] not in seen:
                seen.add(e["id"])
                lines.append(self.g.fact_text(e))
        text = ("Computed from the graph's current facts by a deterministic query "
                f"({form}):\nResult: {result}\nEvidence:\n" + "\n".join(lines))
        cited = sorted({x for e in ev for x in e["episodes"]})
        return text, cited, {"computed": form, "facts": len(lines), "linked": [], "reports": 0, "tables": False}
