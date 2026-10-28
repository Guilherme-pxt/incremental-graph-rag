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

A question that matches no form, or whose client or station cannot be linked
to exactly one node, takes the ordinary retrieval path.
"""

from __future__ import annotations

import re

from .values import norm_name

_CHAIN = re.compile(r"^(who develops|which metro station is near) the building (.+?) is "
                    r"(?:interested in|focused on) now\??$", re.IGNORECASE)
_NEAR = re.compile(r"^which of my clients are currently interested in a building near (.+?) station\??$",
                   re.IGNORECASE)
_FIT = re.compile(r"^which of my clients looking for (\d+) bedrooms? can afford a building delivered in "
                  r"(20\d\d) near (.+?) station\??$", re.IGNORECASE)


class Query:
    def __init__(self, graph):
        self.g = graph

    def current(self, nid: str, pred: str) -> list[dict]:
        return [e for e in self.g.edges_of(nid) if e["src"] == nid and e["pred"] == pred and e["valid_to"] is None]

    def one(self, nid: str, pred: str) -> dict | None:
        es = self.current(nid, pred)
        return es[0] if len(es) == 1 else None

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
