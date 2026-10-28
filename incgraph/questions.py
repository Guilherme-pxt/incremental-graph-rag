"""Typed gold questions per broker, derived from the generator's ground truth.

Every question is asked to one broker's system and is scored against that
broker's own episodes. A gold answer is a list of atoms that must all appear
in the answer line:

* ``money``  an amount in dollars ("$950,000", "950k" and "$0.95M" all match 950000)
* ``month``  a year and month ("June 2027", "Jun 2027", "2027-06")
* ``int``    a small count ("3", "three")
* ``entity`` a client, building, developer, neighborhood or station, matched by
             any of its unambiguous names
* ``word``   a keyword (objections)

``stale`` atoms are values the same slot held at another time (a superseded
budget, the building a client dropped); ``forbidden`` atoms are entities that
must not be listed; ``leak`` atoms are another broker's data. Types:

    single_hop, temporal_current, as_of, multi_hop, aggregation, unanswerable, isolation
"""

from __future__ import annotations

import datetime as dt
import random

from .synth import BROKERS, CLIENTS, MONTHS, TODAY, World, slug

TYPES = ["single_hop", "temporal_current", "as_of", "multi_hop", "aggregation", "unanswerable", "isolation"]


def money(v: int) -> dict:
    return {"t": "money", "v": int(v)}


def month(v: str) -> dict:
    return {"t": "month", "v": v}


def integer(v: int) -> dict:
    return {"t": "int", "v": int(v)}


def word(v: str) -> dict:
    return {"t": "word", "v": v}


def date_text(d: str) -> str:
    x = dt.date.fromisoformat(d)
    return f"{MONTHS[x.month - 1]} {x.day}, {x.year}"


def midpoint(a: str, b: str) -> str:
    da, db = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    return (da + (db - da) / 2).isoformat()


class QB:
    def __init__(self, w: World, broker: str, rng: random.Random):
        self.w, self.b, self.rng = w, broker, rng
        self.rows: list[dict] = []
        self.clients = [e for e in w.entities.values() if e["type"] == "client" and e["broker"] == broker]
        self.clients.sort(key=lambda e: [c[0] for c in CLIENTS[broker]].index(e["name"]))
        listed = {f["subject"] for f in w.broker_facts(broker) if f["subject"].startswith("bld:")}
        self.buildings = sorted(listed)

    def ent(self, eid: str) -> dict:
        e = self.w.entities[eid]
        names = [e["name"]] + [a for a in e["aliases"] if a.lower() not in {"the aurora", "aurora"}]
        if e["type"] == "client":
            names.append(e["name"].split()[-1])  # the surname alone identifies a client within a broker
        return {"t": "entity", "id": eid, "names": sorted(set(names))}

    def name(self, cid: str) -> str:
        e = self.w.entities[cid]
        return e["name"] if self.rng.random() < 0.75 else e["contact"] if " " in e["contact"] else e["name"]

    def cur(self, subj: str, pred: str):
        f = self.w.value_at(self.b, subj, pred)
        return f["object"] if f else None

    def add(self, qtype: str, text: str, kind: str, required: list[dict], evidence: list[str],
            stale: list[dict] = (), forbidden: list[dict] = (), leak: list[dict] = (), confusion: list[dict] = ()):
        self.rows.append({"id": f"{self.b}-q{len(self.rows) + 1:02d}", "broker": self.b, "type": qtype,
                          "question": text, "kind": kind, "required": required, "stale": list(stale),
                          "forbidden": list(forbidden), "leak": list(leak), "confusion": list(confusion),
                          "evidence": sorted(set(evidence))})

    def ev(self, subj: str, pred: str, day: str | None = None) -> list[str]:
        f = self.w.value_at(self.b, subj, pred, day)
        return [f["episode"]] if f else []

    def atom(self, pred: str, v):
        if pred in ("budget", "price"):
            return money(v)
        if pred in ("deadline", "delivery"):
            return month(v)
        if pred in ("bedrooms",):
            return integer(v)
        return self.ent(v)

    # ------------------------------------------------------------ question families

    def build(self) -> list[dict]:
        c = self.clients
        w, b = self.w, self.b
        hist = lambda s, p: w.history(b, s, p)
        bname = lambda bid: w.entities[bid]["name"]

        # single hop
        b1, b2 = self.rng.sample(self.buildings, 2)
        self.add("single_hop", f"Who is the developer of {bname(b1)}?", "value",
                 [self.ent(self.cur(b1, "developer"))], self.ev(b1, "developer"))
        self.add("single_hop", f"Which metro station is {bname(b2)} close to?", "value",
                 [self.ent(self.cur(b2, "near_station"))], self.ev(b2, "near_station"))
        cz = c[6]
        self.add("single_hop", f"Which neighborhood does {self.name(cz['id'])} want to live in?", "value",
                 [self.ent(self.cur(cz["id"], "wants_neighborhood"))], self.ev(cz["id"], "wants_neighborhood"))
        visits = [f for f in w.broker_facts(b) if f["subject"] == c[3]["id"] and f["predicate"] == "visited"]
        vset = sorted({f["object"] for f in visits})
        self.add("single_hop", f"Which buildings has {self.name(c[3]['id'])} visited?", "list",
                 [self.ent(x) for x in vset], [f["episode"] for f in visits],
                 forbidden=[self.ent(x) for x in self.buildings if x not in vset])
        obj = [f for f in w.broker_facts(b) if f["subject"] == c[3]["id"] and f["predicate"] == "objection"][0]
        self.add("single_hop", f"What worried {self.name(c[3]['id'])} about {bname(obj['about'])}?", "value",
                 [word(obj["object"])], [obj["episode"]])

        # temporal: the current value of a slot that changed
        for cl, pred, text in [(c[0], "budget", "What is {n}'s budget now?"),
                               (c[1], "interested_in", "Which building is {n} interested in now?"),
                               (c[2], "bedrooms", "How many bedrooms does {n} want now?")]:
            h = hist(cl["id"], pred)
            self.add("temporal_current", text.format(n=self.name(cl["id"])), "value",
                     [self.atom(pred, h[-1]["object"])], [h[-1]["episode"]],
                     stale=[self.atom(pred, f["object"]) for f in h[:-1]])
        changed_price = [x for x in self.buildings if len(hist(x, "price")) > 1]
        changed_delivery = [x for x in self.buildings if len(hist(x, "delivery")) > 1]
        bp, bd = changed_price[0], changed_delivery[0]
        h = hist(bp, "price")
        self.add("temporal_current", f"What is the current starting price at {bname(bp)}?", "value",
                 [money(h[-1]["object"])], [h[-1]["episode"]], stale=[money(f["object"]) for f in h[:-1]])
        h = hist(bd, "delivery")
        self.add("temporal_current", f"When will {bname(bd)} be delivered?", "value",
                 [month(h[-1]["object"])], [h[-1]["episode"]], stale=[month(f["object"]) for f in h[:-1]])

        # as of a past date
        for subj, pred, text, k in [(c[0]["id"], "budget", "What was {n}'s budget on {d}?", 1),
                                    (c[1]["id"], "interested_in", "Which building was {n} interested in on {d}?", 0),
                                    (bp, "price", "What was the starting price at {n} on {d}?", 0),
                                    (c[2]["id"], "bedrooms", "How many bedrooms did {n} want on {d}?", 0)]:
            h = hist(subj, pred)
            f = h[k]
            day = midpoint(f["valid_from"], f["valid_to"])
            n = bname(subj) if subj.startswith("bld:") else self.name(subj)
            self.add("as_of", text.format(n=n, d=date_text(day)), "value", [self.atom(pred, f["object"])],
                     [f["episode"]], stale=[self.atom(pred, g["object"]) for g in h if g is not f])

        # multi hop
        for cl, pred, text in [(c[1], "developer", "Who develops the building {n} is interested in now?"),
                               (c[5], "near_station", "Which metro station is near the building {n} is focused on now?")]:
            now_b = self.cur(cl["id"], "interested_in")
            old = [f["object"] for f in hist(cl["id"], "interested_in")[:-1]]
            stale = [self.ent(self.cur(o, pred)) for o in old if self.cur(o, pred) != self.cur(now_b, pred)]
            self.add("multi_hop", text.format(n=self.name(cl["id"])), "value", [self.ent(self.cur(now_b, pred))],
                     self.ev(cl["id"], "interested_in") + self.ev(now_b, pred), stale=stale)
        by_station: dict[str, list[str]] = {}
        for cl in c:
            bi = self.cur(cl["id"], "interested_in")
            by_station.setdefault(self.cur(bi, "near_station"), []).append(cl["id"])
        stn = sorted(by_station, key=lambda s: (-len(by_station[s]), s))[0]
        who = by_station[stn]
        self.add("multi_hop", f"Which of my clients are currently interested in a building near "
                              f"{w.entities[stn]['name']} station?", "list", [self.ent(x) for x in who],
                 [e for x in who for e in self.ev(x, "interested_in")],
                 forbidden=[self.ent(x["id"]) for x in c if x["id"] not in who])
        self.fit_question()

        # aggregation
        beds = {cl["id"]: self.cur(cl["id"], "bedrooms") for cl in c}
        n_beds = max(sorted(set(beds.values())), key=lambda k: list(beds.values()).count(k))
        self.add("aggregation", f"How many of my clients want {n_beds} bedrooms right now?", "count",
                 [integer(list(beds.values()).count(n_beds))], [e for x in beds for e in self.ev(x, "bedrooms")])
        budgets = sorted(self.cur(cl["id"], "budget") for cl in c)
        cut = (budgets[3] // 50_000) * 50_000
        if cut <= budgets[2]:
            cut = budgets[2] + 5_000
        n_above = sum(1 for v in budgets if v > cut)
        self.add("aggregation", f"How many of my clients have a budget above ${cut:,} right now?", "count",
                 [integer(n_above)], [e for cl in c for e in self.ev(cl["id"], "budget")])
        top = max(c, key=lambda cl: self.cur(cl["id"], "budget"))
        self.add("aggregation", "Which of my clients has the highest budget right now?", "value",
                 [self.ent(top["id"])], self.ev(top["id"], "budget"),
                 forbidden=[self.ent(x["id"]) for x in c if x is not top])

        # unanswerable near misses
        unlisted = [f"bld:{slug(x)}" for x in [e["name"] for e in w.entities.values() if e["type"] == "building"]
                    if f"bld:{slug(x)}" not in self.buildings]
        self.add("unanswerable", "When will Aurora Gardens be delivered?", "idk", [], [])
        self.add("unanswerable", f"What is the starting price at {bname(sorted(unlisted)[0])}?", "idk", [], [])
        self.add("unanswerable", f"How many parking spaces come with units at {bname(b1)}?", "idk", [], [])

        # isolation probes: another broker's clients, same first name first when there is one
        mine_first = {cl["name"].split()[0] for cl in c}
        foreign = [e for e in w.entities.values() if e["type"] == "client" and e["broker"] != b]
        foreign.sort(key=lambda e: (e["name"].split()[0] not in mine_first, e["id"]))
        picks = [foreign[0]] + self.rng.sample(foreign[1:], 2)
        for fe in picks:
            same = [x for x in c if x["name"].split()[0] == fe["name"].split()[0]]
            self.add("isolation", f"What is {fe['name']}'s budget?", "idk", [], [],
                     leak=[money(f["object"]) for f in w.history(fe["broker"], fe["id"], "budget")],
                     confusion=[money(f["object"]) for x in same for f in hist(x["id"], "budget")])
        return self.rows

    def fit_question(self) -> None:
        """The owner's example: clients wanting N bedrooms who can afford a building delivered in year Y near station S."""
        w, b = self.w, self.b
        best = None
        for stn in sorted({self.cur(x, "near_station") for x in self.buildings}):
            for year in ("2026", "2027", "2028"):
                blds = [x for x in self.buildings if self.cur(x, "near_station") == stn
                        and self.cur(x, "delivery").startswith(year)]
                for n in (1, 2, 3, 4):
                    fits, near = [], []
                    for cl in self.clients:
                        if self.cur(cl["id"], "bedrooms") != n:
                            continue
                        ok = [x for x in blds if n in self.units(x)]
                        if not ok:
                            continue
                        (fits if any(self.cur(x, "price") <= self.cur(cl["id"], "budget") for x in ok)
                         else near).append(cl["id"])
                    if fits:
                        score = (len(near) > 0, -abs(len(fits) - 2), stn, year, n)
                        if best is None or score > best[0]:
                            best = (score, stn, year, n, fits, blds)
        _, stn, year, n, fits, blds = best
        ev = [e for x in fits for p in ("bedrooms", "budget") for e in self.ev(x, p)]
        ev += [e for x in blds for p in ("price", "delivery", "near_station") for e in self.ev(x, p)]
        self.add("multi_hop", f"Which of my clients looking for {n} bedrooms can afford a building delivered in "
                              f"{year} near {w.entities[stn]['name']} station?", "list", [self.ent(x) for x in fits],
                 ev, forbidden=[self.ent(x["id"]) for x in self.clients if x["id"] not in fits])

    def units(self, bid: str) -> set[int]:
        return {f["object"] for f in self.w.broker_facts(self.b) if f["subject"] == bid
                and f["predicate"] == "unit_bedrooms"}


def build_questions(w: World, rng: random.Random) -> list[dict]:
    out = []
    for broker, _ in BROKERS:
        out.extend(QB(w, broker, rng).build())
    return out


__all__ = ["TYPES", "TODAY", "build_questions"]
