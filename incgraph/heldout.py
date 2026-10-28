"""Held-out questions for the closed-form parser, written before the parser existed.

Purpose: the seven question patterns of the second and third runs were written
against the generator's own wording. This set asks the same kinds of question
the way a broker types them (abbreviations such as "3br" or "dorm", slang such
as "rn" or "atm", typos, other word orders, missing words, "back in March"),
16 of them in Portuguese, plus 12 questions the parser must NOT turn into a
form (single-hop lookups, questions about attributes nobody records, questions
about another broker's clients).

Every question carries a gold form in the closed schema below, written by hand,
and a gold answer computed by code: ``evaluate`` runs the gold form on the
generator's true facts (not on any graph), honouring validity intervals. A
month-level date ("back in March") is accepted only when the gold answer is
the same on every day of that month, which ``build`` checks.

The schema (what a form may say):

    operation   join | count | max | min          ("none": not a question for a form)
    target      client | building
    filters     [field, op, value], all must hold
    can_afford  [field, op, value] building conditions: the client qualifies if some building
                meeting them has a starting price <= the client's budget and offers the
                client's bedroom count
    metric      the numeric field max / min compare (join, count: none)
    return      name (default) or interested_in.<building field> (the building the client
                is focused on); a field other than name only with join
    as_of       now | YYYY-MM-DD (gold may say YYYY-MM: any day of that month)

    ops         = != < <= > >= year      (= and != on a multi-valued field mean contains / lacks;
                                          year: a month field falls in that year)

This file and data/heldout.jsonl were committed on their own, before any parser
code, and are not edited afterwards.
"""

from __future__ import annotations

import calendar
import json
from pathlib import Path

from .synth import TODAY
from .values import norm_name

OPERATIONS = ["join", "count", "max", "min"]
TARGETS = ["client", "building"]
OPS = ["=", "!=", "<", "<=", ">", ">=", "year"]

# field -> (value kind, multi-valued)
CLIENT_FIELDS = {"name": ("name", False), "budget": ("money", False), "bedrooms": ("int", False),
                 "wants_neighborhood": ("place", False), "deadline": ("month", False),
                 "interested_in": ("building", False), "visited": ("building", True)}
BUILDING_FIELDS = {"name": ("name", False), "developer": ("place", False), "located_in": ("place", False),
                   "near_station": ("place", False), "price": ("money", False), "delivery": ("month", False),
                   "unit_bedrooms": ("int", True)}
CLIENT_FIELDS.update({f"interested_in.{k}": v for k, v in BUILDING_FIELDS.items()})
FIELDS = {"client": CLIENT_FIELDS, "building": BUILDING_FIELDS}
NUMERIC = {"money", "int", "month"}

# ------------------------------------------------------------------ the questions
# (broker, language, question, gold form). Forms use names as a broker would; evaluate() resolves them.

N, NOW = "now", "now"


def F(op, target="client", filters=(), can_afford=(), metric=None, ret="name", as_of=NOW):
    return {"operation": op, "target": target, "filters": [list(f) for f in filters],
            "can_afford": [list(f) for f in can_afford], "metric": metric, "return": ret, "as_of": as_of}


NONE = {"operation": "none"}

QUESTIONS = [
    # ---------------- broker a (Lena Ferraz)
    ("a", "en", "how many clients r looking for 3br rn?", F("count", filters=[("bedrooms", "=", 3)])),
    ("a", "en", "anyone w/ budget over 900k", F("join", filters=[("budget", ">", 900000)])),
    ("a", "en", "who's got the smallest budget atm", F("min", metric="budget")),
    ("a", "en", "which clients are after a place near harbor central right now",
     F("join", filters=[("interested_in.near_station", "=", "Harbor Central")])),
    ("a", "en", "who devlops the bulding Ruth is lookin at", F("join", filters=[("name", "=", "Ruth Castellano")],
                                                                ret="interested_in.developer")),
    ("a", "en", "back in March how many of my clients wanted 2 bedrooms?",
     F("count", filters=[("bedrooms", "=", 2)], as_of="2026-03")),
    ("a", "en", "highest budget back in feb, who was it", F("max", metric="budget", as_of="2026-02")),
    ("a", "en", "3 bed clients who can afford smth near Riverside delivered 2027",
     F("join", filters=[("bedrooms", "=", 3)], can_afford=[("near_station", "=", "Riverside"),
                                                             ("delivery", "year", 2027)])),
    ("a", "en", "how many buildings deliver in 2027", F("count", target="building",
                                                        filters=[("delivery", "year", 2027)])),
    ("a", "en", "cheapest building rn?", F("min", target="building", metric="price")),
    ("a", "pt", "quantos clientes procurando 3 dorm até 800 mil?",
     F("count", filters=[("bedrooms", "=", 3), ("budget", "<=", 800000)])),
    ("a", "pt", "quem tá interessado em prédio perto da estação Mill Street hoje?",
     F("join", filters=[("interested_in.near_station", "=", "Mill Street")])),
    ("a", "pt", "qual cliente tem o maior orçamento?", F("max", metric="budget")),
    ("a", "pt", "em 15 de março, quantos clientes tinham orçamento acima de 700 mil?",
     F("count", filters=[("budget", ">", 700000)], as_of="2026-03-15")),
    ("a", "pt", "prédios com 3 quartos abaixo de 700 mil", F("join", target="building",
                                                             filters=[("unit_bedrooms", "=", 3),
                                                                      ("price", "<", 700000)])),
    ("a", "en", "whats kate's budget now", NONE),
    ("a", "pt", "qual o prazo da Ingrid pra mudar?", NONE),
    ("a", "en", "how many clients want a penthouse with a pool?", NONE),
    ("a", "pt", "quantos clientes do Tomas querem 3 quartos?", NONE),
    # ---------------- broker b (Tomas Ekwueme)
    ("b", "en", "how many ppl want 4 bedrooms", F("count", filters=[("bedrooms", "=", 4)])),
    ("b", "en", "clients currently into Mill House?", F("join", filters=[("interested_in", "=", "Mill House")])),
    ("b", "en", "biggest budget rn", F("max", metric="budget")),
    ("b", "en", "who's eyeing a building near riverside now",
     F("join", filters=[("interested_in.near_station", "=", "Riverside")])),
    ("b", "en", "who builds the place Kenji wants now", F("join", filters=[("name", "=", "Kenji Albright")],
                                                          ret="interested_in.developer")),
    ("b", "en", "on March 1, how many clients were interested in Saltmarsh Gardens?",
     F("count", filters=[("interested_in", "=", "Saltmarsh Gardens")], as_of="2026-03-01")),
    ("b", "en", "3br clients that can afford smth near Harbor Central delivered 2027",
     F("join", filters=[("bedrooms", "=", 3)], can_afford=[("near_station", "=", "Harbor Central"),
                                                             ("delivery", "year", 2027)])),
    ("b", "en", "how many buildings under 600k", F("count", target="building", filters=[("price", "<", 600000)])),
    ("b", "en", "lowest budget among my clients back in March", F("min", metric="budget", as_of="2026-03")),
    ("b", "en", "what station is near the building Oskar is into now",
     F("join", filters=[("name", "=", "Oskar Brandt")], ret="interested_in.near_station")),
    ("b", "pt", "quantos clientes querem 3 quartos hoje?", F("count", filters=[("bedrooms", "=", 3)])),
    ("b", "pt", "quem tem orçamento acima de 1 milhão?", F("join", filters=[("budget", ">", 1000000)])),
    ("b", "pt", "qual o cliente com menor orçamento agora", F("min", metric="budget")),
    ("b", "pt", "prédios entregues em 2027 perto da estação Harbor Central",
     F("join", target="building", filters=[("delivery", "year", 2027), ("near_station", "=", "Harbor Central")])),
    ("b", "pt", "qual a construtora do prédio que a Leah quer?",
     F("join", filters=[("name", "=", "Leah Hartmann")], ret="interested_in.developer")),
    ("b", "en", "Marina's budget rn?", NONE),
    ("b", "pt", "o Samuel visitou quais prédios?", NONE),
    ("b", "en", "how many clients want a sea view?", NONE),
    ("b", "en", "what's Daniel Reyes's budget", NONE),
    # ---------------- broker c (Priya Valdivia)
    ("c", "en", "how many of the clients want 3 bdrm", F("count", filters=[("bedrooms", "=", 3)])),
    ("c", "en", "who's interested in N1 rn", F("join", filters=[("interested_in", "=", "Northbank One")])),
    ("c", "en", "top budget?", F("max", metric="budget")),
    ("c", "en", "clients after a building near Riverside today",
     F("join", filters=[("interested_in.near_station", "=", "Riverside")])),
    ("c", "en", "who develops amir's building now", F("join", filters=[("name", "=", "Amir Haddad")],
                                                      ret="interested_in.developer")),
    ("c", "en", "3br clients able to afford anything near Mill Street delivered in 2028",
     F("join", filters=[("bedrooms", "=", 3)], can_afford=[("near_station", "=", "Mill Street"),
                                                             ("delivery", "year", 2028)])),
    ("c", "en", "on march 20 who had the lowest budget", F("min", metric="budget", as_of="2026-03-20")),
    ("c", "en", "count buildings w/ 4 bedroom units", F("count", target="building",
                                                        filters=[("unit_bedrooms", "=", 4)])),
    ("c", "en", "most expensive building rn", F("max", target="building", metric="price")),
    ("c", "en", "how many clients had budget over 700k back on Feb 20",
     F("count", filters=[("budget", ">", 700000)], as_of="2026-02-20")),
    ("c", "pt", "quantas pessoas querem 1 quarto?", F("count", filters=[("bedrooms", "=", 1)])),
    ("c", "pt", "quem estava interessado no Mill House em 1 de março?",
     F("join", filters=[("interested_in", "=", "Mill House")], as_of="2026-03-01")),
    ("c", "pt", "qual o menor orçamento entre meus clientes hoje? quem é?", F("min", metric="budget")),
    ("c", "pt", "clientes de 3 dorm que conseguem pagar algo perto do metrô Riverside com entrega em 2027",
     F("join", filters=[("bedrooms", "=", 3)], can_afford=[("near_station", "=", "Riverside"),
                                                             ("delivery", "year", 2027)])),
    ("c", "pt", "qual estação fica perto do prédio que o Daniel Weber quer agora?",
     F("join", filters=[("name", "=", "Daniel Weber")], ret="interested_in.near_station")),
    ("c", "pt", "quantos prédios custam menos de 500 mil?", F("count", target="building",
                                                              filters=[("price", "<", 500000)])),
    ("c", "en", "Hugo's deadline?", NONE),
    ("c", "pt", "qual o orçamento da Bia hoje?", NONE),
    ("c", "en", "how many clients have pets?", NONE),
    ("c", "pt", "quanto a Kate Moreau pode pagar?", NONE),
]

# Gold answers for the questions that must not become a form: what the right answer is, for scoring the
# retrieval path that answers them (subject, predicate) as a lookup, or None for "I don't know".
NONE_GOLD = {
    "whats kate's budget now": ("Katherine Moreau", "budget"),
    "qual o prazo da Ingrid pra mudar?": ("Ingrid Sato", "deadline"),
    "Marina's budget rn?": ("Marina Lindqvist", "budget"),
    "o Samuel visitou quais prédios?": ("Samuel Okonjo", "visited"),
    "Hugo's deadline?": ("Hugo Tamsin", "deadline"),
    "qual o orçamento da Bia hoje?": ("Beatriz Lund", "budget"),
}


# ------------------------------------------------------------------ ground truth


class Truth:
    """The broker's true facts, with validity, and the names that refer to each entity."""

    def __init__(self, broker: str, facts: list[dict], entities: dict[str, dict]):
        self.broker = broker
        self.facts = [f for f in facts if f["broker"] == broker]
        self.entities = entities
        self.clients = sorted(e["id"] for e in entities.values() if e["type"] == "client" and e["broker"] == broker)
        self.buildings = sorted({f["subject"] for f in self.facts if f["subject"].startswith("bld:")})
        self.alias: dict[str, set[str]] = {}
        for e in entities.values():
            if e["broker"] in (None, broker):
                names = [e["name"], *e["aliases"], e.get("contact") or ""]
                if e["type"] == "client":
                    names += [e["name"].split()[0], e["name"].split()[-1]]
                for n in names:
                    if n:
                        self.alias.setdefault(_key(n), set()).add(e["id"])

    def resolve(self, name: str) -> set[str]:
        return self.alias.get(_key(name), set())

    def value(self, subj: str, pred: str, at: str):
        """Single-valued: the object that held at ``at`` (a YYYY-MM-DD day), or None."""
        for f in self.facts:
            if f["subject"] == subj and f["predicate"] == pred and f["valid_from"] <= at and (
                    f["valid_to"] is None or at < f["valid_to"]):
                return f["object"]
        return None

    def values(self, subj: str, pred: str, at: str) -> set:
        """Multi-valued: every object stated on or before ``at``."""
        return {f["object"] for f in self.facts if f["subject"] == subj and f["predicate"] == pred
                and f["valid_from"] <= at}

    def field(self, eid: str, field: str, at: str):
        if field == "name":
            return eid
        if field.startswith("interested_in."):
            b = self.value(eid, "interested_in", at)
            return self.field(b, field.split(".", 1)[1], at) if b else None
        target = "client" if eid.startswith("cli:") else "building"
        kind, multi = FIELDS[target][field]
        return self.values(eid, field, at) if multi else self.value(eid, field, at)

    def exists(self, eid: str, at: str) -> bool:
        return any(f["subject"] == eid and f["valid_from"] <= at for f in self.facts)


def _key(name: str) -> str:
    return " ".join(w for w in norm_name(str(name)).split() if w not in ("station", "metro", "estacao", "estação"))


def _test(truth: Truth, actual, op: str, value, kind: str, multi: bool) -> bool:
    if actual is None or (multi and not actual):
        return False
    if kind in ("name", "place", "building"):
        targets = truth.resolve(value)
        hit = (bool(actual & targets) if multi else actual in targets)
        return hit if op == "=" else (not hit if op == "!=" else False)
    if op == "year":
        return str(actual)[:4] == str(value)
    vals = actual if multi else {actual}
    cmp = {"=": lambda a, b: a == b, "!=": lambda a, b: a != b, "<": lambda a, b: a < b,
           "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}[op]
    return any(cmp(v, value) for v in vals) if op != "!=" else all(cmp(v, value) for v in vals)


def evaluate(truth: Truth, form: dict, at: str) -> list | int:
    """Run a form on the true facts at day ``at``: a sorted list of entity ids or values, or a count."""
    target = form["target"]
    pool = truth.clients if target == "client" else truth.buildings
    pool = [e for e in pool if truth.exists(e, at)]
    fields = FIELDS[target]
    out = []
    for e in pool:
        ok = all(_test(truth, truth.field(e, f, at), op, v, *fields[f]) for f, op, v in form["filters"])
        if ok and form["can_afford"]:
            budget, beds = truth.value(e, "budget", at), truth.value(e, "bedrooms", at)
            ok = budget is not None and beds is not None and any(
                all(_test(truth, truth.field(b, f, at), op, v, *BUILDING_FIELDS[f]) for f, op, v in form["can_afford"])
                and (truth.value(b, "price", at) or 10**12) <= budget and beds in truth.values(b, "unit_bedrooms", at)
                for b in truth.buildings if truth.exists(b, at))
        if ok:
            out.append(e)
    op = form["operation"]
    if op == "count":
        return len(out)
    if op in ("max", "min"):
        scored = [(truth.field(e, form["metric"], at), e) for e in out]
        scored = [(v, e) for v, e in scored if v is not None]
        best = (max if op == "max" else min)(v for v, _ in scored)
        return sorted(e for v, e in scored if v == best)
    if form["return"] == "name":
        return sorted(out)
    return sorted({truth.field(e, form["return"], at) for e in out} - {None})


def days(as_of: str) -> list[str]:
    if as_of == "now":
        return [TODAY]
    if len(as_of) == 7:
        y, m = map(int, as_of.split("-"))
        return [f"{y}-{m:02d}-{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]
    return [as_of]


def entity_atom(entities: dict, eid: str) -> dict:
    e = entities[eid]
    names = [e["name"]] + [a for a in e["aliases"] if a.lower() not in {"the aurora", "aurora"}]
    if e["type"] == "client":
        names.append(e["name"].split()[-1])
    return {"t": "entity", "id": eid, "names": sorted(set(names))}


def value_atom(entities: dict, field: str, v) -> dict:
    if isinstance(v, str) and v in entities:
        return entity_atom(entities, v)
    kind = FIELDS["client"].get(field, FIELDS["building"].get(field, ("name", False)))[0]
    if kind == "money":
        return {"t": "money", "v": int(v)}
    if kind == "month":
        return {"t": "month", "v": v}
    return {"t": "int", "v": int(v)}


def build(facts: list[dict], entities: dict[str, dict]) -> list[dict]:
    rows = []
    truths = {b: Truth(b, facts, entities) for b in "abc"}
    for i, (broker, lang, text, form) in enumerate(QUESTIONS, start=1):
        t = truths[broker]
        row = {"id": f"h{i:02d}", "broker": broker, "lang": lang, "question": text, "gold_form": form,
               "must_parse": form["operation"] != "none", "type": "heldout", "stale": [], "leak": [],
               "confusion": [], "evidence": []}
        if form["operation"] == "none":
            look = NONE_GOLD.get(text)
            if look is None:
                row.update(kind="idk", required=[], forbidden=[])
            else:
                (cid,) = t.resolve(look[0]) & set(t.clients)
                pred = look[1]
                if pred == "visited":
                    vals = sorted(t.values(cid, pred, TODAY))
                    row.update(kind="list", required=[entity_atom(entities, v) for v in vals],
                               forbidden=[entity_atom(entities, b) for b in t.buildings if b not in vals])
                else:
                    row.update(kind="value", required=[value_atom(entities, pred, t.value(cid, pred, TODAY))],
                               forbidden=[])
            rows.append(row)
            continue
        assert form["operation"] in OPERATIONS and form["target"] in TARGETS, text
        for f, op, _ in form["filters"] + [["can_afford." + x[0], x[1], x[2]] for x in form["can_afford"]]:
            assert op in OPS and f.removeprefix("can_afford.") in FIELDS["building" if f.startswith("can_afford.")
                                                                         else form["target"]], (text, f)
        answers = [evaluate(t, form, d) for d in days(form["as_of"])]
        assert all(a == answers[0] for a in answers), f"answer changes within {form['as_of']}: {text}"
        ans = answers[0]
        assert ans, f"empty or zero answer: {text}"
        pool = t.clients if form["target"] == "client" else t.buildings
        if form["operation"] == "count":
            row.update(kind="count", required=[{"t": "int", "v": ans}], forbidden=[], gold_answer=ans)
        elif form["operation"] == "join" and form["return"] == "name":
            row.update(kind="list", required=[entity_atom(entities, e) for e in ans],
                       forbidden=[entity_atom(entities, e) for e in pool if e not in ans],
                       gold_answer=[entities[e]["name"] for e in ans])
        elif form["operation"] == "join":
            row.update(kind="value", required=[value_atom(entities, form["return"].split(".")[-1], v) for v in ans],
                       forbidden=[], gold_answer=[entities[v]["name"] if v in entities else v for v in ans])
        else:
            row.update(kind="value", required=[entity_atom(entities, e) for e in ans],
                       forbidden=[entity_atom(entities, e) for e in pool if e not in ans],
                       gold_answer=[entities[e]["name"] for e in ans])
        rows.append(row)
    return rows


def write(data_dir: str | Path) -> dict:
    d = Path(data_dir)
    facts = [json.loads(x) for x in (d / "facts.jsonl").read_text().splitlines() if x.strip()]
    entities = {e["id"]: e for e in (json.loads(x) for x in (d / "entities.jsonl").read_text().splitlines() if x)}
    rows = build(facts, entities)
    (d / "heldout.jsonl").write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
    return {"questions": len(rows), "must_parse": sum(r["must_parse"] for r in rows),
            "portuguese": sum(r["lang"] == "pt" for r in rows),
            "portuguese_must_parse": sum(r["lang"] == "pt" and r["must_parse"] for r in rows)}
