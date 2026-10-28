"""A closed form, filled by the model and validated and run by code (fourth run).

Instead of seven regular expressions tied to the generator's wording, one short
LLM call turns any question into a form of the closed schema in
``heldout.py``: an operation (join, count, max, min, or none), a target
(client or building), filters (field, operator, value), optional
``can_afford`` building conditions, a metric, a return field and a date.
The model writes it as a few short lines; nothing else.

Code then decides whether to trust it:

* the operation, target, fields and operators must be in the closed lists, and
  the operator must fit the field (names take = or !=, amounts and counts take
  comparisons, ``year`` only months);
* every value must normalize (an amount, a count, a month, a year, or a name
  that resolves to at least one node of the right type in this broker's graph;
  a client name must resolve to exactly one client);
* max and min need a numeric metric; a return field other than the name is only
  allowed with join, and only as interested_in.<building field>, so a plain
  lookup ("Kate's budget") is not a form;
* the date must be "now" or a day on or before today.

A form that fails any check, or an operation "none", sends the question down the
ordinary retrieval path, as before. A valid form runs on the graph's edges at
its date (``valid_from <= date < valid_to``), and the model receives the result,
the facts that prove it and a plain description of what was understood, which
is also appended to the answer so the broker can see how the question was read.
"""

from __future__ import annotations

import datetime as dt
import re

from .heldout import BUILDING_FIELDS, FIELDS, NUMERIC, OPERATIONS, OPS, TARGETS
from .resolve import building_match
from .synth import TODAY
from .values import norm_name, parse_int, parse_month

SYSTEM = "You turn a real-estate broker's question into a small query form. You reply with the form only."

SCHEMA = """Fill this form for the broker's question about the broker's own clients and buildings.

op: join (list the matching clients or buildings), count, max, min, or none
target: client or building
as_of: now, or a date YYYY-MM-DD (today is """ + TODAY + """; a month alone means a day in that month)
filter: <field> <operator> <value>          (any number of lines; all must hold)
afford: <building field> <operator> <value> (client questions about what they can afford: some building meeting
                                             these has a price within the client's budget and their bedroom count)
metric: <field>                             (max and min only)
return: <field>                             (join only: name, or interested_in.<building field>)

Client fields: name, budget, bedrooms, wants_neighborhood, deadline, interested_in (the building the client is
focused on), visited, interested_in.<building field> (a field of that building)
Building fields: name, developer, located_in, near_station, price, delivery, unit_bedrooms
Operators: = != < <= > >= year (year: a delivery or deadline in that year)
Amounts are plain dollars (800k = 800000, 1 milhão = 1000000). Use op none for a question about one known client's
or building's own field, for anything these fields cannot express, and for questions about another broker's clients.

Example: anyone wanting 2 bedrooms with a budget under 500k?
op: join
target: client
as_of: now
filter: bedrooms = 2
filter: budget < 500000

Example: quantos prédios entregam em 2028?
op: count
target: building
as_of: now
filter: delivery year 2028

Example: who had the highest budget on 3 Feb?
op: max
target: client
as_of: 2026-02-03
metric: budget

Example: what neighborhood is the building Jo Smith likes in
op: join
target: client
as_of: now
filter: name = Jo Smith
return: interested_in.located_in

Example: 2br clients who could pay for something near Elm Square delivered in 2027
op: join
target: client
as_of: now
filter: bedrooms = 2
afford: near_station = Elm Square
afford: delivery year 2027

Example: what is Jo Smith's deadline
op: none"""


def form_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{SCHEMA}\n\nQuestion: {question}"}]


def parse(text: str) -> dict:
    """The model's lines into a raw form (strings); unknown lines are ignored."""
    raw = {"operation": None, "target": None, "as_of": "now", "filters": [], "can_afford": [], "metric": None,
           "return": "name"}
    for line in text.splitlines():
        m = re.match(r"^\s*(op|operation|target|as_of|filter|afford|metric|return)\s*:\s*(.*?)\s*$", line, re.I)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip().strip("`")
        if key in ("op", "operation"):
            raw["operation"] = val.lower()
        elif key in ("target", "metric", "return"):
            raw[key] = val
        elif key == "as_of":
            raw["as_of"] = val
        else:
            fm = re.match(r"^([\w.]+)\s+(!=|<=|>=|=|<|>|year)\s+(.+)$", val, re.I)
            raw["filters" if key == "filter" else "can_afford"].append(
                [fm.group(1), fm.group(2).lower(), fm.group(3).strip().strip("\"'")] if fm else [val, None, None])
    return raw


# ------------------------------------------------------------------ validation


def money(v) -> int | None:
    """Dollars from '800000', '800k', '$800,000', '800 mil', '1,2 milhão', '1.2M'."""
    s = str(v).lower().replace("$", "").replace("us", "").strip()
    m = re.match(r"^([\d.,]+)\s*(k|mil|thousand|m|mi|mm|million|milhao|milhão|milhoes|milhões)?$", s)
    if not m:
        return None
    num, suf = m.group(1), m.group(2) or ""
    if suf in ("", "k", "mil", "thousand") and re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", num):
        num = re.sub(r"[.,]", "", num)  # thousands separators, either convention
    num = num.replace(",", ".")
    try:
        x = float(num)
    except ValueError:
        return None
    x *= {"k": 1e3, "mil": 1e3, "thousand": 1e3}.get(suf, 1e6 if suf else 1)
    return int(round(x)) if x >= 1000 else None


def day(v: str) -> str | None:
    s = str(v).strip().lower()
    if s in ("now", "today", "hoje", "agora", ""):
        return None
    try:
        if re.fullmatch(r"20\d\d-\d\d-\d\d", s):
            d = dt.date.fromisoformat(s)
        elif re.fullmatch(r"20\d\d-\d\d", s):
            d = dt.date.fromisoformat(s + "-15")
        else:
            return "invalid"
    except ValueError:
        return "invalid"
    return d.isoformat() if d.isoformat() <= TODAY else "invalid"


class Invalid(ValueError):
    pass


def resolve_names(graph, value: str, kind: str) -> list[str]:
    """Nodes of this broker's graph a name refers to: exact name or alias first, then word-prefix rules."""
    etype = {"building": "building", "name": None, "place": None}[kind]
    types = [etype] if etype else ["client", "building", "developer", "neighborhood", "station"]
    key = lambda s: " ".join(w for w in norm_name(s).split() if w not in ("station", "metro", "estacao", "estação"))
    exact = [n for n in graph.nodes if graph.nodes[n]["type"] in types
             and any(key(x) == key(value) for x in graph.names(n))]
    if exact:
        return sorted(exact)
    return sorted(n for n in graph.nodes if graph.nodes[n]["type"] in types and graph.nodes[n]["type"] != "client"
                  and any(building_match(value, x) for x in graph.names(n)))


def validate(raw: dict, graph) -> dict:
    """The normalized form, or raises Invalid with the reason. Values become ints, months, years or node ids."""
    op = (raw.get("operation") or "").lower()
    if op == "none" or not op:
        raise Invalid("operation none")
    if op not in OPERATIONS:
        raise Invalid(f"unknown operation {op!r}")
    target = (raw.get("target") or "").lower()
    if target not in TARGETS:
        raise Invalid(f"unknown target {target!r}")

    def norm_filter(field, o, value, fields, ctx):
        if field not in fields:
            raise Invalid(f"unknown {ctx} field {field!r}")
        if o not in OPS:
            raise Invalid(f"unknown operator {o!r}")
        kind, multi = fields[field]
        if kind in ("name", "place", "building"):
            if o not in ("=", "!="):
                raise Invalid(f"operator {o} on name field {field}")
            if field == "name" and target == "client" and ctx == "filter":
                from .retrieve import link
                hits = [n for n in link(graph, str(value)) if graph.nodes[n]["type"] == "client"]
                if len(hits) != 1:
                    raise Invalid(f"client name {value!r} matches {len(hits)} clients")
                return [field, o, hits]
            hits = resolve_names(graph, str(value), kind)
            if not hits:
                raise Invalid(f"{value!r} names nothing in the graph")
            return [field, o, hits]
        if o == "year":
            if kind != "month" or not re.fullmatch(r"20\d\d", str(value).strip()):
                raise Invalid(f"year operator needs a month field and a year, got {field} {value!r}")
            return [field, o, int(value)]
        if multi and o != "=":
            raise Invalid(f"operator {o} on multi-valued field {field}")
        v = money(value) if kind == "money" else parse_int(value) if kind == "int" else parse_month(str(value))
        if v is None:
            raise Invalid(f"value {value!r} is not a {kind}")
        return [field, o, v]

    # "interested_in.name" is the building itself: the same filter as "interested_in" (found in the fourth run,
    # where the executor had no rule for it and matched nothing)
    raw_filters = [["interested_in" if f == "interested_in.name" else f, o, v] for f, o, v in raw.get("filters", [])]
    filters = [norm_filter(f, o, v, FIELDS[target], "filter") for f, o, v in raw_filters]
    afford = [norm_filter(f, o, v, BUILDING_FIELDS, "afford") for f, o, v in raw.get("can_afford", [])]
    if afford and target != "client":
        raise Invalid("afford conditions need target client")
    metric = raw.get("metric") or None
    metric = None if metric in (None, "", "-", "none") else metric
    if op in ("max", "min"):
        if metric not in FIELDS[target] or FIELDS[target][metric][0] not in NUMERIC or FIELDS[target][metric][1]:
            raise Invalid(f"{op} needs a single-valued numeric metric, got {metric!r}")
    elif metric:
        raise Invalid(f"metric with operation {op}")
    ret = (raw.get("return") or "name").strip()
    if ret == "interested_in.name":
        ret = "interested_in"  # the building a client is focused on is the client's own field
        if op != "join" or target != "client" or all(f[0] == "name" for f in filters):
            raise Invalid("the building one named client is focused on is a lookup, not a form")
    if ret not in ("name", "interested_in"):
        if op != "join" or target != "client" or not ret.startswith("interested_in.") or ret not in FIELDS["client"]:
            raise Invalid(f"return {ret!r} not allowed")
    elif op == "join" and not afford and all(f[0] == "name" for f in filters):
        raise Invalid("a join on names alone is a lookup, not a form")
    at = day(raw.get("as_of") or "now")
    if at == "invalid":
        raise Invalid(f"date {raw.get('as_of')!r}")
    return {"operation": op, "target": target, "filters": filters, "can_afford": afford, "metric": metric,
            "return": ret, "as_of": at or "now"}


# ------------------------------------------------------------------ running it on the graph


class Runner:
    def __init__(self, graph):
        self.g = graph

    def edges(self, nid: str, pred: str, at: str | None) -> list[dict]:
        out = []
        for e in self.g.edges_of(nid):
            if e["src"] != nid or e["pred"] != pred:
                continue
            if at is None:
                ok = e["valid_to"] is None
            else:
                ok = e["valid_from"] <= at and (e["valid_to"] is None or at < e["valid_to"])
            if ok:
                out.append(e)
        return out

    def field(self, nid: str, field: str, at: str | None) -> list[dict]:
        """The edges giving a node's value of ``field`` at ``at`` (interested_in.x follows the building)."""
        if field.startswith("interested_in."):
            out = []
            for b in self.edges(nid, "interested_in", at)[:1]:
                sub = self.field(b["dst"], field.split(".", 1)[1], at)
                if sub:
                    out += [b] + sub
            return out
        return self.edges(nid, field, at)

    @staticmethod
    def val(e: dict):
        return e["dst"] if e["dst"] is not None else e["value"]

    def test(self, nid: str, flt: list, at: str | None, fields: dict) -> list[dict] | None:
        """The edges proving the filter holds for ``nid``, or None."""
        field, op, value = flt
        if field == "name":
            return [] if (nid in value) == (op == "=") else None
        es = self.field(nid, field, at)
        last = field.split(".")[-1]
        es_val = [e for e in es if e["pred"] == last]
        if not es_val:
            return None
        cmp = {"=": lambda a: a == value, "!=": lambda a: a != value, "<": lambda a: a < value,
               "<=": lambda a: a <= value, ">": lambda a: a > value, ">=": lambda a: a >= value,
               "year": lambda a: str(a)[:4] == str(value)}
        if isinstance(value, list):
            ok = [e for e in es_val if self.val(e) in value]
            hit = bool(ok)
            return (es[:-len(es_val)] + ok if hit else None) if op == "=" else (es if not hit else None)
        ok = [e for e in es_val if cmp[op](self.val(e))]
        return (es[:-len(es_val)] + ok) if ok else None

    def exists(self, nid: str, at: str | None) -> bool:
        return at is None or any(e["valid_from"] <= at for e in self.g.edges_of(nid) if e["src"] == nid)

    def run(self, form: dict) -> tuple[list | int, list[dict], list[str]]:
        """(result, evidence edges, matched node ids). Result: node ids or values (join), an int (count)."""
        at = None if form["as_of"] == "now" else form["as_of"]
        target = form["target"]
        ev, matched = [], []
        for nid in self.g.of_type(target):
            if not self.exists(nid, at):
                continue
            proofs = [self.test(nid, f, at, FIELDS[target]) for f in form["filters"]]
            if any(p is None for p in proofs):
                continue
            proof = [e for p in proofs for e in p]
            if form["can_afford"]:
                budget, beds = self.edges(nid, "budget", at), self.edges(nid, "bedrooms", at)
                if not budget or not beds:
                    continue
                found = None
                for b in self.g.of_type("building"):
                    ps = [self.test(b, f, at, BUILDING_FIELDS) for f in form["can_afford"]]
                    price = self.edges(b, "price", at)
                    units = [e for e in self.edges(b, "unit_bedrooms", at) if e["value"] == beds[0]["value"]]
                    if all(p is not None for p in ps) and price and units and price[0]["value"] <= budget[0]["value"]:
                        found = [e for p in ps for e in p] + [price[0], units[0]]
                        break
                if found is None:
                    continue
                proof += budget[:1] + beds[:1] + found
            if form["metric"]:
                m = self.field(nid, form["metric"], at)
                if not m:
                    continue
                proof += m
            matched.append(nid)
            ev += proof
        op = form["operation"]
        if op == "count":
            return len(matched), ev, matched
        if op in ("max", "min"):
            vals = {n: self.val(self.field(n, form["metric"], at)[-1]) for n in matched}
            if not vals:
                return [], [], []
            best = (max if op == "max" else min)(vals.values())
            top = [n for n in matched if vals[n] == best]
            keep = {id(e) for n in top for e in self.field(n, form["metric"], at)}
            return top, [e for e in ev if id(e) in keep], top
        if form["return"] == "name":
            return matched, ev, matched
        vals = []
        for n in matched:
            chain = self.field(n, form["return"], at)  # the client's building edge, then the returned field
            ev += chain
            for e in chain[-1:]:
                if self.val(e) not in vals:
                    vals.append(self.val(e))
        return vals, ev, matched

    def show(self, v, field: str = "") -> str:
        from .graph import fmt_value
        if isinstance(v, str) and v in self.g.nodes:
            return self.g.nodes[v]["name"]
        pred = field.split(".")[-1]
        return fmt_value(pred, v) if pred in ("budget", "price", "deadline", "delivery") else str(v)

    def describe(self, form: dict) -> str:
        """What was understood, in plain words: 'counted clients with bedrooms = 3, budget <= $800,000, now'."""
        verb = {"join": "listed", "count": "counted", "max": "found the highest " + str(form["metric"]) + " among",
                "min": "found the lowest " + str(form["metric"]) + " among"}[form["operation"]]
        conds = []
        for f, o, v in form["filters"]:
            shown = " or ".join(self.show(x) for x in v) if isinstance(v, list) else str(v) if o == "year" \
                else self.show(v, f)
            conds.append(f"{f} {o} {shown}")
        for f, o, v in form["can_afford"]:
            shown = " or ".join(self.show(x) for x in v) if isinstance(v, list) else str(v) if o == "year" \
                else self.show(v, f)
            conds.append(f"can afford a building with {f} {o} {shown}")
        what = f"{form['target']}s" + (f" with {', '.join(conds)}" if conds else "")
        ret = "" if form["return"] == "name" else f", returning {form['return']}"
        when = "now" if form["as_of"] == "now" else f"as of {form['as_of']}"
        return f"{verb} {what}{ret}, {when}"

    def context(self, form: dict) -> tuple[str, list[str], str]:
        result, ev, _ = self.run(form)
        if isinstance(result, int):
            shown = str(result)
        else:
            shown = ", ".join(self.show(v, form["return"]) for v in result) or "none"
        seen, lines = set(), []
        for e in ev:
            if e["id"] not in seen:
                seen.add(e["id"])
                lines.append(self.g.fact_text(e))
        understood = self.describe(form)
        text = (f"Computed from the graph by a query the question was turned into.\nUnderstood as: {understood}.\n"
                f"Result: {shown}\nEvidence:\n" + ("\n".join(lines[:60]) or "(no matching facts)"))
        return text, sorted({x for e in ev for x in e["episodes"]}), understood


def understand(llm, graph, question: str, ref: str, max_tokens: int = 80) -> dict:
    """One short call; returns the raw text, the raw form, the validated form or the reason it was rejected."""
    res = llm.chat(form_messages(question), max_tokens=max_tokens, tag="form", ref=ref)
    raw = parse(res.text)
    try:
        form = validate(raw, graph)
        return {"text": res.text, "raw": raw, "form": form, "reason": None}
    except Invalid as exc:
        return {"text": res.text, "raw": raw, "form": None, "reason": str(exc)}
