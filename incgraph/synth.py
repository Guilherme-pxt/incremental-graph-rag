"""Seeded synthetic world: a fictional brokerage in an invented city, and its episode stream.

Velmora is an invented city. Three brokers work there, each with their own
clients, and all of them sell from the same catalogue of eight invented
buildings. Every broker produces a stream of episodes (WhatsApp-style chat
messages and the broker's own notes), in English, from January to June 2026.
Facts change over time on purpose: clients raise or cut budgets, change the
number of bedrooms, switch buildings; buildings change price and delivery date.

Confounders, also on purpose:
* two clients called Marina at two different brokers, and two clients called
  Daniel at the same broker;
* nicknames: a chat contact saved as "Kate Moreau" or "Fil Arantes" is the
  client the notes call "Katherine Moreau" or "Felipe Arantes";
* one building under several names ("Northbank One" and "N1", "Quarry Lofts"
  and "the Lofts"), and two buildings that share a word ("Aurora Residences"
  and "Aurora Park"), so "the Aurora" is ambiguous without context.

The generator keeps the ground truth: every entity with the surface forms used
for it, every fact with the episode that stated it and its validity interval,
and, per episode, which entity each surface form refers to. Graph quality and
answers are scored against that. Output is byte-identical for a given seed.
Nothing here describes a real company or person.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

TODAY = "2026-06-30"  # "now" for every question: the day after the last episode

NEIGHBORHOODS = ["Northbank", "Old Quarry", "Saltmarsh", "Linden Hill", "Harbor Row"]
STATIONS = ["Riverside", "Quarry Gate", "Harbor Central", "Mill Street", "Fenwick Square"]
DEVELOPERS = ["Halden Group", "Brightwater Homes", "Corvane Developments", "Tessaro Building Co"]

# name, aliases used in text (unambiguous), developer, neighborhood, station, unit bedrooms, price, delivery
BUILDINGS = [
    ("Aurora Residences", ["Aurora Res."], "Halden Group", "Northbank", "Riverside", [2, 3], 640_000, "2027-06"),
    ("Aurora Park", [], "Brightwater Homes", "Saltmarsh", "Harbor Central", [1, 2], 412_000, "2026-11"),
    ("Linden Court", ["the Linden"], "Corvane Developments", "Linden Hill", "Mill Street", [3, 4], 985_000, "2028-03"),
    ("Quarry Lofts", ["the Lofts"], "Tessaro Building Co", "Old Quarry", "Quarry Gate", [1, 2], 357_000, "2026-09"),
    ("Harbor Point", ["Harbor Point Tower"], "Halden Group", "Harbor Row", "Harbor Central", [2, 3], 724_000, "2027-02"),
    ("Saltmarsh Gardens", ["the Gardens"], "Brightwater Homes", "Saltmarsh", "Riverside", [3], 593_000, "2027-10"),
    ("Mill House", ["Mill House tower"], "Corvane Developments", "Old Quarry", "Mill Street", [2, 3, 4], 836_000, "2027-08"),
    ("Northbank One", ["N1"], "Tessaro Building Co", "Northbank", "Riverside", [1, 2, 3], 508_000, "2028-01"),
]

BROKERS = [("a", "Lena Ferraz"), ("b", "Tomas Ekwueme"), ("c", "Priya Valdivia")]

# full name, chat contact name (how the client is saved in the broker's phone), short forms used in notes
CLIENTS = {
    "a": [("Katherine Moreau", "Kate Moreau", ["Kate Moreau", "Katherine M."]),
          ("Marina Okafor", "Marina Okafor", ["Marina O."]),
          ("Jonah Pellegrini", "Jonah P", ["Jonah P."]),
          ("Ingrid Sato", "Ingrid Sato", ["Ingrid S."]),
          ("Felipe Arantes", "Fil Arantes", ["Fil Arantes"]),
          ("Ruth Castellano", "Ruth C", ["Ruth C."]),
          ("William Thorne", "Will Thorne", ["Will Thorne"])],
    "b": [("Marina Lindqvist", "Marina L", ["Marina L."]),
          ("Oskar Brandt", "Oskar Brandt", ["Oskar B."]),
          ("Leah Hartmann", "Leah H", ["Leah H."]),
          ("Samuel Okonjo", "Sam Okonjo", ["Sam Okonjo"]),
          ("Teresa Vilar", "Teresa Vilar", ["Teresa V."]),
          ("Kenji Albright", "Kenji A", ["Kenji A."]),
          ("Nadia Ferreira", "Nadia Ferreira", ["Nadia F."])],
    "c": [("Daniel Reyes", "Dani Reyes", ["Dani Reyes", "Daniel R."]),
          ("Daniel Weber", "Daniel W", ["Daniel W."]),
          ("Beatriz Lund", "Bia Lund", ["Bia Lund"]),
          ("Hugo Tamsin", "Hugo Tamsin", ["Hugo T."]),
          ("Clara Novak", "Clara Novak", ["Clara N."]),
          ("Amir Haddad", "Amir H", ["Amir H."]),
          ("Elisabeth Grant", "Liz Grant", ["Liz Grant"])],
}

# What happens to client i of every broker after the introduction (coverage by construction).
PLANS = [
    ["budget", "visit", "budget"],          # budget changes twice
    ["visit", "switch", "visit"],           # switches building
    ["bedrooms", "budget", "visit"],        # needs another bedroom count, budget follows
    ["visit", "objection", "visit"],        # two visits and an objection
    ["deadline", "visit"],                  # moves the deadline
    ["budget", "switch"],                   # cuts the budget and switches building
    ["visit"],                              # nothing changes
]

SINGLE = {"budget", "bedrooms", "wants_neighborhood", "deadline", "interested_in",
          "developer", "located_in", "near_station", "price", "delivery"}
MULTI = {"visited", "objection", "unit_bedrooms"}
PREDICATES = sorted(SINGLE | MULTI)
ENTITY_OBJECTS = {"wants_neighborhood": "neighborhood", "interested_in": "building", "visited": "building",
                  "developer": "developer", "located_in": "neighborhood", "near_station": "station"}

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
OBJECTIONS = [("the kitchen", "kitchen", "the kitchen is too small"), ("the noise", "noise", "the street noise"),
              ("the parking", "parking", "there is only one parking spot"), ("the view", "view", "the view is blocked")]


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-").replace("--", "-")


def money_text(v: int, rng: random.Random) -> str:
    k = v // 1000
    return rng.choice([f"${v:,}", f"{k}k", f"${k}k", f"{v:,} dollars"])


def month_text(m: str, rng: random.Random) -> str:
    y, mo = m.split("-")
    return rng.choice([f"{MONTHS[int(mo) - 1]} {y}", f"{MONTHS[int(mo) - 1][:3]} {y}", f"{y}-{mo}"])


def add_days(d: str, n: int) -> str:
    return (dt.date.fromisoformat(d) + dt.timedelta(days=n)).isoformat()


def human_date(d: str) -> str:
    x = dt.date.fromisoformat(d)
    return f"{MONTHS[x.month - 1]} {x.day}"


class World:
    """Ground truth for one seed: entities, episodes, facts, questions."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.entities: dict[str, dict] = {}
        self.episodes: list[dict] = []
        self.facts: list[dict] = []
        self._pending: dict[str, list[dict]] = {}

    # ------------------------------------------------------------ entities

    def entity(self, eid: str, etype: str, name: str, broker: str | None, aliases=(), **extra) -> str:
        if eid not in self.entities:
            self.entities[eid] = {"id": eid, "type": etype, "name": name, "broker": broker,
                                  "aliases": sorted(set(aliases) - {name}), **extra}
        return eid

    def build_catalog(self) -> None:
        for n in NEIGHBORHOODS:
            self.entity(f"nbh:{slug(n)}", "neighborhood", n, None)
        for s in STATIONS:
            self.entity(f"stn:{slug(s)}", "station", s, None, aliases=[f"{s} station"])
        for d in DEVELOPERS:
            self.entity(f"dev:{slug(d)}", "developer", d, None)
        for b in BUILDINGS:
            self.entity(f"bld:{slug(b[0])}", "building", b[0], None, aliases=b[1])

    # ------------------------------------------------------------ episodes

    def emit(self, broker: str, day: str, kind: str, text: str, mentions: list[tuple[str, str]],
             facts: list[tuple], contact: str | None = None, speaker: str = "broker",
             thread: str | None = None) -> None:
        """Queue an episode; ids are assigned in chronological order by finalize()."""
        self._pending.setdefault(broker, []).append({
            "broker": broker, "ts": day, "kind": kind, "thread": thread, "contact": contact,
            "speaker": speaker, "text": text, "mentions": [{"surface": s, "entity": e} for s, e in mentions],
            "_facts": facts})

    def finalize(self) -> None:
        for broker, _ in BROKERS:
            eps = sorted(self._pending.get(broker, []), key=lambda e: e["ts"])  # stable: keeps emit order per day
            open_slots: dict[tuple, dict] = {}
            for i, ep in enumerate(eps, start=1):
                ep["id"] = f"{broker}-{i:03d}"
                ep["fact_ids"] = []
                for subj, pred, obj, extra in ep.pop("_facts"):
                    f = {"id": f"f-{broker}-{len(self.facts) + 1:04d}", "broker": broker, "subject": subj,
                         "predicate": pred, "object": obj, "valid_from": ep["ts"], "valid_to": None,
                         "episode": ep["id"], "superseded_by": None, **extra}
                    if pred in SINGLE:
                        prev = open_slots.get((subj, pred))
                        if prev and prev["object"] == obj:
                            prev.setdefault("restated_in", []).append(ep["id"])
                            ep["fact_ids"].append(prev["id"])
                            continue
                        if prev:
                            prev["valid_to"] = ep["ts"]
                            prev["superseded_by"] = ep["id"]
                        open_slots[(subj, pred)] = f
                    self.facts.append(f)
                    ep["fact_ids"].append(f["id"])
                self.episodes.append(ep)

    # ------------------------------------------------------------ the stream

    def generate(self) -> None:
        self.build_catalog()
        for broker, broker_name in BROKERS:
            self.entity(f"brk:{broker}", "broker", broker_name, broker)
            self.broker_stream(broker)
        self.finalize()

    def broker_stream(self, broker: str) -> None:
        rng = self.rng
        # the catalogue this broker works with: both Auroras always (the confounder), plus four others
        others = [b for b in BUILDINGS if not b[0].startswith("Aurora")]
        rng.shuffle(others)
        listed = [b for b in BUILDINGS if b[0].startswith("Aurora")] + sorted(others[:4], key=BUILDINGS.index)
        state = {}  # building id -> {"price", "delivery"}
        day0 = dt.date(2026, 1, 5)
        for i, b in enumerate(listed):
            name, aliases, dev, nbh, stn, units, price, delivery = b
            bid = f"bld:{slug(name)}"
            state[bid] = {"price": price, "delivery": delivery, "units": units, "station": stn, "name": name}
            d1 = (day0 + dt.timedelta(days=2 * i)).isoformat()
            alias_note = " (everyone calls it N1)" if name == "Northbank One" else ""
            unit_txt = " and ".join([", ".join(str(u) for u in units[:-1]), str(units[-1])]) if len(units) > 1 else str(units[0])
            text = rng.choice([
                f"Listing sheet: {name}{alias_note}, a project by {dev}, in {nbh}, a short walk from {stn} station. "
                f"Units with {unit_txt} bedrooms, prices from {money_text(price, rng)}, delivery {month_text(delivery, rng)}.",
                f"New project in the catalogue: {name}{alias_note}. Developer {dev}. Neighborhood {nbh}, next to {stn} "
                f"station. {unit_txt} bedroom units, starting at {money_text(price, rng)}. Keys in {month_text(delivery, rng)}.",
            ])
            m1 = [(name, bid), (dev, f"dev:{slug(dev)}"), (nbh, f"nbh:{slug(nbh)}"), (stn, f"stn:{slug(stn)}")]
            if alias_note:
                m1.append(("N1", bid))
            self.emit(broker, d1, "note", text, m1,
                      [(bid, "developer", f"dev:{slug(dev)}", {}), (bid, "located_in", f"nbh:{slug(nbh)}", {}),
                       (bid, "near_station", f"stn:{slug(stn)}", {})]
                      + [(bid, "unit_bedrooms", u, {}) for u in units]
                      + [(bid, "price", price, {}), (bid, "delivery", delivery, {})])
        self.building_updates(broker, state)
        for ci, spec in enumerate(CLIENTS[broker]):
            self.client_story(broker, ci, spec, state)
        self.small_talk(broker)

    def bsurf(self, bid: str) -> str:
        """How a building is written in a message: its name, or now and then one of its unambiguous aliases."""
        e = self.entities[bid]
        return self.rng.choice(e["aliases"]) if e["aliases"] and self.rng.random() < 0.4 else e["name"]

    def building_updates(self, broker: str, state: dict) -> None:
        rng = self.rng
        bids = sorted(state)
        rng.shuffle(bids)
        # two price changes and one delivery delay per broker, between March and June
        for k, bid in enumerate(bids[:2]):
            b = state[bid]
            new = b["price"] + rng.choice([-1, 1]) * rng.randrange(12, 40) * 1000 + 1000 * (k + 1)
            day = f"2026-{rng.randrange(3, 6):02d}-{rng.randrange(1, 28):02d}"
            short = self.bsurf(bid)
            verb = "went up" if new > b["price"] else "dropped"
            text = rng.choice([f"Heads up: {short} prices {verb}, now from {money_text(new, rng)}.",
                               f"Developer update: {short} new starting price {money_text(new, rng)}."])
            self.emit(broker, day, "note", text, [(short, bid)], [(bid, "price", new, {})])
            b["price"] = new
            b.setdefault("history", []).append(("price", day))
        bid = bids[2]
        b = state[bid]
        y, m = map(int, b["delivery"].split("-"))
        m += rng.choice([3, 4, 6])
        y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
        new = f"{y}-{m:02d}"
        day = f"2026-{rng.randrange(3, 6):02d}-{rng.randrange(1, 28):02d}"
        short = self.bsurf(bid)
        text = rng.choice([f"{short} delivery delayed to {month_text(new, rng)}.",
                           f"Bad news from the developer: {short} keys only in {month_text(new, rng)} now."])
        self.emit(broker, day, "note", text, [(short, bid)], [(bid, "delivery", new, {})])
        b["delivery"] = new

    def client_story(self, broker: str, ci: int, spec: tuple, state: dict) -> None:
        rng = self.rng
        full, contact, shorts = spec
        first = full.split()[0]
        cid = self.entity(f"cli:{broker}:{slug(full)}", "client", full, broker,
                          aliases=[contact] + shorts, contact=contact)
        thread = f"{broker}-t{ci + 1:02d}"
        bids = sorted(state)
        b0 = rng.choice(bids)
        beds = rng.choice(state[b0]["units"])
        # budgets are unique across the world, and never equal to a building price
        while True:
            budget = rng.randrange(300, 1150) * 1000 + rng.choice([5000, 15000, 25000])
            if budget not in self._budgets and budget not in {b[6] for b in BUILDINGS}:
                break
        self._budgets.add(budget)
        nbh = self.entities[f"nbh:{slug(NEIGHBORHOODS[rng.randrange(len(NEIGHBORHOODS))])}"]["name"]
        deadline = rng.choice([f"2026-{rng.randrange(9, 13):02d}", f"2027-{rng.randrange(1, 13):02d}",
                               f"2028-{rng.randrange(1, 13):02d}"])
        day = dt.date(2026, 1, 12) + dt.timedelta(days=ci * 9 + rng.randrange(0, 5))
        d = day.isoformat()

        def chat(day_s, text, mentions, facts, speaker="client"):
            # the thread itself names the client: every chat episode mentions its contact
            self.emit(broker, day_s, "chat", text, mentions + [(contact, cid)], facts, contact=contact,
                      speaker=speaker, thread=thread)

        def note(day_s, text, mentions, facts):
            self.emit(broker, day_s, "note", text, mentions, facts)

        def ref():
            s = rng.choice([full] + shorts)
            return s, [(s, cid)]

        nid = f"nbh:{slug(nbh)}"
        chat(d, rng.choice([
            f"Hi, this is {full}. I'm looking for a {beds}-bedroom apartment in {nbh}, budget around {money_text(budget, rng)}.",
            f"Hello! {full} here. We want {WORDS[beds]} bedroom{'s' if beds > 1 else ''}, ideally in {nbh}. "
            f"Our budget is {money_text(budget, rng)}.",
        ]), [(full, cid), (nbh, nid)],
            [(cid, "bedrooms", beds, {}), (cid, "wants_neighborhood", nid, {}), (cid, "budget", budget, {})])
        b0n = self.bsurf(b0)
        d = add_days(d, rng.randrange(1, 4))
        chat(d, rng.choice([
            f"{b0n} caught my eye. I need to move in by {month_text(deadline, rng)}.",
            f"I liked what I saw about {b0n}. The move has to happen by {month_text(deadline, rng)}.",
        ]), [(b0n, b0)], [(cid, "interested_in", b0, {}), (cid, "deadline", deadline, {})])
        current = b0
        for step in PLANS[ci]:
            d = add_days(d, rng.randrange(14, 36))
            if step == "budget":
                old = budget
                while True:
                    delta = rng.randrange(40, 160) * 1000
                    budget = old + delta if ci != 5 else old - delta
                    if budget not in self._budgets and budget > 250_000:
                        break
                self._budgets.add(budget)
                if rng.random() < 0.5:
                    chat(d, rng.choice([f"Good news, we can go up to {money_text(budget, rng)} now."
                                        if budget > old else
                                        f"We had to cut the budget, {money_text(budget, rng)} max."]),
                         [], [(cid, "budget", budget, {})])
                else:
                    s, m = ref()
                    verb = "raised" if budget > old else "lowered"
                    note(d, f"Note: {s} {verb} the budget to {money_text(budget, rng)}.", m,
                         [(cid, "budget", budget, {})])
            elif step == "bedrooms":
                new = 3 if beds != 3 else 4
                text = rng.choice([f"Actually we need {WORDS[new]} bedrooms now, the family is growing.",
                                   f"Change of plans: {new} bedrooms, not {beds}."])
                chat(d, text, [], [(cid, "bedrooms", new, {})])
                beds = new
            elif step == "switch":
                cands = [b for b in bids if b != current and beds in state[b]["units"]] or [b for b in bids if b != current]
                new = rng.choice(cands)
                on, nn = self.bsurf(current), self.bsurf(new)
                if rng.random() < 0.5:
                    chat(d, f"I've changed my mind, I'd rather go for {nn}. Forget {on.rstrip('.')}.",
                         [(nn, new), (on, current)], [(cid, "interested_in", new, {})])
                else:
                    s, m = ref()
                    note(d, f"Note: {s} dropped {on} and is now focused on {nn}.", m + [(on, current), (nn, new)],
                         [(cid, "interested_in", new, {})])
                current = new
            elif step == "visit":
                cands = [current] + [b for b in bids if b != current]
                target = cands[0] if rng.random() < 0.6 else rng.choice(cands[1:])
                tn = state[target]["name"]
                # "the Aurora" is ambiguous (two Aurora buildings); used only when the client is focused on one
                surface = ("the Aurora" if tn.startswith("Aurora") and target == current and rng.random() < 0.7
                           else self.bsurf(target))
                s, m = ref()
                note(d, rng.choice([f"Visited {surface} with {s} on {human_date(d)}.",
                                    f"Showed {surface} to {s} today."]), m + [(surface, target)],
                     [(cid, "visited", target, {"date": d})])
            elif step == "objection":
                tn = self.bsurf(current)
                obj = rng.choice(OBJECTIONS)
                chat(d, f"About {tn}: honestly {obj[2]}, that worries me.", [(tn, current)],
                     [(cid, "objection", obj[1], {"about": current})])
            elif step == "deadline":
                y, m = map(int, deadline.split("-"))
                m += rng.choice([4, 6, 8])
                y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
                deadline = f"{y}-{m:02d}"
                chat(d, f"We can wait a bit longer, the move can be by {month_text(deadline, rng)}.",
                     [], [(cid, "deadline", deadline, {})])
            if rng.random() < 0.2:  # the broker answers in the same thread, addressing the client by first name
                greet = contact.split()[0] if rng.random() < 0.5 else first
                chat(add_days(d, 1), rng.choice([f"Thanks {greet}, I'll send options this week.",
                                                 f"Noted, {greet}! I'll get back to you soon."]),
                     [(greet, cid)], [], speaker="broker")

    def small_talk(self, broker: str) -> None:
        rng = self.rng
        lines = ["Note: office closed on Friday for the holiday.", "Note: renew the portal subscription.",
                 "Note: printer on the second floor is broken again.", "Note: team lunch moved to Thursday."]
        for i, line in enumerate(rng.sample(lines, 2)):
            self.emit(broker, f"2026-{2 + 2 * i:02d}-{rng.randrange(1, 28):02d}", "note", line, [], [])

    # ------------------------------------------------------------ truth views

    _budgets: set = set()

    def broker_facts(self, broker: str) -> list[dict]:
        return [f for f in self.facts if f["broker"] == broker]

    def value_at(self, broker: str, subj: str, pred: str, day: str | None = None):
        """Value of a single-valued slot as of ``day`` (None = now), from the broker's own episodes."""
        for f in self.broker_facts(broker):
            if f["subject"] == subj and f["predicate"] == pred:
                if day is None and f["valid_to"] is None:
                    return f
                if day is not None and f["valid_from"] <= day and (f["valid_to"] is None or day < f["valid_to"]):
                    return f
        return None

    def history(self, broker: str, subj: str, pred: str) -> list[dict]:
        return [f for f in self.broker_facts(broker) if f["subject"] == subj and f["predicate"] == pred]


def generate(seed: int = 42) -> tuple[World, list[dict]]:
    from .questions import build_questions
    World._budgets = set()
    w = World(seed)
    w.generate()
    return w, build_questions(w, random.Random(seed + 1))


def write(out_dir: str | Path, seed: int = 42) -> dict:
    w, questions = generate(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def dump(name, rows):
        (out / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    dump("episodes.jsonl", w.episodes)
    dump("facts.jsonl", w.facts)
    dump("entities.jsonl", sorted(w.entities.values(), key=lambda e: e["id"]))
    dump("questions.jsonl", questions)
    counts = {"episodes": len(w.episodes), "facts": len(w.facts), "entities": len(w.entities),
              "questions": len(questions)}
    return counts
