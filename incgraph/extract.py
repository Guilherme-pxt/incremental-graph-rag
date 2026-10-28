"""Per-episode fact extraction with the local LLM, and normalization of what it returns.

One call per episode, a fixed schema of thirteen predicates over five entity
types, and one ``subject | predicate | object`` line per fact. Lines instead of
JSON because generation is the slow part on a shared 7B model: the same facts
cost about a fifth of the output tokens. The prompt shows up to two earlier episodes of the same
chat as context (so "I", "we" and "it" can be resolved) and asks for facts
from the new episode only.

Both graph systems use exactly this call for every episode. The batch graph
therefore pays the same extraction cost per episode as the incremental one;
what differs is what each does with the output (merging, time, communities)
and how often it has to pay again.
"""

from __future__ import annotations

import json

from .synth import ENTITY_OBJECTS, PREDICATES
from .values import parse_int, parse_money, parse_month

ENTITY_TYPES = ["client", "building", "developer", "neighborhood", "station"]
SUBJECT_TYPE = {p: ("building" if p in {"developer", "located_in", "near_station", "unit_bedrooms", "price",
                                         "delivery"} else "client") for p in PREDICATES}

SYSTEM = "You extract facts from a real-estate broker's messages into a knowledge graph."

SCHEMA = """Write one fact per line as: subject | predicate | object
Predicates for a client (subject is the client's name):
  budget (the most the client can pay, in dollars), bedrooms (number the client wants),
  wants_neighborhood, deadline (month the client must move by), interested_in (the building the client is
  focused on now), visited (a building the client visited or was shown), objection (a few words on what worries the client)
Predicates for a building (subject is the building's name):
  developer, located_in (neighborhood), near_station (metro station), unit_bedrooms (one line per bedroom count),
  price (starting price, in dollars), delivery (delivery month)
If the message gives another name for an entity, add: name | alias | other name
Rules:
- Only facts the NEW message states. Earlier messages are context, to know who "I", "we" or "it" is.
- In a chat, "I" and "we" are the client of the chat header. Use the client's full name if the message gives it, otherwise the header name.
- Write names as they appear. Never invent or complete a surname.
- Greetings, thanks and office notes have no facts: write NONE.

Example. Chat with the client saved as "Jo Smith", written by the client: We can stretch to 450k now, and we'd love the Elm.
Jo Smith | budget | 450000
Jo Smith | interested_in | the Elm

Example. Note written by the broker: Elm Tower (also called ET) by Acme Homes, in Riverbend: 1 and 2 bedroom units from $300,000, keys in May 2027.
Elm Tower | alias | ET
Elm Tower | developer | Acme Homes
Elm Tower | located_in | Riverbend
Elm Tower | unit_bedrooms | 1
Elm Tower | unit_bedrooms | 2
Elm Tower | price | 300000
Elm Tower | delivery | 2027-05

Example. Note written by the broker: Showed Oak Plaza to Ana Lee; she now prefers it to Elm Tower and needs 3 bedrooms.
Ana Lee | visited | Oak Plaza
Ana Lee | interested_in | Oak Plaza
Ana Lee | bedrooms | 3

Example. Chat with the client saved as "Jo Smith", written by the broker: Thanks Jo, I'll send options soon.
NONE"""


def episode_header(ep: dict, broker_name: str) -> str:
    if ep["kind"] == "chat":
        who = "the client" if ep["speaker"] == "client" else f"the broker ({broker_name})"
        return f'[{ep["id"]}] {ep["ts"]}, WhatsApp chat with the client saved as "{ep["contact"]}"; written by {who}'
    return f'[{ep["id"]}] {ep["ts"]}, note written by the broker ({broker_name})'


def episode_text(ep: dict, broker_name: str) -> str:
    return f'{episode_header(ep, broker_name)}:\n{ep["text"]}'


def extraction_messages(ep: dict, context: list[dict], broker_name: str) -> list[dict]:
    parts = [SCHEMA, ""]
    if context:
        parts.append("Earlier messages in this chat (context only):")
        parts.extend(episode_text(c, broker_name) for c in context)
        parts.append("")
    parts.append("NEW message:")
    parts.append(episode_text(ep, broker_name))
    parts.append("\nFacts:")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


_PRED_ALIASES = {"price_from": "price", "starting_price": "price", "neighborhood": "located_in",
                 "station": "near_station", "wants_bedrooms": "bedrooms", "delivery_date": "delivery",
                 "interested": "interested_in", "location": "located_in"}


def parse_lines(text: str) -> dict:
    """``subject | predicate | object`` lines into the raw dict ``clean`` expects; anything else is ignored."""
    facts, aliases = [], {}
    for line in text.splitlines():
        parts = [p.strip().strip("`*-").strip() for p in line.split("|")]
        if len(parts) != 3 or not all(parts):
            continue
        subj, pred, obj = parts
        pred = pred.lower().replace(" ", "_")
        pred = pred.split("_", 1)[1] if pred.split("_", 1)[0] in ("client", "building") and "_" in pred else pred
        pred = _PRED_ALIASES.get(pred, pred)
        if pred == "alias":
            aliases.setdefault(subj, []).append(obj)
        else:
            facts.append({"subject": subj, "predicate": pred, "object": obj})
    ents = [{"name": k, "type": "", "aliases": v} for k, v in aliases.items()]
    return {"entities": ents, "facts": facts}


def normalize_object(pred: str, obj):
    """Literal predicates become int / 'YYYY-MM' / short text; entity predicates stay names. None = unusable."""
    if pred in ("budget", "price"):
        return parse_money(obj)
    if pred in ("bedrooms", "unit_bedrooms"):
        v = parse_int(obj)
        return v if v is not None and 0 < v < 10 else None
    if pred in ("deadline", "delivery"):
        return parse_month(obj)
    if isinstance(obj, (dict, list)) or obj is None:
        return None
    s = str(obj).strip()
    if pred == "objection":
        return s.lower()[:80] or None
    return s or None


def clean(raw: dict) -> dict:
    """Validate the LLM output: known predicates, usable values, one entity record per name."""
    ents: dict[str, dict] = {}
    for e in raw.get("entities") or []:
        if not isinstance(e, dict) or not str(e.get("name") or "").strip():
            continue
        name = str(e["name"]).strip()
        etype = str(e.get("type") or "").lower().strip()
        aliases = [str(a).strip() for a in (e.get("aliases") or []) if isinstance(a, (str, int)) and str(a).strip()]
        ents.setdefault(name, {"name": name, "type": etype if etype in ENTITY_TYPES else "", "aliases": []})
        ents[name]["aliases"] = sorted(set(ents[name]["aliases"]) | set(aliases) - {name})
    facts = []
    for f in raw.get("facts") or []:
        if not isinstance(f, dict):
            continue
        pred = str(f.get("predicate") or "").strip().lower().replace(" ", "_")
        subj = str(f.get("subject") or "").strip()
        if pred not in PREDICATES or not subj:
            continue
        obj = normalize_object(pred, f.get("object"))
        if obj is None:
            continue
        facts.append({"subject": subj, "predicate": pred, "object": obj})
        # every name a fact uses is an entity, typed by the predicate when the LLM did not list it
        ents.setdefault(subj, {"name": subj, "type": SUBJECT_TYPE[pred], "aliases": []})
        if not ents[subj]["type"]:
            ents[subj]["type"] = SUBJECT_TYPE[pred]
        if pred in ENTITY_OBJECTS:
            ents.setdefault(obj, {"name": obj, "type": ENTITY_OBJECTS[pred], "aliases": []})
            if not ents[obj]["type"]:
                ents[obj]["type"] = ENTITY_OBJECTS[pred]
    seen, uniq = set(), []
    for f in facts:
        k = (f["subject"], f["predicate"], json.dumps(f["object"]))
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return {"entities": [e for e in ents.values() if e["type"]], "facts": uniq}


def thread_context(episodes: list[dict], ep: dict, n: int = 2) -> list[dict]:
    """The ``n`` episodes before ``ep`` in the same chat thread (notes have no thread, so no context)."""
    if not ep.get("thread"):
        return []
    before = [e for e in episodes if e.get("thread") == ep["thread"] and e["id"] < ep["id"]]
    return before[-n:]


def extract(llm, ep: dict, context: list[dict], broker_name: str, max_tokens: int = 200,
            tag: str = "inc:extract") -> dict:
    res = llm.chat(extraction_messages(ep, context, broker_name), max_tokens=max_tokens, tag=tag, ref=ep["id"])
    out = clean(parse_lines(res.text))
    out["raw"] = res.text
    return out
