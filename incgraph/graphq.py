"""Graph quality against the generator's ground truth.

**Mapping nodes to true entities.** Every node remembers each (episode, name)
that was resolved to it. The generator recorded, per episode, which true
entity each surface form refers to. A mention maps to the true entity of the
same type mentioned in that episode whose recorded surface form is closest to
the extracted name; when the episode has no entity of that type (the
extractor pulled a name from the chat context), the name is looked up among
all of the broker's true names and aliases. A node's entity is the majority
over its mentions.

* duplication rate  extra nodes per true entity: (nodes mapped to entities -
                    distinct entities covered) / distinct entities covered
* wrong-merge rate  share of nodes whose mentions map to two or more true entities
* relation precision / recall  graph facts (mapped subject, predicate, mapped
                    object or value) against every true fact the broker's
                    episodes stated, history included; objections match when
                    the true keyword is in the extracted words
* current-value accuracy  share of true single-valued slots (a client's
                    budget, a building's price, ...) for which the graph's open
                    values are exactly the true current value
* closed correctly  share of superseded true facts that the graph holds as a
                    closed edge, and with the exact closing date
"""

from __future__ import annotations

from collections import Counter

from .synth import SINGLE
from .values import norm_name


def _overlap(a: str, b: str) -> float:
    x, y = set(norm_name(a).split()), set(norm_name(b).split())
    return len(x & y) / max(1, len(x | y))


class Truth:
    def __init__(self, broker: str, episodes: list[dict], facts: list[dict], entities: dict[str, dict]):
        self.broker = broker
        self.episodes = {e["id"]: e for e in episodes if e["broker"] == broker}
        self.facts = [f for f in facts if f["broker"] == broker]
        self.entities = entities
        self.names: dict[str, set[str]] = {}
        for e in entities.values():
            if e["broker"] in (None, broker):
                for n in [e["name"], *e["aliases"], e.get("contact") or ""]:
                    if n:
                        self.names.setdefault(norm_name(n), set()).add(e["id"])

    def map_mention(self, episode: str, surface: str, etype: str) -> str | None:
        prefix = {"client": "cli:", "building": "bld:", "developer": "dev:", "neighborhood": "nbh:",
                  "station": "stn:"}[etype]
        ep = self.episodes.get(episode)
        cands = [(m["surface"], m["entity"]) for m in (ep["mentions"] if ep else []) if m["entity"].startswith(prefix)]
        exact = {eid for s, eid in cands if norm_name(s) == norm_name(surface)}
        if len(exact) == 1:
            return exact.pop()
        ids = sorted({eid for _, eid in cands})
        if len(ids) == 1:
            return ids[0]
        if ids:
            scored = sorted(((max(_overlap(surface, s) for s, e in cands if e == eid), eid) for eid in ids),
                            reverse=True)
            if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                return scored[0][1]
        glob = {x for x in self.names.get(norm_name(surface), set()) if x.startswith(prefix)}
        return glob.pop() if len(glob) == 1 else None

    def gold_triples(self) -> set[tuple]:
        return {(f["subject"], f["predicate"], f["object"]) for f in self.facts}


def evaluate_graph(graph, truth: Truth) -> dict:
    node_gold: dict[str, str | None] = {}
    impure = 0
    mapped_nodes = 0
    for nid, node in graph.nodes.items():
        ms = [truth.map_mention(ep, s, node["type"]) for ep, s in node["mentions"]]
        ms = [m for m in ms if m]
        if not ms:
            node_gold[nid] = None
            continue
        mapped_nodes += 1
        c = Counter(ms)
        node_gold[nid] = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        if len(c) > 1:
            impure += 1
    covered = Counter(g for g in node_gold.values() if g)
    mentioned = {m["entity"] for e in truth.episodes.values() for m in e["mentions"]}
    by_type: dict[str, dict] = {}
    for etype, prefix in [("client", "cli:"), ("building", "bld:"), ("place", ("dev:", "nbh:", "stn:"))]:
        ents = {e for e in mentioned if e.startswith(prefix)}
        nodes = [n for n, gld in node_gold.items() if gld and gld.startswith(prefix)]
        cov = {node_gold[n] for n in nodes}
        by_type[etype] = {"true_entities": len(ents), "nodes": len(nodes),
                          "duplication_rate": (len(nodes) - len(cov)) / max(1, len(cov)),
                          "entities_split": sum(1 for e in cov if covered[e] > 1),
                          "entity_recall": len(cov & ents) / max(1, len(ents))}

    def triple(e):
        s = node_gold.get(e["src"])
        o = node_gold.get(e["dst"]) if e["dst"] is not None else e["value"]
        return (s, e["pred"], o)

    gold = truth.gold_triples()
    gold_obj = {(s, o) for s, p, o in gold if p == "objection"}
    pred_triples = set()
    hits = 0
    for e in graph.edges:
        t = triple(e)
        pred_triples.add(t)
    for s, p, o in pred_triples:
        if p == "objection":
            ok = any(gs == s and isinstance(o, str) and go in o for gs, go in gold_obj)
        else:
            ok = (s, p, o) in gold
        hits += ok
    found = 0
    for s, p, o in gold:
        if p == "objection":
            found += any(ps == s and pp == p and isinstance(po, str) and o in po for ps, pp, po in pred_triples)
        else:
            found += (s, p, o) in pred_triples

    # temporal: the current value of every true single-valued slot, and the superseded facts
    slots: dict[tuple, list[dict]] = {}
    for f in truth.facts:
        if f["predicate"] in SINGLE:
            slots.setdefault((f["subject"], f["predicate"]), []).append(f)
    cur_ok = latest_ok = 0
    for (s, p), fs in slots.items():
        current = [f for f in fs if f["valid_to"] is None][0]["object"]
        edges = [e for e in graph.edges if node_gold.get(e["src"]) == s and e["pred"] == p]
        open_vals = {triple(e)[2] for e in edges if e["valid_to"] is None}
        cur_ok += open_vals == {current}
        if edges:  # what a reader following dates would conclude: the most recently stated value
            last = max(edges, key=lambda e: (max(e["dates"]), e["id"]))
            latest_ok += triple(last)[2] == current
    superseded = [f for f in truth.facts if f["valid_to"] is not None]
    closed = closed_date = 0
    for f in superseded:
        es = [e for e in graph.edges if node_gold.get(e["src"]) == f["subject"] and e["pred"] == f["predicate"]
              and triple(e)[2] == f["object"]]
        if any(e["valid_to"] is not None for e in es):
            closed += 1
            closed_date += any(e["valid_to"] == f["valid_to"] for e in es)
    multi_open = sum(1 for (s, p) in slots
                     if len({triple(e)[2] for e in graph.edges if node_gold.get(e["src"]) == s and e["pred"] == p
                             and e["valid_to"] is None}) > 1)
    return {"nodes": len(graph.nodes), "edges": len(graph.edges), "unmapped_nodes": len(graph.nodes) - mapped_nodes,
            "wrong_merge_rate": impure / max(1, mapped_nodes), "wrong_merges": impure,
            "duplication_rate": (sum(covered.values()) - len(covered)) / max(1, len(covered)),
            "by_type": by_type,
            "relation_precision": hits / max(1, len(pred_triples)), "relation_recall": found / max(1, len(gold)),
            "relation_counts": {"graph": len(pred_triples), "true": len(gold), "correct": hits, "found": found},
            "slots": len(slots), "current_value_accuracy": cur_ok / max(1, len(slots)),
            "latest_stated_accuracy": latest_ok / max(1, len(slots)),
            "slots_with_conflicting_open_values": multi_open,
            "superseded_facts": len(superseded), "closed": closed, "closed_with_exact_date": closed_date}
