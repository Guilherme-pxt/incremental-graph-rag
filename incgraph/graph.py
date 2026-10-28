"""A per-broker knowledge graph in plain dicts, with or without time.

Nodes are entities (client, building, developer, neighborhood, station).
Edges are facts: ``src --pred--> dst`` when the object is an entity, or
``src --pred--> value`` when it is a literal (an amount, a month, a count, a
short text). Every edge keeps the ids of the episodes that stated it.

With ``temporal=True`` (the incremental graph) an edge also has
``valid_from`` and ``valid_to``. For a single-valued predicate (a budget, the
building a client is focused on, a price), a new value closes the open edge
(``valid_to`` = the new episode's date, ``invalidated_by`` = its id) instead
of overwriting it, so the graph can answer both "now" and "as of". With
``temporal=False`` (the batch graph) every distinct fact is kept side by side
with the dates it was stated, and nothing is ever closed.

There is one graph per broker and no structure is shared between graphs, so
a question asked to one broker's graph cannot reach another broker's data.
"""

from __future__ import annotations

import json

from .synth import ENTITY_OBJECTS, MONTHS, MULTI, SINGLE
from .values import norm_name


def fmt_value(pred: str, v) -> str:
    if pred in ("budget", "price"):
        return f"${int(v):,}"
    if pred in ("deadline", "delivery"):
        y, m = str(v).split("-")
        return f"{MONTHS[int(m) - 1]} {y}"
    return str(v)


class Graph:
    def __init__(self, broker: str, temporal: bool):
        self.broker = broker
        self.temporal = temporal
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.name_index: dict[tuple[str, str], set[str]] = {}  # (type, normalized name or alias) -> node ids
        self.thread_index: dict[str, str] = {}                 # chat thread -> client node
        self.by_node: dict[str, list[int]] = {}                # node id -> indexes of incident edges

    # ------------------------------------------------------------ nodes

    def add_node(self, etype: str, name: str, episode: str, thread: str | None = None) -> str:
        nid = f"{self.broker}:n{len(self.nodes) + 1:03d}"
        self.nodes[nid] = {"id": nid, "type": etype, "name": name, "aliases": [], "thread": thread,
                           "mentions": [], "created_in": episode}
        self._index(nid, name)
        if thread:
            self.thread_index[thread] = nid
        return nid

    def _index(self, nid: str, name: str) -> None:
        self.name_index.setdefault((self.nodes[nid]["type"], norm_name(name)), set()).add(nid)

    def add_alias(self, nid: str, alias: str) -> None:
        node = self.nodes[nid]
        if alias and alias != node["name"] and alias not in node["aliases"]:
            node["aliases"].append(alias)
            self._index(nid, alias)

    def names(self, nid: str) -> list[str]:
        node = self.nodes[nid]
        return [node["name"]] + node["aliases"]

    def lookup(self, etype: str, name: str) -> set[str]:
        return set(self.name_index.get((etype, norm_name(name)), set()))

    def of_type(self, etype: str) -> list[str]:
        return [nid for nid, n in self.nodes.items() if n["type"] == etype]

    # ------------------------------------------------------------ edges

    def _add_edge(self, src: str, pred: str, obj, episode: str, ts: str) -> dict:
        is_ent = pred in ENTITY_OBJECTS
        e = {"id": len(self.edges), "src": src, "pred": pred, "dst": obj if is_ent else None,
             "value": None if is_ent else obj, "valid_from": ts, "valid_to": None, "invalidated_by": None,
             "episodes": [episode], "dates": [ts]}
        self.edges.append(e)
        self.by_node.setdefault(src, []).append(e["id"])
        if is_ent:
            self.by_node.setdefault(obj, []).append(e["id"])
        return e

    @staticmethod
    def obj(e: dict):
        return e["dst"] if e["dst"] is not None else e["value"]

    def edges_of(self, nid: str) -> list[dict]:
        return [self.edges[i] for i in self.by_node.get(nid, [])]

    def open_edges(self, src: str, pred: str) -> list[dict]:
        return [e for e in self.edges_of(src) if e["src"] == src and e["pred"] == pred and e["valid_to"] is None]

    def assert_fact(self, src: str, pred: str, obj, episode: str, ts: str) -> str:
        """Record that ``episode`` states ``src pred obj``. Returns what happened:
        'restated', 'added', or 'replaced' (an open value of a single-valued predicate was closed)."""
        same = [e for e in self.edges_of(src) if e["src"] == src and e["pred"] == pred and self.obj(e) == obj
                and (e["valid_to"] is None or not self.temporal)]
        if same:
            e = same[0]
            if episode not in e["episodes"]:
                e["episodes"].append(episode)
                e["dates"].append(ts)
            return "restated"
        action = "added"
        if self.temporal and pred in SINGLE:
            for e in self.open_edges(src, pred):
                e["valid_to"] = ts
                e["invalidated_by"] = episode
                action = "replaced"
        self._add_edge(src, pred, obj, episode, ts)
        return action

    # ------------------------------------------------------------ text

    def obj_text(self, e: dict) -> str:
        return self.nodes[e["dst"]]["name"] if e["dst"] is not None else fmt_value(e["pred"], e["value"])

    def fact_text(self, e: dict) -> str:
        cites = "".join(f"[{x}]" for x in e["episodes"][:3])
        base = f"{cites} {self.nodes[e['src']]['name']} | {e['pred']} | {self.obj_text(e)}"
        if not self.temporal or e["pred"] in MULTI:  # events and set members have dates, not validity
            return f"{base} | stated {', '.join(sorted(set(e['dates'])))}"
        if e["valid_to"] is None:
            return f"{base} | current, since {e['valid_from']}"
        return f"{base} | superseded: held from {e['valid_from']} to {e['valid_to']}"

    def summary_row(self, nid: str) -> str:
        """One line per entity for the type tables: current values (temporal) or every value with its dates."""
        node = self.nodes[nid]
        parts: dict[str, list[str]] = {}
        for e in self.edges_of(nid):
            if e["src"] != nid or (self.temporal and e["valid_to"] is not None):
                continue
            v = self.obj_text(e)
            if not self.temporal:
                v += f" ({e['dates'][0]})"
            parts.setdefault(e["pred"], []).append(v + "".join(f" [{x}]" for x in e["episodes"][:1]))
        body = "; ".join(f"{p} {', '.join(vs)}" for p, vs in sorted(parts.items()))
        return f"{node['name']}: {body}" if body else f"{node['name']}: (no facts)"

    # ------------------------------------------------------------ persistence

    def to_json(self) -> dict:
        return {"broker": self.broker, "temporal": self.temporal, "nodes": list(self.nodes.values()),
                "edges": self.edges}

    def dump(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_json(), fh, indent=1, sort_keys=True)
