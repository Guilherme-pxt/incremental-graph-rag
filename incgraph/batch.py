"""System 2: batch GraphRAG. Extract from every episode, merge by name, find communities, summarize them.

This follows the indexing recipe of Microsoft's GraphRAG (Edge et al., "From
Local to Global: A Graph RAG Approach to Query-Focused Summarization", 2024),
scaled down:

1. extraction per text unit, here one episode (the same LLM call the
   incremental graph makes, so the two graphs start from identical extractions);
2. entities merged when their normalized name and type are equal, as GraphRAG
   merges entities by title; no aliases, no chat-thread key, no LLM merges;
3. every distinct fact kept with the dates it was stated; nothing is closed,
   because a batch index has no notion of a fact ending;
4. communities by Louvain modularity (GraphRAG uses Leiden; at 40 nodes the
   two give the same kind of partition), and one LLM report per community of
   three or more entities.

The graph is a function of the whole corpus, so absorbing new episodes means
building it again: re-extracting everything (or replaying an extraction cache)
and, in any case, recomputing communities and their reports.
"""

from __future__ import annotations

import time

from .extract import extract, thread_context
from .graph import Graph
from .synth import ENTITY_OBJECTS
from .values import norm_name


def louvain(nodes: list[str], edges: list[tuple[str, str, float]]) -> dict[str, int]:
    """Deterministic Louvain (Blondel et al. 2008): local moves, then aggregation, until nothing moves."""
    adj: dict[str, dict[str, float]] = {n: {} for n in nodes}
    for a, b, w in edges:
        if a == b:
            continue
        adj[a][b] = adj[a].get(b, 0.0) + w
        adj[b][a] = adj[b].get(a, 0.0) + w
    membership = {n: n for n in nodes}  # original node -> current super-node
    level_nodes = sorted(nodes)
    level_adj = adj
    while True:
        m2 = sum(sum(nb.values()) for nb in level_adj.values())  # 2m
        if m2 == 0:
            break
        comm = {n: n for n in level_nodes}
        k = {n: sum(level_adj[n].values()) for n in level_nodes}
        tot = dict(k)
        moved_any = False
        improved = True
        while improved:
            improved = False
            for n in level_nodes:
                c0 = comm[n]
                links: dict[str, float] = {}
                for nb, w in level_adj[n].items():
                    if nb != n:  # a super-node's self-loop is its internal weight: part of k, not a link
                        links[comm[nb]] = links.get(comm[nb], 0.0) + w
                tot[c0] -= k[n]
                best, best_gain = c0, links.get(c0, 0.0) - tot[c0] * k[n] / m2
                for c in sorted(links):
                    gain = links[c] - tot[c] * k[n] / m2
                    if gain > best_gain + 1e-12:
                        best, best_gain = c, gain
                tot[best] += k[n]
                if best != c0:
                    comm[n] = best
                    improved = moved_any = True
        if not moved_any:
            break
        membership = {o: comm[s] for o, s in membership.items()}
        new_nodes = sorted(set(comm.values()))
        new_adj: dict[str, dict[str, float]] = {c: {} for c in new_nodes}
        for a, nbs in level_adj.items():
            for b, w in nbs.items():
                ca, cb = comm[a], comm[b]
                new_adj[ca][cb] = new_adj[ca].get(cb, 0.0) + w  # ca == cb keeps the internal weight
        level_nodes, level_adj = new_nodes, new_adj
    labels = sorted(set(membership.values()))
    index = {c: i for i, c in enumerate(labels)}
    return {n: index[membership[n]] for n in nodes}


SUMMARY_SYSTEM = "You write short reports about part of a real-estate broker's knowledge graph."


def summary_messages(graph: Graph, members: list[str]) -> list[dict]:
    lines = [graph.fact_text(e) for e in graph.edges if e["src"] in members]
    names = ", ".join(graph.nodes[n]["name"] for n in members)
    body = "\n".join(lines[:60])
    return [{"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": f"Entities: {names}\nFacts (with the dates they were stated):\n{body}\n\n"
                                        "Write a report of at most 4 sentences on these entities and how they "
                                        "relate. When a value was stated more than once with different values, "
                                        "give the most recent one and its date."}]


class BatchGraph:
    def __init__(self, broker: str, broker_name: str, llm, min_community: int = 3, summary_tokens: int = 120):
        self.broker = broker
        self.broker_name = broker_name
        self.llm = llm
        self.min_community = min_community
        self.summary_tokens = summary_tokens

    def build(self, episodes: list[dict], summaries: bool = True) -> dict:
        """Index ``episodes`` (this broker's) from scratch. Returns the graph, communities, reports and cost."""
        eps = [e for e in episodes if e["broker"] == self.broker]
        n_calls = len(self.llm.ledger)
        g = Graph(self.broker, temporal=False)
        keys: dict[tuple[str, str], str] = {}
        python_s = 0.0
        for ep in eps:
            out = extract(self.llm, ep, thread_context(eps, ep), self.broker_name, tag="batch:extract")
            t0 = time.perf_counter()
            types = {e["name"]: e["type"] for e in out["entities"]}

            def node(name: str, etype: str) -> str:
                key = (etype, norm_name(name))
                if key not in keys:
                    keys[key] = g.add_node(etype, name, ep["id"])
                g.nodes[keys[key]]["mentions"].append([ep["id"], name])
                return keys[key]

            for f in out["facts"]:
                src = node(f["subject"], types.get(f["subject"]) or "client")
                obj = node(f["object"], ENTITY_OBJECTS[f["predicate"]]) if f["predicate"] in ENTITY_OBJECTS \
                    else f["object"]
                g.assert_fact(src, f["predicate"], obj, ep["id"], ep["ts"])
            python_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        weights: dict[tuple[str, str], float] = {}
        for e in g.edges:
            if e["dst"] is not None:
                key = tuple(sorted((e["src"], e["dst"])))
                weights[key] = weights.get(key, 0.0) + 1.0
        part = louvain(sorted(g.nodes), [(a, b, w) for (a, b), w in sorted(weights.items())])
        communities: dict[int, list[str]] = {}
        for nid, c in sorted(part.items()):
            communities.setdefault(c, []).append(nid)
        python_s += time.perf_counter() - t0
        reports: dict[int, str] = {}
        if summaries:
            for c, members in sorted(communities.items()):
                if len(members) >= self.min_community:
                    res = self.llm.chat(summary_messages(g, members), max_tokens=self.summary_tokens,
                                        tag="batch:summary", ref=f"{self.broker}:c{c}")
                    reports[c] = res.text.strip()
        calls = self.llm.ledger[n_calls:]
        cost = {"episodes": len(eps), "llm_calls": len(calls),
                "extract_calls": sum(1 for x in calls if x["tag"] == "batch:extract"),
                "summary_calls": sum(1 for x in calls if x["tag"] == "batch:summary"),
                "prompt_tokens": sum(x["prompt_tokens"] for x in calls),
                "completion_tokens": sum(x["completion_tokens"] for x in calls),
                "llm_s": sum(x["latency_s"] for x in calls),
                "summary_llm_s": sum(x["latency_s"] for x in calls if x["tag"] == "batch:summary"),
                "summary_tokens": sum(x["prompt_tokens"] + x["completion_tokens"] for x in calls
                                      if x["tag"] == "batch:summary"),
                "python_s": python_s}
        return {"graph": g, "communities": communities, "reports": reports, "cost": cost,
                "membership": part}
