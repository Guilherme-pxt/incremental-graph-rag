"""System 3: the incremental temporal graph. Each episode is read once and folded into the graph.

ingest(episode):
    1. extract facts from the episode (one LLM call, the earlier messages of the chat as context)
    2. resolve every name to a node (``resolve.Resolver``: thread, exact, rules, and the LLM only if ambiguous)
    3. assert every fact; a new value of a single-valued predicate closes the open edge instead of
       overwriting it, so nothing the broker was ever told is lost

The cost of absorbing an episode does not depend on how many came before it,
apart from the resolver's candidate lists, which grow with the number of
entities, not episodes.
"""

from __future__ import annotations

import time

from .extract import extract, thread_context
from .graph import Graph
from .resolve import Resolver
from .synth import ENTITY_OBJECTS


class IncrementalGraph:
    def __init__(self, broker: str, broker_name: str, llm, episodes: list[dict], use_llm_merge: bool = True,
                 tag: str = "inc"):
        self.broker = broker
        self.broker_name = broker_name
        self.llm = llm
        self.episodes = [e for e in episodes if e["broker"] == broker]  # the broker's own stream, for chat context
        self.graph = Graph(broker, temporal=True)
        self.resolver = Resolver(self.graph, llm, broker_name, use_llm=use_llm_merge)
        self.stats: list[dict] = []
        self.tag = tag
        self.extractions: dict[str, dict] = {}

    def ingest(self, ep: dict) -> dict:
        if ep["broker"] != self.broker:
            raise ValueError(f"episode {ep['id']} belongs to broker {ep['broker']}, not {self.broker}")
        n_calls = len(self.llm.ledger)
        ctx = thread_context(self.episodes, ep)
        out = extract(self.llm, ep, ctx, self.broker_name, tag=f"{self.tag}:extract")
        self.extractions[ep["id"]] = out
        t0 = time.perf_counter()
        types = {e["name"]: e["type"] for e in out["entities"]}
        ids: dict[tuple[str, str], str] = {}

        def node(name: str, etype: str) -> str:
            if (name, etype) not in ids:
                ids[(name, etype)] = self.resolver.resolve(name, etype, ep, ctx)
            return ids[(name, etype)]

        actions = []
        for f in out["facts"]:
            src = node(f["subject"], types.get(f["subject"]) or "client")
            obj = f["object"]
            if f["predicate"] in ENTITY_OBJECTS:
                obj = node(obj, ENTITY_OBJECTS[f["predicate"]])
            actions.append(self.graph.assert_fact(src, f["predicate"], obj, ep["id"], ep["ts"]))
        for e in out["entities"]:
            if e["aliases"] and (e["name"], e["type"]) in ids:
                for a in e["aliases"]:
                    self.graph.add_alias(ids[(e["name"], e["type"])], a)
        calls = self.llm.ledger[n_calls:]
        row = {"episode": ep["id"], "python_s": time.perf_counter() - t0, "facts": len(out["facts"]),
               "actions": {a: actions.count(a) for a in sorted(set(actions))},
               "llm_calls": len(calls), "merge_calls": sum(1 for c in calls if c["tag"].endswith(":merge")),
               "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
               "completion_tokens": sum(c["completion_tokens"] for c in calls),
               "llm_s": sum(c["latency_s"] for c in calls)}
        self.stats.append(row)
        return row
