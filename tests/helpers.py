"""Shared fixtures: one generated world per test process, a config, and scripted LLMs that never touch Ollama."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from incgraph.graph import fmt_value
from incgraph.llm import LLM, ChatResult, ScriptedProvider
from incgraph.pipeline import load_data
from incgraph.synth import write

_DIR = None


def data_dir() -> Path:
    global _DIR
    if _DIR is None:
        _DIR = tempfile.TemporaryDirectory()
        write(_DIR.name, seed=42)
    return Path(_DIR.name)


def config(tmp: Path) -> dict:
    return {"run": {"seed": 42, "data_dir": str(data_dir()), "results_dir": str(tmp / "results"),
                    "cache": str(tmp / "cache.sqlite")},
            "llm": {"base_url": "http://localhost:9", "model": "scripted", "timeout_s": 1},
            "hybrid": {"top_k": 8, "candidates": 30, "rrf_k": 60, "lsa_dims": 16},
            "graph_retrieval": {"max_lines": 70, "hop2": 25, "search": 10, "max_reports": 3},
            "batch": {"min_community": 3, "summary_tokens": 120}, "answer": {"max_tokens": 60},
            "cost": {"new_fraction": 0.25}}


def oracle_lines(data: dict, episode_id: str) -> str:
    """What a perfect extractor would write for an episode, using the surface forms the episode used."""
    ep = next(e for e in data["episodes"] if e["id"] == episode_id)
    facts = {f["id"]: f for f in data["facts"]}
    surf = {}
    for m in ep["mentions"]:
        surf.setdefault(m["entity"], m["surface"])
    if ep["kind"] == "chat":
        for m in ep["mentions"]:
            if m["entity"].startswith("cli:") and m["surface"] == ep["contact"]:
                surf.setdefault(m["entity"], ep["contact"])
    name = lambda eid: surf.get(eid) or data["entities"][eid]["name"]
    lines = []
    for fid in ep["fact_ids"]:
        f = facts[fid]
        obj = f["object"]
        if isinstance(obj, str) and ":" in obj and obj.split(":")[0] in ("bld", "dev", "nbh", "stn", "cli"):
            o = name(obj)
        elif f["predicate"] in ("budget", "price", "deadline", "delivery"):
            o = fmt_value(f["predicate"], obj)
        else:
            o = str(obj)
        lines.append(f"{name(f['subject'])} | {f['predicate']} | {o}")
    if "everyone calls it N1" in ep["text"]:
        lines.append("Northbank One | alias | N1")
    return "\n".join(lines) or "NONE"


def oracle_llm(answer: str = "ANSWER: I don't know", cache: Path | None = None) -> tuple[LLM, ScriptedProvider]:
    data = load_data(data_dir())

    def fn(messages):
        text = messages[-1]["content"]
        if "NEW message:" in text:
            ep_id = re.search(r"NEW message:\n\[([abc]-\d{3})\]", text).group(1)
            return oracle_lines(data, ep_id)
        if "Reply with the number only" in text:
            return ChatResult(text="1", latency_s=0.01, top_logprobs=[{"1": -0.1, "0": -2.5}],
                              prompt_tokens=100, completion_tokens=1)
        if "Write a report" in text:
            return "A short community report."
        return answer

    provider = ScriptedProvider(fn)
    return LLM(provider, cache), provider
