"""Run everything once and derive every table: ingestion, batch build, answers, metrics, cost.

Order matters only for which run pays the model: the incremental graph reads
the stream first, so its extraction calls are the uncached ones, and the batch
build replays the identical calls from the cache. The ledger keeps the latency
and token counts measured when each call was actually made, so both systems
are charged the same real cost per extraction.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import tomllib
from pathlib import Path

import requests

from .answer import answer_messages
from .batch import BatchGraph
from .graphq import Truth, evaluate_graph
from .incremental import IncrementalGraph
from .questions import TYPES
from .retrieve import GraphRetriever, HybridIndex
from .scoring import bootstrap_ci, paired_bootstrap, score
from .synth import BROKERS

SYSTEMS = ["hybrid", "batch_graph", "incremental_graph"]
BROKER_NAMES = dict(BROKERS)


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def load_data(data_dir: str | Path) -> dict:
    d = Path(data_dir)
    ents = read_jsonl(d / "entities.jsonl")
    return {"episodes": read_jsonl(d / "episodes.jsonl"), "facts": read_jsonl(d / "facts.jsonl"),
            "entities": {e["id"]: e for e in ents}, "questions": read_jsonl(d / "questions.jsonl")}


def ollama_ps(base_url: str) -> list[dict]:
    """Read-only snapshot of what the shared server has loaded (recorded before and after the run)."""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/ps", timeout=10)
        return [{k: m.get(k) for k in ("name", "context_length", "expires_at", "size_vram")}
                for m in r.json().get("models", [])]
    except (requests.RequestException, ValueError):
        return []


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------ the run


def build_systems(cfg: dict, data: dict, llm) -> dict:
    eps = data["episodes"]
    out = {"incremental": {}, "batch": {}, "hybrid": {}, "ablation_no_llm_merge": {}}
    for broker, name in BROKERS:
        b_eps = [e for e in eps if e["broker"] == broker]
        ig = IncrementalGraph(broker, name, llm, eps, use_llm_merge=True)
        t0 = time.perf_counter()
        for i, ep in enumerate(b_eps, start=1):
            ig.ingest(ep)
            if i % 10 == 0 or i == len(b_eps):
                log(f"incremental {broker}: {i}/{len(b_eps)} episodes, {time.perf_counter() - t0:.0f} s")
        out["incremental"][broker] = ig
    for broker, name in BROKERS:
        res = BatchGraph(broker, name, llm, cfg["batch"]["min_community"], cfg["batch"]["summary_tokens"]).build(eps)
        log(f"batch {broker}: {res['cost']['extract_calls']} extractions, {res['cost']['summary_calls']} reports")
        out["batch"][broker] = res
    for broker, name in BROKERS:
        # ablation: the same extractions (replayed), ambiguous merges decided by "oldest candidate", no LLM
        ab = IncrementalGraph(broker, name, llm, eps, use_llm_merge=False, tag="ablation")
        for ep in (e for e in eps if e["broker"] == broker):
            ab.ingest(ep)
        out["ablation_no_llm_merge"][broker] = ab
    for broker, _ in BROKERS:
        t0 = time.perf_counter()
        out["hybrid"][broker] = HybridIndex([e for e in eps if e["broker"] == broker], BROKER_NAMES,
                                            cfg["hybrid"]["lsa_dims"], cfg["hybrid"]["candidates"],
                                            cfg["hybrid"]["rrf_k"])
        out.setdefault("hybrid_index_s", {})[broker] = time.perf_counter() - t0
    out["hybrid_shared"] = HybridIndex(eps, BROKER_NAMES, cfg["hybrid"]["lsa_dims"], cfg["hybrid"]["candidates"],
                                       cfg["hybrid"]["rrf_k"])
    return out


def contexts(cfg: dict, systems: dict, q: dict) -> dict[str, tuple[str, list[str], dict]]:
    b = q["broker"]
    k = cfg["hybrid"]["top_k"]
    ctx, ids = systems["hybrid"][b].context(q["question"], k)
    res = {"hybrid": (ctx, ids, {})}
    bt = systems["batch"][b]
    gr = GraphRetriever(bt["graph"], bt["reports"], bt["membership"], **cfg["graph_retrieval"])
    res["batch_graph"] = gr.context(q["question"])
    ig = systems["incremental"][b]
    res["incremental_graph"] = GraphRetriever(ig.graph, **cfg["graph_retrieval"]).context(q["question"])
    if q["type"] == "isolation":
        ctx, ids = systems["hybrid_shared"].context(q["question"], k)
        res["hybrid_shared_index"] = (ctx, ids, {})
    return res


def answer_all(cfg: dict, data: dict, systems: dict, llm) -> list[dict]:
    records = []
    qs = data["questions"]
    t0 = time.perf_counter()
    for i, q in enumerate(qs, start=1):
        for system, (ctx, ids, info) in contexts(cfg, systems, q).items():
            res = llm.chat(answer_messages(BROKER_NAMES[q["broker"]], ctx, q["question"]),
                           max_tokens=cfg["answer"]["max_tokens"], tag=f"answer:{system}", ref=q["id"])
            records.append({"qid": q["id"], "broker": q["broker"], "type": q["type"], "system": system,
                            "question": q["question"], "answer": res.text, "context_chars": len(ctx),
                            "context_episodes": ids[:40], "retrieval": info, "latency_s": res.latency_s,
                            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                            **score(q, res.text)})
        if i % 9 == 0 or i == len(qs):
            log(f"answers: {i}/{len(qs)} questions, {time.perf_counter() - t0:.0f} s")
    return records


# ------------------------------------------------------------------ metrics


def rate(rows: list[dict], key: str) -> float | None:
    return sum(r[key] for r in rows) / len(rows) if rows else None


def answer_metrics(records: list[dict], questions: list[dict]) -> dict:
    by_sys = {s: [r for r in records if r["system"] == s] for s in SYSTEMS}
    order = [q["id"] for q in questions]
    out = {"per_system": {}, "per_type": {}, "paired": {}, "per_type_paired": {}}
    for s, rows in by_sys.items():
        rows = sorted(rows, key=lambda r: order.index(r["qid"]))
        answerable = [r for r in rows if r["type"] not in ("unanswerable", "isolation")]
        lo, hi = bootstrap_ci([r["correct"] for r in rows])
        cur = [r for r in rows if r["type"] in ("temporal_current", "multi_hop")
               and next(q for q in questions if q["id"] == r["qid"])["stale"]]
        asof = [r for r in rows if r["type"] == "as_of"]
        iso = [r for r in rows if r["type"] == "isolation"]
        out["per_system"][s] = {
            "n": len(rows), "accuracy": rate(rows, "correct"), "ci95": [lo, hi],
            "answerable_accuracy": rate(answerable, "correct"),
            "false_idk_rate": rate(answerable, "declined"),
            "stale_rate_now": rate(cur, "stale"), "n_now_with_history": len(cur),
            "wrong_period_rate_as_of": rate(asof, "stale"),
            "isolation_leaks": sum(r["leak"] for r in iso), "isolation_confusions": sum(r["confusion"] for r in iso),
            "declined_unanswerable": rate([r for r in rows if r["type"] == "unanswerable"], "declined"),
            "declined_isolation": rate(iso, "declined"),
            "supported_share_of_correct": (sum(r["supported"] for r in answerable)
                                           / max(1, sum(r["correct"] for r in answerable))),
            "median_answer_latency_s": sorted(r["latency_s"] for r in rows)[len(rows) // 2],
            "median_prompt_tokens": sorted(r["prompt_tokens"] for r in rows)[len(rows) // 2]}
        out["per_type"][s] = {t: {"n": len(x), "accuracy": rate(x, "correct")}
                              for t in TYPES for x in [[r for r in rows if r["type"] == t]]}
    vec = {s: [r["correct"] for r in sorted(by_sys[s], key=lambda r: order.index(r["qid"]))] for s in SYSTEMS}
    for a, b in [("hybrid", "incremental_graph"), ("batch_graph", "incremental_graph"), ("hybrid", "batch_graph")]:
        out["paired"][f"{b} vs {a}"] = paired_bootstrap(vec[a], vec[b])
        out["per_type_paired"][f"{b} vs {a}"] = {}
        for t in TYPES:
            idx = [i for i, qid in enumerate(order) if next(q for q in questions if q["id"] == qid)["type"] == t]
            out["per_type_paired"][f"{b} vs {a}"][t] = paired_bootstrap([vec[a][i] for i in idx],
                                                                        [vec[b][i] for i in idx])
    shared = [r for r in records if r["system"] == "hybrid_shared_index"]
    out["shared_index_probe"] = {"n": len(shared), "leaks": sum(r["leak"] for r in shared),
                                 "declined": sum(r["declined"] for r in shared)}
    return out


def cost_tables(cfg: dict, data: dict, systems: dict, llm) -> dict:
    """What it costs each system to absorb the last N episodes of every broker's stream."""
    frac = cfg["cost"]["new_fraction"]
    extract_cost = {}  # episode -> measured cost of its one extraction call
    for c in llm.ledger:
        if c["tag"] == "inc:extract":
            extract_cost[c["ref"]] = c
    per_broker = {}
    for broker, _ in BROKERS:
        b_eps = [e for e in data["episodes"] if e["broker"] == broker]
        n_new = max(1, round(len(b_eps) * frac))
        new_ids = {e["id"] for e in b_eps[-n_new:]}
        stats = [s for s in systems["incremental"][broker].stats if s["episode"] in new_ids]
        inc = {"llm_calls": sum(s["llm_calls"] for s in stats), "merge_calls": sum(s["merge_calls"] for s in stats),
               "prompt_tokens": sum(s["prompt_tokens"] for s in stats),
               "completion_tokens": sum(s["completion_tokens"] for s in stats),
               "llm_s": sum(s["llm_s"] for s in stats), "python_s": sum(s["python_s"] for s in stats)}
        bc = systems["batch"][broker]["cost"]
        rebuild = {k: bc[k] for k in ("llm_calls", "prompt_tokens", "completion_tokens", "llm_s", "python_s")}
        cached = {"llm_calls": bc["summary_calls"], "tokens": bc["summary_tokens"], "llm_s": bc["summary_llm_s"],
                  "python_s": bc["python_s"]}
        # staying as fresh as the incremental graph: rebuild after every new episode (extraction cache assumed)
        per_episode_fresh = {"llm_calls": bc["summary_calls"] * n_new, "llm_s": bc["summary_llm_s"] * n_new}
        per_broker[broker] = {"episodes": len(b_eps), "new_episodes": n_new, "incremental": inc,
                              "rebuild": rebuild, "rebuild_cached_extraction": cached,
                              "rebuild_every_episode_cached_extraction": per_episode_fresh,
                              "hybrid_reindex_s": systems["hybrid_index_s"][broker]}
    tot = lambda path: sum(_dig(per_broker[b], path) for b in per_broker)
    n_new = tot("new_episodes")
    summary = {
        "new_episodes": n_new, "total_episodes": tot("episodes"),
        "incremental": {"llm_calls": tot("incremental.llm_calls"), "merge_calls": tot("incremental.merge_calls"),
                        "tokens": tot("incremental.prompt_tokens") + tot("incremental.completion_tokens"),
                        "llm_s": tot("incremental.llm_s"), "python_s": tot("incremental.python_s")},
        "rebuild": {"llm_calls": tot("rebuild.llm_calls"),
                    "tokens": tot("rebuild.prompt_tokens") + tot("rebuild.completion_tokens"),
                    "llm_s": tot("rebuild.llm_s"), "python_s": tot("rebuild.python_s")},
        "rebuild_cached_extraction": {"llm_calls": tot("rebuild_cached_extraction.llm_calls"),
                                      "tokens": tot("rebuild_cached_extraction.tokens"),
                                      "llm_s": tot("rebuild_cached_extraction.llm_s"),
                                      "python_s": tot("rebuild_cached_extraction.python_s")},
        "hybrid_reindex_s": tot("hybrid_reindex_s"),
        "extraction_per_episode": {
            "median_s": _median([c["latency_s"] for c in extract_cost.values()]),
            "mean_s": sum(c["latency_s"] for c in extract_cost.values()) / max(1, len(extract_cost)),
            "mean_prompt_tokens": sum(c["prompt_tokens"] for c in extract_cost.values()) / max(1, len(extract_cost)),
            "mean_completion_tokens": (sum(c["completion_tokens"] for c in extract_cost.values())
                                       / max(1, len(extract_cost)))}}
    return {"per_broker": per_broker, "summary": summary}


def _dig(d: dict, path: str):
    for k in path.split("."):
        d = d[k]
    return d


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def graph_quality(data: dict, systems: dict) -> dict:
    out = {}
    for key, getter in [("incremental_graph", lambda b: systems["incremental"][b].graph),
                        ("batch_graph", lambda b: systems["batch"][b]["graph"]),
                        ("incremental_no_llm_merge", lambda b: systems["ablation_no_llm_merge"][b].graph)]:
        per = {}
        for broker, _ in BROKERS:
            truth = Truth(broker, data["episodes"], data["facts"], data["entities"])
            per[broker] = evaluate_graph(getter(broker), truth)
        out[key] = {"per_broker": per, "overall": _pool(per)}
    routes = {}
    for broker, _ in BROKERS:
        for row in systems["incremental"][broker].resolver.log:
            routes[row["route"]] = routes.get(row["route"], 0) + 1
    out["incremental_resolution_routes"] = routes
    return out


def _pool(per: dict) -> dict:
    """Pool the per-broker counts into overall rates (micro-averages, not means of rates)."""
    s = lambda k: sum(p[k] for p in per.values())
    rc = lambda k: sum(p["relation_counts"][k] for p in per.values())
    nodes_mapped = sum(p["nodes"] - p["unmapped_nodes"] for p in per.values())
    by_type = {}
    for t in ("client", "building", "place"):
        ents = sum(p["by_type"][t]["true_entities"] for p in per.values())
        nodes = sum(p["by_type"][t]["nodes"] for p in per.values())
        cov = sum(round(p["by_type"][t]["entity_recall"] * p["by_type"][t]["true_entities"]) for p in per.values())
        by_type[t] = {"true_entities": ents, "nodes": nodes,
                      "entities_split": sum(p["by_type"][t]["entities_split"] for p in per.values()),
                      "duplication_rate": (nodes - cov) / max(1, cov), "entity_recall": cov / max(1, ents)}
    return {"nodes": s("nodes"), "edges": s("edges"), "unmapped_nodes": s("unmapped_nodes"),
            "wrong_merges": s("wrong_merges"), "wrong_merge_rate": s("wrong_merges") / max(1, nodes_mapped),
            "by_type": by_type,
            "duplication_rate": sum(by_type[t]["nodes"] for t in by_type) / max(1, sum(
                round(by_type[t]["entity_recall"] * by_type[t]["true_entities"]) for t in by_type)) - 1,
            "relation_precision": rc("correct") / max(1, rc("graph")), "relation_recall": rc("found") / max(1, rc("true")),
            "relation_counts": {k: rc(k) for k in ("graph", "true", "correct", "found")},
            "slots": s("slots"), "current_value_accuracy": sum(p["current_value_accuracy"] * p["slots"]
                                                               for p in per.values()) / max(1, s("slots")),
            "latest_stated_accuracy": sum(p["latest_stated_accuracy"] * p["slots"] for p in per.values())
            / max(1, s("slots")),
            "slots_with_conflicting_open_values": s("slots_with_conflicting_open_values"),
            "superseded_facts": s("superseded_facts"), "closed": s("closed"),
            "closed_with_exact_date": s("closed_with_exact_date")}


def save_artifacts(results: Path, systems: dict, llm) -> None:
    g = results / "graphs"
    g.mkdir(parents=True, exist_ok=True)
    reports, stats, rlog = {}, [], []
    for broker, _ in BROKERS:
        systems["incremental"][broker].graph.dump(g / f"incremental_{broker}.json")
        systems["batch"][broker]["graph"].dump(g / f"batch_{broker}.json")
        bt = systems["batch"][broker]
        reports[broker] = {str(c): {"members": [bt["graph"].nodes[n]["name"] for n in bt["communities"][c]],
                                    "report": bt["reports"].get(c)} for c in bt["communities"]}
        stats += systems["incremental"][broker].stats
        rlog += systems["incremental"][broker].resolver.log
    (results / "communities.json").write_text(json.dumps(reports, indent=1, sort_keys=True))
    write_jsonl(results / "ingest_stats.jsonl", stats)
    write_jsonl(results / "resolution_log.jsonl", rlog)
    extractions = []
    for broker, _ in BROKERS:
        for ep_id, ex in systems["incremental"][broker].extractions.items():
            extractions.append({"episode": ep_id, "facts": ex["facts"], "raw": ex["raw"]})
    write_jsonl(results / "extractions.jsonl", extractions)
    write_jsonl(results / "llm_ledger.jsonl", llm.ledger)


def run(cfg: dict, llm, results_dir: str | Path, data_dir: str | Path) -> dict:
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_all = time.perf_counter()
    ps_before = ollama_ps(cfg["llm"]["base_url"])
    data = load_data(data_dir)
    log(f"data: {len(data['episodes'])} episodes, {len(data['facts'])} facts, {len(data['questions'])} questions")
    t0 = time.perf_counter()
    systems = build_systems(cfg, data, llm)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    records = answer_all(cfg, data, systems, llm)
    answer_s = time.perf_counter() - t0
    metrics = answer_metrics(records, data["questions"])
    quality = graph_quality(data, systems)
    cost = cost_tables(cfg, data, systems, llm)
    save_artifacts(results, systems, llm)
    write_jsonl(results / "answers.jsonl", records)
    (results / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True))
    (results / "graph_quality.json").write_text(json.dumps(quality, indent=1, sort_keys=True))
    (results / "cost.json").write_text(json.dumps(cost, indent=1, sort_keys=True))
    uncached = [c for c in llm.ledger if not c["cached"]]
    meta = {"started": started, "wall_s": time.perf_counter() - t_all, "build_s": build_s, "answer_s": answer_s,
            "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "config": cfg, "llm_calls": len(llm.ledger), "llm_calls_uncached": len(uncached),
            "llm_s_uncached": sum(c["latency_s"] for c in uncached), "llm_s_all_measured":
                sum(c["latency_s"] for c in llm.ledger), "ollama_ps_before": ps_before,
            "ollama_ps_after": ollama_ps(cfg["llm"]["base_url"]),
            "calls_by_tag": {t: sum(1 for c in llm.ledger if c["tag"] == t) for t in sorted({c["tag"] for c in llm.ledger})}}
    (results / "run_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    from .report import write_report
    write_report(results)
    log(f"done in {meta['wall_s']:.0f} s ({meta['llm_calls_uncached']} uncached LLM calls, "
        f"{meta['llm_s_uncached']:.0f} s of model time)")
    return meta
