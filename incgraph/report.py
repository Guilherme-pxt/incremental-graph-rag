"""results/report.md from the result files (no model calls)."""

from __future__ import annotations

import json
from pathlib import Path

from .questions import TYPES

LABEL = {"hybrid": "Hybrid (BM25 + LSA + RRF over episodes)", "batch_graph": "Batch GraphRAG",
         "incremental_graph": "Incremental temporal graph"}


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def ci(c) -> str:
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def delta(d) -> str:
    return f"{100 * d['delta']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]"


def write_report(results_dir) -> Path:
    r = Path(results_dir)
    m = json.loads((r / "metrics.json").read_text())
    q = json.loads((r / "graph_quality.json").read_text())
    c = json.loads((r / "cost.json").read_text())
    meta = json.loads((r / "run_meta.json").read_text())
    out = ["# Report", "",
           f"Run started {meta['started']}, wall time {meta['wall_s']:.0f} s; {meta['llm_calls_uncached']} uncached "
           f"LLM calls ({meta['llm_s_uncached']:.0f} s of model time) out of {meta['llm_calls']} calls.", ""]

    out += ["## Answers", "", "| system | accuracy (95% CI) | answerable | false \"I don't know\" | stale (now) | "
            "wrong period (as of) | unanswerable declined | isolation declined | isolation leaks | "
            "correct and cites a gold episode | median prompt tokens | median answer latency |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, v in m["per_system"].items():
        out.append(f"| {LABEL[s]} | {pct(v['accuracy'])} {ci(v['ci95'])} | {pct(v['answerable_accuracy'])} | "
                   f"{pct(v['false_idk_rate'])} | {pct(v['stale_rate_now'])} | {pct(v['wrong_period_rate_as_of'])} | "
                   f"{pct(v['declined_unanswerable'])} | {pct(v['declined_isolation'])} | {v['isolation_leaks']} | "
                   f"{pct(v['supported_share_of_correct'])} | {v['median_prompt_tokens']} | "
                   f"{v['median_answer_latency_s']:.2f} s |")
    sh = m["shared_index_probe"]
    out += ["", f"Isolation probes against one index shared by all brokers (hybrid, for contrast): "
                f"{sh['leaks']} leaks and {sh['declined']} declined out of {sh['n']}.", ""]
    out += ["### Accuracy by question type", "",
            "| system | " + " | ".join(f"{t} ({m['per_type']['hybrid'][t]['n']})" for t in TYPES) + " |",
            "|---|" + "---|" * len(TYPES)]
    for s in m["per_type"]:
        out.append(f"| {LABEL[s]} | " + " | ".join(pct(m["per_type"][s][t]["accuracy"]) for t in TYPES) + " |")
    out += ["", "### Paired bootstrap (accuracy, points, 95% CI, 10,000 resamples)", "",
            "| comparison | all | " + " | ".join(TYPES) + " |", "|---|---|" + "---|" * len(TYPES)]
    for k, v in m["paired"].items():
        out.append(f"| {k} | {delta(v)} | " + " | ".join(delta(m['per_type_paired'][k][t]) for t in TYPES) + " |")

    out += ["", "## Graph quality against the ground truth", "",
            "| graph | nodes | wrong-merge rate | duplication rate | clients: nodes / true | buildings: nodes / true | "
            "relation precision | relation recall | current-value accuracy | latest-stated value correct | "
            "superseded facts closed (exact date) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in ("incremental_graph", "incremental_no_llm_merge", "batch_graph"):
        o = q[g]["overall"]
        bt = o["by_type"]
        out.append(f"| {g} | {o['nodes']} | {pct(o['wrong_merge_rate'])} ({o['wrong_merges']}) | "
                   f"{pct(o['duplication_rate'])} | {bt['client']['nodes']} / {bt['client']['true_entities']} | "
                   f"{bt['building']['nodes']} / {bt['building']['true_entities']} | {pct(o['relation_precision'])} | "
                   f"{pct(o['relation_recall'])} | {pct(o['current_value_accuracy'])} | "
                   f"{pct(o['latest_stated_accuracy'])} | {o['closed']} ({o['closed_with_exact_date']}) "
                   f"of {o['superseded_facts']} |")
    out += ["", f"Resolution routes (incremental): {json.dumps(q['incremental_resolution_routes'], sort_keys=True)}", ""]

    s = c["summary"]
    n = s["new_episodes"]
    out += ["## Cost of absorbing new episodes", "",
            f"{n} new episodes (the last quarter of each broker's stream, {s['total_episodes']} in total).", "",
            "| strategy | LLM calls | tokens | LLM seconds | python seconds | LLM seconds per new episode |",
            "|---|---|---|---|---|---|"]
    rows = [("incremental: fold the new episodes in", s["incremental"]),
            ("batch: rebuild from scratch", s["rebuild"]),
            ("batch: rebuild, extractions replayed from a cache", s["rebuild_cached_extraction"])]
    for label, v in rows:
        tok = v.get("tokens")
        out.append(f"| {label} | {v['llm_calls']} | {tok:,} | {v['llm_s']:.0f} | {v['python_s']:.3f} | "
                   f"{v['llm_s'] / n:.1f} |")
    out.append(f"| hybrid: re-index | 0 | 0 | 0 | {s['hybrid_reindex_s']:.3f} | 0 |")
    e = s["extraction_per_episode"]
    out += ["", f"One extraction call: median {e['median_s']:.1f} s, mean {e['mean_s']:.1f} s, "
                f"{e['mean_prompt_tokens']:.0f} prompt and {e['mean_completion_tokens']:.0f} output tokens on average.",
            "", f"Calls by tag: {json.dumps(meta['calls_by_tag'], sort_keys=True)}", ""]
    path = r / "report.md"
    path.write_text("\n".join(out) + "\n")
    return path
