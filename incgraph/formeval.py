"""Fourth run: the closed-form parser on the held-out set and on the original 81 questions.

Systems compared on the held-out set (58 questions, ``data/heldout.jsonl``):

* hybrid             BM25 + LSA + RRF over episodes (as in every run)
* patterns           the third run's incremental graph: seven regular expressions, else graph retrieval
* form               the incremental graph with the closed-form parser (``form.py``), else graph retrieval

and on the original 81 questions, ``form`` against the third run's ``results/answers.jsonl``.

Measured: answer accuracy; for the questions that should become a form, how
often the form matches the gold form exactly, how often the parser falls back to
retrieval, and how often the form was valid but wrong and so was the answer
("wrong but confident"); for the questions that should not, how often a form
was triggered anyway.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import pipeline as P
from .answer import answer_messages
from .form import Runner, understand
from .heldout import Truth, days
from .retrieve import GraphRetriever
from .scoring import bootstrap_ci, paired_bootstrap, score

STRICT = {"<": "<", "<=": "<=", ">": ">", ">=": ">="}
LENIENT = {"<": "<", "<=": "<", ">": ">", ">=": ">"}


def canon(form: dict, graph, truth: Truth, ops: dict) -> dict:
    """A validated or gold form in comparable terms: values as sets of true entity ids, months, numbers."""

    def val(v):
        if isinstance(v, list):  # node ids from a validated form
            return frozenset(x for n in v for x in truth.resolve(graph.nodes[n]["name"]))
        if isinstance(v, str) and not v[:1].isdigit():  # a name in a gold form
            return frozenset(truth.resolve(v))
        return v

    def flt(fs):
        return sorted((f, ops.get(o, o), repr(val(v))) for f, o, v in fs)

    return {"operation": form["operation"], "target": form["target"], "filters": flt(form["filters"]),
            "can_afford": flt(form["can_afford"]), "metric": form.get("metric"), "return": form.get("return", "name")}


def same_date(pred: str, gold: str) -> bool:
    if gold == "now":
        return pred == "now"
    return pred in days(gold)


def form_match(pred: dict | None, gold: dict, graph, truth: Truth, ops: dict) -> bool:
    if pred is None or gold["operation"] == "none":
        return False
    return canon(pred, graph, truth, ops) == canon(gold, graph, truth, ops) and same_date(pred["as_of"], gold["as_of"])


def answer(llm, q: dict, ctx: str, system: str, max_tokens: int) -> str:
    res = llm.chat(answer_messages(P.BROKER_NAMES[q["broker"]], ctx, q["question"]), max_tokens=max_tokens,
                   tag=f"answer:{system}", ref=q["id"])
    return res.text


def run_form(llm, q: dict, graph, cfg: dict) -> dict:
    u = understand(llm, graph, q["question"], ref=q["id"])
    if u["form"] is not None:
        ctx, cited, understood = Runner(graph).context(u["form"])
        text = answer(llm, q, ctx, "form", cfg["answer"]["max_tokens"]).rstrip() + f"\nUNDERSTOOD: {understood}"
    else:
        ctx, cited, _ = GraphRetriever(graph, structured=False, **cfg["graph_retrieval"]).context(q["question"])
        text = answer(llm, q, ctx, "form", cfg["answer"]["max_tokens"])
    return {"form_text": u["text"], "form": u["form"], "reason": u["reason"], "answer": text,
            "triggered": u["form"] is not None}


def evaluate(cfg: dict, llm, data_dir: str | Path, results_dir: str | Path) -> dict:
    t0 = time.perf_counter()
    data = P.load_data(data_dir)
    heldout = P.read_jsonl(Path(data_dir) / "heldout.jsonl")
    systems = P.build_systems(cfg, data, llm)
    truths = {b: Truth(b, data["facts"], data["entities"]) for b in "abc"}
    out = Path(results_dir) / "form"
    out.mkdir(parents=True, exist_ok=True)
    k = cfg["hybrid"]["top_k"]
    mt = cfg["answer"]["max_tokens"]

    rows = []
    for i, q in enumerate(heldout, start=1):
        g = systems["incremental"][q["broker"]].graph
        hctx, _ = systems["hybrid"][q["broker"]].context(q["question"], k)
        pctx, _, pinfo = GraphRetriever(g, structured=True, **cfg["graph_retrieval"]).context(q["question"])
        f = run_form(llm, q, g, cfg)
        r = {"id": q["id"], "broker": q["broker"], "lang": q["lang"], "question": q["question"],
             "must_parse": q["must_parse"], "form_text": f["form_text"], "form_reason": f["reason"],
             "form_triggered": f["triggered"],
             "form_parsed": f["form"], "pattern_matched": bool(pinfo.get("computed"))}
        truth = truths[q["broker"]]
        r["form_exact"] = form_match(f["form"], q["gold_form"], g, truth, STRICT)
        r["form_exact_lenient"] = form_match(f["form"], q["gold_form"], g, truth, LENIENT)
        for system, text in (("hybrid", answer(llm, q, hctx, "hybrid", mt)),
                             ("patterns", answer(llm, q, pctx, "incremental_graph", mt)), ("form", f["answer"])):
            r[system] = {"answer": text, **score(q, text)}
        rows.append(r)
        if i % 10 == 0 or i == len(heldout):
            P.log(f"held-out: {i}/{len(heldout)}, {time.perf_counter() - t0:.0f} s")
    P.write_jsonl(out / "heldout_answers.jsonl", rows)

    ref = {(r["qid"], r["system"]): r for r in P.read_jsonl(Path(results_dir) / "answers.jsonl")}
    orig = []
    for i, q in enumerate(data["questions"], start=1):
        g = systems["incremental"][q["broker"]].graph
        f = run_form(llm, q, g, cfg)
        orig.append({"id": q["id"], "type": q["type"], "question": q["question"],
                     "must_parse": q["type"] in ("multi_hop", "aggregation"), "form_parsed": f["form"],
                     "form_text": f["form_text"], "form_reason": f["reason"], "form_triggered": f["triggered"],
                     "form_answer": {"answer": f["answer"], **score(q, f["answer"])},
                     "third_run_correct": ref[(q["id"], "incremental_graph")]["correct"],
                     "hybrid_correct": ref[(q["id"], "hybrid")]["correct"]})
        if i % 27 == 0:
            P.log(f"original 81: {i}/81, {time.perf_counter() - t0:.0f} s")
    P.write_jsonl(out / "original_answers.jsonl", orig)
    metrics = summarize(rows, orig)
    metrics["wall_s"] = time.perf_counter() - t0
    metrics["uncached_calls"] = sum(1 for c in llm.ledger if not c["cached"])
    metrics["uncached_s"] = sum(c["latency_s"] for c in llm.ledger if not c["cached"])
    metrics["form_calls"] = sum(1 for c in llm.ledger if c["tag"] == "form")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True))
    return metrics


def _rate(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize(rows: list[dict], orig: list[dict]) -> dict:
    mp = [r for r in rows if r["must_parse"]]
    no = [r for r in rows if not r["must_parse"]]
    m = {"heldout": {"n": len(rows), "must_parse": len(mp), "must_not_parse": len(no)}}
    h = m["heldout"]
    for s in ("hybrid", "patterns", "form"):
        acc = [r[s]["correct"] for r in rows]
        h[s] = {"accuracy": _rate(acc), "ci95": bootstrap_ci(acc),
                "accuracy_must_parse": _rate(r[s]["correct"] for r in mp),
                "accuracy_must_not_parse": _rate(r[s]["correct"] for r in no),
                "accuracy_pt": _rate(r[s]["correct"] for r in rows if r["lang"] == "pt"),
                "accuracy_en": _rate(r[s]["correct"] for r in rows if r["lang"] == "en"),
                "accuracy_pt_must_parse": _rate(r[s]["correct"] for r in mp if r["lang"] == "pt")}
    h["form_exact_match"] = _rate(r["form_exact"] for r in mp)
    h["form_exact_match_lenient_ops"] = _rate(r["form_exact_lenient"] for r in mp)
    h["form_exact_match_pt"] = _rate(r["form_exact"] for r in mp if r["lang"] == "pt")
    h["fallback_rate"] = _rate(not r["form_triggered"] for r in mp)
    h["false_trigger_rate"] = _rate(r["form_triggered"] for r in no)
    h["false_triggers"] = [r["id"] for r in no if r["form_triggered"]]
    h["patterns_matched_must_parse"] = sum(r["pattern_matched"] for r in mp)
    # wrong but confident: a valid form ran, it was not the gold form, and the answer is wrong
    wbc = [r for r in rows if r["form_triggered"] and not r["form_exact"] and not r["form"]["correct"]]
    h["wrong_but_confident"] = len(wbc)
    h["wrong_but_confident_ids"] = [r["id"] for r in wbc]
    h["wrong_but_confident_rate_of_triggered"] = len(wbc) / max(1, sum(r["form_triggered"] for r in rows))
    h["triggered_exact_but_answer_wrong"] = [r["id"] for r in rows if r["form_triggered"] and r["form_exact"]
                                             and not r["form"]["correct"]]
    h["triggered_not_exact_but_answer_right"] = [r["id"] for r in rows if r["form_triggered"] and not r["form_exact"]
                                                 and r["form"]["correct"]]
    h["fallback_ids"] = [r["id"] for r in mp if not r["form_triggered"]]
    vec = {s: [r[s]["correct"] for r in rows] for s in ("hybrid", "patterns", "form")}
    h["paired"] = {"form vs patterns": paired_bootstrap(vec["patterns"], vec["form"]),
                   "form vs hybrid": paired_bootstrap(vec["hybrid"], vec["form"]),
                   "patterns vs hybrid": paired_bootstrap(vec["hybrid"], vec["patterns"])}
    mpv = {s: [r[s]["correct"] for r in mp] for s in vec}
    h["paired_must_parse"] = {"form vs patterns": paired_bootstrap(mpv["patterns"], mpv["form"]),
                              "form vs hybrid": paired_bootstrap(mpv["hybrid"], mpv["form"])}

    o = {"n": len(orig)}
    acc = [r["form_answer"]["correct"] for r in orig]
    o["accuracy"], o["ci95"] = _rate(acc), bootstrap_ci(acc)
    o["third_run_accuracy"] = _rate(r["third_run_correct"] for r in orig)
    o["paired vs third run"] = paired_bootstrap([r["third_run_correct"] for r in orig], acc)
    o["paired vs hybrid"] = paired_bootstrap([r["hybrid_correct"] for r in orig], acc)
    types = sorted({r["type"] for r in orig})
    o["per_type"] = {t: {"form": _rate(r["form_answer"]["correct"] for r in orig if r["type"] == t),
                         "third_run": _rate(r["third_run_correct"] for r in orig if r["type"] == t),
                         "n": sum(1 for r in orig if r["type"] == t)} for t in types}
    omp = [r for r in orig if r["must_parse"]]
    ono = [r for r in orig if not r["must_parse"]]
    o["fallback_rate_structured"] = _rate(not r["form_triggered"] for r in omp)
    o["false_trigger_rate_other"] = _rate(r["form_triggered"] for r in ono)
    o["false_triggers"] = [r["id"] for r in ono if r["form_triggered"]]
    o["fixed"] = [r["id"] for r in orig if r["form_answer"]["correct"] and not r["third_run_correct"]]
    o["broken"] = [r["id"] for r in orig if not r["form_answer"]["correct"] and r["third_run_correct"]]
    o["wrong_but_confident"] = [r["id"] for r in orig if r["form_triggered"] and not r["form_answer"]["correct"]]
    m["original"] = o
    return m
