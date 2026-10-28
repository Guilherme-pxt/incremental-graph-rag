"""Command line entry point: ``python -m incgraph <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from . import pipeline as P
from .llm import LLMError, build_llm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="incgraph", description=(
        "An incremental temporal knowledge graph against batch GraphRAG and hybrid search, "
        "on a synthetic real-estate brokerage."))
    p.add_argument("--config", default="config.toml", help="TOML config (default: config.toml)")
    p.add_argument("--results-dir", help="override [run].results_dir")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the LLM response cache")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate the episode stream, ground truth and gold questions (seeded)")
    s.add_argument("--seed", type=int)

    sub.add_parser("run", help="ingest, build, answer every question with every system, evaluate, report")
    s = sub.add_parser("eval", help="the same as run, replaying the cache; any cache miss is an error")
    s.add_argument("--allow-miss", action="store_true", help="call the model on a cache miss instead of failing")
    sub.add_parser("report", help="rebuild results/report.md from the result files, no model calls")
    sub.add_parser("run-all", help="synth, then run")

    s = sub.add_parser("ask", help="answer one question for one broker with one system (builds from the cache)")
    s.add_argument("broker", choices=["a", "b", "c"])
    s.add_argument("question")
    s.add_argument("--system", choices=P.SYSTEMS, default="incremental_graph")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = P.load_config(args.config)
    if args.results_dir:
        cfg["run"]["results_dir"] = args.results_dir
    run_cfg = cfg["run"]

    def llm():
        return build_llm(cfg["llm"], None if args.no_cache else run_cfg["cache"])

    try:
        if args.command in ("synth", "run-all"):
            from .synth import write
            seed = getattr(args, "seed", None) or run_cfg["seed"]
            print(json.dumps(write(run_cfg["data_dir"], seed)))
        if args.command in ("run", "run-all", "eval"):
            m = llm()
            if args.command == "eval" and not args.allow_miss:
                m.allow_miss = False
            P.run(cfg, m, run_cfg["results_dir"], run_cfg["data_dir"])
            m.close()
        elif args.command == "report":
            from .report import write_report
            write_report(run_cfg["results_dir"])
        elif args.command == "ask":
            from .answer import answer_messages
            m = llm()
            data = P.load_data(run_cfg["data_dir"])
            systems = P.build_systems(cfg, data, m)
            q = {"broker": args.broker, "question": args.question, "type": "", "id": "cli"}
            ctx, ids, info = P.contexts(cfg, systems, q)[args.system]
            res = m.chat(answer_messages(P.BROKER_NAMES[args.broker], ctx, args.question),
                         max_tokens=cfg["answer"]["max_tokens"], tag=f"answer:{args.system}", ref="cli")
            print(ctx, "\n---\n", res.text, sep="")
            m.close()
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
