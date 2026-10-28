import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from incgraph import pipeline as P
from incgraph.graph import fmt_value
from incgraph.incremental import IncrementalGraph
from incgraph.retrieve import GraphRetriever, HybridIndex, link
from incgraph.scoring import atom_in, paired_bootstrap, score
from incgraph.synth import BROKERS, SINGLE, write

from .helpers import config, data_dir, oracle_llm


def gold_answer(q: dict) -> str:
    """Render a question's gold atoms the way a correct answer would state them."""
    if q["kind"] == "idk":
        return "ANSWER: I don't know"
    parts = []
    for a in q["required"]:
        if a["t"] == "money":
            parts.append(fmt_value("budget", a["v"]))
        elif a["t"] == "month":
            parts.append(fmt_value("delivery", a["v"]))
        elif a["t"] == "entity":
            parts.append(a["names"][0])
        else:
            parts.append(str(a["v"]))
    return f"ANSWER: {', '.join(parts)} [{(q['evidence'] or ['a-001'])[0]}]"


class SynthTests(unittest.TestCase):
    def test_byte_identical_for_a_seed(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            write(d1, 42)
            write(d2, 42)
            for name in ("episodes.jsonl", "facts.jsonl", "entities.jsonl", "questions.jsonl"):
                self.assertEqual((Path(d1) / name).read_bytes(), (Path(d2) / name).read_bytes(), name)

    def test_confounders_exist(self):
        data = P.load_data(data_dir())
        clients = [e for e in data["entities"].values() if e["type"] == "client"]
        marinas = {e["broker"] for e in clients if e["name"].startswith("Marina ")}
        daniels = [e for e in clients if e["name"].startswith("Daniel ")]
        self.assertEqual(marinas, {"a", "b"})
        self.assertEqual({e["broker"] for e in daniels}, {"c"})
        self.assertEqual(len(daniels), 2)
        texts = " ".join(e["text"] + " " + (e["contact"] or "") for e in data["episodes"])
        for form in ("the Aurora", "N1", "Kate Moreau", "Fil Arantes"):
            self.assertIn(form, texts)

    def test_every_single_valued_slot_has_one_open_fact_and_contiguous_history(self):
        data = P.load_data(data_dir())
        slots = {}
        for f in data["facts"]:
            if f["predicate"] in SINGLE:
                slots.setdefault((f["broker"], f["subject"], f["predicate"]), []).append(f)
        changed = 0
        for fs in slots.values():
            self.assertEqual(sum(f["valid_to"] is None for f in fs), 1)
            for a, b in zip(fs, fs[1:]):
                self.assertEqual(a["valid_to"], b["valid_from"])
            changed += len(fs) > 1
        self.assertGreaterEqual(changed, 18)  # facts really do change over time

    def test_question_mix(self):
        qs = P.load_data(data_dir())["questions"]
        self.assertEqual(len(qs), 81)
        per = {}
        for q in qs:
            per[(q["broker"], q["type"])] = per.get((q["broker"], q["type"]), 0) + 1
        self.assertEqual(set(per.values()) - {3, 4, 5}, set())
        for q in qs:
            if q["kind"] != "idk":
                self.assertTrue(q["required"] and q["evidence"], q["id"])
            if q["type"] == "isolation":
                self.assertTrue(q["leak"], q["id"])


class ScoringTests(unittest.TestCase):
    def test_gold_answers_score_correct_and_idk_scores_wrong(self):
        for q in P.load_data(data_dir())["questions"]:
            self.assertTrue(score(q, gold_answer(q))["correct"], q["id"])
            if q["kind"] != "idk":
                self.assertFalse(score(q, "ANSWER: I don't know")["correct"], q["id"])

    def test_stale_leak_and_forbidden(self):
        q = {"kind": "value", "required": [{"t": "money", "v": 968000}], "stale": [{"t": "money", "v": 870000}],
             "forbidden": [], "leak": [], "confusion": [], "evidence": ["a-039"]}
        s = score(q, "ANSWER: $870k [a-020]")
        self.assertEqual((s["correct"], s["stale"]), (False, True))
        s = score(q, "ANSWER: 968k [a-039]")
        self.assertEqual((s["correct"], s["supported"]), (True, True))
        probe = {"kind": "idk", "required": [], "stale": [], "forbidden": [], "evidence": [],
                 "leak": [{"t": "money", "v": 808000}], "confusion": [{"t": "money", "v": 1089000}]}
        self.assertTrue(score(probe, "ANSWER: $808,000 [b-028]")["leak"])
        self.assertTrue(score(probe, "ANSWER: 1,089,000 dollars")["confusion"])
        lst = {"kind": "list", "required": [{"t": "entity", "names": ["Ruth Castellano", "Castellano"]}],
               "forbidden": [{"t": "entity", "names": ["Marina Okafor", "Okafor"]}], "stale": [], "evidence": []}
        self.assertTrue(score(lst, "ANSWER: Ruth Castellano")["correct"])
        self.assertFalse(score(lst, "ANSWER: Ruth Castellano, Marina Okafor")["correct"])
        self.assertTrue(atom_in({"t": "entity", "names": ["Aurora Res."]}, "it is aurora res"))

    def test_paired_bootstrap(self):
        d = paired_bootstrap([0, 0, 1, 1], [1, 1, 1, 1])
        self.assertAlmostEqual(d["delta"], 0.5)
        self.assertGreaterEqual(d["ci95"][0], 0.0)


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = P.load_data(data_dir())
        llm, _ = oracle_llm()
        cls.ig = IncrementalGraph("a", "Lena Ferraz", llm, cls.data["episodes"])
        for ep in (e for e in cls.data["episodes"] if e["broker"] == "a"):
            cls.ig.ingest(ep)

    def test_link_ignores_a_first_name_with_another_surname(self):
        g = self.ig.graph
        self.assertEqual(link(g, "What is Marina Lindqvist's budget?"), [])
        names = [g.nodes[n]["name"] for n in link(g, "What is Marina Okafor's budget now?")]
        self.assertEqual(names, ["Marina Okafor"])

    def test_graph_context_has_current_and_history_with_citations(self):
        ctx, cited, info = GraphRetriever(self.ig.graph).context("What is Katherine Moreau's budget now?")
        self.assertEqual(info["linked"], ["Katherine Moreau"])
        self.assertIn("| budget |", ctx)
        current, history = ctx.split("History (superseded values")
        self.assertIn("current since", current)
        self.assertNotIn("superseded", current.split("\n", 1)[1])  # no superseded line among current facts
        self.assertIn("superseded, held", history)
        self.assertTrue(cited)

    def test_multi_hop_forms_are_answered_by_the_query_and_match_gold(self):
        from incgraph.query import Query
        n = 0
        for q in self.data["questions"]:
            out = Query(self.ig.graph).run(q["question"]) if q["broker"] == "a" else None
            if q["broker"] != "a":
                continue
            self.assertEqual(out is not None, q["type"] in ("multi_hop", "aggregation"), q["id"])
            if out:
                n += 1
                self.assertTrue(all(atom_in(a, out[2]) for a in q["required"]), q["id"])
                self.assertFalse(any(atom_in(a, out[2]) for a in q["forbidden"]), q["id"])
        self.assertEqual(n, 7)  # 4 multi-hop and 3 aggregation questions for broker a

    def test_structured_context_carries_result_and_evidence(self):
        q = "Which of my clients are currently interested in a building near Harbor Central station?"
        ctx, cited, info = GraphRetriever(self.ig.graph, structured=True).context(q)
        self.assertTrue(ctx.startswith("Computed from the graph"))
        self.assertIn("Result: ", ctx)
        self.assertIn("near_station | Harbor Central", ctx)
        self.assertTrue(cited)
        _, _, plain = GraphRetriever(self.ig.graph, structured=False).context(q)
        self.assertNotIn("computed", plain)

    def test_tables_for_questions_about_all_clients(self):
        _, _, info = GraphRetriever(self.ig.graph).context("How many of my clients want 2 bedrooms right now?")
        self.assertTrue(info["tables"])

    def test_hybrid_is_per_broker(self):
        eps = [e for e in self.data["episodes"] if e["broker"] == "a"]
        ctx, ids = HybridIndex(eps, P.BROKER_NAMES, dims=16).context("What is Marina Lindqvist's budget?", 8)
        self.assertTrue(all(i.startswith("a-") for i in ids))
        self.assertNotIn("Lindqvist", ctx)


class EndToEndTests(unittest.TestCase):
    """The whole pipeline with an oracle extractor and an oracle answerer: no Ollama, no network."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cfg = config(Path(cls.tmp.name))
        # the same question text is asked to every broker; the system prompt names the broker
        qs = {(P.BROKER_NAMES[q["broker"]], q["question"]): q for q in P.load_data(data_dir())["questions"]}
        llm, cls.provider = oracle_llm()
        inner = cls.provider.fn

        def fn(messages):
            text = messages[-1]["content"]
            if "\nQuestion: " in text:
                broker = messages[0]["content"].split("assistant of ", 1)[1].split(",", 1)[0]
                return gold_answer(qs[(broker, text.rsplit("\nQuestion: ", 1)[1])])
            return inner(messages)

        cls.provider.fn = fn
        with contextlib.redirect_stdout(io.StringIO()):
            cls.meta = P.run(cfg, llm, cfg["run"]["results_dir"], data_dir())
        cls.results = Path(cfg["run"]["results_dir"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_artifacts_and_report(self):
        for name in ("answers.jsonl", "metrics.json", "graph_quality.json", "cost.json", "run_meta.json",
                     "report.md", "graphs/incremental_a.json", "resolution_log.jsonl", "llm_ledger.jsonl"):
            self.assertTrue((self.results / name).exists(), name)
        self.assertIn("## Cost of absorbing new episodes", (self.results / "report.md").read_text())

    def test_perfect_extraction_gives_a_perfect_incremental_graph(self):
        q = json.loads((self.results / "graph_quality.json").read_text())
        inc, bat = q["incremental_graph"]["overall"], q["batch_graph"]["overall"]
        self.assertEqual((inc["wrong_merges"], inc["duplication_rate"]), (0, 0.0))
        self.assertEqual(inc["current_value_accuracy"], 1.0)
        self.assertEqual(inc["closed_with_exact_date"], inc["superseded_facts"])
        self.assertGreater(bat["duplication_rate"], 0.2)  # name-only merging splits nicknames and aliases
        self.assertEqual(bat["closed"], 0)

    def test_oracle_answers_are_all_correct_and_nothing_leaks(self):
        m = json.loads((self.results / "metrics.json").read_text())
        for s in P.SYSTEMS:
            self.assertEqual(m["per_system"][s]["accuracy"], 1.0, s)
            self.assertEqual(m["per_system"][s]["isolation_leaks"], 0, s)

    def test_cost_accounting(self):
        c = json.loads((self.results / "cost.json").read_text())["summary"]
        self.assertEqual(c["rebuild"]["llm_calls"] - c["rebuild_cached_extraction"]["llm_calls"], c["total_episodes"])
        self.assertEqual(c["incremental"]["llm_calls"] - c["incremental"]["merge_calls"], c["new_episodes"])

    def test_no_request_left_the_process(self):
        self.assertTrue(self.provider.calls)
        self.assertEqual(json.loads((self.results / "run_meta.json").read_text())["ollama_ps_before"], [])

    def test_graphs_are_per_broker(self):
        for broker, _ in BROKERS:
            g = json.loads((self.results / f"graphs/incremental_{broker}.json").read_text())
            eps = {ep for e in g["edges"] for ep in e["episodes"]}
            self.assertTrue(eps and all(ep.startswith(f"{broker}-") for ep in eps), broker)


if __name__ == "__main__":
    unittest.main()
