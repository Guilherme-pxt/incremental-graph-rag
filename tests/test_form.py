import json
import unittest
from pathlib import Path

from incgraph.form import Invalid, Runner, money, parse, validate
from incgraph.formeval import STRICT, form_match
from incgraph.heldout import Truth
from incgraph.incremental import IncrementalGraph
from incgraph.pipeline import load_data
from incgraph.scoring import score

from .helpers import oracle_llm

ROOT = Path(__file__).resolve().parent.parent


def lines(form: dict) -> str:
    """A gold form written the way the model is asked to write it."""
    if form["operation"] == "none":
        return "op: none"
    out = [f"op: {form['operation']}", f"target: {form['target']}", f"as_of: {form['as_of']}"]
    out += [f"filter: {f} {o} {v}" for f, o, v in form["filters"]]
    out += [f"afford: {f} {o} {v}" for f, o, v in form["can_afford"]]
    if form["metric"]:
        out.append(f"metric: {form['metric']}")
    if form["return"] != "name":
        out.append(f"return: {form['return']}")
    return "\n".join(out)


class FormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data(ROOT / "data")
        cls.heldout = [json.loads(x) for x in (ROOT / "data" / "heldout.jsonl").read_text().splitlines()]
        llm, _ = oracle_llm()
        cls.graphs = {}
        for b in "abc":
            ig = IncrementalGraph(b, "", llm, cls.data["episodes"])
            for ep in (e for e in cls.data["episodes"] if e["broker"] == b):
                ig.ingest(ep)
            cls.graphs[b] = ig.graph

    def bad(self, text: str, broker: str = "a"):
        with self.assertRaises(Invalid):
            validate(parse(text), self.graphs[broker])

    def test_money_normalization(self):
        for s, v in [("800000", 800000), ("800k", 800000), ("$800,000", 800000), ("800 mil", 800000),
                     ("1 milhão", 1000000), ("1,2 milhão", 1200000), ("1.2M", 1200000), ("1.000.000", 1000000)]:
            self.assertEqual(money(s), v, s)
        self.assertIsNone(money("lots"))

    def test_validation_rejects(self):
        self.bad("op: none")
        self.bad("op: average\ntarget: client")
        self.bad("op: count\ntarget: lead\nfilter: bedrooms = 3")
        self.bad("op: count\ntarget: client\nfilter: pets = 1")                       # field not in the list
        self.bad("op: count\ntarget: client\nfilter: bedrooms ~ 3")                   # operator not in the list
        self.bad("op: count\ntarget: client\nfilter: wants_neighborhood > Saltmarsh")  # comparison on a name
        self.bad("op: count\ntarget: client\nfilter: budget year 2027")               # year on an amount
        self.bad("op: count\ntarget: client\nfilter: budget > lots")                  # value does not normalize
        self.bad("op: count\ntarget: client\nfilter: interested_in = Aurora Gardens")  # names nothing in the graph
        self.bad("op: max\ntarget: client")                                           # max without a metric
        self.bad("op: max\ntarget: client\nmetric: wants_neighborhood")               # non-numeric metric
        self.bad("op: count\ntarget: client\nreturn: interested_in.developer")        # return only with join
        self.bad("op: join\ntarget: client\nfilter: name = Katherine Moreau\nreturn: budget")  # a lookup
        self.bad("op: join\ntarget: client\nfilter: name = Katherine Moreau")          # names alone: a lookup
        self.bad("op: count\ntarget: client\nas_of: 2026-09-01\nfilter: bedrooms = 3")  # a future date
        self.bad("op: join\ntarget: client\nfilter: name = Daniel\nreturn: interested_in.developer", "c")  # 2 Daniels

    def test_interested_in_name_is_the_building(self):
        """Fourth-run bug: interested_in.name was accepted but matched nothing."""
        g = self.graphs["b"]
        a = validate(parse("op: join\ntarget: client\nfilter: interested_in.name = Mill House"), g)
        b = validate(parse("op: join\ntarget: client\nfilter: interested_in = Mill House"), g)
        self.assertEqual(a, b)
        names = sorted(g.nodes[n]["name"] for n in Runner(g).run(a)[0])
        self.assertEqual(names, ["Kenji Albright", "Marina Lindqvist", "Nadia Ferreira"])
        self.bad("op: join\ntarget: client\nfilter: name = Oskar Brandt\nreturn: interested_in.name", "b")  # lookup
        f = validate(parse("op: join\ntarget: client\nfilter: bedrooms = 4\nreturn: interested_in.name"), g)
        res, _, _ = Runner(g).run(f)
        self.assertEqual([g.nodes[n]["name"] for n in res], ["Saltmarsh Gardens", "Mill House"])

    def test_validation_normalizes(self):
        f = validate(parse("op: count\ntarget: client\nas_of: 2026-03\nfilter: budget <= 800 mil\n"
                           "filter: interested_in.near_station = Riverside station"), self.graphs["a"])
        self.assertEqual(f["as_of"], "2026-03-15")
        self.assertEqual(f["filters"][0], ["budget", "<=", 800000])
        self.assertEqual([self.graphs["a"].nodes[n]["name"] for n in f["filters"][1][2]], ["Riverside"])

    def test_every_gold_form_reproduces_the_gold_answer_on_a_perfect_graph(self):
        """parse -> validate -> run on the oracle graph agrees with heldout.evaluate on the true facts."""
        n = 0
        for q in self.heldout:
            if not q["must_parse"]:
                continue
            g = self.graphs[q["broker"]]
            form = validate(parse(lines(q["gold_form"])), g)
            runner = Runner(g)
            result, ev, _ = runner.run(form)
            shown = str(result) if isinstance(result, int) else ", ".join(runner.show(v, form["return"]) for v in result)
            self.assertTrue(score(q, f"ANSWER: {shown}")["correct"], (q["id"], shown))
            self.assertTrue(form_match(form, q["gold_form"], g, Truth(q["broker"], self.data["facts"],
                                                                     self.data["entities"]), STRICT), q["id"])
            self.assertTrue(isinstance(result, int) or ev, q["id"])  # evidence for every non-empty result
            n += 1
        self.assertEqual(n, 46)

    def test_describe_says_what_was_understood(self):
        g = self.graphs["a"]
        f = validate(parse("op: count\ntarget: client\nfilter: bedrooms = 3\nfilter: budget <= 800000"), g)
        self.assertEqual(Runner(g).describe(f), "counted clients with bedrooms = 3, budget <= $800,000, now")
        ctx, cited, understood = Runner(g).context(f)
        self.assertIn("Result: 1", ctx)
        self.assertTrue(cited)


if __name__ == "__main__":
    unittest.main()


class FormEvalEndToEndTests(unittest.TestCase):
    """form-eval with a scripted model (gold forms for held-out questions, 'op: none' otherwise): no Ollama."""

    def test_runs_and_scores(self):
        import contextlib
        import io
        import shutil
        import tempfile

        from incgraph import pipeline as P
        from incgraph.formeval import evaluate

        from .helpers import config, data_dir
        from .test_pipeline import gold_answer
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            data = d / "data"
            shutil.copytree(data_dir(), data)
            shutil.copy(ROOT / "data" / "heldout.jsonl", data / "heldout.jsonl")
            heldout = {q["question"]: q for q in map(json.loads, (data / "heldout.jsonl").read_text().splitlines())}
            original = {(P.BROKER_NAMES[q["broker"]], q["question"]): q for q in P.load_data(data)["questions"]}
            cfg = config(d)
            llm, provider = oracle_llm()
            inner = provider.fn

            def fn(messages):
                text = messages[-1]["content"]
                if text.startswith("Fill this form"):
                    q = heldout.get(text.rsplit("Question: ", 1)[1])
                    return lines(q["gold_form"]) if q else "op: none"
                if "\nQuestion: " in text:
                    broker = messages[0]["content"].split("assistant of ", 1)[1].split(",", 1)[0]
                    question = text.rsplit("\nQuestion: ", 1)[1]
                    q = heldout.get(question) if question in heldout and heldout[question]["broker"] == \
                        next(k for k, v in P.BROKER_NAMES.items() if v == broker) else original.get((broker, question))
                    return gold_answer(q) if q else "ANSWER: I don't know"
                return inner(messages)

            provider.fn = fn
            with contextlib.redirect_stdout(io.StringIO()):
                P.run(cfg, llm, cfg["run"]["results_dir"], data)
                m = evaluate(cfg, llm, data, cfg["run"]["results_dir"])
            h = m["heldout"]
            self.assertEqual(h["form_exact_match"], 1.0)
            self.assertEqual(h["fallback_rate"], 0.0)
            self.assertEqual(h["false_trigger_rate"], 0.0)
            self.assertEqual(h["form"]["accuracy"], 1.0)
            self.assertEqual(h["wrong_but_confident"], 0)
            self.assertEqual(m["original"]["n"], 81)
            self.assertTrue((Path(cfg["run"]["results_dir"]) / "form" / "heldout_answers.jsonl").exists())
