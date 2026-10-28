import unittest
from unittest import mock

from incgraph.batch import louvain
from incgraph.extract import clean, extraction_messages, parse_lines, thread_context
from incgraph.graph import Graph
from incgraph.llm import LLM, REQUIRED_OPTIONS, ChatResult, LLMError, OllamaProvider, ScriptedProvider, choice_probs
from incgraph.resolve import Resolver, building_match, first_compatible, person_level
from incgraph.values import money_amounts, months_in, norm_name, parse_money, parse_month, small_ints

EP_CHAT = {"id": "a-010", "broker": "a", "ts": "2026-02-01", "kind": "chat", "thread": "a-t01",
           "contact": "Kate Moreau", "speaker": "client", "text": "We can go up to 950k now."}
EP_NOTE = {"id": "a-011", "broker": "a", "ts": "2026-02-02", "kind": "note", "thread": None, "contact": None,
           "speaker": "broker", "text": "Showed the Aurora to Kate Moreau today."}


class ValueTests(unittest.TestCase):
    def test_money(self):
        self.assertEqual(parse_money("$950,000"), 950000)
        self.assertEqual(parse_money("950k"), 950000)
        self.assertEqual(parse_money("$1013k"), 1013000)
        self.assertEqual(parse_money("1.2 million"), 1200000)
        self.assertEqual(parse_money(640000), 640000)
        self.assertIsNone(parse_money("3"))
        self.assertEqual(money_amounts("from $640k to 1,089,000 dollars in 2027"), [640000, 1089000])

    def test_months(self):
        self.assertEqual(parse_month("June 2027"), "2027-06")
        self.assertEqual(parse_month("Jun 2027"), "2027-06")
        self.assertEqual(parse_month("2027-06"), "2027-06")
        self.assertEqual(parse_month("06/2027"), "2027-06")
        self.assertEqual(months_in("on March 21, 2026 and in Dec 2027"), ["2026-03", "2027-12"])

    def test_small_ints_ignore_dates_amounts_and_ids(self):
        self.assertEqual(small_ints("3 [a-017]"), [3])
        self.assertEqual(small_ints("three bedrooms"), [3])
        self.assertEqual(small_ints("2027-06 and $950k"), [])

    def test_norm_name(self):
        self.assertEqual(norm_name("The Aurora"), "aurora")
        self.assertEqual(norm_name("Ms. Moreau"), "moreau")
        self.assertEqual(norm_name("Jonah P."), "jonah p")


class ExtractTests(unittest.TestCase):
    def test_parse_lines_tolerates_type_prefixes_and_aliases(self):
        raw = parse_lines("Elm Tower | building price | 300k\nElm Tower | alias | ET\nnot a fact\nNONE")
        out = clean(raw)
        self.assertEqual(out["facts"], [{"subject": "Elm Tower", "predicate": "price", "object": 300000}])
        self.assertEqual(out["entities"][0]["aliases"], ["ET"])
        self.assertEqual(out["entities"][0]["type"], "building")

    def test_clean_drops_unknown_predicates_and_bad_values(self):
        raw = {"facts": [{"subject": "Jo", "predicate": "salary", "object": 1},
                         {"subject": "Jo", "predicate": "bedrooms", "object": "lots"},
                         {"subject": "Jo", "predicate": "deadline", "object": "May 2027"},
                         {"subject": "Jo", "predicate": "interested_in", "object": "Elm Tower"}]}
        out = clean(raw)
        self.assertEqual([f["predicate"] for f in out["facts"]], ["deadline", "interested_in"])
        self.assertEqual(out["facts"][0]["object"], "2027-05")
        types = {e["name"]: e["type"] for e in out["entities"]}
        self.assertEqual(types, {"Jo": "client", "Elm Tower": "building"})

    def test_prompt_has_context_and_new_message(self):
        ctx = [dict(EP_CHAT, id="a-005", text="Hi, this is Katherine Moreau.")]
        msgs = extraction_messages(EP_CHAT, ctx, "Lena Ferraz")
        body = msgs[-1]["content"]
        self.assertLess(body.index("[a-005]"), body.index("NEW message:"))
        self.assertIn('saved as "Kate Moreau"', body)
        self.assertTrue(body.startswith("Write one fact per line"))  # fixed prefix first, for prompt reuse

    def test_thread_context(self):
        eps = [dict(EP_CHAT, id=f"a-00{i}") for i in range(1, 5)] + [dict(EP_NOTE, id="a-003b")]
        self.assertEqual([e["id"] for e in thread_context(eps, eps[3])], ["a-002", "a-003"])
        self.assertEqual(thread_context(eps, EP_NOTE), [])


class GraphTests(unittest.TestCase):
    def test_temporal_graph_closes_instead_of_overwriting(self):
        g = Graph("a", temporal=True)
        c = g.add_node("client", "Kate Moreau", "a-001")
        self.assertEqual(g.assert_fact(c, "budget", 800000, "a-001", "2026-01-01"), "added")
        self.assertEqual(g.assert_fact(c, "budget", 800000, "a-002", "2026-01-05"), "restated")
        self.assertEqual(g.assert_fact(c, "budget", 950000, "a-003", "2026-02-01"), "replaced")
        old, new = g.edges
        self.assertEqual((old["valid_to"], old["invalidated_by"], old["episodes"]), ("2026-02-01", "a-003",
                                                                                    ["a-001", "a-002"]))
        self.assertIsNone(new["valid_to"])
        self.assertEqual(g.open_edges(c, "budget"), [new])
        self.assertTrue(g.fact_text(old).startswith("[a-001][a-002] superseded, held 2026-01-01 to 2026-02-01: "))
        self.assertTrue(g.fact_text(new).startswith("[a-003] current since 2026-02-01: Kate Moreau | budget"))

    def test_multi_valued_facts_accumulate(self):
        g = Graph("a", temporal=True)
        c = g.add_node("client", "Kate", "a-001")
        b1, b2 = g.add_node("building", "Elm", "a-001"), g.add_node("building", "Oak", "a-001")
        g.assert_fact(c, "visited", b1, "a-001", "2026-01-01")
        g.assert_fact(c, "visited", b2, "a-002", "2026-01-02")
        self.assertEqual([e["valid_to"] for e in g.edges], [None, None])

    def test_batch_graph_keeps_both_values(self):
        g = Graph("a", temporal=False)
        c = g.add_node("client", "Kate", "a-001")
        g.assert_fact(c, "budget", 800000, "a-001", "2026-01-01")
        g.assert_fact(c, "budget", 950000, "a-003", "2026-02-01")
        self.assertEqual([e["valid_to"] for e in g.edges], [None, None])
        self.assertEqual(g.fact_text(g.edges[1]), "[a-003] Kate | budget | $950,000 | stated 2026-02-01")

    def test_louvain_two_triangles(self):
        e = [("a", "b", 1), ("b", "c", 1), ("a", "c", 1), ("d", "e", 1), ("e", "f", 1), ("d", "f", 1), ("c", "d", 1)]
        part = louvain(list("abcdef"), e)
        self.assertEqual(len({part["a"], part["b"], part["c"]}), 1)
        self.assertEqual(len({part["d"], part["e"], part["f"]}), 1)
        self.assertNotEqual(part["a"], part["d"])


class ResolveTests(unittest.TestCase):
    def test_name_rules(self):
        self.assertTrue(first_compatible("kate", "katherine"))
        self.assertTrue(first_compatible("dani", "daniel"))
        self.assertFalse(first_compatible("fil", "felipe"))  # not in the table: left to the LLM
        self.assertEqual(person_level("Jonah Pellegrini", "Jonah P"), 2)
        self.assertEqual(person_level("Kate Moreau", "Katherine Moreau"), 3)
        self.assertEqual(person_level("Fil Arantes", "Felipe Arantes"), 1)
        self.assertEqual(person_level("Marina Lindqvist", "Marina Okafor"), 0)
        self.assertEqual(person_level("Daniel W.", "Daniel Reyes"), 0)
        self.assertTrue(building_match("Aurora Res.", "Aurora Residences"))
        self.assertTrue(building_match("the Lofts", "Quarry Lofts"))
        self.assertFalse(building_match("N1", "Northbank One"))

    def test_thread_key_wins(self):
        g = Graph("a", temporal=True)
        r = Resolver(g, llm=None)
        first = r.resolve("Katherine Moreau", "client", dict(EP_CHAT, id="a-001"), [])
        again = r.resolve("Kate Moreau", "client", EP_CHAT, [])
        self.assertEqual(first, again)
        self.assertEqual([x["route"] for x in r.log], ["thread-new", "thread"])
        self.assertIn("Kate Moreau", g.names(first))

    def test_ambiguous_building_goes_to_the_llm_and_is_not_learned(self):
        g = Graph("a", temporal=True)
        res_ = g.add_node("building", "Aurora Residences", "a-001")
        park = g.add_node("building", "Aurora Park", "a-002")
        seen = []

        def fn(messages):
            seen.append(messages[-1]["content"])
            return ChatResult("2", 0.1, top_logprobs=[{"2": -0.05, "1": -3.0, "0": -5.0}])

        llm = LLM(ScriptedProvider(fn))
        r = Resolver(g, llm)
        nid = r.resolve("the Aurora", "building", EP_NOTE, [])
        self.assertEqual(nid, park)
        self.assertIn("1. Aurora Residences", seen[0])
        self.assertIn("2. Aurora Park", seen[0])
        self.assertEqual(llm.ledger[-1]["tag"], "inc:merge")
        self.assertNotIn("the Aurora", g.names(park))  # an ambiguous form never becomes an alias
        self.assertNotIn("the Aurora", g.names(res_))

    def test_llm_zero_creates_a_new_node(self):
        g = Graph("c", temporal=True)
        g.add_node("client", "Daniel Reyes", "c-001")
        g.add_node("client", "Daniel Weber", "c-002")
        llm = LLM(ScriptedProvider(lambda m: ChatResult("0", 0.1, top_logprobs=[{"0": -0.1, "1": -3.0}])))
        nid = Resolver(g, llm).resolve("Daniel", "client", dict(EP_NOTE, id="c-003", broker="c"), [])
        self.assertEqual(len(g.nodes), 3)
        self.assertEqual(g.nodes[nid]["name"], "Daniel")

    def test_without_llm_the_oldest_candidate_wins(self):
        g = Graph("a", temporal=True)
        first = g.add_node("building", "Aurora Residences", "a-001")
        g.add_node("building", "Aurora Park", "a-002")
        r = Resolver(g, llm=None, use_llm=False)
        self.assertEqual(r.resolve("the Aurora", "building", EP_NOTE, []), first)
        self.assertEqual(r.log[-1]["route"], "first")


class LLMTests(unittest.TestCase):
    def test_every_request_pins_keep_alive_and_options(self):
        body = OllamaProvider().payload([{"role": "user", "content": "x"}], 50, False, 0)
        self.assertEqual(body["keep_alive"], -1)
        self.assertEqual(body["model"], "qwen2.5-coder:7b")
        self.assertEqual({k: body["options"][k] for k in REQUIRED_OPTIONS}, {"num_ctx": 24576, "temperature": 0,
                                                                             "seed": 42})
        self.assertEqual(body["options"]["num_predict"], 50)

    def test_backoff_then_error_without_hammering(self):
        p = OllamaProvider(backoff_s=(1.0, 2.0))
        bad = mock.Mock(status_code=503, text="busy")
        with mock.patch("incgraph.llm.requests.post", return_value=bad) as post, \
                mock.patch("incgraph.llm.time.sleep") as sleep:
            with self.assertRaises(LLMError):
                p.chat([{"role": "user", "content": "x"}])
        self.assertEqual(post.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.0, 2.0])

    def test_only_api_chat_is_called(self):
        ok = mock.Mock(status_code=200)
        ok.json.return_value = {"message": {"content": "hi"}, "prompt_eval_count": 5, "eval_count": 1}
        with mock.patch("incgraph.llm.requests.post", return_value=ok) as post:
            OllamaProvider("m", "http://h:1").chat([{"role": "user", "content": "x"}])
        self.assertEqual(post.call_args.args[0], "http://h:1/api/chat")

    def test_cache_replays_measured_cost(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            calls = []
            prov = ScriptedProvider(lambda m: calls.append(1) or ChatResult("x", 2.5, prompt_tokens=100,
                                                                             completion_tokens=7))
            llm = LLM(prov, Path(d) / "c.sqlite")
            llm.chat([{"role": "user", "content": "q"}], tag="t", ref="r")
            again = llm.chat([{"role": "user", "content": "q"}], tag="t", ref="r")
            self.assertEqual(len(calls), 1)
            self.assertTrue(again.cached)
            self.assertEqual((again.latency_s, again.prompt_tokens), (2.5, 100))
            self.assertEqual([e["cached"] for e in llm.ledger], [False, True])
            llm.allow_miss = False
            with self.assertRaises(LLMError):
                llm.chat([{"role": "user", "content": "other"}])
            llm.close()

    def test_choice_probs(self):
        p = choice_probs([{"1": -0.1, "2": -2.0, "x": -0.5}], ["0", "1", "2"])
        self.assertGreater(p["1"], p["2"])
        self.assertEqual(p["0"], 0.0)
        self.assertEqual(choice_probs([], ["0", "1"], "1"), {"0": 0.0, "1": 1.0})


if __name__ == "__main__":
    unittest.main()


class AggregationQueryTests(unittest.TestCase):
    """The aggregation step of query.py on a hand-built temporal graph."""

    def setUp(self):
        from incgraph.query import Query
        g = Graph("a", temporal=True)
        self.kate = g.add_node("client", "Kate Moreau", "a-001")
        self.ruth = g.add_node("client", "Ruth Castellano", "a-002")
        self.jo = g.add_node("client", "Jonah Pellegrini", "a-003")
        g.assert_fact(self.kate, "budget", 800000, "a-001", "2026-01-10")
        g.assert_fact(self.kate, "budget", 950000, "a-010", "2026-03-01")   # raised later
        g.assert_fact(self.ruth, "budget", 900000, "a-002", "2026-01-12")
        g.assert_fact(self.jo, "budget", 950000, "a-003", "2026-01-15")
        for c, n in ((self.kate, 2), (self.ruth, 3), (self.jo, 3)):
            g.assert_fact(c, "bedrooms", n, "a-004", "2026-01-20")
        g.assert_fact(self.jo, "bedrooms", 2, "a-011", "2026-04-01")         # changed later
        self.q = Query(g)

    def test_value_at_respects_validity(self):
        self.assertEqual(self.q.value_at(self.kate, "budget")["value"], 950000)
        self.assertEqual(self.q.value_at(self.kate, "budget", "2026-02-01")["value"], 800000)
        self.assertEqual(self.q.value_at(self.kate, "budget", "2026-03-01")["value"], 950000)  # valid_to is exclusive
        self.assertIsNone(self.q.value_at(self.kate, "budget", "2026-01-01"))  # before it was stated

    def test_count_where_now_and_as_of(self):
        now, ev = self.q.count_where("client", "bedrooms", lambda v: v == 3)
        self.assertEqual(now, [self.ruth])
        self.assertEqual([e["episodes"] for e in ev], [["a-004"]])
        then, _ = self.q.count_where("client", "bedrooms", lambda v: v == 3, "2026-02-15")
        self.assertEqual(then, [self.ruth, self.jo])

    def test_extreme_returns_ties_and_uses_the_date(self):
        top, ev = self.q.extreme("client", "budget", highest=True)
        self.assertEqual(top, [self.kate, self.jo])
        self.assertEqual(len(ev), 2)
        self.assertEqual(self.q.extreme("client", "budget", highest=True, at="2026-02-01")[0], [self.jo])
        self.assertEqual(self.q.extreme("client", "budget", highest=False, at="2026-02-01")[0], [self.kate])

    def test_recognized_forms(self):
        out = self.q.run("How many of my clients want 3 bedrooms right now?")
        self.assertEqual(out[2], "1 (Ruth Castellano)")
        self.assertTrue(self.q.run("How many of my clients want 3 bedrooms on February 15, 2026?")[2].startswith("2 "))
        self.assertTrue(self.q.run("How many of my clients have a budget above $900,000 right now?")[2].startswith("2 "))
        self.assertTrue(self.q.run("How many of my clients have a budget under 900k?")[2].startswith("0"))
        self.assertEqual(self.q.run("Which of my clients has the lowest budget right now?")[2], "Ruth Castellano")
        self.assertIsNone(self.q.run("How many buildings deliver in 2027?"))  # not a covered form: retrieval path
