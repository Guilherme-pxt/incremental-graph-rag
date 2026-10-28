import json
import tempfile
import unittest
from pathlib import Path

from incgraph.heldout import write
from incgraph.scoring import score

from .test_pipeline import gold_answer

ROOT = Path(__file__).resolve().parent.parent


class HeldOutTests(unittest.TestCase):
    """data/heldout.jsonl is regenerated from incgraph/heldout.py and the committed facts, byte for byte."""

    def test_regenerates_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("facts.jsonl", "entities.jsonl"):
                (Path(d) / name).write_bytes((ROOT / "data" / name).read_bytes())
            counts = write(d)
            self.assertEqual((Path(d) / "heldout.jsonl").read_bytes(), (ROOT / "data" / "heldout.jsonl").read_bytes())
        self.assertGreaterEqual(counts["must_parse"], 40)
        self.assertGreaterEqual(counts["portuguese_must_parse"], 15)
        self.assertGreaterEqual(counts["questions"] - counts["must_parse"], 10)

    def test_gold_answers_score_correct(self):
        for line in (ROOT / "data" / "heldout.jsonl").read_text().splitlines():
            q = json.loads(line)
            self.assertTrue(score(q, gold_answer(q))["correct"], q["id"])
            if q["kind"] != "idk":
                self.assertFalse(score(q, "ANSWER: I don't know")["correct"], q["id"])


if __name__ == "__main__":
    unittest.main()
