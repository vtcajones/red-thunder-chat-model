"""Training-data invariants that protect contextual labels and evaluation integrity."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("chat_context_model", ROOT / "scripts/chat-context-model.py")
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)
CATALOGUE = ROOT / "SpheroMcQueen/ChatResponses.json"
DATA = ROOT / "docs/training/chat-context-teacher-v1.json"
EXPANDED_DATA = ROOT / "docs/training/chat-context-teacher-v2.json"


class ContextModelDataTests(unittest.TestCase):
    def setUp(self):
        self.records, self.cases, self.data = MODEL.read_inputs(CATALOGUE, DATA)
        self.by_id = {case["id"]: case for case in self.cases}

    def test_same_latest_words_require_different_contextual_choices(self):
        mater = self.by_id["test-mater-him-1"]
        doc = self.by_id["test-doc-him-1"]
        unresolved = self.by_id["test-unresolved-pronoun-1"]
        self.assertEqual(mater["user"], doc["user"])
        self.assertEqual(doc["user"], unresolved["user"])
        self.assertIn(694, mater["preferred"])
        self.assertIn(745, doc["preferred"])
        self.assertEqual(set(unresolved["preferred"]), {0})
        self.assertNotEqual(mater["context"], doc["context"])

    def test_recent_spoken_duplicates_are_excluded_even_with_different_ids(self):
        case = self.by_id["test-avoid-repeat-1"]
        # Records 247 and 780 both say "I am speed." History played only 780.
        self.assertNotIn(247, case["eligible"])
        self.assertNotIn(780, case["eligible"])
        self.assertIn(782, case["preferred"])
        self.assertIn(424, case["preferred"])

    def test_no_match_keeps_the_full_recorded_vocabulary_available(self):
        case = self.by_id["test-weather-1"]
        self.assertEqual(set(case["preferred"]), {0})
        self.assertEqual(len(case["eligible"]), 319)
        self.assertEqual(len(self.records), 319)

    def test_duplicate_conversation_cannot_leak_into_a_test_group(self):
        data = json.loads(DATA.read_text())
        original = data["groups"][0]
        data["groups"].append({**original, "id": "leaked-test", "split": "test"})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "leaks across splits"):
                MODEL.read_inputs(CATALOGUE, path)

    def test_holdout_reference_ids_have_no_positive_training_labels(self):
        train = {ident for case in self.cases if case["split"] == "train" for ident in case["preferred"]}
        holdout = {ident for case in self.cases if case["holdoutTopic"] for ident in case["preferred"]}
        self.assertTrue(holdout)
        self.assertTrue(train.isdisjoint(holdout))

    def test_previous_test_cases_remain_outside_new_training_and_evaluation(self):
        _, expanded, _ = MODEL.read_inputs(CATALOGUE, EXPANDED_DATA)
        old = {(json.dumps(c["history"]), c["user"]) for c in self.cases if c["split"] == "test"}
        new = {(json.dumps(c["history"]), c["user"]) for c in expanded}
        self.assertTrue(old.isdisjoint(new))

    def test_same_latest_question_distinguishes_win_loss_and_tie(self):
        _, cases, _ = MODEL.read_inputs(CATALOGUE, EXPANDED_DATA)
        cases = {c["id"]: c for c in cases}
        win = cases["v2-outcome-0-win-1"]
        loss = cases["v2-outcome-0-lose-1"]
        tie = cases["v2-outcome-0-tie-1"]
        self.assertEqual(win["user"], loss["user"])
        self.assertEqual(loss["user"], tie["user"])
        self.assertIn(350, win["preferred"])
        self.assertIn(313, loss["preferred"])
        self.assertEqual(set(tie["preferred"]), {673})
        self.assertIn(673, win["rejected"])
        self.assertIn(673, loss["rejected"])

    def test_named_character_can_be_answerable_or_require_silence(self):
        _, cases, _ = MODEL.read_inputs(CATALOGUE, EXPANDED_DATA)
        cases = {c["id"]: c for c in cases}
        known = cases["v2-known-694-1"]
        unknown = cases["v2-unknown-694-1"]
        self.assertEqual(known["history"], unknown["history"])
        self.assertIn(694, known["preferred"])
        self.assertEqual(set(unknown["preferred"]), {0})
        self.assertIn(694, unknown["rejected"])

    def test_dialogue_refinement_preserves_the_unread_final_test_and_validation(self):
        parent = json.loads(EXPANDED_DATA.read_text())
        refined = json.loads((ROOT / "docs/training/chat-context-teacher-v3.json").read_text())
        MODEL.read_inputs(CATALOGUE, refined)
        for split in ("test", "validation"):
            self.assertEqual([g for g in parent["groups"] if g["split"] == split],
                             [g for g in refined["groups"] if g["split"] == split])


if __name__ == "__main__":
    unittest.main()
