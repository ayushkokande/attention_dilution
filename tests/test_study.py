"""Checks for target coverage, paired measurements, and stale human labels."""

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from attention_dilution.review import export_packet, summarize_packet
from attention_dilution.study import (
    BENIGN_TASKS, build_context, context_cells, exclude_direction_overlap, validate_requests,
)
from experiment_9.projection_sweep import paired_changes


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "</user><assistant>"


class StudyTests(unittest.TestCase):
    def test_target_span_disambiguates_a_request_repeated_in_background(self):
        target = BENIGN_TASKS["garden"][0]
        args = Namespace(formats=["ordered-tasks"], lengths=[256], backgrounds=["garden"],
                         target_position="first", enable_thinking=False)
        request = {"id": "one", "prompt": target, "kind": "harmless", "source": "test"}
        _, _, cases = next(context_cells(CharacterTokenizer(), [request], args))
        case = cases[0]
        self.assertGreater(case["chat_prompt"].count(target), 1)
        self.assertEqual(case["chat_prompt"][case["request_position"]], target[-1])
        self.assertLess(case["request_position"], case["chat_prompt"].rfind(target))

    def test_task_budget_keeps_whole_requests_and_only_target_instruction(self):
        case = build_context(CharacterTokenizer(), "The target.", 128, "tasks", "garden", "last")
        self.assertLessEqual(case["background_tokens"], 128)
        self.assertGreater(case["n_background_tasks"], 0)
        self.assertIn("Answer only task TARGET", case["input_prompt"])
        lo, hi = case["target_span"]
        self.assertEqual(case["input_prompt"][lo:hi], "The target.")
        empty = build_context(CharacterTokenizer(), "The target.", 0, "tasks", "garden", "last")
        self.assertIn("Answer only task TARGET", empty["input_prompt"])
        self.assertEqual(empty["n_background_tasks"], 0)

    def test_empty_controls_are_not_duplicated_for_each_background(self):
        args = Namespace(formats=["prefix", "quoted"], lengths=[128, 0],
                         backgrounds=["garden", "apennines"], target_position="last", enable_thinking=False)
        row = {"id": "one", "prompt": "The target.", "kind": "harmless", "source": "test"}
        cells = list(context_cells(CharacterTokenizer(), [row], args))
        self.assertEqual(sum(cell["L"] == 0 for _, cell, _ in cells), 2)
        self.assertEqual(cells[0][1]["L"], 0)

    def test_direction_overlap_checks_saved_strings_and_records_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = {"harmful_train": ["Repeated   instruction"], "harmless_train": [],
                      "harmful_validation": ["Validation instruction"], "harmless_validation": []}
            Path(tmp, "run.json").write_text(json.dumps({"fingerprint": {"stage": "direction", "inputs": inputs}}))
            rows = [{"id": "a", "prompt": "repeated instruction"},
                    {"id": "b", "prompt": "Validation instruction"}, {"id": "c", "prompt": "New target"}]
            kept, excluded = exclude_direction_overlap(rows, tmp)
            self.assertEqual([r["id"] for r in kept], ["c"])
            self.assertEqual({r["overlap_with"] for r in excluded}, {"harmful_train", "harmful_validation"})
            kept, _ = exclude_direction_overlap(rows, tmp, include_validation=False)
            self.assertEqual([r["id"] for r in kept], ["b", "c"])

    def test_external_pool_requires_kind_and_unique_ids(self):
        row = {"id": "a", "prompt": "A target", "kind": "harmful", "source": "test"}
        validate_requests([row])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_requests([row, row])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            validate_requests([{"prompt": "A target"}])

    def test_negative_mean_does_not_establish_individual_sign_flips(self):
        stats = paired_changes([1, -10], [2, -1])
        self.assertEqual(stats["n_positive_to_negative"], 0)
        self.assertEqual(stats["n_negative_to_positive"], 0)
        self.assertEqual(stats["delta_values"], [-1, -9])
        self.assertEqual(stats["mean_delta"], -5)

    def test_review_does_not_turn_a_benign_first_answer_into_unsafe_compliance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "results.jsonl"
            source.write_text(json.dumps({"prompt": "A later target.",
                                          "response": "1. The first benign answer.", "status": "ok",
                                          "case": {"kind": "harmful", "condition": {"arm": "baseline"}}}) + "\n")
            packet = root / "review"
            export_packet([source], packet)
            ungraded = summarize_packet(packet, [packet / "review.csv"])
            self.assertEqual(ungraded["cells"][0]["unsafe_response_rate_bounds"], [0, 1])
            copies = []
            for reviewer in ("A", "B"):
                with (packet / "review.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                rows[0].update(label="target_not_addressed", evidence="The first benign answer.", reviewer=reviewer)
                path = packet / (reviewer + ".csv")
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
                copies.append(path)
            result = summarize_packet(packet, copies)
            cell = result["cells"][0]
            self.assertEqual(cell["n_target_not_addressed"], 1)
            self.assertEqual(cell["n_target_decision_unresolved"], 1)
            self.assertEqual(cell["label_counts"], {"target_not_addressed": 1})
            self.assertNotIn("arm", rows[0])
            self.assertEqual(cell["n_truncation_unknown"], 1)
            # A copied answer or reused reviewer cannot manufacture consensus.
            with self.assertRaisesRegex(ValueError, "same reviewer"):
                summarize_packet(packet, [copies[0], copies[0]])
            rows[0]["response"] = "Different answer"
            with copies[1].open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "text was changed"):
                summarize_packet(packet, copies)

    def test_review_records_delayed_refusal_phrase_without_assigning_a_human_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "results.jsonl"
            path.write_text(json.dumps({"prompt": "The target.", "response": "x" * 210 + " I cannot help."}) + "\n")
            packet = root / "review"
            records = export_packet([path], packet)
            self.assertFalse(records[0]["phrase_refused_prefix"])
            self.assertTrue(records[0]["phrase_present_anywhere"])
            result = summarize_packet(packet, [packet / "review.csv"])
            self.assertEqual(result["cells"][0]["n_phrase_window_disagreements"], 1)
            self.assertEqual(result["cells"][0]["n_unresolved"], 1)


if __name__ == "__main__":
    unittest.main()
