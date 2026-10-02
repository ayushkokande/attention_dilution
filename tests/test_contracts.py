"""Regression checks for failures found in the original experiment pipeline."""

import ast
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from attention_dilution.prompts import request_token_position
from attention_dilution.shared import (
    assert_disjoint_pools, check_context_budget, check_direction_metadata,
    load_split_manifest, looks_like_refusal, prepare_run, strip_think_block,
    summarize_responses, validate_ranges, wilson_interval,
)

ROOT = Path(__file__).resolve().parent.parent


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class ContractTests(unittest.TestCase):
    def test_all_tracked_python_sources_compile(self):
        # Exclude generated runs and virtual environments. This catches the
        # duplicate baseline's late __future__ import without importing torch.
        paths = list((ROOT / "attention_dilution").glob("*.py"))
        paths += list(ROOT.glob("experiment_*/*.py"))
        paths += list((ROOT / "Suraj").glob("*.py"))
        for path in paths:
            with self.subTest(path=path):
                compile(path.read_text(), str(path), "exec")

    def test_baseline_has_one_entry_point_and_one_main(self):
        tree = ast.parse((ROOT / "experiment_1/baseline_benchmark.py").read_text())
        self.assertEqual(sum(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body), 1)

    def test_historical_leaky_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            validate_ranges({"train": [0, 256], "test": [200, 250]})

    def test_core_manifest_is_separated(self):
        core = load_split_manifest()
        self.assertEqual(core["alpaca_post_filter"]["train"], [0, 256])
        self.assertEqual(core["alpaca_post_filter"]["test"], [256, 356])

    def test_strings_are_checked_when_row_numbering_differs(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint_pools({"train": ["A  repeated prompt"], "test": ["a repeated prompt"]})

    def test_request_and_readout_are_different_positions(self):
        text = "<user>request</user><assistant>"
        position = request_token_position(CharacterTokenizer(), text, "request")
        self.assertEqual(text[position], "t")
        self.assertLess(position, len(text) - 1)

    def test_context_budget_includes_template_and_output(self):
        check_context_budget(32000, 256, 32768)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            check_context_budget(32768 + 20, 256, 32768)

    def test_failure_is_not_scored_as_non_refusal(self):
        stats = summarize_responses([
            {"status": "ok", "response": "I cannot help", "refused": True},
            {"status": "error", "refused": None},
            {"status": "empty", "response": "", "refused": False},
        ])
        self.assertEqual(stats["n_evaluated"], 1)
        self.assertEqual(stats["refusal_rate"], 1.0)
        self.assertEqual(stats["n_failed"], 1)
        self.assertEqual(stats["n_empty"], 1)
        self.assertIsNone(summarize_responses([])["refusal_rate"])

    def test_phrase_score_does_not_establish_safe_completion(self):
        # This deliberately demonstrates the heuristic's limitation.
        self.assertTrue(looks_like_refusal("I cannot help with that, but here is an answer."))
        self.assertEqual(strip_think_block("<think>x</think> answer"), "answer")
        self.assertEqual(strip_think_block("Quoted <think>x</think> text"), "Quoted <think>x</think> text")

    def test_wilson_interval_handles_boundary_samples(self):
        self.assertIsNone(wilson_interval(0, 0))
        lo, hi = wilson_interval(0, 100)
        self.assertAlmostEqual(lo, 0)
        self.assertAlmostEqual(hi, .0369934982)
        lo, hi = wilson_interval(100, 100)
        self.assertLess(lo, 1)
        self.assertAlmostEqual(hi, 1)

    def test_resume_rejects_changed_prompts_or_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "direction.pt"
            artifact.write_bytes(b"first")
            args = Namespace(model="test/model", output_dir=str(Path(tmp) / "run"), resume=False)
            prepare_run("test", args, {"prompts": ["first"]}, artifacts=(artifact,))
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare_run("test", args, {"prompts": ["first"]}, artifacts=(artifact,))
            args.resume = True
            prepare_run("test", args, {"prompts": ["first"]}, artifacts=(artifact,))
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare_run("test", args, {"prompts": ["other"]}, artifacts=(artifact,))
            artifact.write_bytes(b"second")
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare_run("test", args, {"prompts": ["first"]}, artifacts=(artifact,))

    def test_stale_direction_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "template settings"):
            check_direction_metadata({"model": "qwen"}, "qwen", False)
        check_direction_metadata({"model": "qwen", "enable_thinking": False}, "qwen", False)
        with self.assertRaisesRegex(ValueError, "thinking settings"):
            check_direction_metadata({"model": "qwen", "enable_thinking": True}, "qwen", False)

    def test_core_command_help_does_not_load_model_dependencies(self):
        for stage in ["baseline", "direction", "context", "projection"]:
            with self.subTest(stage=stage):
                result = subprocess.run([sys.executable, "-m", "attention_dilution", stage, "--help"],
                                        cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--model", result.stdout)


if __name__ == "__main__":
    unittest.main()
