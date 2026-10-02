"""CPU checks with a tiny decoder; no Hugging Face downloads."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from attention_dilution.activations import collect_residuals
from attention_dilution.generation import generate_dataset
from attention_dilution.shared import directional_ablation, summarize_responses


if torch is not None:
    class Encoding(dict):
        def to(self, device):
            return Encoding({key: value.to(device) for key, value in self.items()})

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, texts, **kwargs):
            width = max(map(len, texts))
            ids, masks = [], []
            for text in texts:
                padding = width - len(text)
                ids.append([0] * padding + [ord(c) % 32 for c in text])
                masks.append([0] * padding + [1] * len(text))
            return Encoding(input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks))

        def batch_decode(self, ids, **kwargs):
            responses = {1: "I cannot assist.", 2: "An answer.", 3: ""}
            return [responses[int(row[0])] for row in ids]

    class Block(torch.nn.Module):
        def __init__(self, amount):
            super().__init__()
            self.amount = amount

        def forward(self, hidden):
            return hidden + self.amount

    class TinyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(1), Block(2)])
            self.fail = False

        def forward(self, input_ids, attention_mask, use_cache):
            hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
            for layer in self.layers:
                hidden = layer(hidden)
                if self.fail:
                    raise RuntimeError("test forward failure")
            return SimpleNamespace(last_hidden_state=hidden)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyBase()
            self.config = SimpleNamespace(hidden_size=2)
            self.generate_calls = 0
            self.fail_generation = False

        def forward(self, **kwargs):
            raise AssertionError("Residual capture must not construct language-model logits")

        def generate(self, input_ids, attention_mask, **kwargs):
            self.generate_calls += 1
            if self.fail_generation:
                raise RuntimeError("test generation failure")
            new = torch.tensor([[1, 99, 0, 0]] * len(input_ids))
            return torch.cat([input_ids, new], dim=1)


@unittest.skipIf(torch is None, "CPU decoder checks require PyTorch")
class DecoderTests(unittest.TestCase):
    def test_capture_uses_base_decoder_and_correct_left_padded_positions(self):
        model, tokenizer = TinyModel(), TinyTokenizer()
        residuals = collect_residuals(model, tokenizer, ["abcTAIL", "xTAIL"], [0, 1],
                                     2, "cpu", request_positions=[2, 0])
        self.assertEqual(tuple(residuals.shape), (2, 2, 2, 2))
        self.assertEqual(residuals[0, 0, 1, 0].item(), ord("c") % 32 + 1)
        self.assertEqual(residuals[1, 1, 1, 0].item(), ord("x") % 32 + 3)
        self.assertEqual(residuals[1, 1, 0, 0].item(), ord("L") % 32 + 3)
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_capture_removes_hooks_on_failure(self):
        model = TinyModel()
        model.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "forward failure"):
            collect_residuals(model, TinyTokenizer(), ["abc"], [0, 1], 1, "cpu")
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_ablation_changes_only_the_selected_direction_and_cleans_up(self):
        model = TinyModel()
        enc = TinyTokenizer()(["a"])
        with directional_ablation(model, torch.tensor([1., 0.])):
            hidden = model.model(**enc, use_cache=False).last_hidden_state
            self.assertEqual(hidden[0, 0, 0].item(), 0)
            self.assertEqual(hidden[0, 0, 1].item(), ord("a") % 32 + 3)
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))
        hidden = model.model(**enc, use_cache=False).last_hidden_state
        self.assertEqual(hidden[0, 0, 0].item(), ord("a") % 32 + 3)

    def test_generation_resume_does_not_repeat_completed_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TinyModel()
            kwargs = dict(device="cpu", batch_size=2, max_new_tokens=4,
                          temperature=0, context_budget=128, path=Path(tmp) / "rows.jsonl")
            rows = generate_dataset(model, TinyTokenizer(), ["abc", "xy"], ["a", "b"], **kwargs)
            self.assertEqual(summarize_responses(rows)["refusal_rate"], 1)
            self.assertFalse(rows[0]["truncated"])
            resumed = generate_dataset(model, TinyTokenizer(), ["abc", "xy"], ["a", "b"],
                                       resume=True, **kwargs)
            self.assertEqual(len(resumed), 2)
            self.assertEqual(model.generate_calls, 1)

    def test_failed_generation_has_no_refusal_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TinyModel()
            model.fail_generation = True
            rows = generate_dataset(model, TinyTokenizer(), ["a"], ["a"], device="cpu",
                                    batch_size=1, max_new_tokens=4, temperature=0,
                                    context_budget=128, path=Path(tmp) / "rows.jsonl")
            self.assertEqual(summarize_responses(rows)["n_failed"], 1)
            self.assertIsNone(summarize_responses(rows)["refusal_rate"])


if __name__ == "__main__":
    unittest.main()
