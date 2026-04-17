"""
Rerun ONLY the harmful/ablate condition from Experiment 3 with a larger
max_new_tokens budget, so we can read what the ablated model actually produces
past the 48-token cap used in intervene_refusal.py.

Saves to results/intervention_results_full.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import transformers
from tqdm.auto import tqdm

if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    _hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    transformers.TRANSFORMERS_CACHE = os.path.join(_hf_home, "hub")


def _patch_qwen2_rope_theta_for_transformer_lens() -> None:
    try:
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    except ImportError:
        return
    if getattr(Qwen2Config, "_tl_rope_theta_shim", False):
        return

    def _rope_theta_prop(self) -> float:
        d = getattr(self, "__dict__", {})
        if d.get("rope_theta") is not None:
            return float(d["rope_theta"])
        rp = getattr(self, "rope_parameters", None)
        if isinstance(rp, dict) and rp.get("rope_theta") is not None:
            return float(rp["rope_theta"])
        return 1_000_000.0

    Qwen2Config.rope_theta = property(_rope_theta_prop)  # type: ignore[assignment]
    Qwen2Config._tl_rope_theta_shim = True


_patch_qwen2_rope_theta_for_transformer_lens()

from transformer_lens import HookedTransformer

from utils import get_device, load_harmful_prompts, looks_like_refusal

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DIRECTIONS_PATH = Path("results/refusal_directions.pt")
OUTPUT_PATH = Path("results/intervention_results_full.json")

N_HELD_OUT = 16
MAX_NEW_TOKENS = 2000


@torch.inference_mode()
def _generate(model: HookedTransformer, prompt_text: str, device: str) -> str:
    input_ids = model.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    out = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        prepend_bos=False,
        verbose=False,
        return_type="tokens",
    )
    new_ids = out[0, input_ids.shape[1]:]
    return model.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _format(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    device, dtype = get_device()
    print(f"Device: {device}, dtype: {dtype}")

    payload = torch.load(DIRECTIONS_PATH, map_location="cpu", weights_only=False)
    best_layer = int(torch.argmax(payload["separation_scores"]).item())
    n_train = int(payload["n_prompts"])
    u = payload["refusal_dirs_normalized"][best_layer].to(device=device, dtype=dtype)
    print(f"Ablating refusal direction at layer {best_layer} (n_train={n_train}).")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device=device, dtype=dtype, trust_remote_code=True
    )
    model.eval()

    harmful_eval = load_harmful_prompts(n_train + N_HELD_OUT)[n_train : n_train + N_HELD_OUT]

    def ablate_hook(resid, hook):
        proj = (resid * u).sum(dim=-1, keepdim=True)
        return resid - proj * u

    fwd_hooks = [(f"blocks.{L}.hook_resid_post", ablate_hook) for L in range(model.cfg.n_layers)]

    results: list[dict] = []
    with model.hooks(fwd_hooks=fwd_hooks):
        for idx, prompt in enumerate(tqdm(harmful_eval, desc="harmful/ablate(long)")):
            text = _format(model.tokenizer, prompt)
            response = _generate(model, text, device)
            results.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "response": response,
                    "refused": looks_like_refusal(response),
                }
            )

    rate = sum(1 for r in results if r["refused"]) / len(results)
    print(f"\nRefusal rate (ablated, long): {rate:.1%}")

    summary = {
        "model": MODEL_NAME,
        "best_layer": best_layer,
        "n_held_out": N_HELD_OUT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "condition": "harmful_ablate_long",
        "refusal_rate": rate,
        "samples": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
