"""
Experiment 3: causal intervention via the refusal direction from Experiment 2.

Two interventions on held-out prompts:
  - ABLATION (harmful prompts): at every block's hook_resid_post, project the
    layer-L normalized refusal direction out of the residual stream. Expectation:
    refusal rate drops (the model complies).
  - ADDITION (harmless prompts): at layer L's hook_resid_post, add the raw
    mean-difference vector (mean_harmful - mean_harmless) to the residual stream.
    Expectation: refusal rate rises (refusal is induced).

If both effects appear, the layer-L direction is not merely correlated with
refusal: it is a causal lever on it.
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
        if isinstance(rp, dict):
            t = rp.get("rope_theta")
            if t is not None:
                return float(t)
        if rp is not None and hasattr(rp, "get"):
            t = rp.get("rope_theta")  # type: ignore[union-attr]
            if t is not None:
                return float(t)
        return 1_000_000.0

    Qwen2Config.rope_theta = property(_rope_theta_prop)  # type: ignore[assignment]
    Qwen2Config._tl_rope_theta_shim = True


_patch_qwen2_rope_theta_for_transformer_lens()

from transformer_lens import HookedTransformer

from utils import (
    get_device,
    load_harmful_prompts,
    load_harmless_prompts,
    looks_like_refusal,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DIRECTIONS_PATH = Path("results/refusal_directions.pt")
OUTPUT_PATH = Path("results/intervention_results.json")

N_HELD_OUT = 16
MAX_NEW_TOKENS = 48


def _format(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


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


def _run_condition(
    model: HookedTransformer,
    prompts: list[str],
    fwd_hooks: list,
    device: str,
    label: str,
) -> list[dict]:
    results: list[dict] = []
    with model.hooks(fwd_hooks=fwd_hooks):
        for idx, prompt in enumerate(tqdm(prompts, desc=label)):
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
    return results


def _rate(xs: list[dict]) -> float:
    return sum(1 for r in xs if r["refused"]) / len(xs) if xs else 0.0


def main() -> None:
    device, dtype = get_device()
    print(f"Device: {device}, dtype: {dtype}")

    if not DIRECTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"{DIRECTIONS_PATH} not found. Run src/extract_refusal_direction.py first."
        )
    payload = torch.load(DIRECTIONS_PATH, map_location="cpu", weights_only=False)

    separation = payload["separation_scores"]
    best_layer = int(torch.argmax(separation).item())
    n_train = int(payload["n_prompts"])
    print(
        f"Best layer from extraction: {best_layer} "
        f"(separation={float(separation[best_layer]):.3f}); training n={n_train}"
    )

    u = payload["refusal_dirs_normalized"][best_layer].to(device=device, dtype=dtype)
    add_vec = payload["refusal_dirs"][best_layer].to(device=device, dtype=dtype)
    print(f"||refusal_dir[L={best_layer}]|| = {add_vec.norm().item():.3f}")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device=device, dtype=dtype, trust_remote_code=True
    )
    model.eval()

    harmful_all = load_harmful_prompts(n_train + N_HELD_OUT)
    harmless_all = load_harmless_prompts(n_train + N_HELD_OUT)
    harmful_eval = harmful_all[n_train : n_train + N_HELD_OUT]
    harmless_eval = harmless_all[n_train : n_train + N_HELD_OUT]

    def ablate_hook(resid, hook):
        proj = (resid * u).sum(dim=-1, keepdim=True)
        return resid - proj * u

    def add_hook(resid, hook):
        return resid + add_vec

    n_layers = model.cfg.n_layers
    ablate_hooks = [(f"blocks.{L}.hook_resid_post", ablate_hook) for L in range(n_layers)]
    add_hooks = [(f"blocks.{best_layer}.hook_resid_post", add_hook)]

    print("\n--- Baseline: harmful, no intervention ---")
    harmful_baseline = _run_condition(model, harmful_eval, [], device, "harmful/base")
    print("\n--- Ablate refusal direction on harmful (expect LESS refusal) ---")
    harmful_ablate = _run_condition(model, harmful_eval, ablate_hooks, device, "harmful/ablate")

    print("\n--- Baseline: harmless, no intervention ---")
    harmless_baseline = _run_condition(model, harmless_eval, [], device, "harmless/base")
    print("\n--- Add refusal direction on harmless (expect MORE refusal) ---")
    harmless_add = _run_condition(model, harmless_eval, add_hooks, device, "harmless/add")

    rates = {
        "harmful_baseline": _rate(harmful_baseline),
        "harmful_ablate": _rate(harmful_ablate),
        "harmless_baseline": _rate(harmless_baseline),
        "harmless_add": _rate(harmless_add),
    }

    print("\n=== Refusal rates ===")
    for k, v in rates.items():
        print(f"  {k:22s}: {v:.1%}")
    print(
        "\nDeltas: "
        f"harmful {rates['harmful_ablate'] - rates['harmful_baseline']:+.1%}, "
        f"harmless {rates['harmless_add'] - rates['harmless_baseline']:+.1%}"
    )

    summary = {
        "model": MODEL_NAME,
        "best_layer": best_layer,
        "n_held_out": N_HELD_OUT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "refusal_dir_norm": float(add_vec.norm().item()),
        "rates": rates,
        "samples": {
            "harmful_baseline": harmful_baseline,
            "harmful_ablate": harmful_ablate,
            "harmless_baseline": harmless_baseline,
            "harmless_add": harmless_add,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
