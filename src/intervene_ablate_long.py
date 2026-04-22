"""Harmful-prompt ablation with a longer generation budget."""

from __future__ import annotations

import json

import torch
from tqdm.auto import tqdm

from utils import (
    MODEL_NAME,
    ensure_transformers_cache_attr,
    format_chat_prompt,
    get_device,
    load_harmful_prompts,
    looks_like_refusal,
    results_dir,
    strip_think_block,
)

ensure_transformers_cache_attr()

from transformer_lens import HookedTransformer  # noqa: E402

N_HELD_OUT = 16
MAX_NEW_TOKENS = 512
ENABLE_THINKING = False


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


def main() -> None:
    device, dtype = get_device()
    print(f"Device: {device}, dtype: {dtype}, model: {MODEL_NAME}")

    rdir = results_dir()
    directions_path = rdir / "refusal_directions.pt"
    payload = torch.load(directions_path, map_location="cpu", weights_only=False)
    best_layer = int(torch.argmax(payload["separation_scores"]).item())
    n_train = int(payload["n_prompts"])
    u = payload["refusal_dirs_normalized"][best_layer].to(device=device, dtype=dtype)
    print(f"Ablating refusal direction at layer {best_layer} (n_train={n_train}).")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device=device, dtype=dtype, trust_remote_code=True
    )
    model.eval()

    harmful_eval = load_harmful_prompts(n_train + N_HELD_OUT)[n_train: n_train + N_HELD_OUT]

    def ablate_hook(resid, hook):
        proj = (resid * u).sum(dim=-1, keepdim=True)
        return resid - proj * u

    fwd_hooks = [
        (f"blocks.{L}.hook_resid_post", ablate_hook) for L in range(model.cfg.n_layers)
    ]

    results: list[dict] = []
    with model.hooks(fwd_hooks=fwd_hooks):
        for idx, prompt in enumerate(tqdm(harmful_eval, desc="harmful/ablate(long)")):
            text = format_chat_prompt(model.tokenizer, prompt, enable_thinking=ENABLE_THINKING)
            raw = _generate(model, text, device)
            results.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "response_raw": raw,
                    "response": strip_think_block(raw),
                    "refused": looks_like_refusal(raw),
                }
            )

    rate = sum(1 for r in results if r["refused"]) / len(results)
    print(f"\nRefusal rate (ablated, long): {rate:.1%}")

    out_path = rdir / "intervention_results_full.json"
    summary = {
        "model": MODEL_NAME,
        "best_layer": best_layer,
        "n_held_out": N_HELD_OUT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "enable_thinking": ENABLE_THINKING,
        "condition": "harmful_ablate_long",
        "refusal_rate": rate,
        "samples": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
