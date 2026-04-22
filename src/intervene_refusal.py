"""Forward hooks: ablate refusal direction on harmful prompts; add it on harmless prompts."""

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
    load_harmless_prompts,
    looks_like_refusal,
    results_dir,
    strip_think_block,
)

ensure_transformers_cache_attr()

from transformer_lens import HookedTransformer  # noqa: E402

N_HELD_OUT = 16
MAX_NEW_TOKENS = 64
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


def _run_condition(
    model: HookedTransformer,
    prompts: list[str],
    fwd_hooks: list,
    device: str,
    label: str,
) -> list[dict]:
    out: list[dict] = []
    with model.hooks(fwd_hooks=fwd_hooks):
        for idx, prompt in enumerate(tqdm(prompts, desc=label)):
            text = format_chat_prompt(model.tokenizer, prompt, enable_thinking=ENABLE_THINKING)
            raw = _generate(model, text, device)
            out.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "response_raw": raw,
                    "response": strip_think_block(raw),
                    "refused": looks_like_refusal(raw),
                }
            )
    return out


def _rate(xs: list[dict]) -> float:
    return sum(1 for r in xs if r["refused"]) / len(xs) if xs else 0.0


def main() -> None:
    device, dtype = get_device()
    print(f"Device: {device}, dtype: {dtype}, model: {MODEL_NAME}")

    directions_path = results_dir() / "refusal_directions.pt"
    if not directions_path.is_file():
        raise FileNotFoundError(
            f"{directions_path} not found. Run src/extract_refusal_direction.py first."
        )
    payload = torch.load(directions_path, map_location="cpu", weights_only=False)

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
    harmful_eval = harmful_all[n_train: n_train + N_HELD_OUT]
    harmless_eval = harmless_all[n_train: n_train + N_HELD_OUT]

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
        "enable_thinking": ENABLE_THINKING,
        "refusal_dir_norm": float(add_vec.norm().item()),
        "rates": rates,
        "samples": {
            "harmful_baseline": harmful_baseline,
            "harmful_ablate": harmful_ablate,
            "harmless_baseline": harmless_baseline,
            "harmless_add": harmless_add,
        },
    }
    out_path = results_dir() / "intervention_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
