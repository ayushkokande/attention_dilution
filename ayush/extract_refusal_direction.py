"""Compute per-layer refusal directions from residual activations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import (
    MODEL_NAME,
    ensure_transformers_cache_attr,
    format_chat_prompt,
    get_device,
    load_harmful_prompts,
    load_harmless_prompts,
    results_dir,
)

ensure_transformers_cache_attr()

from transformer_lens import HookedTransformer  # noqa: E402

N_PROMPTS = 32
ENABLE_THINKING = False


def _load_hooked_transformer(
    model_name: str, device: str, dtype: torch.dtype
) -> HookedTransformer:
    kwargs = dict(model_name=model_name, device=device, dtype=dtype, trust_remote_code=True)
    try:
        return HookedTransformer.from_pretrained(**kwargs)
    except Exception as e:
        print(f"from_pretrained failed ({e!r}); retrying with from_pretrained_no_processing.")
        return HookedTransformer.from_pretrained_no_processing(**kwargs)


@torch.inference_mode()
def _resid_post_last_token(model: HookedTransformer, prompt_text: str) -> torch.Tensor:
    _, cache = model.run_with_cache(prompt_text, return_type=None)
    n_layers = model.cfg.n_layers
    layers = []
    for L in range(n_layers):
        h = cache[f"blocks.{L}.hook_resid_post"][:, -1, :].detach().cpu().float()
        layers.append(h.squeeze(0))
    return torch.stack(layers, dim=0)  # [n_layers, d_model]


def _separation_score(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    norm_dirs: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-8
    n_layers = norm_dirs.shape[0]
    scores = []
    for L in range(n_layers):
        d = norm_dirs[L]
        ph = (harmful_acts[:, L, :] * d).sum(dim=-1)
        ps = (harmless_acts[:, L, :] * d).sum(dim=-1)
        mh, sh = ph.mean(), ph.std(unbiased=False)
        ms, ss = ps.mean(), ps.std(unbiased=False)
        denom = sh + ss
        scores.append(((mh - ms) / (denom + eps)).item())
    return torch.tensor(scores, dtype=torch.float32)


def main() -> None:
    device, dtype = get_device()
    print(f"Device: {device}, dtype: {dtype}, model: {MODEL_NAME}")

    model = _load_hooked_transformer(MODEL_NAME, device, dtype)
    model.eval()
    print(f"n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")

    harmful_prompts = load_harmful_prompts(N_PROMPTS)
    harmless_prompts = load_harmless_prompts(N_PROMPTS)
    if len(harmless_prompts) < N_PROMPTS:
        raise RuntimeError(
            f"Only collected {len(harmless_prompts)} harmless prompts "
            f"(need {N_PROMPTS}); relax filters or use more Alpaca rows."
        )

    print("\nSample harmful prompts:")
    for p in harmful_prompts[:3]:
        print(f"  - {p[:120]}{'...' if len(p) > 120 else ''}")
    print("\nSample harmless prompts:")
    for p in harmless_prompts[:3]:
        print(f"  - {p[:120]}{'...' if len(p) > 120 else ''}")

    harmful_list: list[torch.Tensor] = []
    for prompt in tqdm(harmful_prompts, desc="Harmful prompts"):
        text = format_chat_prompt(model.tokenizer, prompt, enable_thinking=ENABLE_THINKING)
        harmful_list.append(_resid_post_last_token(model, text))

    harmless_list: list[torch.Tensor] = []
    for prompt in tqdm(harmless_prompts, desc="Harmless prompts"):
        text = format_chat_prompt(model.tokenizer, prompt, enable_thinking=ENABLE_THINKING)
        harmless_list.append(_resid_post_last_token(model, text))

    harmful_acts = torch.stack(harmful_list, dim=0)  # [N, L, D]
    harmless_acts = torch.stack(harmless_list, dim=0)

    mean_harmful = harmful_acts.mean(dim=0)
    mean_harmless = harmless_acts.mean(dim=0)
    refusal_dirs = mean_harmful - mean_harmless
    refusal_dirs_normalized = F.normalize(refusal_dirs, p=2, dim=-1)

    separation_scores = _separation_score(
        harmful_acts, harmless_acts, refusal_dirs_normalized
    )

    top5 = torch.topk(separation_scores, k=min(5, separation_scores.numel()))
    print("\nTop 5 layers by separation score:")
    for rank, (val, idx) in enumerate(zip(top5.values.tolist(), top5.indices.tolist()), start=1):
        print(f"  {rank}. layer {idx}: {val:.4f}")

    out_dir = results_dir()
    directions_path = out_dir / "refusal_directions.pt"
    activations_path = out_dir / "activations.pt"

    torch.save(
        {
            "refusal_dirs": refusal_dirs,
            "refusal_dirs_normalized": refusal_dirs_normalized,
            "mean_harmful": mean_harmful,
            "mean_harmless": mean_harmless,
            "separation_scores": separation_scores,
            "model_name": MODEL_NAME,
            "enable_thinking": ENABLE_THINKING,
            "n_prompts": N_PROMPTS,
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
        },
        directions_path,
    )
    print(f"\nSaved directions to {directions_path}")

    torch.save(
        {"harmful": harmful_acts, "harmless": harmless_acts},
        activations_path,
    )
    print(f"Saved activations to {activations_path}")


if __name__ == "__main__":
    main()
