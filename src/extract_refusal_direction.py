"""
Experiment 2: extract a per-layer refusal direction (harmful vs harmless mean difference)
from last-token residual stream activations (Arditi et al. 2024 style).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from tqdm.auto import tqdm

# transformer_lens expects transformers.TRANSFORMERS_CACHE (removed in transformers v5+).
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    _hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    transformers.TRANSFORMERS_CACHE = os.path.join(_hf_home, "hub")


def _patch_qwen2_rope_theta_for_transformer_lens() -> None:
    """HF transformers v5+ Qwen2Config uses rope_parameters; transformer_lens reads rope_theta."""
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

from utils import get_device, load_harmful_prompts, load_harmless_prompts

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_PROMPTS = 32
OUTPUT_PATH = "results/refusal_directions.pt"
ACTIVATIONS_PATH = "results/activations.pt"


def _load_hooked_transformer(model_name: str, device: str, dtype: torch.dtype) -> HookedTransformer:
    kwargs = dict(model_name=model_name, device=device, dtype=dtype, trust_remote_code=True)
    try:
        return HookedTransformer.from_pretrained(**kwargs)
    except Exception as e:
        print(f"from_pretrained failed ({e!r}); retrying with from_pretrained_no_processing.")
        return HookedTransformer.from_pretrained_no_processing(**kwargs)


def _format_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def _resid_post_last_token(
    model: HookedTransformer, prompt_text: str
) -> torch.Tensor:
    """Residual stream after each block at the last prompt token: [n_layers, d_model] on CPU."""
    _, cache = model.run_with_cache(prompt_text, return_type=None)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
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
    """Per-layer d-prime style score; harmful_acts: [Nh, L, D], norm_dirs: [L, D]."""
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
    print(f"Device: {device}, dtype: {dtype}")

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
        text = _format_prompt(model.tokenizer, prompt)
        harmful_list.append(_resid_post_last_token(model, text))

    harmless_list: list[torch.Tensor] = []
    for prompt in tqdm(harmless_prompts, desc="Harmless prompts"):
        text = _format_prompt(model.tokenizer, prompt)
        harmless_list.append(_resid_post_last_token(model, text))

    harmful_acts = torch.stack(harmful_list, dim=0)  # [N, L, D]
    harmless_acts = torch.stack(harmless_list, dim=0)

    mean_harmful = harmful_acts.mean(dim=0)
    mean_harmless = harmless_acts.mean(dim=0)
    refusal_dirs = mean_harmful - mean_harmless
    refusal_dirs_normalized = F.normalize(refusal_dirs, p=2, dim=-1)

    separation_scores = _separation_score(harmful_acts, harmless_acts, refusal_dirs_normalized)

    top5 = torch.topk(separation_scores, k=min(5, separation_scores.numel()))
    print("\nTop 5 layers by separation score:")
    for rank, (val, idx) in enumerate(zip(top5.values.tolist(), top5.indices.tolist()), start=1):
        print(f"  {rank}. layer {idx}: {val:.4f}")

    out_dir = Path(OUTPUT_PATH).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "refusal_dirs": refusal_dirs,
            "refusal_dirs_normalized": refusal_dirs_normalized,
            "mean_harmful": mean_harmful,
            "mean_harmless": mean_harmless,
            "separation_scores": separation_scores,
            "model_name": MODEL_NAME,
            "n_prompts": N_PROMPTS,
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
        },
        OUTPUT_PATH,
    )
    print(f"\nSaved directions to {OUTPUT_PATH}")

    torch.save(
        {"harmful": harmful_acts, "harmless": harmless_acts},
        ACTIVATIONS_PATH,
    )
    print(f"Saved activations to {ACTIVATIONS_PATH}")


if __name__ == "__main__":
    main()
