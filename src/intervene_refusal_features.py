"""Zero out specific transcoder features during generation on harmful prompts.

Reads a feature list from ``results/<slug>/refusal_features.json``. The file is
produced by hand after inspecting the attribution graphs from
``src/trace_refusal_circuit.py`` and has the shape::

    {
      "features": [
        {"layer": 18, "pos_from_end": -1, "feature_idx": 12345, "label": "refusal_1"},
        ...
      ],
      "value": 0.0
    }

``pos_from_end`` is interpreted relative to the tokenized prompt (``-1`` =
last prompt token). The intervention uses an open-ended slice starting at that
position, so the features stay pinned to ``value`` for every generated token.
If ``features`` is empty, the script just reports the baseline to confirm the
pipeline works.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm

from utils import (
    MODEL_NAME,
    TRANSCODER_SET,
    ensure_transformers_cache_attr,
    format_chat_prompt,
    get_device,
    load_harmful_prompts,
    looks_like_refusal,
    results_dir,
    strip_think_block,
)

ensure_transformers_cache_attr()

from circuit_tracer import ReplacementModel  # noqa: E402

MAX_NEW_TOKENS = 64


def _load_feature_config(path: Path) -> tuple[list[dict], float]:
    if not path.is_file():
        print(
            f"[warn] {path} not found — running baseline only. "
            "Create it after inspecting the attribution graphs."
        )
        return [], 0.0
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    feats = list(cfg.get("features", []))
    value = float(cfg.get("value", 0.0))
    return feats, value


def _resolve_intervention_tuples(
    model: ReplacementModel,
    formatted_prompt: str,
    features: list[dict],
    value: float,
):
    """Convert ``pos_from_end`` entries to an open-ended slice over generation."""
    if not features:
        return []
    ids = model.tokenizer(formatted_prompt, return_tensors="pt")["input_ids"]
    n_tokens = int(ids.shape[1])
    tuples = []
    for feat in features:
        layer = int(feat["layer"])
        feature_idx = int(feat["feature_idx"])
        pos_from_end = int(feat.get("pos_from_end", -1))
        abs_pos = n_tokens + pos_from_end if pos_from_end < 0 else pos_from_end
        abs_pos = max(0, min(n_tokens - 1, abs_pos))
        tuples.append((layer, slice(abs_pos, None, None), feature_idx, float(value)))
    return tuples


def _resolve_dtype(device: str, override: Optional[str]) -> torch.dtype:
    if override is not None:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[override]
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-held-out", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--feature-config", type=Path, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument(
        "--backend",
        choices=["transformerlens", "nnsight"],
        default="transformerlens",
    )
    args = parser.parse_args()

    rdir = results_dir()
    config_path = args.feature_config or (rdir / "refusal_features.json")
    features, value = _load_feature_config(config_path)

    device, _ = get_device()
    dtype = _resolve_dtype(device, args.dtype)
    print(f"Device: {device}, dtype: {dtype}, model: {MODEL_NAME}")

    directions_path = rdir / "refusal_directions.pt"
    if directions_path.is_file():
        payload = torch.load(directions_path, map_location="cpu", weights_only=False)
        n_train = int(payload["n_prompts"])
    else:
        n_train = 32
        print(f"[warn] {directions_path} missing; defaulting n_train={n_train}")

    harmful_eval = load_harmful_prompts(n_train + args.n_held_out)[
        n_train: n_train + args.n_held_out
    ]

    print(f"Loading ReplacementModel with transcoders {TRANSCODER_SET} ...")
    model = ReplacementModel.from_pretrained(
        MODEL_NAME,
        TRANSCODER_SET,
        dtype=dtype,
        backend=args.backend,
    )

    baseline_rows: list[dict] = []
    feature_rows: list[dict] = []

    for idx, prompt in enumerate(tqdm(harmful_eval, desc="harmful/feature-ablate")):
        formatted = format_chat_prompt(model.tokenizer, prompt, enable_thinking=False)

        baseline_raw = model.feature_intervention_generate(
            formatted,
            [],
            do_sample=False,
            verbose=False,
            max_new_tokens=args.max_new_tokens,
        )[0]
        baseline_rows.append(
            {
                "index": idx,
                "prompt": prompt,
                "response_raw": baseline_raw,
                "response": strip_think_block(baseline_raw),
                "refused": looks_like_refusal(baseline_raw),
            }
        )

        tuples = _resolve_intervention_tuples(model, formatted, features, value)
        intervened_raw = model.feature_intervention_generate(
            formatted,
            tuples,
            do_sample=False,
            verbose=False,
            max_new_tokens=args.max_new_tokens,
        )[0]
        feature_rows.append(
            {
                "index": idx,
                "prompt": prompt,
                "n_features_ablated": len(tuples),
                "response_raw": intervened_raw,
                "response": strip_think_block(intervened_raw),
                "refused": looks_like_refusal(intervened_raw),
            }
        )

    def _rate(xs: list[dict]) -> float:
        return sum(1 for r in xs if r["refused"]) / len(xs) if xs else 0.0

    rates = {
        "harmful_baseline": _rate(baseline_rows),
        "harmful_feature_ablate": _rate(feature_rows),
    }
    print("\n=== Refusal rates ===")
    for k, v in rates.items():
        print(f"  {k:26s}: {v:.1%}")
    print(
        f"\nDelta (feature_ablate - baseline): "
        f"{rates['harmful_feature_ablate'] - rates['harmful_baseline']:+.1%}"
    )

    out_path = rdir / "intervention_feature_results.json"
    summary = {
        "model": MODEL_NAME,
        "transcoders": TRANSCODER_SET,
        "feature_config": str(config_path.relative_to(rdir.parent))
        if config_path.exists()
        else None,
        "features": features,
        "ablation_value": value,
        "n_held_out": len(harmful_eval),
        "max_new_tokens": args.max_new_tokens,
        "rates": rates,
        "samples": {
            "harmful_baseline": baseline_rows,
            "harmful_feature_ablate": feature_rows,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
