"""Controlled inert-prefix sweep with intact and post-block ablated arms."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_dilution.generation import generate_dataset
from attention_dilution.prompts import (
    BENIGN_SEED_PASSAGE, adaptive_batch_size, build_filler, wrap_prompt,
)
from attention_dilution.shared import (
    MODEL_NAME, CONTEXT_BUDGET, check_direction_metadata, directional_ablation,
    format_chat_prompt, load_core_pool, load_model, prepare_run, record_environment,
    slug_from_model, summarize_responses, validate_context_lengths, write_json,
)

SWEEP_HARMFUL_N = 100
DEFAULT_LENGTHS = [0, 128, 512, 1024, 2048, 4096, 8192, 16384]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--refusal-dir", default=None)
    p.add_argument("--ablation-layer", type=int, default=None)
    p.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    p.add_argument("--harmful-n", type=int, default=SWEEP_HARMFUL_N)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--arms", choices=["both", "baseline", "ablated"], default="both")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    validate_context_lengths(args.lengths)
    pool = load_core_pool("test", "advbench")
    if not 0 < args.harmful_n <= len(pool):
        raise ValueError(f"--harmful-n must be in [1, {len(pool)}] for the held-out split")
    harmful = pool[:args.harmful_n]
    arms = ["baseline", "ablated"] if args.arms == "both" else [args.arms]
    artifacts, direction, layer, meta = (), None, None, None
    if "ablated" in arms:
        if not args.refusal_dir:
            raise ValueError("--refusal-dir is required for the ablated arm")
        refusal_dir = Path(args.refusal_dir)
        meta_path = refusal_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        check_direction_metadata(meta, args.model, args.enable_thinking)
        layer = args.ablation_layer if args.ablation_layer is not None else meta["canonical_layer"]
        artifacts = (refusal_dir / "d_hat_all_layers.pt", meta_path)
    out_dir = prepare_run(
        "context", args, {"harmful": harmful, "filler": BENIGN_SEED_PASSAGE},
        artifacts=artifacts,
    )
    import torch
    torch.manual_seed(args.seed)
    if artifacts:
        bank = torch.load(artifacts[0], map_location="cpu", weights_only=True)
        if not 0 <= layer < bank.shape[0]:
            raise ValueError("Ablation layer outside direction bank")
        direction = bank[layer].float()
        if not torch.isclose(direction.norm(), torch.tensor(1.0), atol=1e-4):
            raise ValueError("Direction bank contains a non-unit vector")
    model, tok, device = load_model(args.model, args.dtype, args.revision)
    record_environment(out_dir, model, tok)
    if meta and meta.get("model_revision") and meta["model_revision"] != getattr(model.config, "_commit_hash", None):
        raise ValueError("Direction and evaluation model revisions differ")
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        "model": args.model, "ablation_layer": layer, "lengths": args.lengths,
        "harmful_n": len(harmful), "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "seed": args.seed,
        "intervention": "decoder block output", "cells": {},
    }
    for length in args.lengths:
        filler = build_filler(tok, length)
        actual = len(tok(filler, add_special_tokens=False)["input_ids"]) if filler else 0
        formatted = [format_chat_prompt(tok, wrap_prompt(filler, p),
                                         enable_thinking=args.enable_thinking) for p in harmful]
        batch_size = args.batch_size if args.batch_size is not None else adaptive_batch_size(length)
        for arm in arms:
            label = f"{arm}_L{length}"
            intervention = directional_ablation(model, direction) if arm == "ablated" else nullcontext()
            with intervention:
                rows = generate_dataset(
                    model, tok, formatted, harmful, device=device, batch_size=batch_size,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    context_budget=args.context_budget, path=out_dir / (label + ".jsonl"),
                    resume=args.resume,
                )
            summary["cells"][label] = {
                "L": length, "arm": arm, "filler_tok": actual,
                "batch_size": batch_size, **summarize_responses(rows),
            }
            write_json(summary_path, summary)
            print(label, summary["cells"][label])
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
