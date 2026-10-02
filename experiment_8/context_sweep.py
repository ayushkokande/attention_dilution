"""Target-aware context sweep with explicit ablation and steering controls."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_dilution.generation import generate_dataset
from attention_dilution.prompts import adaptive_batch_size
from attention_dilution.study import (
    BACKGROUNDS, BENIGN_TASKS, add_study_arguments, context_cells, load_requests,
)
from attention_dilution.shared import (
    MODEL_NAME, CONTEXT_BUDGET, check_direction_metadata, directional_ablation,
    directional_addition, load_model, prepare_run, record_environment,
    summarize_responses, validate_context_lengths, write_json,
)

SWEEP_HARMFUL_N = 100
DEFAULT_LENGTHS = [0, 512, 4096]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--refusal-dir", default=None)
    p.add_argument("--ablation-layer", "--direction-layer", type=int, default=None,
                   help="Layer where the vector was extracted; ablation still acts at every block")
    p.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    p.add_argument("--harmful-n", "--n-prompts", type=int, default=SWEEP_HARMFUL_N)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--arms", choices=["both", "baseline", "ablated", "steered"], default="baseline")
    p.add_argument("--steering-alpha", type=float, help="Activation units; required for steered arm")
    p.add_argument("--steering-layer", type=int, help="One block output; defaults to vector's source layer")
    p.add_argument("--random-direction", action="store_true", help="Seeded random unit-vector control")
    add_study_arguments(p)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    validate_context_lengths(args.lengths)
    if args.arms == "steered" and args.steering_alpha is None:
        raise ValueError("--steering-alpha is required for the steered arm")
    if args.steering_alpha is not None and not math.isfinite(args.steering_alpha):
        raise ValueError("Steering coefficient must be finite")
    if args.arms != "steered" and (args.steering_alpha is not None or args.steering_layer is not None):
        raise ValueError("Steering settings require --arms steered")
    if args.arms == "baseline" and args.random_direction:
        raise ValueError("A random direction requires an intervention arm")
    requests, excluded = load_requests(args)
    targets = [r["prompt"] for r in requests]
    arms = ["baseline", "ablated"] if args.arms == "both" else [args.arms]
    artifacts, direction, layer, meta = (), None, None, None
    if any(arm != "baseline" for arm in arms) and not args.refusal_dir:
        raise ValueError("--refusal-dir is required for an intervention arm")
    if args.refusal_dir:
        refusal_dir = Path(args.refusal_dir)
        meta_path = refusal_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        check_direction_metadata(meta, args.model, args.enable_thinking)
        layer = args.ablation_layer if args.ablation_layer is not None else meta["canonical_layer"]
        artifacts = (refusal_dir / "d_hat_all_layers.pt", meta_path, refusal_dir / "run.json")
    out_dir = prepare_run(
        "context", args, {"requests": requests, "excluded": excluded,
                          "backgrounds": BACKGROUNDS, "benign_tasks": BENIGN_TASKS},
        artifacts=artifacts,
    )
    import torch
    torch.manual_seed(args.seed)
    if any(arm != "baseline" for arm in arms):
        bank = torch.load(artifacts[0], map_location="cpu", weights_only=True)
        if not 0 <= layer < bank.shape[0]:
            raise ValueError("Ablation layer outside direction bank")
        direction = bank[layer].float()
        if not torch.isclose(direction.norm(), torch.tensor(1.0), atol=1e-4):
            raise ValueError("Direction bank contains a non-unit vector")
        if args.random_direction:
            direction = torch.randn(direction.shape, generator=torch.Generator().manual_seed(args.seed))
            direction = direction / direction.norm()
    model, tok, device = load_model(args.model, args.dtype, args.revision)
    record_environment(out_dir, model, tok)
    if meta and meta.get("model_revision") and meta["model_revision"] != getattr(model.config, "_commit_hash", None):
        raise ValueError("Direction and evaluation model revisions differ")
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        "model": args.model, "direction_source_layer": layer, "lengths": args.lengths,
        "n_requests": len(requests), "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "seed": args.seed,
        "intervention": "decoder block output", "split": args.split,
        "target_grading": "pending; phrase-refusal rates are diagnostic only", "cells": {},
    }
    steering_layer = args.steering_layer if args.steering_layer is not None else layer
    for cell_label, cell, cases in context_cells(tok, requests, args):
        length = cell["L"]
        formatted = [case["chat_prompt"] for case in cases]
        batch_size = args.batch_size if args.batch_size is not None else adaptive_batch_size(length)
        for arm in arms:
            label = f"{arm}_{cell_label}"
            intervention = nullcontext()
            if arm == "ablated":
                intervention = directional_ablation(model, direction)
            elif arm == "steered":
                intervention = directional_addition(model, direction, steering_layer, args.steering_alpha)
            metadata = [{**{k: v for k, v in case.items() if k != "chat_prompt"},
                        "condition": {**cell, "arm": arm,
                        "random_direction": args.random_direction if arm != "baseline" else False,
                        "direction_source_layer": layer if arm != "baseline" else None,
                        "steering_layer": steering_layer if arm == "steered" else None,
                        "steering_alpha": args.steering_alpha if arm == "steered" else None}}
                        for case in cases]
            with intervention:
                rows = generate_dataset(
                    model, tok, formatted, targets, device=device, batch_size=batch_size,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    context_budget=args.context_budget, path=out_dir / (label + ".jsonl"),
                    resume=args.resume, case_metadata=metadata,
                )
            summary["cells"][label] = {
                **cell, "arm": arm, "background_tokens": cases[0]["background_tokens"],
                "n_background_tasks": cases[0]["n_background_tasks"],
                "batch_size": batch_size, **summarize_responses(rows),
                "by_kind": {kind: summarize_responses([r for r in rows if r["case"]["kind"] == kind])
                            for kind in sorted({r["kind"] for r in requests})},
            }
            write_json(summary_path, summary)
            print(label, summary["cells"][label])
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
