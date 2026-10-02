"""Projection onto per-layer directions at request and readout token positions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_dilution.activations import collect_residuals
from attention_dilution.prompts import (
    BENIGN_SEED_PASSAGE, adaptive_batch_size, build_filler, request_token_position,
    wrap_prompt,
)
from attention_dilution.shared import (
    MODEL_NAME, CONTEXT_BUDGET, check_direction_metadata, format_chat_prompt,
    load_core_pool, load_model, prepare_run, record_environment,
    validate_context_lengths, write_json,
)

DEFAULT_LENGTHS = [0, 128, 512, 1024, 2048, 4096, 8192, 16384]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--refusal-dir", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Defaults to the selected layer in direction metadata")
    p.add_argument("--harmful-n", type=int, default=100)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def summarize(values):
    import numpy as np
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()), "std": float(v.std()), "n": len(v),
        "min": float(v.min()), "max": float(v.max()),
        "q05": float(np.quantile(v, .05)), "q25": float(np.quantile(v, .25)),
        "q50": float(np.quantile(v, .5)), "q75": float(np.quantile(v, .75)),
        "q95": float(np.quantile(v, .95)), "values": v.tolist(),
    }


def maybe_plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, position in zip(axes, ["readout", "request"]):
        for layer in summary["layers"]:
            stats = [summary["cells"][f"L{n}"]["by_position"][position][str(layer)]
                     for n in summary["lengths"] if f"L{n}" in summary["cells"]]
            lengths = [n for n in summary["lengths"] if f"L{n}" in summary["cells"]]
            means = [s["mean"] for s in stats]
            ax.plot(lengths, means, "o-", label=f"layer {layer}")
        ax.set_xscale("symlog", linthresh=128)
        ax.set_title(position + " token")
        ax.set_xlabel("Filler tokens")
        ax.axhline(0, color="black", linewidth=.5)
        ax.legend()
    axes[0].set_ylabel("Mean residual projection")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    validate_context_lengths(args.lengths)
    refusal_dir = Path(args.refusal_dir)
    meta_path = refusal_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    check_direction_metadata(meta, args.model, args.enable_thinking)
    layers = args.layers if args.layers is not None else [meta["canonical_layer"]]
    if len(set(layers)) != len(layers):
        raise ValueError("Layer indices must be distinct")
    harmful = load_core_pool("test", "advbench")
    if not 0 < args.harmful_n <= len(harmful):
        raise ValueError(f"--harmful-n must be in [1, {len(harmful)}]")
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    harmful = harmful[:args.harmful_n]
    if args.max_prompts is not None:
        harmful = harmful[:args.max_prompts]
    bank_path = refusal_dir / "d_hat_all_layers.pt"
    out_dir = prepare_run(
        "projection", args, {"harmful": harmful, "filler": BENIGN_SEED_PASSAGE},
        artifacts=(bank_path, meta_path),
    )
    import torch
    torch.manual_seed(args.seed)
    bank = torch.load(bank_path, map_location="cpu", weights_only=True).float()
    if any(not 0 <= layer < bank.shape[0] for layer in layers):
        raise ValueError("Layer indices outside direction bank")
    directions = bank[layers]
    if not torch.allclose(directions.norm(dim=-1), torch.ones(len(layers)), atol=1e-4):
        raise ValueError("Direction bank contains a non-unit vector")
    model, tok, device = load_model(args.model, args.dtype, args.revision)
    record_environment(out_dir, model, tok)
    if meta.get("model_revision") and meta["model_revision"] != getattr(model.config, "_commit_hash", None):
        raise ValueError("Direction and evaluation model revisions differ")
    summary_path = out_dir / "projection_sweep.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        "model": args.model, "layers": layers, "lengths": args.lengths,
        "harmful_n": len(harmful), "positions": {
            "request": "last token overlapping the request",
            "readout": "last token of the full chat template",
        }, "cells": {},
    }
    for length in args.lengths:
        if args.resume and f"L{length}" in summary["cells"]:
            continue
        filler = build_filler(tok, length)
        formatted = [format_chat_prompt(tok, wrap_prompt(filler, p),
                                         enable_thinking=args.enable_thinking) for p in harmful]
        positions = [request_token_position(tok, text, p) for text, p in zip(formatted, harmful)]
        bs = args.batch_size if args.batch_size is not None else adaptive_batch_size(length)
        residuals = collect_residuals(
            model, tok, formatted, layers, bs, device,
            context_budget=args.context_budget, request_positions=positions,
        )
        projections = (residuals * directions[:, None, None, :]).sum(dim=-1)
        by_position = {
            name: {str(layer): summarize(projections[slot, :, pos].tolist())
                   for slot, layer in enumerate(layers)}
            for pos, name in enumerate(["readout", "request"])
        }
        summary["cells"][f"L{length}"] = {
            "L": length, "batch_size": bs,
            "filler_tok": len(tok(filler, add_special_tokens=False)["input_ids"]) if filler else 0,
            "by_position": by_position, "by_layer": by_position["readout"],
        }
        write_json(summary_path, summary)
        del residuals, projections
    if not args.no_plot:
        maybe_plot(summary, out_dir / "projection_sweep.png")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
