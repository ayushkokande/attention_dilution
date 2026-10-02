"""Projection onto per-layer directions at request and readout token positions."""

from __future__ import annotations

import argparse
import json
import random
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_dilution.activations import collect_residuals
from attention_dilution.prompts import adaptive_batch_size
from attention_dilution.study import (
    BACKGROUNDS, BENIGN_TASKS, add_study_arguments, context_cells, load_requests,
)
from attention_dilution.shared import (
    MODEL_NAME, CONTEXT_BUDGET, check_direction_metadata,
    load_model, prepare_run, record_environment,
    validate_context_lengths, write_json,
)

DEFAULT_LENGTHS = [0, 512, 4096]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--refusal-dir", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Defaults to the selected layer in direction metadata")
    p.add_argument("--harmful-n", "--n-prompts", type=int, default=100)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    add_study_arguments(p)
    return p.parse_args()


def summarize(values):
    v = sorted(float(x) for x in values)
    if not v or not all(math.isfinite(x) for x in v):
        raise ValueError("Nonempty finite projections required")

    def quantile(p):
        index = p * (len(v) - 1)
        lo, hi = math.floor(index), math.ceil(index)
        return v[lo] + (v[hi] - v[lo]) * (index - lo)

    return {
        "mean": statistics.mean(v), "std": statistics.pstdev(v), "n": len(v),
        "min": v[0], "max": v[-1],
        "q05": quantile(.05), "q25": quantile(.25), "q50": quantile(.5),
        "q75": quantile(.75), "q95": quantile(.95), "values": list(values),
    }


def paired_changes(values, baseline, *, seed=0):
    """Paired bootstrap resamples requests, never repeated context cells."""
    if len(values) != len(baseline) or not values:
        raise ValueError("Nonempty paired projections of equal length required")
    delta = [v - b for v, b in zip(values, baseline)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(2000))
    return {"delta_values": delta, "mean_delta": sum(delta) / len(delta),
            "mean_delta_ci95": [means[49], means[1949]],
            "n_positive_to_negative": sum(b > 0 and v < 0 for b, v in zip(baseline, values)),
            "n_negative_to_positive": sum(b < 0 and v > 0 for b, v in zip(baseline, values)),
            "bootstrap_unit": "request", "bootstrap_resamples": 2000}


def maybe_plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, position in zip(axes, ["readout", "request"]):
        groups = {(c["format"], c["background"]) for c in summary["cells"].values() if c["L"]}
        if not groups:
            groups = {(c["format"], "empty") for c in summary["cells"].values()}
        for fmt, background in sorted(groups):
            cells = sorted((c for c in summary["cells"].values()
                            if c["format"] == fmt and c["background"] in ("empty", background)),
                           key=lambda c: c["L"])
            for layer in summary["layers"]:
                means = [c["by_position"][position][str(layer)]["mean"] for c in cells]
                ax.plot([c["L"] for c in cells], means, "o-", label=f"{fmt}/{background}/L{layer}")
        ax.set_xscale("symlog", linthresh=128)
        ax.set_title(position + " token")
        ax.set_xlabel("Background token allowance")
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
    requests, excluded = load_requests(args)
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    if args.max_prompts is not None:
        requests = requests[:args.max_prompts]
    bank_path = refusal_dir / "d_hat_all_layers.pt"
    out_dir = prepare_run(
        "projection", args, {"requests": requests, "excluded": excluded,
                             "backgrounds": BACKGROUNDS, "benign_tasks": BENIGN_TASKS},
        artifacts=(bank_path, meta_path, refusal_dir / "run.json"),
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
        "n_requests": len(requests), "split": args.split, "positions": {
            "request": "last token overlapping the request",
            "readout": "last token of the full chat template",
        }, "cells": {},
    }
    baselines = {}
    for label, cell, cases in context_cells(tok, requests, args):
        length = cell["L"]
        if args.resume and label in summary["cells"]:
            if length == 0:
                baselines[cell["format"]] = summary["cells"][label]["by_position"]
            continue
        formatted = [case["chat_prompt"] for case in cases]
        positions = [case["request_position"] for case in cases]
        bs = args.batch_size if args.batch_size is not None else adaptive_batch_size(length)
        residuals = collect_residuals(
            model, tok, formatted, layers, bs, device,
            context_budget=args.context_budget, request_positions=positions,
        )
        projections = (residuals * directions[:, None, None, :]).sum(dim=-1)
        norms = residuals.norm(dim=-1)
        cosines = projections / norms.clamp_min(1e-12)
        by_position = {
            name: {str(layer): summarize(projections[slot, :, pos].tolist())
                   for slot, layer in enumerate(layers)}
            for pos, name in enumerate(["readout", "request"])
        }
        for pos, name in enumerate(["readout", "request"]):
            for slot, layer in enumerate(layers):
                stats = by_position[name][str(layer)]
                stats.update({"norms": norms[slot, :, pos].tolist(),
                              "cosines": cosines[slot, :, pos].tolist(),
                              "n_negative": sum(v < 0 for v in stats["values"])})
                if cell["format"] in baselines:
                    base = baselines[cell["format"]][name][str(layer)]["values"]
                    stats.update(paired_changes(stats["values"], base, seed=args.seed))
        if length == 0:
            baselines[cell["format"]] = by_position
        row_path = out_dir / (label + ".jsonl")
        tmp = row_path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as handle:
            for index, case in enumerate(cases):
                measured = {}
                for name in ("readout", "request"):
                    measured[name] = {}
                    for layer in layers:
                        stats = by_position[name][str(layer)]
                        measured[name][str(layer)] = {
                            "projection": stats["values"][index], "norm": stats["norms"][index],
                            "cosine": stats["cosines"][index],
                            "delta_from_empty": stats.get("delta_values", [None] * len(cases))[index],
                        }
                handle.write(json.dumps({"index": index, "case": case, "by_position": measured}) + "\n")
        tmp.replace(row_path)
        summary["cells"][label] = {
            **cell, "batch_size": bs,
            "background_tokens": cases[0]["background_tokens"],
            "request_ids": [r["id"] for r in requests],
            "by_position": by_position, "by_layer": by_position["readout"],
        }
        write_json(summary_path, summary)
        del residuals, projections, norms, cosines
    if not args.no_plot:
        maybe_plot(summary, out_dir / "projection_sweep.png")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
