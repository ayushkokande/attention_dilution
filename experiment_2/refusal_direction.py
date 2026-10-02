"""Extract per-layer directions and select one on separate validation prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_dilution.activations import collect_residuals
from attention_dilution.generation import generate_dataset
from attention_dilution.shared import (
    MODEL_NAME, CONTEXT_BUDGET, assert_disjoint_pools, directional_ablation,
    format_chat_prompt, load_core_pool, load_model, prepare_run, record_environment,
    slug_from_model, summarize_responses, write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--n-harmful", type=int, default=256)
    p.add_argument("--n-harmless", type=int, default=256)
    p.add_argument("--harmful-file", default=None)
    p.add_argument("--harmless-file", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--causal-ablation-n", type=int, default=24)
    p.add_argument("--causal-max-new-tokens", type=int, default=256)
    p.add_argument("--causal-depth-min", type=float, default=.35)
    p.add_argument("--causal-depth-max", type=float, default=.9)
    p.add_argument("--causal-layer-step", type=int, default=2)
    p.add_argument("--layer-criterion", choices=["causal", "norm"], default="causal")
    p.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _load_jsonl_prompts(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    prompts = [row.get("prompt") or row.get("instruction") or row.get("goal") for row in rows]
    if any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError(f"Missing prompt text in {path}")
    return prompts


def training_pool(args, source, count, override):
    if override:
        return _load_jsonl_prompts(override)
    pool = load_core_pool("train", source)
    if not 0 < count <= len(pool):
        raise ValueError(f"Training pool size must be in [1, {len(pool)}]")
    return pool[:count]


def main():
    args = parse_args()
    if not 0 <= args.causal_depth_min <= args.causal_depth_max <= 1 or args.causal_layer_step <= 0:
        raise ValueError("Invalid causal layer sweep")
    if args.layer_criterion == "causal" and args.causal_ablation_n <= 0:
        raise ValueError("Causal selection requires a nonempty validation pool")
    harmful = training_pool(args, "advbench", args.n_harmful, args.harmful_file)
    harmless = training_pool(args, "alpaca_post_filter", args.n_harmless, args.harmless_file)
    val_h = load_core_pool("validation", "advbench")
    val_b = load_core_pool("validation", "alpaca_post_filter")
    if not 0 <= args.causal_ablation_n <= min(len(val_h), len(val_b)):
        raise ValueError("Requested validation size exceeds the manifest")
    val_h, val_b = val_h[:args.causal_ablation_n], val_b[:args.causal_ablation_n]
    # Check actual strings as well as the row ranges in splits.json.
    harmful_pools = {"train": harmful, "test": load_core_pool("test", "advbench")}
    harmless_pools = {"train": harmless, "test": load_core_pool("test", "alpaca_post_filter")}
    if val_h:
        harmful_pools["validation"] = val_h
        harmless_pools["validation"] = val_b
    assert_disjoint_pools(harmful_pools)
    assert_disjoint_pools(harmless_pools)
    inputs = {"harmful_train": harmful, "harmless_train": harmless,
              "harmful_validation": val_h, "harmless_validation": val_b}
    out_dir = prepare_run("direction", args, inputs)
    import torch
    torch.manual_seed(args.seed)
    model, tok, device = load_model(args.model, args.dtype, args.revision)
    record_environment(out_dir, model, tok)
    n_layers = len(model.model.layers)
    layer_indices = list(range(n_layers))

    def format_all(prompts):
        return [format_chat_prompt(tok, p, enable_thinking=args.enable_thinking) for p in prompts]

    bank_path = out_dir / "d_hat_all_layers.pt"
    norms_path = out_dir / "direction_norms.json"
    if args.resume and bank_path.exists() and norms_path.exists():
        bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        norms = json.loads(norms_path.read_text())["norms"]
    else:
        acts_h = collect_residuals(model, tok, format_all(harmful), layer_indices,
                                  args.batch_size, device, context_budget=args.context_budget)
        mean_h = acts_h[:, :, 0].mean(dim=1)
        del acts_h
        acts_b = collect_residuals(model, tok, format_all(harmless), layer_indices,
                                  args.batch_size, device, context_budget=args.context_budget)
        diff = mean_h - acts_b[:, :, 0].mean(dim=1)
        del acts_b, mean_h
        norm_tensor = diff.norm(dim=-1)
        if bool((norm_tensor < 1e-8).any()):
            raise ValueError("A layer has no measurable difference of means")
        bank = diff / norm_tensor[:, None]
        norms = norm_tensor.tolist()
        tmp = bank_path.with_suffix(".pt.tmp")
        torch.save(bank, tmp)
        tmp.replace(bank_path)
        write_json(norms_path, {"norms": norms})

    norm_best = max(range(n_layers), key=lambda layer: norms[layer])
    records = []
    lo = max(0, round(n_layers * args.causal_depth_min))
    hi = min(n_layers - 1, round(n_layers * args.causal_depth_max))
    candidates = list(range(lo, hi + 1, args.causal_layer_step))
    if args.layer_criterion == "causal" and not candidates:
        raise ValueError("Causal sweep has no candidate layers")

    if args.layer_criterion == "causal":
        for layer in candidates:
            stats = {}
            with directional_ablation(model, bank[layer]):
                for label, prompts in [("harmful", val_h), ("harmless", val_b)]:
                    rows = generate_dataset(
                        model, tok, format_all(prompts), prompts, device=device,
                        batch_size=args.batch_size, max_new_tokens=args.causal_max_new_tokens,
                        temperature=0.0, context_budget=args.context_budget,
                        path=out_dir / f"validation_L{layer}_{label}.jsonl", resume=args.resume,
                    )
                    stats[label] = summarize_responses(rows)
                    if stats[label]["n_evaluated"] != len(prompts):
                        raise ValueError(f"Incomplete validation at layer {layer}; start a new run")
            records.append({
                "layer": layer, "harmful_refusal_post_ablate": stats["harmful"]["refusal_rate"],
                "harmless_refusal_post_ablate": stats["harmless"]["refusal_rate"],
                "harmful_ci95": stats["harmful"]["refusal_rate_ci95"],
                "norm": norms[layer], "status": "ok",
            })
            write_json(out_dir / "layer_selection.json", {"records": records})
        selected = min(records, key=lambda r: (
            r["harmful_refusal_post_ablate"], r["harmless_refusal_post_ablate"],
            abs(r["layer"] - norm_best),
        ))["layer"]
    else:
        selected = norm_best

    meta = {
        "model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
        "enable_thinking": args.enable_thinking, "n_layers": n_layers,
        "d_model": int(bank.shape[1]), "n_harmful": len(harmful), "n_harmless": len(harmless),
        "diff_norms_per_layer": norms, "norm_best_layer": norm_best,
        "causal_best_layer": selected if args.layer_criterion == "causal" else None,
        "canonical_layer": selected, "default_layer": selected,
        "layer_criterion": args.layer_criterion, "causal_sweep_records": records,
        "causal_judge_config": {"judge": "looks_like_refusal", "character_window": 200,
                                "max_new_tokens": args.causal_max_new_tokens},
        "intervention": "decoder block output",
        "activation_position": "last token of the full chat template",
        "selection_tied_layers": [
            r["layer"] for r in records
            if r["harmful_refusal_post_ablate"] == min(x["harmful_refusal_post_ablate"] for x in records)
        ] if records else [],
    }
    torch.save({"V_refusal": bank[selected].clone(), "layer": selected},
               out_dir / "d_hat_best_layer.pt")
    write_json(out_dir / "meta.json", meta)
    print(f"Selected layer {selected}; results: {out_dir}")


if __name__ == "__main__":
    main()
