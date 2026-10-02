"""Baseline phrase-refusal rates on AdvBench and filtered Alpaca."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_dilution.generation import generate_dataset
from attention_dilution.shared import (
    MODEL_NAME, CONTEXT_BUDGET, format_chat_prompt, load_harmful_prompts,
    load_harmless_prompts, load_model, prepare_run, record_environment,
    summarize_responses, write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--alpaca-n", type=int, default=512)
    p.add_argument("--advbench-n", type=int, default=520)
    p.add_argument("--splits", choices=["both", "harmful", "harmless"], default="both")
    p.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.alpaca_n <= 0 or args.advbench_n <= 0:
        raise ValueError("Dataset sizes must be positive")
    pools = {}
    if args.splits in ("both", "harmful"):
        pools["harmful"] = load_harmful_prompts(args.advbench_n)
    if args.splits in ("both", "harmless"):
        pools["harmless"] = load_harmless_prompts(args.alpaca_n)
    out_dir = prepare_run("baseline", args, pools)
    import torch
    torch.manual_seed(args.seed)
    model, tok, device = load_model(args.model, args.dtype, args.revision)
    record_environment(out_dir, model, tok)
    summary = {"model": args.model, "splits": {}}
    for name, prompts in pools.items():
        formatted = [format_chat_prompt(tok, p, enable_thinking=args.enable_thinking) for p in prompts]
        filename = "baseline_advbench.jsonl" if name == "harmful" else "baseline_alpaca.jsonl"
        rows = generate_dataset(
            model, tok, formatted, prompts, device=device, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            context_budget=args.context_budget, path=out_dir / filename, resume=args.resume,
        )
        summary["splits"][name] = summarize_responses(rows)
        summary[name + "_refusal_rate"] = summary["splits"][name]["refusal_rate"]
        write_json(out_dir / "baseline_summary.json", summary)
        print(name, summary["splits"][name])
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
