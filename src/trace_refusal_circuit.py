"""Run `circuit-tracer` attribution on refusal-triggering prompts.

For each prompt we save:
- a raw attribution graph (``*.pt``) under ``results/<slug>/graphs/``
- pruned JSON graph files under ``results/<slug>/graph_files/`` that the
  ``circuit-tracer`` server can visualize

By default we trace three prompts:
  1. ``harmful_baseline``: a harmful prompt the model refuses
  2. ``harmful_ablate``: the same prompt after residual-stream ablation
     (pulled from ``intervention_results.json``)
  3. ``harmless_baseline``: a matched harmless prompt

Comparing the three graphs is the cheapest way to find candidate "refusal
features" for later intervention.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from utils import (
    MODEL_NAME,
    MODEL_SLUG,
    TRANSCODER_SET,
    ensure_transformers_cache_attr,
    format_chat_prompt,
    get_device,
    load_harmful_prompts,
    load_harmless_prompts,
    results_dir,
)

ensure_transformers_cache_attr()

from circuit_tracer import ReplacementModel, attribute  # noqa: E402
from circuit_tracer.utils import create_graph_files  # noqa: E402


@dataclass
class TracePrompt:
    name: str
    slug: str
    prompt: str
    notes: str = ""


def _pick_baseline_refused_prompt(
    intervention_path: Path,
) -> Optional[tuple[int, str]]:
    """Return ``(index, prompt)`` of the first held-out harmful prompt that the
    baseline refused. Returns ``None`` if no such sample exists."""
    if not intervention_path.is_file():
        return None
    with open(intervention_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for s in data.get("samples", {}).get("harmful_baseline", []):
        if s.get("refused"):
            return int(s["index"]), str(s["prompt"])
    return None


def _select_prompts(
    rdir: Path,
    fallback_harmful_index: int = 0,
    fallback_harmless_index: int = 0,
) -> list[TracePrompt]:
    intervention_path = rdir / "intervention_results.json"

    pick = _pick_baseline_refused_prompt(intervention_path)
    if pick is not None:
        h_idx, harmful_text = pick
    else:
        h_idx = fallback_harmful_index
        harmful_text = load_harmful_prompts(h_idx + 1)[h_idx]

    harmless_text = load_harmless_prompts(fallback_harmless_index + 1)[fallback_harmless_index]

    return [
        TracePrompt(
            name="harmful_baseline",
            slug=f"harmful-baseline-{h_idx}",
            prompt=harmful_text,
            notes=(
                "Held-out harmful prompt refused by the baseline model. "
                "Attributes from the first generated token of the refusal."
            ),
        ),
        TracePrompt(
            name="harmless_baseline",
            slug=f"harmless-baseline-{fallback_harmless_index}",
            prompt=harmless_text,
            notes="Matched harmless prompt from Alpaca.",
        ),
    ]


def _resolve_dtype(device: str, override: Optional[str]) -> torch.dtype:
    if override is not None:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[override]
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harmful-index",
        type=int,
        default=0,
        help="Fallback AdvBench index to use if intervention_results.json is absent.",
    )
    parser.add_argument(
        "--harmless-index",
        type=int,
        default=0,
        help="Alpaca index for the matched harmless prompt.",
    )
    parser.add_argument(
        "--max-n-logits",
        type=int,
        default=10,
        help="How many logits (at the first generated token) to attribute from.",
    )
    parser.add_argument(
        "--desired-logit-prob",
        type=float,
        default=0.95,
        help="Stop accumulating logits once cumulative prob >= this threshold.",
    )
    parser.add_argument(
        "--max-feature-nodes",
        type=int,
        default=4096,
        help="Prune feature-node budget; lower is faster.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for circuit-tracer's backward passes.",
    )
    parser.add_argument(
        "--node-threshold",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--offload",
        choices=["cpu", "disk", "none"],
        default="cpu",
        help="Offload transcoders during attribution to save memory.",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument(
        "--backend",
        choices=["transformerlens", "nnsight"],
        default="transformerlens",
    )
    args = parser.parse_args()

    device, _ = get_device()
    dtype = _resolve_dtype(device, args.dtype)
    offload = None if args.offload == "none" else args.offload
    print(f"Device: {device}, dtype: {dtype}, offload: {offload}, backend: {args.backend}")

    rdir = results_dir()
    graphs_dir = rdir / "graphs"
    graph_files_dir = rdir / "graph_files"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    graph_files_dir.mkdir(parents=True, exist_ok=True)

    trace_prompts = _select_prompts(rdir, args.harmful_index, args.harmless_index)

    print(f"\nLoading ReplacementModel ({MODEL_NAME}) with transcoders {TRANSCODER_SET} ...")
    model = ReplacementModel.from_pretrained(
        MODEL_NAME,
        TRANSCODER_SET,
        dtype=dtype,
        backend=args.backend,
    )

    index: list[dict] = []
    for tp in trace_prompts:
        formatted = format_chat_prompt(model.tokenizer, tp.prompt, enable_thinking=False)
        print(f"\n=== {tp.name} :: {tp.slug} ===")
        print(f"prompt (first 140 chars): {tp.prompt[:140]}")
        print("attributing...")

        graph = attribute(
            prompt=formatted,
            model=model,
            max_n_logits=args.max_n_logits,
            desired_logit_prob=args.desired_logit_prob,
            batch_size=args.batch_size,
            max_feature_nodes=args.max_feature_nodes,
            offload=offload,
            verbose=True,
        )

        graph_pt = graphs_dir / f"{tp.slug}.pt"
        graph.to_pt(graph_pt)
        print(f"  saved raw graph -> {graph_pt}")

        create_graph_files(
            graph_or_path=graph_pt,
            slug=tp.slug,
            output_path=str(graph_files_dir),
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )
        print(f"  wrote pruned graph files -> {graph_files_dir}/ (slug={tp.slug})")

        index.append(
            {
                "name": tp.name,
                "slug": tp.slug,
                "prompt": tp.prompt,
                "formatted_prompt": formatted,
                "graph_path": str(graph_pt.relative_to(rdir.parent)),
                "notes": tp.notes,
            }
        )

    index_path = rdir / "graph_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "model_slug": MODEL_SLUG,
                "transcoders": TRANSCODER_SET,
                "graphs_dir": str(graphs_dir.relative_to(rdir.parent)),
                "graph_files_dir": str(graph_files_dir.relative_to(rdir.parent)),
                "items": index,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nWrote {index_path}")
    print(
        f"\nTo view the graphs interactively, run:\n"
        f"  python src/visualize_refusal_graph.py --data-dir {graph_files_dir}"
    )


if __name__ == "__main__":
    main()
