"""Phase 2: plots for the context-length scaling sweep.

Reads ``results/<slug>/scaling/sweep_rows.jsonl`` produced by
``ayush/sweep_context_scaling.py`` and emits three figures:

    scaling_asr.png                   # ASR (= 1 - refusal rate) vs N
    scaling_refusal_projection.png    # mean proj onto r^(l*) at harmful_end
                                      # and at final position, vs N
    scaling_attn_dilution.png         # per-head attn-to-harmful vs N at the
                                      # best refusal layer, with the top-K
                                      # "candidate guardrail heads" highlighted

All figures go next to the JSONL under results/<slug>/scaling/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import results_dir


TOP_K_GUARDRAIL_HEADS = 5


def _load_rows(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _by_N(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(int(r["N"]), []).append(r)
    return out


def _plot_asr(by_N: dict, out_path: Path) -> None:
    Ns = sorted(by_N)
    refusal_rate = [
        sum(r["refused"] for r in by_N[N]) / len(by_N[N]) for N in Ns
    ]
    asr = [1.0 - r for r in refusal_rate]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(Ns, asr, marker="o", color="#c0392b", label="ASR = 1 - refusal rate")
    ax.plot(Ns, refusal_rate, marker="s", color="#2c3e50", label="refusal rate", alpha=0.6)
    ax.set_xscale("symlog", linthresh=128)
    ax.set_xlabel("Bloat length N (tokens)")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Attack success vs. context length")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_projection(by_N: dict, best_layer: int, out_path: Path) -> None:
    Ns = sorted(by_N)
    proj_at_harmful = [
        np.mean([r["proj_at_harmful_end"][best_layer] for r in by_N[N]]) for N in Ns
    ]
    proj_at_final = [
        np.mean([r["proj_at_final"][best_layer] for r in by_N[N]]) for N in Ns
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(Ns, proj_at_harmful, marker="o", label="at last harmful token")
    ax.plot(Ns, proj_at_final, marker="s", label="at final input token")
    ax.set_xscale("symlog", linthresh=128)
    ax.set_xlabel("Bloat length N (tokens)")
    ax.set_ylabel(f"Residual-stream proj onto $\\hat{{r}}^{{(L={best_layer})}}$")
    ax.set_title("Representational dilution of the refusal direction")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_attention_dilution(by_N: dict, best_layer: int, out_path: Path) -> None:
    """Per-head attn-to-harmful at the best layer vs N.

    "Candidate guardrail heads" are the top-K heads with highest mean
    attention mass to the harmful span at N=0, under the hypothesis that
    those are the heads most directly gating refusal.
    """
    Ns = sorted(by_N)
    sample_row = by_N[Ns[0]][0]
    n_heads = len(sample_row["attn_to_harmful"][best_layer])

    # per_N_head_mean[N][H] = mean across prompts
    per_N_head_mean = np.zeros((len(Ns), n_heads), dtype=np.float32)
    for i, N in enumerate(Ns):
        stack = np.array(
            [r["attn_to_harmful"][best_layer] for r in by_N[N]],
            dtype=np.float32,
        )  # [n_prompts, n_heads]
        per_N_head_mean[i] = stack.mean(axis=0)

    top_idx_at_zero = np.argsort(-per_N_head_mean[0])[:TOP_K_GUARDRAIL_HEADS]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for H in range(n_heads):
        color = "#c0392b" if H in top_idx_at_zero else "#95a5a6"
        alpha = 0.9 if H in top_idx_at_zero else 0.35
        lw = 1.6 if H in top_idx_at_zero else 0.8
        label = f"head {H}" if H in top_idx_at_zero else None
        ax.plot(Ns, per_N_head_mean[:, H], marker="o", color=color, alpha=alpha, lw=lw, label=label)

    ax.set_xscale("symlog", linthresh=128)
    ax.set_xlabel("Bloat length N (tokens)")
    ax.set_ylabel("Attn mass from final position → harmful span")
    ax.set_title(
        f"Attention dilution at layer {best_layer}\n"
        f"(top {TOP_K_GUARDRAIL_HEADS} heads at N=0 highlighted as candidate guardrail heads)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    rdir = results_dir()
    scaling_dir = rdir / "scaling"
    jsonl_path = scaling_dir / "sweep_rows.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"{jsonl_path} not found. Run ayush/sweep_context_scaling.py first."
        )

    directions_path = rdir / "refusal_directions.pt"
    payload = torch.load(directions_path, map_location="cpu", weights_only=False)
    best_layer = int(torch.argmax(payload["separation_scores"]).item())

    rows = _load_rows(jsonl_path)
    by_N = _by_N(rows)
    print(f"Loaded {len(rows)} rows across {len(by_N)} context lengths.")

    _plot_asr(by_N, scaling_dir / "scaling_asr.png")
    print(f"Wrote {scaling_dir / 'scaling_asr.png'}")

    _plot_projection(by_N, best_layer, scaling_dir / "scaling_refusal_projection.png")
    print(f"Wrote {scaling_dir / 'scaling_refusal_projection.png'}")

    _plot_attention_dilution(by_N, best_layer, scaling_dir / "scaling_attn_dilution.png")
    print(f"Wrote {scaling_dir / 'scaling_attn_dilution.png'}")


if __name__ == "__main__":
    main()
