"""Figure: refusal detected by a judge that reads only the first N tokens (CPU; uses results/v2/judge_horizon.json)."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/v2/judge_horizon.json"))
NICE = {"qwen3-14b": "Qwen3-14B", "gemma2-9b": "Gemma-2-9B", "llama31-8b": "Llama-3.1-8B"}
C = {"first": "#1f77b4", "middle": "#9467bd", "last": "#d62728"}
H = [24, 64, 128, 256, 512, 1024]
fig, axes = plt.subplots(1, len(d), figsize=(3.3 * len(d), 2.5), sharey=True)
for ax, (m, v) in zip(axes, d.items()):
    for pos in ("first", "middle", "last"):
        e = v[f"H_{pos}"]
        ax.plot(H, [e[str(h)] if str(h) in e else e[h] for h in H], "o-", ms=3, color=C[pos], label=f"harmful item {pos}")
        ax.axhline(e["full_target"], color=C[pos], ls=":", lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(H); ax.set_xticklabels([str(h) for h in H], fontsize=7)
    ax.set_ylim(0, 1.02); ax.set_title(NICE[m], fontsize=9); ax.set_xlabel("tokens read by the judge")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("refusal detected"); axes[0].legend(frameon=False, fontsize=6, loc="lower right")
fig.tight_layout(); fig.savefig(sys.argv[2] if len(sys.argv) > 2 else "paper_v2/figures/fig_horizon.pdf")
