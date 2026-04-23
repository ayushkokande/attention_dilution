"""Figures describing the refusal steering vector and intervention outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import results_dir


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _direction_norms(payload) -> np.ndarray:
    dirs = payload["refusal_dirs"].float()
    return dirs.norm(dim=-1).numpy()


def _cosine_heatmap(payload) -> np.ndarray:
    u = payload["refusal_dirs_normalized"].float().numpy()
    return u @ u.T


def _plot_direction_summary(payload, out_path: Path) -> int:
    scores = payload["separation_scores"]
    scores_np = scores.numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
    norms = _direction_norms(payload)
    best_layer = int(np.argmax(scores_np))
    n_layers = scores_np.shape[0]
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(layers, scores_np, marker="o", markersize=3, color="C0")
    ax.axvline(best_layer, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Separation score")
    ax.set_title(f"Separation by layer (best = {best_layer})")

    ax = axes[1]
    ax.plot(layers, norms, marker="o", markersize=3, color="C3")
    ax.axvline(best_layer, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("||mean_harmful - mean_harmless||")
    ax.set_title("Raw steering-vector magnitude by layer")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return best_layer


def _plot_cosine_heatmap(payload, best_layer: int, out_path: Path) -> None:
    cos = _cosine_heatmap(payload)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax.axhline(best_layer, color="k", linewidth=0.8, alpha=0.6)
    ax.axvline(best_layer, color="k", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Layer j")
    ax.set_ylabel("Layer i")
    ax.set_title("Cosine similarity of normalized refusal directions")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pca_at_layer(payload, acts, layer: int, out_path: Path) -> None:
    harmful = acts["harmful"][:, layer, :].float().numpy()
    harmless = acts["harmless"][:, layer, :].float().numpy()
    u = payload["refusal_dirs_normalized"][layer].float().numpy()

    X = np.concatenate([harmful, harmless], axis=0)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    u_hat = u / (np.linalg.norm(u) + 1e-12)
    proj_uu = Xc @ u_hat[:, None] * u_hat[None, :]
    X_perp = Xc - proj_uu
    cov_perp = X_perp.T @ X_perp / max(1, X_perp.shape[0] - 1)
    v = np.random.default_rng(0).normal(size=cov_perp.shape[0])
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(50):
        v = cov_perp @ v
        v /= np.linalg.norm(v) + 1e-12
    v_hat = v - (v @ u_hat) * u_hat
    v_hat /= np.linalg.norm(v_hat) + 1e-12

    coords = np.stack([Xc @ u_hat, Xc @ v_hat], axis=1)
    nh = harmful.shape[0]
    ch = coords[:nh]
    cs = coords[nh:]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ch[:, 0], ch[:, 1], alpha=0.7, label="harmful", color="C0")
    ax.scatter(cs[:, 0], cs[:, 1], alpha=0.7, label="harmless", color="C1")
    ax.scatter(ch[:, 0].mean(), ch[:, 1].mean(), marker="X", s=120,
               edgecolor="black", color="C0")
    ax.scatter(cs[:, 0].mean(), cs[:, 1].mean(), marker="X", s=120,
               edgecolor="black", color="C1")

    arrow_len = (ch[:, 0].mean() - cs[:, 0].mean())
    ax.annotate(
        "",
        xy=(arrow_len, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )
    ax.text(arrow_len * 0.5, 0.05 * (coords[:, 1].max() - coords[:, 1].min()),
            "refusal direction", ha="center")

    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Projection onto refusal direction")
    ax.set_ylabel("Top orthogonal PC")
    ax.set_title(f"Layer {layer} residual stream (last prompt token)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_ablation_effect(payload, acts, layer: int, out_path: Path) -> None:
    """Simulate residual-stream ablation on the stored activations at one layer."""
    harmful = acts["harmful"][:, layer, :].float()
    harmless = acts["harmless"][:, layer, :].float()
    u = payload["refusal_dirs_normalized"][layer].float()

    ph_pre = (harmful * u).sum(dim=-1).numpy()
    ps_pre = (harmless * u).sum(dim=-1).numpy()
    harmful_post = harmful - (harmful @ u)[:, None] * u
    harmless_post = harmless - (harmless @ u)[:, None] * u
    ph_post = (harmful_post * u).sum(dim=-1).numpy()
    ps_post = (harmless_post * u).sum(dim=-1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = 15
    axes[0].hist(ph_pre, bins=bins, alpha=0.6, color="C0", label="harmful")
    axes[0].hist(ps_pre, bins=bins, alpha=0.6, color="C1", label="harmless")
    axes[0].set_title("Projection onto u (pre-ablation)")
    axes[0].set_xlabel("<x, u>")
    axes[0].legend()

    axes[1].hist(ph_post, bins=bins, alpha=0.6, color="C0", label="harmful")
    axes[1].hist(ps_post, bins=bins, alpha=0.6, color="C1", label="harmless")
    axes[1].set_title("Projection onto u (post-ablation)")
    axes[1].set_xlabel("<x - (x.u)u, u>")

    fig.suptitle(f"Layer {layer} — ablating the refusal direction zeroes its component")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_intervention_rates(
    intervention_path: Path,
    intervention_long_path: Path,
    feature_path: Path,
    out_path: Path,
) -> None:
    if not intervention_path.is_file():
        return
    with open(intervention_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rates = data["rates"]
    labels = ["harmful\nbaseline", "harmful\nablate",
              "harmless\nbaseline", "harmless\nadd"]
    values = [rates["harmful_baseline"], rates["harmful_ablate"],
              rates["harmless_baseline"], rates["harmless_add"]]
    colors = ["C0", "C0", "C1", "C1"]

    if intervention_long_path.is_file():
        with open(intervention_long_path, "r", encoding="utf-8") as f:
            long_rate = json.load(f).get("refusal_rate")
        if long_rate is not None:
            labels.append("harmful\nablate (long)")
            values.append(long_rate)
            colors.append("C0")

    if feature_path.is_file():
        with open(feature_path, "r", encoding="utf-8") as f:
            feat = json.load(f)
        feat_rate = feat.get("rates", {}).get("harmful_feature_ablate")
        if feat_rate is not None:
            labels.append("harmful\nfeature\nablate")
            values.append(feat_rate)
            colors.append("C2")

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(labels)), 4.5))
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=colors, alpha=0.85)
    for x, v in zip(xs, values):
        ax.text(x, v + 0.02, f"{v:.0%}", ha="center", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Refusal rate")
    ax.set_title(
        f"Refusal rates by condition (layer {data['best_layer']}, "
        f"n={data['n_held_out']})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    rdir = results_dir()
    refusal_path = rdir / "refusal_directions.pt"
    activations_path = rdir / "activations.pt"
    intervention_path = rdir / "intervention_results.json"
    intervention_long_path = rdir / "intervention_results_full.json"
    feature_path = rdir / "intervention_feature_results.json"

    if not refusal_path.is_file() or not activations_path.is_file():
        raise FileNotFoundError(
            "Run ayush/extract_refusal_direction.py first to produce "
            f"{refusal_path} and {activations_path}."
        )
    payload = _load_torch(refusal_path)
    acts = _load_torch(activations_path)

    best_layer = _plot_direction_summary(payload, rdir / "steering_summary.png")
    _plot_cosine_heatmap(payload, best_layer, rdir / "steering_cosine_heatmap.png")
    _plot_pca_at_layer(
        payload, acts, best_layer, rdir / f"steering_pca_layer_{best_layer}.png"
    )
    _plot_ablation_effect(
        payload, acts, best_layer,
        rdir / f"steering_ablation_effect_layer_{best_layer}.png",
    )
    _plot_intervention_rates(
        intervention_path,
        intervention_long_path,
        feature_path,
        rdir / "steering_intervention_rates.png",
    )

    print(f"Wrote figures for layer {best_layer} to {rdir}/")


if __name__ == "__main__":
    main()
