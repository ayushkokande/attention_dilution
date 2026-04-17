"""Plot separation by layer and projection histograms for top refusal layers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from scipy.stats import gaussian_kde

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

RESULTS_DIR = Path("results")
REFUSAL_PATH = RESULTS_DIR / "refusal_directions.pt"
ACTIVATIONS_PATH = RESULTS_DIR / "activations.pt"


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _projections_for_layer(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    norm_dir: torch.Tensor,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    d = norm_dir[layer].float()
    ph = (harmful[:, layer, :] * d).sum(dim=-1).numpy()
    ps = (harmless[:, layer, :] * d).sum(dim=-1).numpy()
    return ph, ps


def _plot_kde_or_hist(ax, harmful: np.ndarray, harmless: np.ndarray, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Projection onto normalized refusal direction")
    ax.set_ylabel("Density")

    if _HAS_SCIPY and len(harmful) > 1 and len(harmless) > 1:
        xs = np.linspace(
            min(harmful.min(), harmless.min()),
            max(harmful.max(), harmless.max()),
            200,
        )
        kh = gaussian_kde(harmful)(xs)
        ks = gaussian_kde(harmless)(xs)
        ax.plot(xs, kh, label="harmful", color="C0")
        ax.plot(xs, ks, label="harmless", color="C1")
    else:
        ax.hist(harmful, bins=12, density=True, alpha=0.5, label="harmful", color="C0")
        ax.hist(harmless, bins=12, density=True, alpha=0.5, label="harmless", color="C1")
    ax.legend()


def main() -> None:
    if not REFUSAL_PATH.is_file() or not ACTIVATIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {REFUSAL_PATH} and/or {ACTIVATIONS_PATH}. "
            "Run `python src/extract_refusal_direction.py` from the repo root first."
        )
    payload = _load_torch(REFUSAL_PATH)
    acts = _load_torch(ACTIVATIONS_PATH)

    harmful = acts["harmful"]
    harmless = acts["harmless"]
    norm_dirs = payload["refusal_dirs_normalized"]
    scores = payload["separation_scores"]
    if isinstance(scores, torch.Tensor):
        scores_np = scores.numpy()
    else:
        scores_np = np.asarray(scores)

    n_layers = int(scores_np.shape[0])
    layers_axis = np.arange(n_layers)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers_axis, scores_np, marker="o", markersize=3)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Separation score")
    ax.set_title("Harmful vs harmless separation by layer")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "separation_by_layer.png", dpi=150)
    plt.close(fig)

    topk = min(3, n_layers)
    top = torch.topk(torch.as_tensor(scores_np), k=topk)
    best_idx = int(top.indices[0].item())
    best_val = float(top.values[0].item())

    for rank in range(topk):
        L = int(top.indices[rank].item())
        ph, ps = _projections_for_layer(harmful, harmless, norm_dirs, L)
        fig, ax = plt.subplots(figsize=(6, 4))
        _plot_kde_or_hist(ax, ph, ps, title=f"Layer {L} (separation rank {rank + 1})")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"projection_layer_{L}.png", dpi=150)
        plt.close(fig)

    print(f"Best layer: {best_idx} (separation = {best_val:.4f})")
    print(
        f"Layer {best_idx} shows the clearest refusal signal in this mean-difference direction."
    )


if __name__ == "__main__":
    main()
