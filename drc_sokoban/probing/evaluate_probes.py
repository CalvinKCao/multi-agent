"""
Probe evaluation: F1 metrics, results tables, and heatmap figures.

Replicates the structure of Bush et al. 2025 Figure 4 and Figure 6.
"""

import numpy as np
import pickle
from typing import Dict, Optional
from pathlib import Path


def print_results_table(results: Dict):
    """
    Pretty-print probe F1 results as a table matching Bush et al. Fig 4.

    Example output:
               Tick 0    Tick 1    Tick 2    Obs-baseline
    CA Layer 0:  0.421     0.498     0.512
    CA Layer 1:  0.551     0.598     0.613
    CA Layer 2:  0.562     0.604     0.620     0.350
    CB Layer 0:  0.381     0.432     0.451
    CB Layer 1:  0.480     0.520     0.543
    CB Layer 2:  0.488     0.529     0.552     0.285
    """
    ca = results.get("CA", {})
    cb = results.get("CB", {})

    if not ca:
        print("No results to display.")
        return

    layers = sorted(set(k[0] for k in ca.keys()))
    ticks  = sorted(set(k[1] for k in ca.keys()))

    header = f"{'':20s}" + "".join(f"  Tick {t}" for t in ticks) + "  Obs-baseline"
    print(header)
    print("-" * len(header))

    for concept, label in [("CA", ca), ("CB", cb)]:
        baseline = results.get(f"{concept}_baseline", None)
        for li, layer in enumerate(layers):
            row = f"{concept} Layer {layer}:{'':10s}"
            for tick in ticks:
                key = (layer, tick)
                f1 = label.get(key, 0.0)
                row += f"  {f1:.3f}   "
            if li == len(layers) - 1 and baseline is not None:
                row += f"  {baseline:.3f}"
            print(row)
    print()


def save_results(results: Dict, path: str):
    """Save probe results dict to pickle."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {path}")


def load_results(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_f1_heatmap(results: Dict, concept: str = "CA", save_path: Optional[str] = None):
    """
    Plot F1 as a function of (layer, tick) — Figure 4 style heatmap.

    Args:
        results: output of train_all_probes
        concept: "CA" or "CB"
        save_path: if provided, save figure to this path
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    label_dict = results.get(concept, {})
    if not label_dict:
        return

    layers = sorted(set(k[0] for k in label_dict.keys()))
    ticks  = sorted(set(k[1] for k in label_dict.keys()))

    grid = np.zeros((len(layers), len(ticks)))
    for i, layer in enumerate(layers):
        for j, tick in enumerate(ticks):
            grid[i, j] = label_dict.get((layer, tick), 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels([f"Tick {t}" for t in ticks])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    ax.set_title(f"Probe F1 ({concept})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_spatial_f1(
    results: Dict,
    concept: str = "CA",
    layer: int = -1,
    tick: int = -1,
    grid_size: int = 8,
    save_path: Optional[str] = None,
):
    """
    Plot per-cell F1 as a spatial heatmap over the 8×8 board — Figure 5 style.

    Args:
        layer: layer index (-1 for last layer)
        tick:  tick index (-1 for last tick)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    per_pos_key = f"{concept}_per_pos"
    per_pos = results.get(per_pos_key, {})
    if not per_pos:
        print(f"No per-position results found for {concept}.")
        return

    # Resolve -1 indices
    layers = sorted(set(k[0] for k in per_pos.keys()))
    ticks  = sorted(set(k[1] for k in per_pos.keys()))
    layer = layers[layer] if layer < 0 else layer
    tick  = ticks[tick] if tick < 0 else tick

    key = (layer, tick)
    pos_f1 = per_pos.get(key, {})

    grid = np.zeros((grid_size, grid_size))
    for (x, y), f1 in pos_f1.items():
        grid[y, x] = f1

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="plasma", origin="upper")
    ax.set_title(f"{concept} per-cell F1 | Layer {layer}, Tick {tick}")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved spatial F1 plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def check_probe_sanity(results: Dict, min_delta: float = 0.05) -> bool:
    """
    Check 3: probe beats observation baseline by at least min_delta on CA.
    Returns True if sanity check passes.
    """
    ca = results.get("CA", {})
    baseline = results.get("CA_baseline", 0.0)
    if not ca:
        print("SANITY FAIL: No CA results found.")
        return False

    best_f1 = max(ca.values())
    delta = best_f1 - baseline
    passed = delta >= min_delta
    status = "PASS" if passed else "FAIL"
    print(f"Probe sanity check [{status}]: "
          f"best CA F1={best_f1:.3f}, baseline={baseline:.3f}, "
          f"delta={delta:.3f} (required >{min_delta:.2f})")
    return passed
