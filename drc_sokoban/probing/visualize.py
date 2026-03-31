"""
Phase 6 — Visualisation.

1. Saliency maps: probe F1 as a spatial heatmap over the 8×8 board.
2. Plan prediction overlay: probe's CA/CB prediction on top of the game grid.
3. "Smoking gun" frame: at episode start, the hidden state at the goal cell
   already encodes the correct approach direction before the agent has moved.
4. Tick-progression plots: how probe confidence grows across DRC ticks.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Tile rendering ─────────────────────────────────────────────────────────────

TILE_SYMBOLS = {0: "█", 1: " ", 2: "◇", 3: "◆", 4: "·", 5: "@", 6: "+"}
DIRECTION_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "·"}
DIRECTION_NAMES   = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "NEVER"}


def obs_to_grid_str(obs: np.ndarray) -> str:
    """Render a (7, 8, 8) observation as an ASCII grid."""
    ch_to_sym = {0: "█", 1: " ", 2: "◇", 3: "◆", 4: "·", 5: "@", 6: "+"}
    grid = np.argmax(obs, axis=0)   # (8, 8)
    rows = []
    for r in range(8):
        row = "".join(ch_to_sym.get(int(grid[r, c]), "?") for c in range(8))
        rows.append("|" + row + "|")
    return "\n".join(rows)


# ── Probe confidence maps ──────────────────────────────────────────────────────

def plot_probe_confidence_grid(
    probe,
    hidden_state: np.ndarray,
    layer: int,
    tick: int,
    target_class: int,
    title: str = "Probe confidence",
    save_path: Optional[str] = None,
    obs: Optional[np.ndarray] = None,
):
    """
    Visualise the probe's confidence for `target_class` at every (x, y) cell.

    `hidden_state`: (num_ticks, num_layers, 32, 8, 8) for ONE timestep.
    Returns (8, 8) float array of probabilities, and optionally saves a figure.
    """
    confidence_grid = np.zeros((8, 8), dtype=np.float32)

    for y in range(8):
        for x in range(8):
            feat = hidden_state[tick, layer, :, y, x].reshape(1, -1)  # (1, 32)
            try:
                proba = probe.predict_proba(feat)[0]
                classes = probe.named_steps["clf"].classes_
                idx = np.where(classes == target_class)[0]
                confidence_grid[y, x] = proba[idx[0]] if len(idx) > 0 else 0.0
            except Exception:
                confidence_grid[y, x] = 0.0

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(1, 2 if obs is not None else 1,
                                 figsize=(10 if obs is not None else 5, 5))
        if obs is None:
            axes = [axes]

        # Confidence heatmap
        im = axes[0].imshow(confidence_grid, vmin=0, vmax=1, cmap="hot", origin="upper")
        axes[0].set_title(f"{title}\n(class={DIRECTION_NAMES.get(target_class, target_class)})")
        axes[0].set_xlabel("x (col)"); axes[0].set_ylabel("y (row)")
        plt.colorbar(im, ax=axes[0])

        # Direction arrows overlay
        for y in range(8):
            for x in range(8):
                if confidence_grid[y, x] > 0.3:
                    axes[0].text(x, y, DIRECTION_SYMBOLS.get(target_class, "?"),
                                 ha="center", va="center", fontsize=8,
                                 color="cyan", alpha=confidence_grid[y, x])

        # Optional: game grid side-by-side
        if obs is not None and len(axes) > 1:
            tile_grid = np.argmax(obs, axis=0)   # (8, 8)
            cmap_board = plt.cm.get_cmap("tab10", 7)
            axes[1].imshow(tile_grid, cmap=cmap_board, vmin=0, vmax=6, origin="upper")
            axes[1].set_title("Game board")
            legend_labels = ["wall", "floor", "box", "box-on-tgt", "target", "agent", "agent+tgt"]
            patches = [plt.matplotlib.patches.Patch(color=cmap_board(i), label=legend_labels[i])
                       for i in range(7)]
            axes[1].legend(handles=patches, loc="upper right", fontsize=6)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close(fig)
    except ImportError:
        pass   # matplotlib not available

    return confidence_grid


def plot_tick_progression(
    probe,
    hidden_state: np.ndarray,
    layer: int,
    x: int,
    y: int,
    title: str = "Probe confidence vs. tick",
    save_path: Optional[str] = None,
):
    """
    Show how the probe's confidence evolves across DRC ticks for one cell (x,y).
    Replicates Bush et al. Fig 6 (probe F1 improving with more ticks).
    """
    num_ticks = hidden_state.shape[0]
    tick_probs = np.zeros((num_ticks, 5), dtype=np.float32)

    for tick in range(num_ticks):
        feat = hidden_state[tick, layer, :, y, x].reshape(1, -1)
        try:
            proba = probe.predict_proba(feat)[0]
            classes = probe.named_steps["clf"].classes_
            for ci, c in enumerate(classes):
                if c < 5:
                    tick_probs[tick, c] = proba[ci]
        except Exception:
            pass

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for c in range(5):
            ax.plot(range(num_ticks), tick_probs[:, c],
                    marker="o", label=DIRECTION_NAMES.get(c, str(c)))
        ax.set_xlabel("DRC tick"); ax.set_ylabel("Predicted probability")
        ax.set_xticks(range(num_ticks))
        ax.set_xticklabels([f"Tick {t}" for t in range(num_ticks)])
        ax.legend(fontsize=8); ax.set_title(f"{title} — cell ({x},{y}), layer {layer}")
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
    except ImportError:
        pass

    return tick_probs


def plot_smoking_gun(
    episode: dict,
    probe,
    layer: int,
    tick: int,
    concept: str = "CA",
    t_early: int = 0,
    t_late: int = -1,
    save_path: Optional[str] = None,
):
    """
    "Smoking gun" visualisation:
    Show that at t=0 (start of level), the hidden state at the goal cell
    ALREADY encodes the correct future approach direction — before the agent
    has taken a single step.

    Overlays the probe's prediction on the game board for two timesteps:
    t_early (episode start) and t_late (later in episode, for comparison).
    """
    from drc_sokoban.probing.concept_labeler import label_episode_fast

    ca, cb = label_episode_fast(episode["agent_positions"], episode["box_positions"])
    labels = ca if concept == "CA" else cb

    T = len(episode["hidden_states"])
    if t_late < 0:
        t_late = min(T - 1, T // 3)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for subplot_idx, t in enumerate([t_early, t_late]):
            if t >= T:
                continue
            hs = episode["hidden_states"][t]   # (ticks, layers, 32, 8, 8)
            obs = episode["observations"][t]    # (7, 8, 8)

            # Board
            tile_grid = np.argmax(obs, axis=0)
            cmap_board = plt.cm.get_cmap("Pastel1", 7)
            axes[subplot_idx * 2].imshow(tile_grid, cmap=cmap_board, vmin=0, vmax=6, origin="upper")
            axes[subplot_idx * 2].set_title(f"t={t}: Game board")

            # Probe prediction
            pred_grid = np.full((8, 8), 4, dtype=np.int32)  # default NEVER
            for y in range(8):
                for x in range(8):
                    feat = hs[tick, layer, :, y, x].reshape(1, -1)
                    try:
                        pred_grid[y, x] = probe.predict(feat)[0]
                    except Exception:
                        pass

            arrow_colors = {0: "blue", 1: "red", 2: "green", 3: "orange", 4: "gray"}
            axes[subplot_idx * 2 + 1].imshow(tile_grid, cmap=cmap_board, vmin=0, vmax=6,
                                              origin="upper", alpha=0.4)
            for y in range(8):
                for x in range(8):
                    pred = int(pred_grid[y, x])
                    true = int(labels[t, y, x])
                    sym = DIRECTION_SYMBOLS.get(pred, "?")
                    color = "lime" if pred == true and pred != 4 else arrow_colors.get(pred, "gray")
                    axes[subplot_idx * 2 + 1].text(x, y, sym, ha="center", va="center",
                                                    fontsize=10, color=color, fontweight="bold")

            axes[subplot_idx * 2 + 1].set_title(
                f"t={t}: {concept} probe predictions\n"
                f"(green=correct, other=wrong/NEVER)"
            )

        plt.suptitle("Smoking Gun: Agent's plan is encoded before it acts", fontsize=12)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close(fig)
    except ImportError:
        pass


# ── ASCII summary (no matplotlib needed) ──────────────────────────────────────

def ascii_probe_overlay(
    obs: np.ndarray,
    hs: np.ndarray,
    probe,
    layer: int,
    tick: int,
    ca_labels: Optional[np.ndarray] = None,
) -> str:
    """
    Print an ASCII overlay: game board + probe predictions side by side.
    Works without matplotlib.
    """
    tile_grid = np.argmax(obs, axis=0)          # (8, 8)
    sym_map = {0: "█", 1: " ", 2: "◇", 3: "◆", 4: "·", 5: "@", 6: "+"}

    pred_grid = np.full((8, 8), 4, dtype=np.int32)
    for y in range(8):
        for x in range(8):
            feat = hs[tick, layer, :, y, x].reshape(1, -1)
            try:
                pred_grid[y, x] = int(probe.predict(feat)[0])
            except Exception:
                pass

    lines = [f"{'Board':^10}     {'CA Probe Pred':^10}"]
    if ca_labels is not None:
        lines = [f"{'Board':^10}     {'CA Probe':^10}  {'True CA':^10}"]

    for r in range(8):
        board_row = "".join(sym_map.get(int(tile_grid[r, c]), "?") for c in range(8))
        probe_row = "".join(DIRECTION_SYMBOLS.get(int(pred_grid[r, c]), "?") for c in range(8))
        if ca_labels is not None:
            true_row  = "".join(DIRECTION_SYMBOLS.get(int(ca_labels[r, c]), "?") for c in range(8))
            lines.append(f"|{board_row}|  |{probe_row}|  |{true_row}|")
        else:
            lines.append(f"|{board_row}|  |{probe_row}|")

    return "\n".join(lines)
