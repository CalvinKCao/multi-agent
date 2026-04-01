"""
Visualize generated or loaded Sokoban levels as ASCII and optionally PNG.

Usage:
    # Show 5 random generated 6x6 single-agent levels
    python -m drc_sokoban.scripts.visualize_levels --grid-size 6 --n-boxes 1 --count 5

    # Show 5 MA-style levels with internal walls
    python -m drc_sokoban.scripts.visualize_levels --grid-size 6 --n-boxes 2 --internal-walls 2 --count 5

    # Visualize from the boxoban dataset
    python -m drc_sokoban.scripts.visualize_levels --data-dir data/boxoban_levels --count 3

    # Save PNG grid to file
    python -m drc_sokoban.scripts.visualize_levels --grid-size 6 --count 9 --save levels.png
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from drc_sokoban.envs.boxoban_env import (
    WALL, FLOOR, TARGET, BOX_ON_FLOOR, BOX_ON_TGT, AGENT, AGENT_ON_TGT,
)

_TILE_CHAR = {
    WALL: '#', FLOOR: ' ', TARGET: '.', BOX_ON_FLOOR: '$',
    BOX_ON_TGT: '*', AGENT: '@', AGENT_ON_TGT: '+',
}

_TILE_COLOR = {
    WALL: (60, 60, 60),
    FLOOR: (200, 200, 200),
    TARGET: (255, 200, 100),
    BOX_ON_FLOOR: (180, 100, 50),
    BOX_ON_TGT: (50, 180, 50),
    AGENT: (50, 100, 220),
    AGENT_ON_TGT: (100, 150, 255),
}


def grid_to_ascii(grid: np.ndarray) -> str:
    lines = []
    for r in range(grid.shape[0]):
        lines.append(''.join(_TILE_CHAR.get(int(grid[r, c]), '?')
                             for c in range(grid.shape[1])))
    return '\n'.join(lines)


def grid_to_rgb(grid: np.ndarray, cell_px: int = 24) -> np.ndarray:
    """Render grid as an (H*cell_px, W*cell_px, 3) uint8 image."""
    H, W = grid.shape
    img = np.zeros((H * cell_px, W * cell_px, 3), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            color = _TILE_COLOR.get(int(grid[r, c]), (128, 0, 128))
            img[r*cell_px:(r+1)*cell_px, c*cell_px:(c+1)*cell_px] = color
    return img


def render_grid_batch(grids, cols=3, cell_px=24, gap=4):
    """Tile multiple grids into one image."""
    n = len(grids)
    rows_needed = (n + cols - 1) // cols
    # assume all same size
    gh, gw = grids[0].shape
    tile_h = gh * cell_px
    tile_w = gw * cell_px
    img_h = rows_needed * tile_h + (rows_needed - 1) * gap
    img_w = cols * tile_w + (cols - 1) * gap
    canvas = np.full((img_h, img_w, 3), 240, dtype=np.uint8)

    for idx, g in enumerate(grids):
        row, col = divmod(idx, cols)
        y0 = row * (tile_h + gap)
        x0 = col * (tile_w + gap)
        tile = grid_to_rgb(g, cell_px)
        canvas[y0:y0+tile_h, x0:x0+tile_w] = tile

    return canvas


def main():
    p = argparse.ArgumentParser(description="Visualize Sokoban levels")
    p.add_argument("--grid-size", type=int, default=6)
    p.add_argument("--n-boxes", type=int, default=1)
    p.add_argument("--internal-walls", type=int, default=0)
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default=None,
                   help="If set, load from boxoban dataset instead of generating")
    p.add_argument("--save", type=str, default=None,
                   help="Save PNG to this path (requires PIL)")
    args = p.parse_args()

    grids = []

    if args.data_dir:
        from drc_sokoban.envs.boxoban_env import BoxobanEnv
        env = BoxobanEnv(data_dir=args.data_dir, seed=args.seed,
                         step_penalty=0.0, max_steps=999, max_steps_range=0)
        for _ in range(args.count):
            env.reset()
            grids.append(env._grid.copy())
    else:
        from drc_sokoban.envs.level_generator import LevelGenerator
        gen = LevelGenerator(
            grid_size=args.grid_size,
            n_boxes=args.n_boxes,
            n_internal_walls=args.internal_walls,
            seed=args.seed,
        )
        for _ in range(args.count):
            grids.append(gen())

    # ASCII output
    for i, g in enumerate(grids):
        n_box = int(np.sum((g == BOX_ON_FLOOR) | (g == BOX_ON_TGT)))
        n_tgt = int(np.sum((g == TARGET) | (g == BOX_ON_TGT) | (g == AGENT_ON_TGT)))
        has_agent = int(np.sum((g == AGENT) | (g == AGENT_ON_TGT)))
        print(f"--- Level {i+1} ({g.shape[0]}x{g.shape[1]}, "
              f"{n_box} boxes, {n_tgt} targets, agent={'yes' if has_agent else 'NO'}) ---")
        print(grid_to_ascii(g))
        print()

    if args.save:
        try:
            from PIL import Image
            canvas = render_grid_batch(grids, cols=min(3, len(grids)))
            Image.fromarray(canvas).save(args.save)
            print(f"Saved to {args.save}")
        except ImportError:
            print("PIL not available; skipping PNG save. pip install Pillow")


if __name__ == "__main__":
    main()
