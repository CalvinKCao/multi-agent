"""
Export cooperative generator levels with visual solution walkthroughs (ASCII + PNG).

Each scenario folder includes:
  - initial_two_agent.* — the **MA training** layout: same map with agents A and B
    (positions from MABoxobanEnv.reset; generator only embeds one @, the env splits
    into two bodies).
  - walkthrough / filmstrip — **single-agent** shortest path (same solvability check
    as the generator). Joint two-agent action sequences are not computed here.

Uses single-agent BFS (shortest action sequence) matching BoxobanEnv physics.
Frames are written every ``--stride`` steps (always includes step 0 and final).

Usage:
    python -m drc_sokoban.scripts.export_coop_walkthrough \\
        --out-dir drc_sokoban/proof_coop_walkthrough --stride 3 --png

    # One scenario only
    python -m drc_sokoban.scripts.export_coop_walkthrough --scenario zigzag --stride 2
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from drc_sokoban.envs.boxoban_env import (
    WALL,
    FLOOR,
    TARGET,
    BOX_ON_FLOOR,
    BOX_ON_TGT,
    AGENT_ON_TGT,
    _apply_action,
)
from drc_sokoban.envs.coop_level_generator import SCENARIOS, CoopLevelGenerator
from drc_sokoban.envs.ma_boxoban_env import MABoxobanEnv, _step_agent

from drc_sokoban.scripts.visualize_levels import _TILE_COLOR, grid_to_ascii, grid_to_rgb


def _is_solved(g: np.ndarray) -> bool:
    n_t = int(np.sum((g == TARGET) | (g == BOX_ON_TGT) | (g == AGENT_ON_TGT)))
    n_on = int(np.sum(g == BOX_ON_TGT))
    return n_t > 0 and n_on == n_t


ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


def solve_shortest_actions(init: np.ndarray, max_states: int = 400_000) -> Optional[List[int]]:
    """BFS; returns list of actions 0..3 from init to a solved state, or None."""
    start = init.copy()
    if _is_solved(start):
        return []
    s0 = start.tobytes()
    q = deque([start])
    parent: dict = {s0: None}

    while q:
        if len(parent) > max_states:
            return None
        g = q.popleft()
        g_key = g.tobytes()
        for a in range(4):
            gc = g.copy()
            _, won = _apply_action(gc, a)
            if won:
                cur_key = gc.tobytes()
                parent[cur_key] = (g_key, a)
                path: List[int] = []
                k = cur_key
                while k != s0:
                    p = parent[k]
                    if p is None:
                        return None
                    prev_k, act = p
                    path.append(act)
                    k = prev_k
                path.reverse()
                return path
            nk = gc.tobytes()
            if nk not in parent:
                parent[nk] = (g_key, a)
                q.append(gc)
    return None


def solve_ma_shortest_actions(
    init_grid: np.ndarray,
    init_pos_a: Tuple[int, int],
    init_pos_b: Tuple[int, int],
    max_states: int = 400_000,
) -> Optional[List[Tuple[int, int]]]:
    """
    Two-agent BFS using _step_agent.
    Returns list of joint actions [(aA, aB), ...] to solve, or None.
    """
    from drc_sokoban.envs.ma_boxoban_env import _step_agent

    start_grid = init_grid.copy()
    
    def _is_solved_ma(g: np.ndarray) -> bool:
        n_t = int(np.sum((g == TARGET) | (g == BOX_ON_TGT)))
        n_on = int(np.sum(g == BOX_ON_TGT))
        return n_t > 0 and n_on == n_t

    if _is_solved_ma(start_grid):
        return []

    def _state_key(g, pa, pb):
        return g.tobytes() + bytes(f"|{pa}|{pb}", "ascii")

    s0 = _state_key(start_grid, init_pos_a, init_pos_b)
    q = deque([(start_grid, init_pos_a, init_pos_b)])
    parent: dict = {s0: None}

    while q:
        if len(parent) > max_states:
            return None
        g, pa, pb = q.popleft()
        g_key = _state_key(g, pa, pb)

        for act_a in range(4):
            for act_b in range(4):
                gc = g.copy()
                
                # MABoxobanEnv order: A resolves, then B
                n_pa, _, _ = _step_agent(gc, pa, pb, act_a)
                n_pb, _, _ = _step_agent(gc, pb, n_pa, act_b)

                won = _is_solved_ma(gc)
                cur_key = _state_key(gc, n_pa, n_pb)

                if won:
                    parent[cur_key] = (g_key, (act_a, act_b))
                    path: List[Tuple[int, int]] = []
                    k = cur_key
                    while k != s0:
                        p = parent[k]
                        if p is None:
                            return None
                        prev_k, act = p
                        path.append(act)
                        k = prev_k
                    path.reverse()
                    return path

                if cur_key not in parent:
                    parent[cur_key] = (g_key, (act_a, act_b))
                    q.append((gc, n_pa, n_pb))

    return None


def _steps_to_save(total_actions: int, stride: int) -> List[int]:
    """Indices 0..total_actions inclusive (grid after k actions); sample every stride."""
    if total_actions <= 0:
        return [0]
    out = list(range(0, total_actions + 1, max(1, stride)))
    if out[-1] != total_actions:
        out.append(total_actions)
    # dedupe sorted
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _safe_dir_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


_TILE_CHAR_BASE = {
    WALL: "#",
    FLOOR: " ",
    TARGET: ".",
    BOX_ON_FLOOR: "$",
    BOX_ON_TGT: "*",
}

# MA snapshot: distinct from single-agent @ (training: A = probe side, B = partner)
_COLOR_A = (50, 100, 220)
_COLOR_B = (200, 60, 160)


def _ma_layout_from_raw(raw_grid: np.ndarray, env_seed: int):
    """Strip embedded @ and place A/B like MABoxobanEnv."""

    def _gen():
        return raw_grid.copy()

    env = MABoxobanEnv(
        grid_size=raw_grid.shape[0],
        max_steps=50,
        max_steps_range=0,
        step_penalty=0.0,
        seed=env_seed,
        level_generator=_gen,
    )
    env.reset()
    return env._grid.copy(), env._agent_a, env._agent_b


def ma_snapshot_to_ascii(grid: np.ndarray, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> str:
    H, W = grid.shape
    lines = []
    for r in range(H):
        row_chars = []
        for c in range(W):
            if (r, c) == pos_a:
                row_chars.append("A")
            elif (r, c) == pos_b:
                row_chars.append("B")
            else:
                row_chars.append(_TILE_CHAR_BASE.get(int(grid[r, c]), "?"))
        lines.append("".join(row_chars))
    return "\n".join(lines)


def ma_snapshot_to_rgb(
    grid: np.ndarray,
    pos_a: Tuple[int, int],
    pos_b: Tuple[int, int],
    cell_px: int = 32,
) -> np.ndarray:
    H, W = grid.shape
    img = np.zeros((H * cell_px, W * cell_px, 3), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            if (r, c) == pos_a:
                col = _COLOR_A
            elif (r, c) == pos_b:
                col = _COLOR_B
            else:
                col = _TILE_COLOR.get(int(grid[r, c]), (128, 0, 128))
            img[r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px] = col
    return img


def _stable_scenario_offset(name: str) -> int:
    return sum(ord(c) for c in name) % 9_917


def export_one_scenario(
    scenario: str,
    out_root: str,
    base_seed: int,
    stride: int,
    write_png: bool,
    cell_px: int,
    max_states: int,
) -> Tuple[str, int, int]:
    """
    Returns (folder_path, num_actions, num_frames_written).
    Retries seeds until BFS finds a path.
    """
    folder = os.path.join(out_root, _safe_dir_name(scenario))
    os.makedirs(folder, exist_ok=True)
    ascii_dir = os.path.join(folder, "ascii")
    png_dir = os.path.join(folder, "png")
    os.makedirs(ascii_dir, exist_ok=True)
    if write_png:
        os.makedirs(png_dir, exist_ok=True)

    grid = None
    actions_1a: Optional[List[int]] = None
    actions_ma: Optional[List[Tuple[int, int]]] = None
    used_seed = base_seed
    
    is_mutual_block = scenario == "mutual_block"
    
    for attempt in range(50):
        used_seed = base_seed + attempt
        gen = CoopLevelGenerator(seed=used_seed, scenario=scenario)
        g = gen()
        
        if is_mutual_block:
            # Requires MA solve
            mg, ma, mb = _ma_layout_from_raw(g, used_seed)
            path_ma = solve_ma_shortest_actions(mg, ma, mb, max_states=max_states)
            if path_ma is not None:
                grid = g
                actions_ma = path_ma
                break
        else:
            # Use 1-agent check
            path = solve_shortest_actions(g, max_states=max_states)
            if path is not None:
                grid = g
                actions_1a = path
                break

    if grid is None or (not is_mutual_block and actions_1a is None) or (is_mutual_block and actions_ma is None):
        raise RuntimeError(f"could not solve any generated level for scenario={scenario!r}")

    ma_grid, ma_a, ma_b = _ma_layout_from_raw(grid, used_seed)
    
    if is_mutual_block:
        assert actions_ma is not None
        n_act = len(actions_ma)
        prefixes_ma: List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]] = [(ma_grid.copy(), ma_a, ma_b)]
        
        cur_g, ca, cb = ma_grid.copy(), ma_a, ma_b
        from drc_sokoban.envs.ma_boxoban_env import _step_agent
        for (aa, ab) in actions_ma:
            na, _, _ = _step_agent(cur_g, ca, cb, aa)
            nb, _, _ = _step_agent(cur_g, cb, na, ab)
            
            n_t = int(np.sum((cur_g == TARGET) | (cur_g == BOX_ON_TGT)))
            n_on = int(np.sum(cur_g == BOX_ON_TGT))
            won = n_t > 0 and n_on == n_t
            
            prefixes_ma.append((cur_g.copy(), na, nb))
            ca, cb = na, nb
            if won:
                break
                
        if len(prefixes_ma) != n_act + 1:
            raise RuntimeError(f"replay length mismatch MA: {len(prefixes_ma)} states vs {n_act + 1} for {scenario}")
    else:
        assert actions_1a is not None
        n_act = len(actions_1a)
        prefixes: List[np.ndarray] = [grid.copy()]
        cur = grid.copy()
        for a in actions_1a:
            _, won = _apply_action(cur, int(a))
            prefixes.append(cur.copy())
            if won:
                break
        if len(prefixes) != n_act + 1:
            raise RuntimeError(
                f"replay length mismatch: {len(prefixes)} states vs {n_act + 1} for {scenario}"
            )

    pil_ok = write_png
    if pil_ok:
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            pil_ok = False

    step_indices = _steps_to_save(n_act, stride)
    frames_meta: List[Tuple[int, str]] = []

    for si in step_indices:
        label = f"step_{si:04d}"
        ascii_path = os.path.join(ascii_dir, f"{label}.txt")
        
        if is_mutual_block:
            gk, pa, pb = prefixes_ma[si]
            with open(ascii_path, "w", encoding="utf-8") as f:
                f.write(ma_snapshot_to_ascii(gk, pa, pb))
                f.write("\n")
            if pil_ok:
                from PIL import Image as _PIL_Image
                rgb = ma_snapshot_to_rgb(gk, pa, pb, cell_px=cell_px)
                _PIL_Image.fromarray(rgb).save(os.path.join(png_dir, f"{label}.png"))
        else:
            gk = prefixes[si]
            with open(ascii_path, "w", encoding="utf-8") as f:
                f.write(grid_to_ascii(gk))
                f.write("\n")
            if pil_ok:
                from PIL import Image as _PIL_Image
                rgb = grid_to_rgb(gk, cell_px=cell_px)
                _PIL_Image.fromarray(rgb).save(os.path.join(png_dir, f"{label}.png"))
        frames_meta.append((si, label))

    ma_grid, ma_a, ma_b = _ma_layout_from_raw(grid, used_seed)
    with open(os.path.join(folder, "initial_two_agent.txt"), "w", encoding="utf-8") as f:
        f.write(ma_snapshot_to_ascii(ma_grid, ma_a, ma_b))
        f.write("\n")
    if pil_ok:
        from PIL import Image as _PIL_Image

        _PIL_Image.fromarray(ma_snapshot_to_rgb(ma_grid, ma_a, ma_b, cell_px)).save(
            os.path.join(folder, "initial_two_agent.png")
        )

    md_path = os.path.join(folder, "walkthrough.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {scenario}\n\n")
        f.write("## Two-agent layout (MABoxoban training)\n\n")
        f.write(
            "The coop generator stores **one** `@` in the file; `MABoxobanEnv` strips it and "
            "spawns **A** (blue) and **B** (magenta) — see `initial_two_agent.png` / `.txt`. "
            "That is the actual two-body starting state for IPPO.\n\n"
        )
        if not is_mutual_block:
            f.write(f"- Agent **A** (row,col) = `{ma_a}`  |  **B** = `{ma_b}`\n\n")
            f.write("## Single-agent shortest solution (solvability proof)\n\n")
            f.write(
                "The generator checks **single-agent** solvability. The frames below replay one "
                "optimal **one-agent** path (not a joint two-agent schedule).\n\n"
            )
            f.write(f"- generator seed: `{used_seed}`\n")
            f.write(f"- shortest 1-agent solution: **{n_act}** actions\n")
            f.write(f"- frames: every **{stride}** steps (+ final); see `ascii/` and `png/`.\n\n")
            for si, label in frames_meta:
                f.write(f"## Step {si} / {n_act}\n\n")
                f.write("```\n")
                f.write(grid_to_ascii(prefixes[si]))
                f.write("\n```\n\n")
        else:
            f.write(f"- generator seed: `{used_seed}`\n")
            f.write(f"- shortest MA solution: **{n_act}** joint actions\n")
            f.write(f"- frames: every **{stride}** steps (+ final); see `ascii/` and `png/`.\n\n")
            for si, label in frames_meta:
                gk, pa, pb = prefixes_ma[si]
                f.write(f"## Step {si} / {n_act}\n\n")
                f.write("```\n")
                f.write(ma_snapshot_to_ascii(gk, pa, pb))
                f.write("\n```\n\n")

    actions_path = os.path.join(folder, "solution_actions.txt")
    with open(actions_path, "w", encoding="utf-8") as f:
        if is_mutual_block:
            assert actions_ma is not None
            for i, (aa, ab) in enumerate(actions_ma):
                f.write(f"{i+1:4d}  A:{ACTION_NAMES[aa]:<5s} B:{ACTION_NAMES[ab]:<5s} ({aa}, {ab})\n")
        else:
            assert actions_1a is not None
            for i, a in enumerate(actions_1a):
                f.write(f"{i+1:4d}  {ACTION_NAMES[a]} ({a})\n")

    # Filmstrip (optional quick glance)
    if pil_ok and frames_meta:
        try:
            from PIL import Image

            if is_mutual_block:
                imgs = [
                    ma_snapshot_to_rgb(gk, pa, pb, cell_px=cell_px)
                    for si, label in frames_meta
                    for gk, pa, pb in [prefixes_ma[si]]
                ]
            else:
                imgs = [
                    grid_to_rgb(prefixes[si], cell_px=cell_px)
                    for si, _ in frames_meta
                ]
            gap = 6
            h = max(im.shape[0] for im in imgs)
            w = sum(im.shape[1] for im in imgs) + gap * (len(imgs) - 1)
            canvas = np.full((h, w, 3), 255, dtype=np.uint8)
            x = 0
            for im in imgs:
                canvas[: im.shape[0], x : x + im.shape[1]] = im
                x += im.shape[1] + gap
            Image.fromarray(canvas).save(os.path.join(folder, "filmstrip.png"))
        except ImportError:
            pass

    meta_path = os.path.join(folder, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"scenario={scenario}\n")
        f.write(f"generator_seed={used_seed}\n")
        f.write(f"base_seed_arg={base_seed}\n")
        f.write(f"actions_1agent_shortest={n_act}\n")
        f.write(f"stride={stride}\n")
        f.write(f"frames={len(frames_meta)}\n")
        f.write(f"ma_agent_a_rowcol={ma_a}\n")
        f.write(f"ma_agent_b_rowcol={ma_b}\n")

    return folder, n_act, len(frames_meta)


def main():
    p = argparse.ArgumentParser(description="Export coop levels + visual solution walkthrough")
    p.add_argument(
        "--out-dir",
        type=str,
        default="drc_sokoban/proof_coop_walkthrough",
        help="Root folder; one subfolder per scenario",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Emit ASCII/PNG every this many actions (0 and last always included)",
    )
    p.add_argument("--png", action="store_true", help="Write png/ + filmstrip.png (needs Pillow)")
    p.add_argument("--cell-px", type=int, default=32)
    p.add_argument("--max-states", type=int, default=400_000)
    p.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Single scenario; default: all SCENARIOS",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=9001,
        help="Per-scenario seed = base_seed + hash(scenario) mod 10000 (stable ids)",
    )
    args = p.parse_args()

    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)
    if args.scenario and args.scenario not in SCENARIOS:
        print(f"ERROR: unknown scenario; choose from {SCENARIOS}", file=sys.stderr)
        sys.exit(1)

    root = os.path.abspath(args.out_dir)
    os.makedirs(root, exist_ok=True)

    readme = os.path.join(root, "README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "Cooperative level generator — proof pack.\n\n"
            "initial_two_agent.* = real MA layout (A/B from MABoxobanEnv).\n"
            "filmstrip / walkthrough = single-agent shortest path (solvability check).\n\n"
            "Regenerate: python -m drc_sokoban.scripts.export_coop_walkthrough "
            "--out-dir drc_sokoban/proof_coop_walkthrough --stride 3 --png\n"
        )

    summary = []
    for sc in scenarios:
        sub_seed = args.base_seed + _stable_scenario_offset(sc)
        folder, n_act, n_fr = export_one_scenario(
            sc,
            root,
            base_seed=sub_seed,
            stride=max(1, args.stride),
            write_png=args.png,
            cell_px=args.cell_px,
            max_states=args.max_states,
        )
        summary.append((sc, n_act, n_fr, folder))
        print(f"{sc}: {n_act} actions, {n_fr} frames -> {folder}")

    sum_path = os.path.join(root, "SUMMARY.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        for sc, n_act, n_fr, folder in summary:
            f.write(f"{sc}\tactions={n_act}\tframes={n_fr}\t{folder}\n")
    print(f"Wrote {sum_path}")


if __name__ == "__main__":
    main()
