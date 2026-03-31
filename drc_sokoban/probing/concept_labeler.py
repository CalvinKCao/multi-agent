"""
Concept labeling for Bush et al. 2025 probing methodology.

Two concepts:
  CA — Agent Approach Direction: for each cell (x,y) at time t, the direction
       FROM WHICH the agent will next step onto (x,y).
  CB — Box Push Direction: for each cell (x,y) at time t, the direction
       a box will next be pushed OFF (x,y).

Labels are per-timestep per-cell:
  0 = UP (agent comes from below / box goes up)
  1 = DOWN
  2 = LEFT
  3 = RIGHT
  4 = NEVER (cell never visited / box never pushed in this episode)
"""

import numpy as np
from typing import List, Tuple


# ── Label constants ────────────────────────────────────────────────────────────
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
NEVER = 4
N_CLASSES = 5


def _direction(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
    """
    Direction of movement from from_pos to to_pos (both (x, y) = (col, row)).

    Returns one of UP / DOWN / LEFT / RIGHT.
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dy == -1:
        return UP      # moved up (row decreased)
    if dy == 1:
        return DOWN
    if dx == -1:
        return LEFT
    if dx == 1:
        return RIGHT
    return NEVER


def label_episode(
    agent_positions: List[Tuple[int, int]],
    box_positions: List[List[Tuple[int, int]]],
    grid_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CA and CB concept labels for a single episode.

    Args:
        agent_positions: list of (x, y) tuples, length T
        box_positions:   list of lists of (x, y) tuples, length T
        grid_size:       board side length (8 for Boxoban)

    Returns:
        ca_labels: (T, grid_size, grid_size) int32  — concept CA
        cb_labels: (T, grid_size, grid_size) int32  — concept CB

    NOTE: Labels use (row, col) array indexing internally:
          label[t, y, x]  where y=row, x=col.
    """
    T = len(agent_positions)
    G = grid_size

    ca_labels = np.full((T, G, G), NEVER, dtype=np.int32)
    cb_labels = np.full((T, G, G), NEVER, dtype=np.int32)

    # ── CA: agent approach direction ──────────────────────────────────────────
    # For each step t and cell (x,y), find the NEXT time after t that the
    # agent is AT (x,y).  The direction is from the previous agent position.
    for t in range(T):
        for t2 in range(t + 1, T):
            ax, ay = agent_positions[t2]
            px, py = agent_positions[t2 - 1]
            if (ax, ay) != (px, py):
                # Agent moved — label the cell it moved TO
                d = _direction((px, py), (ax, ay))
                # Only label if not already set from an earlier future visit
                if ca_labels[t, ay, ax] == NEVER:
                    ca_labels[t, ay, ax] = d

    # Efficient forward pass: for each cell record the next visit from current t
    # The above naive O(T * T) can be optimised but T=400 is manageable.

    # ── CB: box push direction ────────────────────────────────────────────────
    # For each step t and cell (x,y), find the NEXT time after t that a box
    # is pushed off (x,y).  Direction = where the box went.
    box_sets = [set(bp) for bp in box_positions]

    for t in range(T):
        for t2 in range(t + 1, T):
            # Find boxes that disappeared between t2-1 and t2
            disappeared = box_sets[t2 - 1] - box_sets[t2]
            appeared    = box_sets[t2] - box_sets[t2 - 1]
            for old_pos in disappeared:
                ox, oy = old_pos
                # Find where it went (closest appeared position)
                if appeared:
                    # Match by proximity (should be distance 1)
                    new_pos = min(appeared, key=lambda p: abs(p[0]-ox) + abs(p[1]-oy))
                    d = _direction(old_pos, new_pos)
                else:
                    d = NEVER
                if cb_labels[t, oy, ox] == NEVER:
                    cb_labels[t, oy, ox] = d

    return ca_labels, cb_labels


def label_episode_fast(
    agent_positions: List[Tuple[int, int]],
    box_positions: List[List[Tuple[int, int]]],
    grid_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Faster CA/CB labeling using forward-scan arrays instead of nested loops.

    Same semantics as label_episode but O(T * G²) instead of O(T² * G²).
    """
    T = len(agent_positions)
    G = grid_size

    ca_labels = np.full((T, G, G), NEVER, dtype=np.int32)
    cb_labels = np.full((T, G, G), NEVER, dtype=np.int32)

    # ── CA: build "next visit array" via backward scan ────────────────────────
    # next_visit[y, x] = (t, direction) of the next time agent visits (x,y)
    # We scan backwards from T-1 → 0, updating as we go.
    next_visit_t   = np.full((G, G), T, dtype=np.int32)   # T means never
    next_visit_dir = np.full((G, G), NEVER, dtype=np.int32)

    for t in range(T - 1, -1, -1):
        # At time t, agent is at agent_positions[t]
        # Record approach direction for cells visited AT t
        if t > 0:
            ax, ay = agent_positions[t]
            px, py = agent_positions[t - 1]
            if (ax, ay) != (px, py):
                d = _direction((px, py), (ax, ay))
                next_visit_t[ay, ax]   = t
                next_visit_dir[ay, ax] = d

        # ca_labels[t] = current next_visit snapshot
        ca_labels[t] = next_visit_dir.copy()
        # Cells with next_visit_t <= t have no future visit from t's perspective
        ca_labels[t][next_visit_t <= t] = NEVER

    # ── CB: build "next push off array" via backward scan ─────────────────────
    box_sets = [set(bp) for bp in box_positions]

    next_push_t   = np.full((G, G), T, dtype=np.int32)
    next_push_dir = np.full((G, G), NEVER, dtype=np.int32)

    for t in range(T - 1, 0, -1):
        disappeared = box_sets[t - 1] - box_sets[t]
        appeared    = box_sets[t] - box_sets[t - 1]
        for old_pos in disappeared:
            ox, oy = old_pos
            if appeared:
                new_pos = min(appeared, key=lambda p: abs(p[0]-ox) + abs(p[1]-oy))
                d = _direction(old_pos, new_pos)
            else:
                d = NEVER
            next_push_t[oy, ox]   = t
            next_push_dir[oy, ox] = d

        cb_labels[t - 1] = next_push_dir.copy()
        cb_labels[t - 1][next_push_t <= (t - 1)] = NEVER

    return ca_labels, cb_labels


def label_episodes_parallel(
    episodes: list,
    grid_size: int = 8,
    n_workers: int = 4,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Label a list of episode dicts in parallel.

    Each episode dict must have 'agent_positions' and 'box_positions' keys.

    Returns:
        ca_list: list of (T, G, G) CA label arrays, one per episode
        cb_list: list of (T, G, G) CB label arrays, one per episode
    """
    import multiprocessing as mp
    from functools import partial

    def _label_one(ep):
        return label_episode_fast(
            ep["agent_positions"], ep["box_positions"], grid_size
        )

    with mp.Pool(n_workers) as pool:
        results = pool.map(_label_one, episodes)

    ca_list = [r[0] for r in results]
    cb_list = [r[1] for r in results]
    return ca_list, cb_list
