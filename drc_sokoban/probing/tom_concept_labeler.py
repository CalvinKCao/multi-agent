"""
Theory of Mind concept labels for the multi-agent probing pipeline.

We label from the perspective of Agent A observing Partner B.

Three concepts:
  TA  -- Partner approach direction
         For each cell (x,y) at timestep t: which direction will partner B
         next enter cell (x,y) from?  Same semantics as the single-agent CA
         but computed on B's trajectory.
         Labels: 0=UP  1=DOWN  2=LEFT  3=RIGHT  4=NEVER (5 classes)

  TB  -- Partner box push direction
         For each cell (x,y) at timestep t: in which direction will partner B
         next push a box off cell (x,y)?
         Labels: same 5-class scheme as TA / single-agent CB.

  TC  -- Partner delivery target (binary)
         For each cell (x,y) at timestep t: will B's *next push that places a box
         on a target* land on (x,y)?  Requires env ``onto_target`` on push events;
         legacy trajectories fall back to "next push destination" only.
         Labels: 0 = no, 1 = yes.

All labels are shape (T, 8, 8) — indexed as label[t, row, col].
"""

import numpy as np
from typing import Any, List, Optional, Tuple

from drc_sokoban.probing.concept_labeler import (
    _direction, NEVER, UP, DOWN, LEFT, RIGHT, N_CLASSES,
)

GRID_SIZE = 8  # default; functions below accept grid_size param for flexibility


def _push_from_xy(push) -> Optional[Tuple[int, int]]:
    """Normalize env trajectory entry to old box cell (x, y) = (col, row), or None."""
    if push is None:
        return None
    if isinstance(push, dict):
        return push.get("from_xy")
    return push  # legacy: stored (x, y) directly


def _push_onto_target(push) -> Optional[bool]:
    """True/False from env; None if legacy pickle had no onto_target field."""
    if push is None:
        return None
    if isinstance(push, dict):
        return bool(push.get("onto_target"))
    return None


def label_partner_ta_tb(
    partner_positions: List[Tuple[int, int]],
    partner_box_pushes: List[Optional[Tuple[int, int]]],
    box_positions_seq:  List[List[Tuple[int, int]]],
    grid_size: int = GRID_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute TA and TB labels for partner B.

    Args:
        partner_positions: list of (x, y) = (col, row) per timestep, length T
        partner_box_pushes: per timestep: None, legacy (x, y) old-box-cell, or
                            dict with from_xy / onto_target from the MA env.
        box_positions_seq:  global box positions at each timestep, length T.
                            (Used to compute TB direction.)
        grid_size:          8 for Boxoban

    Returns:
        ta_labels: (T, G, G) int32
        tb_labels: (T, G, G) int32
    """
    T = len(partner_positions)
    G = grid_size

    ta_labels = np.full((T, G, G), NEVER, dtype=np.int32)
    tb_labels = np.full((T, G, G), NEVER, dtype=np.int32)

    # ── TA: partner approach direction via backward scan ─────────────────────
    next_visit_t   = np.full((G, G), T, dtype=np.int32)
    next_visit_dir = np.full((G, G), NEVER, dtype=np.int32)

    for t in range(T - 1, -1, -1):
        if t > 0:
            px, py = partner_positions[t]
            ox, oy = partner_positions[t - 1]
            if (px, py) != (ox, oy):
                d = _direction((ox, oy), (px, py))
                next_visit_t[py, px]   = t
                next_visit_dir[py, px] = d

        ta_labels[t] = next_visit_dir.copy()
        ta_labels[t][next_visit_t <= t] = NEVER

    # ── TB: partner box push direction via backward scan ─────────────────────
    # We know WHERE a box was pushed from (partner_box_pushes) but need to
    # know WHERE it went.  We recover the direction by comparing consecutive
    # box position sets.
    box_sets = [set(bp) for bp in box_positions_seq]

    next_push_t   = np.full((G, G), T, dtype=np.int32)
    next_push_dir = np.full((G, G), NEVER, dtype=np.int32)

    for t in range(T - 1, 0, -1):
        push_from_xy = _push_from_xy(partner_box_pushes[t])
        if push_from_xy is not None:
            ox, oy = push_from_xy   # (col, row)
            # Figure out where the box landed (appeared between t-1 and t)
            disappeared = box_sets[t - 1] - box_sets[t]
            appeared    = box_sets[t] - box_sets[t - 1]
            if appeared:
                new_pos = min(appeared, key=lambda p: abs(p[0]-ox) + abs(p[1]-oy))
                d = _direction((ox, oy), new_pos)
            else:
                d = NEVER
            next_push_t[oy, ox]   = t
            next_push_dir[oy, ox] = d

        tb_labels[t - 1] = next_push_dir.copy()
        tb_labels[t - 1][next_push_t <= (t - 1)] = NEVER

    return ta_labels, tb_labels


def label_partner_tc(
    partner_box_pushes: List[Optional[Any]],
    box_positions_seq:  List[List[Tuple[int, int]]],
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """
    TC: binary — is cell (x,y) where B's *next box delivered onto a target* lands?

    Uses ``onto_target`` from the env push dict when present.  Legacy trajectories
    that only store (x,y) from_xy fall back to labelling the next push destination
    (any push), which is weaker.

    At each timestep t, at most one cell is 1: that delivery landing cell.
    """
    T = len(partner_box_pushes)
    G = grid_size
    tc_labels = np.zeros((T, G, G), dtype=np.int32)

    box_sets = [set(bp) for bp in box_positions_seq]

    # (t, dest_x, dest_y) for deliveries we count toward TC
    deliveries = []
    for t in range(1, T):
        raw = partner_box_pushes[t]
        push_from = _push_from_xy(raw)
        if push_from is None:
            continue
        ox, oy = push_from
        appeared = box_sets[t] - box_sets[t - 1]
        if not appeared:
            continue
        new_pos = min(appeared, key=lambda p: abs(p[0] - ox) + abs(p[1] - oy))
        onto = _push_onto_target(raw)
        if onto is False:
            continue
        if onto is None:
            # Old pickles: cannot distinguish delivery vs slide; keep prior behaviour.
            deliveries.append((t, new_pos[0], new_pos[1]))
        elif onto:
            deliveries.append((t, new_pos[0], new_pos[1]))

    next_dest = [None] * T
    for t in range(T):
        for ev_t, dx, dy in deliveries:
            if ev_t > t:
                next_dest[t] = (dx, dy)
                break

    for t in range(T):
        if next_dest[t] is not None:
            dx, dy = next_dest[t]
            tc_labels[t, dy, dx] = 1

    return tc_labels


def label_tom_episode(
    agent_b_positions: List[Tuple[int, int]],
    agent_b_box_pushes: List[Optional[Tuple[int, int]]],
    box_positions_seq: List[List[Tuple[int, int]]],
    grid_size: int = GRID_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all three ToM labels for a single episode.

    Returns: (ta_labels, tb_labels, tc_labels) each (T, G, G)
    """
    ta, tb = label_partner_ta_tb(
        agent_b_positions, agent_b_box_pushes, box_positions_seq, grid_size
    )
    tc = label_partner_tc(agent_b_box_pushes, box_positions_seq, grid_size)
    return ta, tb, tc


def count_valid_moves(
    obs: np.ndarray,
    agent_pos: Tuple[int, int],
) -> int:
    """
    Heuristic count of moves B could *try* from the egocentric obs (not full env sim).

    Treats a neighbor as blocked if it is a wall, or a box with no free push-through
    cell (wall / OOB / second box in obs).  Still not identical to the env legal set
    (e.g. partner occlusion), but stricter than wall-only.

    Used for the ambiguity kill test: low count ≈ "obvious", high ≈ "ambiguous".
    """
    from drc_sokoban.envs.boxoban_env import _DELTAS
    x, y = agent_pos   # col, row
    r, c = y, x
    G = obs.shape[1]   # grid height (works for any grid size)

    def has_box(rr, cc):
        if not (0 <= rr < G and 0 <= cc < G):
            return False
        return obs[2, rr, cc] + obs[3, rr, cc] > 0.5

    valid = 0
    for dr, dc in _DELTAS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < G and 0 <= nc < G):
            continue
        if obs[0, nr, nc] > 0.5:
            continue
        if has_box(nr, nc):
            br, bc = nr + dr, nc + dc
            if not (0 <= br < G and 0 <= bc < G):
                continue
            if obs[0, br, bc] > 0.5 or has_box(br, bc):
                continue
        valid += 1
    return valid
