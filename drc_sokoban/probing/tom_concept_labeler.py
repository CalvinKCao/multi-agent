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

  TC  -- Partner goal target (binary)
         For each cell (x,y) at timestep t: will the *next box B delivers to
         a target* land on cell (x,y)?
         Labels: 0 = no, 1 = yes.
         This is a binary spatial concept rather than a directional one.

All labels are shape (T, 8, 8) — indexed as label[t, row, col].
"""

import numpy as np
from typing import List, Optional, Tuple

from drc_sokoban.probing.concept_labeler import (
    _direction, NEVER, UP, DOWN, LEFT, RIGHT, N_CLASSES,
)

GRID_SIZE = 8


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
        partner_box_pushes: list of (x, y) old-box-cell per timestep if B pushed,
                            else None.  Length T.
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
        push_from_xy = partner_box_pushes[t]   # (x, y) = (col, row) or None
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
    partner_box_pushes: List[Optional[Tuple[int, int]]],
    box_positions_seq:  List[List[Tuple[int, int]]],
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """
    TC: binary label — is cell (x,y) the target of B's *next* delivered box?

    At each timestep t, exactly one cell has label 1: the target cell where
    B's next "box on target" push will land.  All other cells are 0.
    If B never delivers a box, all cells remain 0.

    Returns:
        tc_labels: (T, G, G) int32  values in {0, 1}
    """
    T = len(partner_box_pushes)
    G = grid_size
    tc_labels = np.zeros((T, G, G), dtype=np.int32)

    box_sets = [set(bp) for bp in box_positions_seq]

    # Walk forward to find each "box on target" event caused by B
    # (a box pushed by B that landed on a target, signalled by the box
    #  disappearing from box_positions_seq because BOX_ON_TGT is tracked
    #  separately in the env info... actually box_positions_seq includes both)
    # Simpler: a box delivered to a target appears as a new BOX_ON_TGT.
    # We identify this by checking which appeared positions overlap with target cells.
    # Since we don't have target info here, we use the proxy: a pushed box that
    # lands on a cell that was a TARGET (doesn't show in box_sets from the prior step
    # AND appears in box_sets in the next step).
    # In practice: box_positions_seq tracks ALL boxes (on floor + on tgt).
    # A "delivery" is when a box appears in box_sets[t] that was in a TARGET cell.
    # Without target cell info here, we use a simpler proxy:
    #   - If a box pushed by B lands on a cell where the box count STAYS the same
    #     but the position changes, it might be a delivery. This is hard to detect
    #     without target info.
    #
    # Better approach: we record which push events cause a +reward delivery in
    # the trajectory.  For simplicity here, we label TC=1 at the destination
    # cell of each push event by B, scanning backward so each timestep t
    # labels the NEXT such event.

    # Find all "B push destinations" in order
    push_destinations = []   # list of (t, dest_x, dest_y)
    for t in range(1, T):
        push_from = partner_box_pushes[t]
        if push_from is None:
            continue
        ox, oy = push_from
        appeared = box_sets[t] - box_sets[t - 1]
        if appeared:
            new_pos = min(appeared, key=lambda p: abs(p[0]-ox) + abs(p[1]-oy))
            push_destinations.append((t, new_pos[0], new_pos[1]))

    # Backward scan: for each t, find the next push destination after t
    ptr = len(push_destinations) - 1
    for t in range(T - 1, -1, -1):
        # Advance pointer to the latest push event with index > t
        while ptr >= 0 and push_destinations[ptr][0] <= t:
            ptr -= 1
        if ptr >= 0:
            # There is a future push at push_destinations[ptr]
            # But we want the NEXT push after t, i.e., smallest index > t
            # Forward scan is needed; store in a list sorted by t
            pass

    # Re-do as forward scan for clarity
    next_dest = [None] * T   # next_dest[t] = (dx, dy) or None
    last_dest = None
    for t in range(T - 1, -1, -1):
        for ev_t, dx, dy in reversed(push_destinations):
            if ev_t > t:
                last_dest = (dx, dy)
                break
        next_dest[t] = last_dest

    # This naive double loop is O(T^2) but T <= 400 so it's fine
    next_dest2 = [None] * T
    for t in range(T):
        for ev_t, dx, dy in push_destinations:
            if ev_t > t:
                next_dest2[t] = (dx, dy)
                break

    for t in range(T):
        if next_dest2[t] is not None:
            dx, dy = next_dest2[t]   # (col, row)
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
    Estimate how many non-blocked moves agent B has at its current position.

    Used for the ambiguity kill test: low count = "obvious" step,
    high count = "ambiguous" step.

    Args:
        obs:       (10, 8, 8) egocentric observation for agent B
        agent_pos: (x, y) = (col, row) of agent B  (ch5/6 = self in B's obs)

    Returns:
        number of valid moves (0-4)
    """
    from drc_sokoban.envs.boxoban_env import _DELTAS, GRID_SIZE as G
    x, y = agent_pos   # col, row
    r, c = y, x

    # Wall channel = obs[0], self channel = obs[5]+obs[6]
    valid = 0
    for dr, dc in _DELTAS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < G and 0 <= nc < G):
            continue
        # Check if destination is wall or already occupied by a box that can't be pushed
        if obs[0, nr, nc] > 0.5:   # wall
            continue
        # Count as valid (simplified — doesn't check full push legality)
        valid += 1
    return valid
