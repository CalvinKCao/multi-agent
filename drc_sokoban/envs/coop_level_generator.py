"""
Cooperative 6x6 Sokoban templates: bottlenecks, room splits, L-corridors.

Returns the same grid format as LevelGenerator (one AGENT tile); MABoxobanEnv
places the second agent via _extract_agents.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from drc_sokoban.envs.boxoban_env import (
    WALL,
    FLOOR,
    TARGET,
    BOX_ON_FLOOR,
    AGENT,
    AGENT_B,
)
from drc_sokoban.envs.level_generator import LevelGenerator, _grid_solvable_bfs


def _empty_border(sz: int = 6) -> np.ndarray:
    g = np.full((sz, sz), FLOOR, dtype=np.int32)
    g[0, :] = WALL
    g[-1, :] = WALL
    g[:, 0] = WALL
    g[:, -1] = WALL
    return g


def _transform_layout(g: np.ndarray, k_rot: int, flip_lr: bool) -> np.ndarray:
    out = g.copy()
    if flip_lr:
        out = np.fliplr(out)
    if k_rot % 4:
        out = np.rot90(out, k=k_rot % 4)
    return out


def _floor_connected(grid: np.ndarray) -> bool:
    sz = grid.shape[0]
    free = [(r, c) for r in range(sz) for c in range(sz) if grid[r, c] != WALL]
    if not free:
        return False
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    start = free[0]
    vis = {start}
    q = [start]
    while q:
        r, c = q.pop()
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < sz and 0 <= nc < sz and grid[nr, nc] != WALL:
                if (nr, nc) not in vis:
                    vis.add((nr, nc))
                    q.append((nr, nc))
    return len(vis) == len(free)


def _base_horizontal_divide() -> np.ndarray:
    g = _empty_border(6)
    g[2, 1] = WALL
    g[2, 2] = WALL
    g[2, 4] = WALL
    return g


def _base_vertical_divide() -> np.ndarray:
    g = _empty_border(6)
    g[1, 2] = WALL
    g[2, 2] = WALL
    g[4, 2] = WALL
    return g


def _base_l_corridor() -> np.ndarray:
    g = _empty_border(6)
    g[2, 2] = WALL
    g[2, 3] = WALL
    g[3, 2] = WALL
    return g


def _base_center_island() -> np.ndarray:
    g = _empty_border(6)
    g[2, 2] = WALL
    g[3, 2] = WALL
    return g


def _base_corner_chambers() -> np.ndarray:
    """Two small stubs (lighter than four) so random placement stays solvable."""
    g = _empty_border(6)
    g[1, 2] = WALL
    g[2, 1] = WALL
    g[4, 3] = WALL
    return g


def _base_zigzag() -> np.ndarray:
    g = _empty_border(6)
    g[2, 2] = WALL
    g[3, 4] = WALL
    g[4, 2] = WALL
    return g


def _hardcoded_mutual_block() -> np.ndarray:
    """
    A strict forced-collaboration topology.
    Two agents must start on opposite sides. Pushing a box down blocks the
    only connecting corridor, trapping the agent on that side.
    """
    g = _empty_border(6)
    g[1, 2] = WALL
    g[1, 3] = WALL
    g[2, 2] = WALL
    g[2, 3] = WALL
    g[3, 2] = WALL
    g[3, 3] = WALL
    g[2, 1] = BOX_ON_FLOOR
    g[2, 4] = BOX_ON_FLOOR
    g[4, 1] = TARGET
    g[4, 4] = TARGET
    g[1, 1] = AGENT
    g[1, 4] = AGENT_B
    return g


def _hardcoded_handover() -> np.ndarray:
    """
    Agent A has the box but no access to the target.
    Agent B has the target but no access to the box.
    Separated by a wall with a 1-tile handover gap.
    """
    g = _empty_border(6)
    # Wall dividing the room with a gap at (2, 3)
    g[1, 3] = WALL
    g[3, 3] = WALL
    g[4, 3] = WALL
    
    g[2, 1] = BOX_ON_FLOOR
    g[2, 2] = BOX_ON_FLOOR
    g[1, 4] = TARGET
    g[2, 4] = TARGET
    
    g[1, 1] = AGENT
    g[4, 4] = AGENT_B
    return g


def _hardcoded_intersection() -> np.ndarray:
    """
    A central bottleneck tile that both agents must use.
    If one agent leaves their box in the intersection, the other is blocked.
    """
    g = _empty_border(6)
    # 4-room layout with center intersection
    g[2, :] = WALL
    g[:, 3] = WALL
    g[2, 3] = FLOOR # The intersection
    
    g[1, 1] = AGENT
    g[1, 2] = BOX_ON_FLOOR
    g[4, 4] = TARGET
    
    g[4, 1] = AGENT_B
    g[4, 2] = BOX_ON_FLOOR
    g[1, 4] = TARGET
    return g


def _hardcoded_gatekeeper() -> np.ndarray:
    """
    Agent A's path is blocked by a box that only Agent B can clear.
    """
    g = _empty_border(6)
    # Vertical corridor for Agent A
    g[:, 2] = WALL
    g[2, 2] = FLOOR # The gate
    
    g[1, 1] = AGENT
    g[3, 1] = BOX_ON_FLOOR
    g[4, 1] = TARGET
    
    g[2, 1] = BOX_ON_FLOOR # The "blocking" box
    
    g[1, 4] = AGENT_B
    g[4, 4] = TARGET
    g[3, 4] = BOX_ON_FLOOR
    return g


SCENARIO_BASES: Dict[str, Callable[[], np.ndarray]] = {
    "horizontal_divide": _base_horizontal_divide,
    "vertical_divide": _base_vertical_divide,
    "l_corridor": _base_l_corridor,
    "center_island": _base_center_island,
    "corner_chambers": _base_corner_chambers,
    "zigzag": _base_zigzag,
}

HARDCODED_SCENARIOS: Dict[str, Callable[[], np.ndarray]] = {
    "mutual_block": _hardcoded_mutual_block,
    "handover": _hardcoded_handover,
    "intersection": _hardcoded_intersection,
    "gatekeeper": _hardcoded_gatekeeper,
}

SCENARIOS: List[str] = list(SCENARIO_BASES.keys()) + list(HARDCODED_SCENARIOS.keys())


def _try_place_and_solve(
    wall_grid: np.ndarray,
    n_boxes: int,
    rng: np.random.RandomState,
    max_tries: int = 80,
) -> Optional[np.ndarray]:
    floors: List[Tuple[int, int]] = [
        (r, c)
        for r in range(6)
        for c in range(6)
        if wall_grid[r, c] == FLOOR
    ]
    need = 2 * n_boxes + 1
    if len(floors) < need:
        return None

    floors_arr = np.array(floors, dtype=np.int64)

    for _ in range(max_tries):
        perm = rng.permutation(len(floors))
        pick = floors_arr[perm[:need]]
        g = wall_grid.copy()
        for i in range(n_boxes):
            r, c = int(pick[i, 0]), int(pick[i, 1])
            g[r, c] = TARGET
        for i in range(n_boxes):
            r, c = int(pick[n_boxes + i, 0]), int(pick[n_boxes + i, 1])
            g[r, c] = BOX_ON_FLOOR
        ar, ac = int(pick[2 * n_boxes, 0]), int(pick[2 * n_boxes, 1])
        g[ar, ac] = AGENT
        if not _floor_connected(g):
            continue
        if _grid_solvable_bfs(g):
            return g
    return None


class CoopLevelGenerator:
    """
    Template-based 6x6 levels with internal walls tuned for MA bottlenecks.
    """

    def __init__(
        self,
        grid_size: int = 6,
        n_boxes: int = 2,
        seed: Optional[int] = None,
        scenario: Optional[str] = None,
    ):
        if grid_size != 6:
            raise ValueError("CoopLevelGenerator only supports grid_size=6")
        if n_boxes != 2:
            raise ValueError("CoopLevelGenerator only supports n_boxes=2")
        self.grid_size = grid_size
        self.n_boxes = n_boxes
        self.scenario = scenario
        if scenario is not None and scenario not in SCENARIOS:
            raise ValueError(
                f"unknown scenario {scenario!r}; expected one of {SCENARIOS}"
            )
        self.rng = random.Random(seed)
        self._np_rng = np.random.RandomState(
            seed if seed is not None else random.randint(0, 2**31 - 1)
        )
        self.last_scenario = None

    def __call__(self) -> np.ndarray:
        for _ in range(150):
            g = self._try_once()
            if g is not None:
                return g
        self.last_scenario = "fallback"
        return self._fallback()

    def _try_once(self) -> Optional[np.ndarray]:
        pool = [self.scenario] if self.scenario else SCENARIOS
        name = self.rng.choice(pool)
        self.last_scenario = name
        
        k_rot = self.rng.randint(0, 4)
        flip_lr = self.rng.randint(0, 2) == 1
        
        if name in HARDCODED_SCENARIOS:
            g = HARDCODED_SCENARIOS[name]()
            return _transform_layout(g, k_rot, flip_lr)

        base_fn = SCENARIO_BASES[name]
        base = base_fn()
        k_rot = self.rng.randint(0, 4)
        flip_lr = self.rng.randint(0, 2) == 1
        wall_grid = _transform_layout(base, k_rot, flip_lr)
        return _try_place_and_solve(wall_grid, self.n_boxes, self._np_rng)

    def _fallback(self) -> np.ndarray:
        base = _base_horizontal_divide()
        for _ in range(200):
            k = self.rng.randint(0, 4)
            flip = self.rng.randint(0, 2) == 1
            wall_grid = _transform_layout(base, k, flip)
            g = _try_place_and_solve(
                wall_grid, self.n_boxes, self._np_rng, max_tries=120
            )
            if g is not None:
                return g
        # Reverse-pull generator always returns a solvable 6x6 (may lack coop topology).
        lg = LevelGenerator(
            grid_size=6,
            n_boxes=2,
            n_internal_walls=2,
            pull_steps_range=(3, 8),
            seed=self.rng.randint(0, 2**31 - 1),
        )
        return lg()


def make_coop_generator(
    grid_size: int = 6,
    n_boxes: int = 2,
    seed: Optional[int] = None,
    scenario: Optional[str] = None,
):
    """Callable for MABoxobanEnv / training, same idea as make_ma_generator."""
    return CoopLevelGenerator(
        grid_size=grid_size,
        n_boxes=n_boxes,
        seed=seed,
        scenario=scenario,
    )
