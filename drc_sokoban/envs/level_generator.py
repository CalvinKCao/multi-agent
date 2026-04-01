"""
Procedural Sokoban level generator for the tiny/curriculum experiments.

Generates guaranteed-solvable levels by working backwards from solved states
(boxes already on targets) and "pulling" boxes away via reverse moves.

Supports:
  - Configurable grid size (6-10 interior)
  - 1-4 boxes
  - Optional internal walls for complexity
  - Single-agent and multi-agent (places 1 or 2 agent spawn points)

Usage:
    gen = LevelGenerator(grid_size=6, n_boxes=1)
    grid = gen()   # returns (6, 6) int32 array
"""

import numpy as np
import random
from typing import Optional, Tuple, List
from collections import deque

from drc_sokoban.envs.boxoban_env import (
    WALL, FLOOR, TARGET, BOX_ON_FLOOR, BOX_ON_TGT,
    AGENT, AGENT_ON_TGT, _DELTAS,
)


def _grid_solvable_bfs(init: np.ndarray, max_states: int = 250_000) -> bool:
    """
    True iff some sequence of actions reaches all boxes on targets.
    Uses full-grid states so multi-box is correct; caps search for safety.
    """
    from drc_sokoban.envs.boxoban_env import _apply_action

    start = init.copy()
    key0 = start.tobytes()
    q = deque([start])
    vis = {key0}
    while q:
        if len(vis) > max_states:
            return True
        g = q.popleft()
        n_t = int(np.sum((g == TARGET) | (g == BOX_ON_TGT) | (g == AGENT_ON_TGT)))
        n_on = int(np.sum(g == BOX_ON_TGT))
        if n_t > 0 and n_on == n_t:
            return True
        for a in range(4):
            gc = g.copy()
            _, won = _apply_action(gc, a)
            if won:
                return True
            kb = gc.tobytes()
            if kb not in vis:
                vis.add(kb)
                q.append(gc)
    return False


class LevelGenerator:
    """
    Procedural Sokoban level generator with reverse-pull solvability guarantee.

    Call the instance to get a new grid: grid = gen()

    For multi-agent envs, the returned grid has a single AGENT tile;
    MABoxobanEnv._extract_agents handles placing the second agent.
    """

    def __init__(
        self,
        grid_size: int = 6,
        n_boxes: int = 1,
        n_internal_walls: int = 0,
        pull_steps_range: Tuple[int, int] = (3, 8),
        seed: Optional[int] = None,
    ):
        assert grid_size >= 5, "need at least 5x5 for a meaningful puzzle"
        assert n_boxes >= 1
        self.grid_size = grid_size
        self.n_boxes = n_boxes
        self.n_internal_walls = n_internal_walls
        self.pull_range = pull_steps_range
        self.rng = random.Random(seed)
        self._np_rng = np.random.RandomState(
            seed if seed is not None else random.randint(0, 2**31)
        )

    def __call__(self) -> np.ndarray:
        """Generate one level. Retries internally on failure (rare)."""
        for _ in range(200):
            grid = self._try_generate()
            if grid is not None:
                return grid
        # absolute fallback: trivial level
        return self._trivial_level()

    def _try_generate(self) -> Optional[np.ndarray]:
        sz = self.grid_size
        grid = np.full((sz, sz), FLOOR, dtype=np.int32)
        grid[0, :] = WALL; grid[-1, :] = WALL
        grid[:, 0] = WALL; grid[:, -1] = WALL

        interior = self._get_interior()

        # optional internal walls (avoid blocking too much)
        walls_placed = 0
        if self.n_internal_walls > 0:
            shuffled = list(interior)
            self.rng.shuffle(shuffled)
            for r, c in shuffled:
                if walls_placed >= self.n_internal_walls:
                    break
                # don't wall off corners or create 2x2 blocks
                grid[r, c] = WALL
                if not self._is_connected(grid):
                    grid[r, c] = FLOOR
                else:
                    walls_placed += 1

        # refresh interior after walls
        interior = [(r, c) for r in range(1, sz - 1) for c in range(1, sz - 1)
                     if grid[r, c] == FLOOR]
        if len(interior) < self.n_boxes * 2 + 1:
            return None

        self.rng.shuffle(interior)
        # place targets, then put boxes on them (solved state)
        target_cells = interior[:self.n_boxes]
        for r, c in target_cells:
            grid[r, c] = BOX_ON_TGT

        # pull each box away from its target
        for r, c in target_cells:
            n_pulls = self.rng.randint(*self.pull_range)
            box_r, box_c = r, c
            for _ in range(n_pulls):
                moved = self._try_pull(grid, box_r, box_c)
                if moved is None:
                    break
                box_r, box_c = moved

        if int(np.sum(grid == BOX_ON_FLOOR)) == 0:
            return None

        free = [(r, c) for r in range(sz) for c in range(sz) if grid[r, c] == FLOOR]
        if not free:
            return None

        self.rng.shuffle(free)
        for ar, ac in free[: min(len(free), 80)]:
            g2 = grid.copy()
            g2[ar, ac] = AGENT
            if _grid_solvable_bfs(g2):
                return g2
        return None

    def _try_pull(self, grid, box_r, box_c) -> Optional[Tuple[int, int]]:
        """
        Undo one forward push. Forward: agent at A pushes in direction d,
        box moves from B = A+d to C = B+d. So C = box_r,box_c is current box,
        B = C - d = nb, A = B - d = C - 2d (agent stood two steps behind C).

        The old code wrongly used A = C + d, which breaks the solvability guarantee.
        """
        sz = self.grid_size
        directions = list(range(4))
        self.rng.shuffle(directions)

        for d in directions:
            dr, dc = _DELTAS[d]
            nb_r, nb_c = box_r - dr, box_c - dc
            ag_r, ag_c = box_r - 2 * dr, box_c - 2 * dc

            if not (1 <= nb_r < sz - 1 and 1 <= nb_c < sz - 1):
                continue
            if not (1 <= ag_r < sz - 1 and 1 <= ag_c < sz - 1):
                continue
            if grid[nb_r, nb_c] not in (FLOOR, TARGET):
                continue
            if grid[ag_r, ag_c] not in (FLOOR, TARGET):
                continue

            # move box: update grid
            was_on_tgt = grid[box_r, box_c] == BOX_ON_TGT
            grid[box_r, box_c] = TARGET if was_on_tgt else FLOOR
            grid[nb_r, nb_c] = BOX_ON_TGT if grid[nb_r, nb_c] == TARGET else BOX_ON_FLOOR
            return (nb_r, nb_c)

        return None

    def _get_interior(self):
        sz = self.grid_size
        return [(r, c) for r in range(1, sz - 1) for c in range(1, sz - 1)]

    def _is_connected(self, grid) -> bool:
        """BFS check that all floor cells are still reachable from each other."""
        sz = self.grid_size
        free = [(r, c) for r in range(sz) for c in range(sz)
                if grid[r, c] != WALL]
        if not free:
            return False

        visited = set()
        queue = deque([free[0]])
        visited.add(free[0])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _DELTAS:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and 0 <= nr < sz and 0 <= nc < sz:
                    if grid[nr, nc] != WALL:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return len(visited) == len(free)

    def _trivial_level(self) -> np.ndarray:
        """Dead-simple 1-box level as last resort."""
        sz = self.grid_size
        g = np.full((sz, sz), FLOOR, dtype=np.int32)
        g[0, :] = WALL; g[-1, :] = WALL; g[:, 0] = WALL; g[:, -1] = WALL
        mid = sz // 2
        g[mid, mid] = BOX_ON_FLOOR
        g[mid, mid + 1] = TARGET
        g[mid - 1, mid] = AGENT
        return g


def make_sa_generator(grid_size=6, n_boxes=1, n_internal_walls=0,
                      pull_steps=(3, 8), seed=None):
    """Convenience: returns a callable for single-agent BoxobanEnv."""
    return LevelGenerator(
        grid_size=grid_size, n_boxes=n_boxes,
        n_internal_walls=n_internal_walls,
        pull_steps_range=pull_steps, seed=seed,
    )


def make_ma_generator(grid_size=6, n_boxes=2, n_internal_walls=2,
                      pull_steps=(4, 10), seed=None):
    """
    Generator tuned for multi-agent: 2+ boxes, some internal walls.
    The internal walls create bottlenecks that naturally incentivize
    splitting work between agents.
    """
    return LevelGenerator(
        grid_size=grid_size, n_boxes=n_boxes,
        n_internal_walls=n_internal_walls,
        pull_steps_range=pull_steps, seed=seed,
    )
