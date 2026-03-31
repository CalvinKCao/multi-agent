"""
Boxoban environment wrapper.

Boxoban levels are 10×10 (outer ring = wall border, 8×8 interior = playable).
We strip the outer wall ring and expose the 8×8 interior as the observation.

Observation: (7, 8, 8) float32 — 7 binary one-hot channels over the 8×8 board.
  Ch 0: wall       Ch 1: floor     Ch 2: box-on-floor
  Ch 3: box-on-tgt Ch 4: target    Ch 5: agent       Ch 6: agent-on-tgt

Level loading is LAZY: each env discovers file paths at init (fast directory
scan) and loads only one randomly selected file per reset() call.  This avoids
the startup bottleneck of all 32 subprocesses reading 900 files simultaneously.
"""

import numpy as np
import os
import glob
import random
from typing import List, Optional, Tuple

# ── Tile constants (shared with the rest of the project) ──────────────────────
WALL         = 0
FLOOR        = 1
TARGET       = 2
BOX_ON_FLOOR = 3
BOX_ON_TGT   = 4
AGENT        = 5
AGENT_ON_TGT = 6

TILE_TO_CH = {
    WALL: 0, FLOOR: 1, BOX_ON_FLOOR: 2,
    BOX_ON_TGT: 3, TARGET: 4, AGENT: 5, AGENT_ON_TGT: 6,
}

# Sokoban text character → tile constant
_CHAR = {
    "#": WALL, " ": FLOOR, ".": TARGET,
    "$": BOX_ON_FLOOR, "*": BOX_ON_TGT,
    "@": AGENT,        "+": AGENT_ON_TGT,
}

# Action: 0=up 1=down 2=left 3=right  →  (row_delta, col_delta)
_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SIZE    = 8   # playable interior (10×10 board minus 1-cell wall border)
OBS_CHANNELS = 7
NUM_ACTIONS  = 4


# ── Level-file parser ──────────────────────────────────────────────────────────

def _parse_level_file(content: str) -> List[np.ndarray]:
    """
    Parse one Boxoban .txt file → list of (8, 8) int32 grids.

    Each level in the file is 10×10 (outer ring = walls).
    We strip the outer ring and return the 8×8 interior.
    """
    grids: List[np.ndarray] = []
    current: List[str] = []

    def _flush():
        if not current:
            return
        # Pad / trim to exactly 10 rows × 10 cols
        rows = []
        for line in current[:10]:
            row = [_CHAR.get(ch, WALL) for ch in line[:10]]
            while len(row) < 10:
                row.append(WALL)
            rows.append(row)
        while len(rows) < 10:
            rows.append([WALL] * 10)

        full = np.array(rows, dtype=np.int32)   # (10, 10)
        grids.append(full[1:9, 1:9].copy())     # (8, 8) interior
        current.clear()

    for line in content.splitlines():
        if line.startswith(";"):
            _flush()
        else:
            current.append(line.rstrip("\n"))
    _flush()
    return grids


# ── Observation helper ─────────────────────────────────────────────────────────

def _to_obs(grid: np.ndarray) -> np.ndarray:
    """(8, 8) int32 → (7, 8, 8) float32 one-hot."""
    obs = np.zeros((OBS_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for tile, ch in TILE_TO_CH.items():
        obs[ch] = (grid == tile).astype(np.float32)
    return obs


# ── In-place Sokoban step ──────────────────────────────────────────────────────

def _apply_action(grid: np.ndarray, action: int) -> Tuple[float, bool]:
    """
    Apply one action to grid IN-PLACE.  Returns (reward, won).

    Reward shaping (following Bush et al. 2025):
      +1.0   box pushed onto a target
      -1.0   box pushed off a target (strong discouragement)
      +10.0  all boxes on targets (level solved)

    `won` is True only when ALL boxes are on targets.
    """
    dr, dc = _DELTAS[action]
    pos = np.argwhere((grid == AGENT) | (grid == AGENT_ON_TGT))
    if len(pos) == 0:
        return 0.0, True

    ar, ac = int(pos[0, 0]), int(pos[0, 1])
    agent_was_on_tgt = bool(grid[ar, ac] == AGENT_ON_TGT)
    nr, nc = ar + dr, ac + dc

    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
        return 0.0, False

    dest = int(grid[nr, nc])
    reward = 0.0

    if dest == WALL:
        return 0.0, False

    if dest in (BOX_ON_FLOOR, BOX_ON_TGT):
        # Try to push the box
        br, bc = nr + dr, nc + dc
        if not (0 <= br < GRID_SIZE and 0 <= bc < GRID_SIZE):
            return 0.0, False
        box_dest = int(grid[br, bc])
        if box_dest not in (FLOOR, TARGET):
            return 0.0, False   # blocked — can't push
        if dest == BOX_ON_TGT:
            reward -= 1.0       # box pushed OFF a target (penalty)
        grid[br, bc] = BOX_ON_TGT if box_dest == TARGET else BOX_ON_FLOOR
        if box_dest == TARGET:
            reward += 1.0       # box pushed ONTO a target
        # Agent steps into box's old cell
        grid[nr, nc] = AGENT_ON_TGT if dest == BOX_ON_TGT else AGENT
    else:
        # Simple move (floor or target)
        grid[nr, nc] = AGENT_ON_TGT if dest == TARGET else AGENT

    # Restore vacated cell
    grid[ar, ac] = TARGET if agent_was_on_tgt else FLOOR

    # Win check: all targets covered by boxes (not by agent)
    n_targets = int(np.sum(
        (grid == TARGET) | (grid == BOX_ON_TGT) | (grid == AGENT_ON_TGT)
    ))
    n_on_tgt = int(np.sum(grid == BOX_ON_TGT))
    if n_targets > 0 and n_on_tgt == n_targets:
        reward += 10.0
        return reward, True

    return reward, False


# ── Main environment class ─────────────────────────────────────────────────────

class BoxobanEnv:
    """
    Boxoban / Sokoban environment for the DRC probing project.

    obs  = env.reset()                          → (7, 8, 8) float32
    obs, rew, done, info = env.step(action)     → action ∈ {0,1,2,3}
    pos  = env.get_agent_pos()                  → (x, y) = (col, row)
    boxes = env.get_box_positions()             → [(x,y), ...]
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "train",
        difficulty: str = "unfiltered",
        max_steps: int = 400,
        seed: Optional[int] = None,
    ):
        self.max_steps  = max_steps
        self.rng        = random.Random(seed)

        # Lazy load: discover file paths at init (directory scan only — fast)
        self._level_files: List[str] = []
        self._file_cache: dict = {}   # path → List[np.ndarray]

        if data_dir is not None:
            pat = os.path.join(data_dir, difficulty, split, "*.txt")
            self._level_files = sorted(glob.glob(pat))
            if not self._level_files:
                self._level_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

        # Episode state
        self._grid: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._solved: bool = False

    # ── Public interface ───────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._step_count = 0
        self._solved     = False
        self._grid       = self._load_random_level()
        return _to_obs(self._grid)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._step_count += 1
        reward, won = _apply_action(self._grid, action)
        # Level is "solved" only when ALL boxes reach targets (won=True)
        self._solved = self._solved or won
        done = won or (self._step_count >= self.max_steps)
        return _to_obs(self._grid), float(reward), bool(done), {"solved": self._solved}

    def get_agent_pos(self) -> Tuple[int, int]:
        """Return (col, row) = (x, y)."""
        if self._grid is None:
            return (0, 0)
        pos = np.argwhere((self._grid == AGENT) | (self._grid == AGENT_ON_TGT))
        if len(pos) == 0:
            return (0, 0)
        r, c = pos[0]
        return (int(c), int(r))

    def get_box_positions(self) -> List[Tuple[int, int]]:
        if self._grid is None:
            return []
        pos = np.argwhere((self._grid == BOX_ON_FLOOR) | (self._grid == BOX_ON_TGT))
        return [(int(c), int(r)) for r, c in pos]

    def get_target_positions(self) -> List[Tuple[int, int]]:
        if self._grid is None:
            return []
        pos = np.argwhere(
            (self._grid == TARGET) | (self._grid == BOX_ON_TGT) | (self._grid == AGENT_ON_TGT)
        )
        return [(int(c), int(r)) for r, c in pos]

    # ── Level loading ──────────────────────────────────────────────────────────

    def _load_random_level(self) -> np.ndarray:
        if not self._level_files:
            return self._random_level()

        # Pick a random file; parse it on first access, then cache
        path = self.rng.choice(self._level_files)
        if path not in self._file_cache:
            with open(path) as f:
                self._file_cache[path] = _parse_level_file(f.read())

        levels = self._file_cache[path]
        if not levels:
            return self._random_level()
        return self.rng.choice(levels).copy()

    def _random_level(self) -> np.ndarray:
        """Minimal random level (fallback when no dataset available)."""
        g = np.full((GRID_SIZE, GRID_SIZE), FLOOR, dtype=np.int32)
        g[0, :] = WALL; g[-1, :] = WALL
        g[:, 0] = WALL; g[:, -1] = WALL
        interior = [(r, c) for r in range(1, 7) for c in range(1, 7)]
        self.rng.shuffle(interior)
        r0, c0 = interior[0]; r1, c1 = interior[1]; r2, c2 = interior[2]
        g[r0, c0] = AGENT
        g[r1, c1] = BOX_ON_FLOOR
        g[r2, c2] = TARGET
        return g

    # ── Gym-compatible stubs ───────────────────────────────────────────────────

    @property
    def observation_space(self):
        class _S:
            shape = (OBS_CHANNELS, GRID_SIZE, GRID_SIZE)
            dtype = np.float32
        return _S()

    @property
    def action_space(self):
        class _S:
            n = NUM_ACTIONS
        return _S()
