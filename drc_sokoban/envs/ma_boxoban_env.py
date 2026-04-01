"""
Multi-agent Boxoban environment.

Two agents (A and B) cooperate on the same 8x8 grid.  Agent A is the
"observer" whose hidden state we probe for Theory of Mind signals; agent B
is the "partner" whose future intentions we try to predict.

Key design choices:
  - Grid stores only WALL / FLOOR / TARGET / BOX_ON_FLOOR / BOX_ON_TGT.
    Agent positions are maintained separately as (row, col) tuples.
  - Simultaneous moves: A resolves first, then B (A's new position treated
    as an obstacle when resolving B's move).
  - Each agent gets an egocentric 10-channel observation (see _to_ma_obs).
  - Reward is shared / cooperative: both agents get the same scalar reward.

Observation channels (10, 8, 8):
  0  wall
  1  floor   (cells NOT occupied by an agent)
  2  box on floor
  3  box on target
  4  target  (cells NOT occupied by an agent)
  5  self on floor
  6  self on target
  7  partner on floor
  8  partner on target
  9  partner's last move, encoded as a float at partner's position
        no prior move -> 0.0   UP -> 0.25   DOWN -> 0.50
        LEFT -> 0.75           RIGHT -> 1.00
"""

import numpy as np
import os
import glob
import random
from typing import Dict, List, Optional, Tuple

from drc_sokoban.envs.boxoban_env import (
    _parse_level_file,
    WALL, FLOOR, TARGET, BOX_ON_FLOOR, BOX_ON_TGT,
    AGENT, AGENT_ON_TGT,
    _DELTAS, GRID_SIZE,
)

MA_OBS_CHANNELS = 10

_LAST_MOVE_ENC = {0: 0.25, 1: 0.50, 2: 0.75, 3: 1.00}


# --------------------------------------------------------------------------- #
# Observation builder
# --------------------------------------------------------------------------- #

def _to_ma_obs(
    grid,
    self_pos,
    partner_pos,
    partner_last_move,
):
    obs = np.zeros((MA_OBS_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    obs[0] = (grid == WALL).astype(np.float32)
    obs[2] = (grid == BOX_ON_FLOOR).astype(np.float32)
    obs[3] = (grid == BOX_ON_TGT).astype(np.float32)

    sr, sc = self_pos
    pr, pc = partner_pos

    floor_mask = (grid == FLOOR).astype(np.float32)
    tgt_mask   = (grid == TARGET).astype(np.float32)
    floor_mask[sr, sc] = 0.0;  floor_mask[pr, pc] = 0.0
    tgt_mask[sr,  sc]  = 0.0;  tgt_mask[pr,  pc]  = 0.0
    obs[1] = floor_mask
    obs[4] = tgt_mask

    obs[5, sr, sc] = 1.0 if grid[sr, sc] == FLOOR   else 0.0
    obs[6, sr, sc] = 1.0 if grid[sr, sc] == TARGET  else 0.0
    obs[7, pr, pc] = 1.0 if grid[pr, pc] == FLOOR   else 0.0
    obs[8, pr, pc] = 1.0 if grid[pr, pc] == TARGET  else 0.0

    if partner_last_move is not None:
        obs[9, pr, pc] = _LAST_MOVE_ENC.get(partner_last_move, 0.0)

    return obs


# --------------------------------------------------------------------------- #
# Single-agent step on shared grid
# --------------------------------------------------------------------------- #

def _step_agent(grid, agent_pos, obstacle_pos, action):
    """
    Attempt one move.  Returns (new_pos, reward, box_push_info).

    box_push_info is None, or a dict:
      from_xy (col, row), to_xy (col, row), onto_target (bool)
    for the box that moved.  Used for ToM labelling (TB / TC).
    """
    dr, dc = _DELTAS[action]
    ar, ac = agent_pos
    nr, nc = ar + dr, ac + dc

    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
        return agent_pos, 0.0, None
    if (nr, nc) == obstacle_pos:
        return agent_pos, 0.0, None

    dest = int(grid[nr, nc])
    reward = 0.0

    if dest == WALL:
        return agent_pos, 0.0, None

    if dest in (BOX_ON_FLOOR, BOX_ON_TGT):
        br, bc = nr + dr, nc + dc
        if not (0 <= br < GRID_SIZE and 0 <= bc < GRID_SIZE):
            return agent_pos, 0.0, None
        if (br, bc) == obstacle_pos:
            return agent_pos, 0.0, None
        box_dest = int(grid[br, bc])
        if box_dest not in (FLOOR, TARGET):
            return agent_pos, 0.0, None
        if dest == BOX_ON_TGT:
            reward -= 1.0
        grid[br, bc] = BOX_ON_TGT if box_dest == TARGET else BOX_ON_FLOOR
        onto = box_dest == TARGET
        if onto:
            reward += 1.0
        grid[nr, nc] = TARGET if dest == BOX_ON_TGT else FLOOR
        # Box stood on (nr, nc) before the push; lands on (br, bc).  (x, y) = (col, row).
        box_push_info = {
            "from_xy": (int(nc), int(nr)),
            "to_xy":   (int(bc), int(br)),
            "onto_target": bool(onto),
        }
    else:
        box_push_info = None

    return (nr, nc), reward, box_push_info


# --------------------------------------------------------------------------- #
# Main environment class
# --------------------------------------------------------------------------- #

class MABoxobanEnv:
    """
    Two-agent cooperative Boxoban.

    Interface:
        (obs_a, obs_b) = env.reset()
        (obs_a, obs_b), reward, done, info = env.step((action_a, action_b))

    info keys:
        solved          bool
        agent_a_pos     (x, y) = (col, row) after step
        agent_b_pos     (x, y) after step
        box_push_a      None or dict {from_xy, to_xy, onto_target}
        box_push_b      same for B
        box_pushed_by_a (x, y) old box cell if A pushed — alias of from_xy
        box_pushed_by_b (x, y) old box cell if B pushed — alias of from_xy
    """

    def __init__(
        self,
        data_dir=None,
        split="train",
        difficulty="unfiltered",
        max_steps=400,
        seed=None,
    ):
        self.max_steps = max_steps
        self.rng       = random.Random(seed)

        self._level_files = []
        self._file_cache  = {}

        if data_dir is not None:
            pat = os.path.join(data_dir, difficulty, split, "*.txt")
            self._level_files = sorted(glob.glob(pat))
            if not self._level_files:
                self._level_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

        self._grid         = None
        self._agent_a      = None
        self._agent_b      = None
        self._last_move_a  = None
        self._last_move_b  = None
        self._n_boxes      = 0
        self._step_count   = 0
        self._solved       = False

    # ----------------------------------------------------------------------- #

    def reset(self):
        self._step_count  = 0
        self._solved      = False
        self._last_move_a = None
        self._last_move_b = None
        self._grid, self._agent_a, self._agent_b = self._load_random_level()
        self._n_boxes = int(
            np.sum((self._grid == BOX_ON_FLOOR) | (self._grid == BOX_ON_TGT))
        )
        return self._make_obs()

    def step(self, actions):
        action_a, action_b = int(actions[0]), int(actions[1])

        new_a, rew_a, push_a = _step_agent(
            self._grid, self._agent_a, self._agent_b, action_a
        )
        self._agent_a    = new_a
        self._last_move_a = action_a

        new_b, rew_b, push_b = _step_agent(
            self._grid, self._agent_b, self._agent_a, action_b
        )
        self._agent_b    = new_b
        self._last_move_b = action_b

        reward = float(rew_a + rew_b)
        self._step_count += 1

        n_on = int(np.sum(self._grid == BOX_ON_TGT))
        won  = (n_on == self._n_boxes) and (self._n_boxes > 0)
        if won:
            reward += 10.0
            self._solved = True

        done = won or (self._step_count >= self.max_steps)

        info = {
            "solved":          self._solved,
            "agent_a_pos":     (int(self._agent_a[1]), int(self._agent_a[0])),
            "agent_b_pos":     (int(self._agent_b[1]), int(self._agent_b[0])),
            "box_push_a":      push_a,
            "box_push_b":      push_b,
            "box_pushed_by_a": push_a["from_xy"] if push_a else None,
            "box_pushed_by_b": push_b["from_xy"] if push_b else None,
        }
        return self._make_obs(), reward, bool(done), info

    def get_agent_a_pos(self):
        if self._agent_a is None:
            return (0, 0)
        return (int(self._agent_a[1]), int(self._agent_a[0]))

    def get_agent_b_pos(self):
        if self._agent_b is None:
            return (0, 0)
        return (int(self._agent_b[1]), int(self._agent_b[0]))

    def get_box_positions(self):
        if self._grid is None:
            return []
        pos = np.argwhere((self._grid == BOX_ON_FLOOR) | (self._grid == BOX_ON_TGT))
        return [(int(c), int(r)) for r, c in pos]

    def get_target_positions(self):
        if self._grid is None:
            return []
        pos = np.argwhere((self._grid == TARGET) | (self._grid == BOX_ON_TGT))
        return [(int(c), int(r)) for r, c in pos]

    # ----------------------------------------------------------------------- #

    def _make_obs(self):
        obs_a = _to_ma_obs(self._grid, self._agent_a, self._agent_b, self._last_move_b)
        obs_b = _to_ma_obs(self._grid, self._agent_b, self._agent_a, self._last_move_a)
        return obs_a, obs_b

    def _load_random_level(self):
        if self._level_files:
            path = self.rng.choice(self._level_files)
            if path not in self._file_cache:
                with open(path) as f:
                    raw = f.read()
                self._file_cache[path] = _parse_level_file(raw)
            levels = self._file_cache[path]
        else:
            levels = []

        raw_grid = self.rng.choice(levels).copy() if levels else self._random_level()
        return self._extract_agents(raw_grid)

    def _extract_agents(self, raw_grid):
        """
        Strip the single-agent tile from the level and place two agents.
        Agent A gets the original position; B gets a random non-adjacent free cell.
        """
        grid = raw_grid.copy()

        orig = np.argwhere((grid == AGENT) | (grid == AGENT_ON_TGT))
        if len(orig) == 0:
            candidates = np.argwhere(grid == FLOOR)
            orig = candidates[:1] if len(candidates) else np.array([[1, 1]])

        ar, ac = int(orig[0, 0]), int(orig[0, 1])
        grid[ar, ac] = TARGET if grid[ar, ac] == AGENT_ON_TGT else FLOOR
        pos_a = (ar, ac)

        # Prefer cells not adjacent to A so B doesn't block A immediately
        free = [
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if (r, c) != pos_a
            and grid[r, c] in (FLOOR, TARGET)
            and abs(r - ar) + abs(c - ac) > 1
        ]
        if not free:
            free = [
                (r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
                if (r, c) != pos_a and grid[r, c] in (FLOOR, TARGET)
            ]
        if not free:
            free = [(1, 1)]

        pos_b = self.rng.choice(free)
        return grid, pos_a, pos_b

    def _random_level(self):
        g = np.full((GRID_SIZE, GRID_SIZE), FLOOR, dtype=np.int32)
        g[0, :] = WALL; g[-1, :] = WALL; g[:, 0] = WALL; g[:, -1] = WALL
        interior = [(r, c) for r in range(1, 7) for c in range(1, 7)]
        self.rng.shuffle(interior)
        r0, c0 = interior[0]; r1, c1 = interior[1]
        r2, c2 = interior[2]; r3, c3 = interior[3]
        g[r0, c0] = AGENT
        g[r1, c1] = BOX_ON_FLOOR
        g[r2, c2] = TARGET
        g[r3, c3] = BOX_ON_FLOOR
        return g

    @property
    def observation_space(self):
        class _S:
            shape = (MA_OBS_CHANNELS, GRID_SIZE, GRID_SIZE)
            dtype = np.float32
        return _S()

    @property
    def action_space(self):
        class _S:
            n = 4
        return _S()
