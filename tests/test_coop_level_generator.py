"""Tests for cooperative 6x6 Sokoban level generation."""

import numpy as np
import pytest

from drc_sokoban.envs.boxoban_env import (
    WALL,
    FLOOR,
    TARGET,
    BOX_ON_FLOOR,
    BOX_ON_TGT,
    AGENT,
    AGENT_ON_TGT,
)
from drc_sokoban.envs.level_generator import _grid_solvable_bfs
from drc_sokoban.envs.coop_level_generator import CoopLevelGenerator, make_coop_generator


def _box_target_balance(grid: np.ndarray) -> bool:
    n_boxes = int(np.sum((grid == BOX_ON_FLOOR) | (grid == BOX_ON_TGT)))
    n_tgt_cells = int(
        np.sum((grid == TARGET) | (grid == BOX_ON_TGT) | (grid == AGENT_ON_TGT))
    )
    return n_boxes == n_tgt_cells


def _wall_pattern_key(grid: np.ndarray) -> bytes:
    """Interior wall layout only (border stripped conceptually)."""
    return (grid == WALL).astype(np.uint8).tobytes()


class TestCoopLevelGenerator:
    def test_grid_shape(self):
        gen = CoopLevelGenerator(grid_size=6, n_boxes=2, seed=0)
        g = gen()
        assert g.shape == (6, 6)
        assert g.dtype == np.int32

    def test_wall_border(self):
        gen = CoopLevelGenerator(seed=1)
        g = gen()
        assert np.all(g[0, :] == WALL)
        assert np.all(g[-1, :] == WALL)
        assert np.all(g[:, 0] == WALL)
        assert np.all(g[:, -1] == WALL)

    def test_single_agent(self):
        gen = CoopLevelGenerator(seed=2)
        g = gen()
        assert int(np.sum((g == AGENT) | (g == AGENT_ON_TGT))) == 1

    def test_box_target_balance(self):
        gen = CoopLevelGenerator(seed=3)
        for _ in range(20):
            g = gen()
            assert _box_target_balance(g), g

    def test_min_two_boxes(self):
        gen = CoopLevelGenerator(n_boxes=2, seed=4)
        for _ in range(20):
            g = gen()
            n = int(np.sum((g == BOX_ON_FLOOR) | (g == BOX_ON_TGT)))
            assert n >= 2

    def test_has_internal_walls(self):
        gen = CoopLevelGenerator(seed=5)
        for _ in range(20):
            g = gen()
            inner = g[1:-1, 1:-1]
            assert int(np.sum(inner == WALL)) >= 1

    def test_solvable(self):
        gen = CoopLevelGenerator(seed=6)
        for _ in range(20):
            g = gen()
            if gen.last_scenario == "mutual_block":
                assert not _grid_solvable_bfs(g), "mutual_block should fail 1-agent BFS"
            else:
                assert _grid_solvable_bfs(g)

    def test_deterministic_seed(self):
        g1 = CoopLevelGenerator(seed=99)()
        g2 = CoopLevelGenerator(seed=99)()
        assert np.array_equal(g1, g2)

    def test_variety(self):
        gen = CoopLevelGenerator(seed=7)
        keys = set()
        for _ in range(50):
            keys.add(_wall_pattern_key(gen()))
        assert len(keys) >= 4

    @pytest.mark.parametrize(
        "scenario",
        [
            "horizontal_divide",
            "vertical_divide",
            "l_corridor",
            "center_island",
            "corner_chambers",
            "zigzag",
            "mutual_block",
        ],
    )
    def test_each_scenario(self, scenario):
        gen = CoopLevelGenerator(scenario=scenario, seed=10)
        g = gen()
        assert g.shape == (6, 6)
        assert _box_target_balance(g)
        assert int(np.sum((g == BOX_ON_FLOOR) | (g == BOX_ON_TGT))) >= 2
        if scenario == "mutual_block":
            assert not _grid_solvable_bfs(g), "mutual_block should fail 1-agent BFS"
        else:
            assert _grid_solvable_bfs(g)

    def test_ma_env_integration(self):
        from drc_sokoban.envs.ma_boxoban_env import MABoxobanEnv

        gen = make_coop_generator(grid_size=6, n_boxes=2, seed=11)

        def factory():
            return gen()

        env = MABoxobanEnv(
            grid_size=6,
            max_steps=50,
            max_steps_range=0,
            step_penalty=0.0,
            seed=11,
            level_generator=factory,
        )
        obs_a, obs_b = env.reset()
        assert obs_a.shape == (10, 6, 6)
        assert obs_b.shape == (10, 6, 6)

    def test_retry_never_fails(self):
        gen = CoopLevelGenerator(seed=12)
        for _ in range(100):
            g = gen()
            assert g is not None
            assert isinstance(g, np.ndarray)

    def test_make_coop_generator(self):
        gen = make_coop_generator(grid_size=6, n_boxes=2, seed=13)
        g = gen()
        assert g.shape == (6, 6)
        if gen.last_scenario == "mutual_block":
            assert not _grid_solvable_bfs(g)
        else:
            assert _grid_solvable_bfs(g)

    def test_grid_size_six_only(self):
        with pytest.raises(ValueError):
            CoopLevelGenerator(grid_size=8)
