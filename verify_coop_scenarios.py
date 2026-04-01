
import numpy as np
from drc_sokoban.envs.coop_level_generator import CoopLevelGenerator, HARDCODED_SCENARIOS
from drc_sokoban.envs.ma_boxoban_env import MABoxobanEnv
from drc_sokoban.envs.boxoban_env import WALL, FLOOR, TARGET, BOX_ON_FLOOR, BOX_ON_TGT

def render_ascii(grid, pos_a, pos_b):
    H, W = grid.shape
    out = []
    for r in range(H):
        line = ""
        for c in range(W):
            if (r, c) == pos_a:
                line += "A"
            elif (r, c) == pos_b:
                line += "B"
            else:
                t = grid[r, c]
                if t == WALL: line += "#"
                elif t == FLOOR: line += " "
                elif t == TARGET: line += "."
                elif t == BOX_ON_FLOOR: line += "$"
                elif t == BOX_ON_TGT: line += "*"
                else: line += "?"
        out.append(line)
    return "\n".join(out)

def test_scenarios():
    for name in HARDCODED_SCENARIOS.keys():
        print(f"\n--- Scenario: {name} ---")
        gen = CoopLevelGenerator(scenario=name)
        raw_grid = gen()
        
        # We need to simulate how MABoxobanEnv extracts them
        # or just use MABoxobanEnv with a mock level loader
        class MockEnv(MABoxobanEnv):
            def _load_random_level(self):
                return self._extract_agents(raw_grid)
        
        env = MockEnv(grid_size=6)
        obs = env.reset()
        print(render_ascii(env._grid, env._agent_a, env._agent_b))

if __name__ == "__main__":
    test_scenarios()
