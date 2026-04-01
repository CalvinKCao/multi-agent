"""
Estimate how often the MA Boxoban env actually reaches a win.

If *random* actions almost never solve in millions of steps, then a trained
policy sitting at solve=0 for 10–20M steps is not necessarily a bug — the
task may just be extremely sparse.  If random solves occasionally appear,
but training stays at 0, look at learning setup / credit assignment instead.

Usage (on cluster, from repo root):
    python -m drc_sokoban.scripts.ma_diagnose_solves \\
        --data-dir /path/to/boxoban_levels --episodes 2000 --max-steps 400
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from drc_sokoban.envs.ma_boxoban_env import MABoxobanEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--difficulty", type=str, default="unfiltered")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-boxes", type=int, default=None,
                   help="Skip resets until level has at most this many boxes (curriculum sanity).")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = MABoxobanEnv(
        data_dir=args.data_dir,
        split=args.split,
        difficulty=args.difficulty,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    wins = 0
    timeouts = 0
    total_steps = 0
    big_rew_eps = 0  # saw reward >= 10 at least once in episode

    for ep in range(args.episodes):
        env.reset()
        # Optional: only keep "easy" levels by box count
        if args.max_boxes is not None:
            tries = 0
            while env._n_boxes > args.max_boxes and tries < 50:
                env.reset()
                tries += 1

        done = False
        saw_big = False
        steps = 0
        info = {}
        while not done:
            a_a = int(rng.integers(0, 4))
            a_b = int(rng.integers(0, 4))
            _, rew, done, info = env.step((a_a, a_b))
            steps += 1
            total_steps += 1
            if rew >= 9.5:
                saw_big = True
        if info.get("solved") or saw_big:
            wins += 1
        if not info.get("solved"):
            timeouts += 1
        if saw_big:
            big_rew_eps += 1

    print(f"Episodes:        {args.episodes}")
    print(f"Total env steps: {total_steps}")
    print(f"Wins (terminal):   {wins}  ({100.0 * wins / max(args.episodes, 1):.4f}%)")
    print(f"Episodes w/ +10: {big_rew_eps}")
    print(f"Timeouts (approx): {timeouts}")
    if wins == 0:
        print(
            "\nInterpretation: random joint policy almost never completes levels.\n"
            "IPPO at solve=0 for 10–20M env-steps is *consistent* with extreme sparsity.\n"
            "Mitigations: curriculum (medium / max_boxes=1–2), longer training (50–250M),\n"
            "or verify single-agent DRC on the same data-dir reaches non-zero solve first."
        )


if __name__ == "__main__":
    main()
