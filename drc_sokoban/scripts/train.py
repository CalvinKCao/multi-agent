"""
Main training entry point for the DRC agent on Boxoban.

Usage:
    # Full training (PoC — 50M steps, 2-layer DRC)
    python scripts/train.py --data-dir data/boxoban_levels

    # Smoke test (fast, 100k steps, 16 channels)
    python scripts/train.py \\
        --data-dir data/boxoban_levels \\
        --num-envs 4 \\
        --target-steps 100000 \\
        --num-layers 2 \\
        --hidden-channels 16 \\
        --save-path checkpoints/smoke_test \\
        --smoke-test

    # Resume from checkpoint
    python scripts/train.py \\
        --data-dir data/boxoban_levels \\
        --resume checkpoints/agent_10M.pt

    # Killarney: 50M steps, 64 envs, checkpoints under \$PROJECT/\$USER/.../checkpoints/sa_agent*.pt
    sbatch --account=<ccdb-group> slurm_sa_boxoban_50m.sh
"""

import argparse
import os
import sys
import torch

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drc_sokoban.wandb_env import load_wandb_local_env
from drc_sokoban.envs.make_env import make_env
from drc_sokoban.training.ppo import PPOTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train DRC agent on Boxoban")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Path to boxoban-levels dataset root")
    p.add_argument("--split", type=str, default="train",
                   choices=["train", "valid"])
    p.add_argument("--difficulty", type=str, default="unfiltered",
                   choices=["unfiltered", "medium"])

    # Architecture
    p.add_argument("--hidden-channels", type=int, default=32)
    p.add_argument("--num-layers",      type=int, default=2,
                   help="DRC depth D (2 for PoC, 3 for full)")
    p.add_argument("--num-ticks",       type=int, default=3)

    # Training
    p.add_argument("--num-envs",       type=int, default=32)
    p.add_argument("--target-steps",   type=int, default=50_000_000)
    p.add_argument("--learning-rate",  type=float, default=3e-4)
    p.add_argument("--gamma",          type=float, default=0.97)
    p.add_argument("--rollout-steps",  type=int, default=20)
    p.add_argument("--ppo-epochs",     type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--entropy-coef",   type=float, default=0.01)
    p.add_argument("--max-grad-norm",  type=float, default=10.0)

    # Misc
    p.add_argument("--save-path", type=str, default="checkpoints/agent",
                   help="Base path for checkpoint files")
    p.add_argument("--save-every", type=int, default=5_000_000)
    p.add_argument("--resume",    type=str, default=None,
                   help="Resume from this checkpoint path")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-subproc", action="store_true",
                   help="Use DummyVecEnv instead of SubprocVecEnv (easier debugging)")
    p.add_argument("--smoke-test", action="store_true",
                   help="Quick smoke-test mode (overrides some settings)")
    return p.parse_args()


def main():
    args = parse_args()
    load_wandb_local_env()

    if args.smoke_test:
        args.num_envs     = max(args.num_envs, 4)
        args.target_steps = max(args.target_steps, 100_000)
        args.save_every   = args.target_steps  # save only at end
        print("=== SMOKE TEST MODE ===")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    print(f"Creating {args.num_envs} parallel environments...")
    env = make_env(
        n_envs      = args.num_envs,
        data_dir    = args.data_dir,
        split       = args.split,
        difficulty  = args.difficulty,
        max_steps   = 400,
        seed        = args.seed,
        use_subproc = not args.no_subproc,
    )

    cfg = dict(
        num_envs        = args.num_envs,
        hidden_channels = args.hidden_channels,
        num_layers      = args.num_layers,
        num_ticks       = args.num_ticks,
        learning_rate   = args.learning_rate,
        gamma           = args.gamma,
        rollout_steps   = args.rollout_steps,
        ppo_epochs      = args.ppo_epochs,
        minibatch_size  = args.minibatch_size,
        entropy_coef    = args.entropy_coef,
        max_grad_norm   = args.max_grad_norm,
        target_steps    = args.target_steps,
        save_every      = args.save_every,
    )

    trainer = PPOTrainer(env, cfg=cfg, device=args.device)

    if args.resume:
        trainer.load(args.resume)

    print(f"Training DRC agent on {trainer.device}", flush=True)
    print(f"  Layers: {args.num_layers}, Ticks: {args.num_ticks}, "
          f"Channels: {args.hidden_channels}", flush=True)
    print(f"  Target steps: {args.target_steps:,}", flush=True)

    try:
        trainer.train(save_path=args.save_path, log_every=100_000)
    finally:
        env.close()

    print("Done.")


if __name__ == "__main__":
    main()
