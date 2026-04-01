"""
IPPO training entry point for the multi-agent ToM experiment.

Usage:
    # Self-play (Parameter-sharing IPPO, 50M steps)
    python -m drc_sokoban.scripts.train_ma \\
        --data-dir data/boxoban_levels \\
        --target-steps 50000000 \\
        --num-envs 64 \\
        --save-path checkpoints/ma_selfplay

    # Handicapped partner (v2 condition)
    python -m drc_sokoban.scripts.train_ma \\
        --data-dir data/boxoban_levels \\
        --target-steps 10000000 \\
        --num-envs 64 \\
        --partner-noise-eps 0.5 \\
        --save-path checkpoints/ma_handicap \\
        --wandb-run-name ma-ippo-handicap-10M

    # Resume from checkpoint
    python -m drc_sokoban.scripts.train_ma \\
        --data-dir data/boxoban_levels \\
        --resume checkpoints/ma_selfplay_10M.pt \\
        --save-path checkpoints/ma_selfplay

    # Smoke test
    python -m drc_sokoban.scripts.train_ma \\
        --smoke-test --no-subproc
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drc_sokoban.wandb_env import load_wandb_local_env
from drc_sokoban.envs.ma_make_env import make_ma_env
from drc_sokoban.training.ippo import IPPOTrainer


def parse_args():
    p = argparse.ArgumentParser(description="IPPO training for MA Boxoban")

    p.add_argument("--data-dir",   type=str, default=None)
    p.add_argument("--split",      type=str, default="train")
    p.add_argument("--difficulty", type=str, default="unfiltered")

    p.add_argument("--hidden-channels", type=int,   default=32)
    p.add_argument("--num-layers",      type=int,   default=2)
    p.add_argument("--num-ticks",       type=int,   default=3)
    p.add_argument("--num-envs",        type=int,   default=64)
    p.add_argument("--target-steps",    type=int,   default=50_000_000)
    p.add_argument("--learning-rate",   type=float, default=3e-4)
    p.add_argument("--gamma",           type=float, default=0.97)
    p.add_argument("--rollout-steps",   type=int,   default=20)
    p.add_argument("--ppo-epochs",      type=int,   default=4)
    p.add_argument("--minibatch-size",  type=int,   default=256)
    p.add_argument("--entropy-coef",    type=float, default=0.01)
    p.add_argument("--max-grad-norm",   type=float, default=10.0)
    p.add_argument("--partner-noise-eps", type=float, default=0.0,
                   help="Epsilon-greedy noise on partner actions (0=self-play, 0.5=handicapped)")

    p.add_argument("--save-path",    type=str, default="checkpoints/ma_agent")
    p.add_argument("--save-every",   type=int, default=5_000_000)
    p.add_argument("--resume",       type=str, default=None)
    p.add_argument("--partner-ckpt", type=str, default=None,
                   help="Fixed partner checkpoint (v2 condition).  None = self-play.")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       type=str, default="auto")
    p.add_argument("--no-subproc",   action="store_true")
    p.add_argument("--smoke-test",   action="store_true")
    p.add_argument("--wandb-project",  type=str, default="drc-sokoban-ma-tom")
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    load_wandb_local_env()

    if args.smoke_test:
        args.num_envs     = 4
        args.target_steps = 100_000
        args.save_every   = 100_000
        print("=== SMOKE TEST ===")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    print(f"Creating {args.num_envs} MA envs...", flush=True)
    env = make_ma_env(
        n_envs      = args.num_envs,
        data_dir    = args.data_dir,
        split       = args.split,
        difficulty  = args.difficulty,
        max_steps   = 400,
        seed        = args.seed,
        use_subproc = not args.no_subproc,
    )

    run_label = args.wandb_run_name or (
        f"ma-ippo-{'handicap' if args.partner_noise_eps > 0 else 'selfplay'}-"
        f"{args.target_steps // 1_000_000}M"
    )

    cfg = dict(
        num_envs          = args.num_envs,
        hidden_channels   = args.hidden_channels,
        num_layers        = args.num_layers,
        num_ticks         = args.num_ticks,
        learning_rate     = args.learning_rate,
        gamma             = args.gamma,
        rollout_steps     = args.rollout_steps,
        ppo_epochs        = args.ppo_epochs,
        minibatch_size    = args.minibatch_size,
        entropy_coef      = args.entropy_coef,
        max_grad_norm     = args.max_grad_norm,
        target_steps      = args.target_steps,
        save_every        = args.save_every,
        partner_noise_eps = args.partner_noise_eps,
        wandb_project     = args.wandb_project,
        wandb_run_name    = run_label,
    )

    trainer = IPPOTrainer(
        env, cfg=cfg, device=args.device,
        partner_ckpt=args.partner_ckpt,
    )

    if args.resume:
        trainer.load(args.resume)

    print(f"Training on {trainer.device} | {args.num_envs} envs | "
          f"{args.target_steps:,} target steps", flush=True)
    print(f"  Partner noise eps: {args.partner_noise_eps}  "
          f"(0 = full self-play)", flush=True)

    try:
        trainer.train(save_path=args.save_path, log_every=500_000)
    finally:
        env.close()

    print("Done.")


if __name__ == "__main__":
    main()
