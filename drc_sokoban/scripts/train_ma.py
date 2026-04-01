"""
IPPO training for multi-agent Boxoban (standard dataset or generated levels).

Usage:
    # Standard 8x8 self-play
    python -m drc_sokoban.scripts.train_ma --data-dir data/boxoban_levels

    # Tiny 6x6 generated levels (sanity check)
    python -m drc_sokoban.scripts.train_ma --use-generator --grid-size 6 --n-boxes 2 \\
        --target-steps 5000000 --save-path checkpoints/ma_tiny

    # Smoke test
    python -m drc_sokoban.scripts.train_ma --smoke-test --no-subproc
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

    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--difficulty", type=str, default="unfiltered")

    # Generator mode
    p.add_argument("--use-generator", action="store_true")
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--n-boxes", type=int, default=2)
    p.add_argument("--internal-walls", type=int, default=2)

    # Architecture
    p.add_argument("--hidden-channels", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-ticks", type=int, default=3)
    p.add_argument("--no-skip", action="store_true")
    p.add_argument("--no-pool-inject", action="store_true")
    p.add_argument("--no-concat-encoder", action="store_true")

    # Training
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--target-steps", type=int, default=50_000_000)
    p.add_argument("--learning-rate", type=float, default=4e-4)
    p.add_argument("--no-lr-decay", action="store_true")
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--gae-lambda", type=float, default=0.97)
    p.add_argument("--rollout-steps", type=int, default=20)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--step-penalty", type=float, default=-0.01)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--max-steps-range", type=int, default=5)
    p.add_argument("--partner-noise-eps", type=float, default=0.0)

    # Misc
    p.add_argument("--save-path", type=str, default="checkpoints/ma_agent")
    p.add_argument("--save-every", type=int, default=5_000_000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--additional-steps", type=int, default=None)
    p.add_argument("--partner-ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-subproc", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--wandb-project", type=str, default="drc-sokoban-ma-tom")
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    load_wandb_local_env()

    if args.smoke_test:
        args.num_envs = 4
        args.target_steps = 100_000
        args.save_every = 100_000
        if not args.use_generator and args.data_dir is None:
            args.use_generator = True
            args.grid_size = 6
        print("=== SMOKE TEST ===")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    level_gen = None
    if args.use_generator:
        from drc_sokoban.envs.level_generator import make_ma_generator
        level_gen = make_ma_generator(
            grid_size=args.grid_size, n_boxes=args.n_boxes,
            n_internal_walls=args.internal_walls, seed=args.seed,
        )

    grid_sz = args.grid_size
    print(f"Creating {args.num_envs} MA envs "
          f"({'generator ' + str(grid_sz) + 'x' + str(grid_sz) if level_gen else 'dataset'})",
          flush=True)

    env = make_ma_env(
        n_envs=args.num_envs,
        data_dir=args.data_dir,
        split=args.split,
        difficulty=args.difficulty,
        max_steps=args.max_steps,
        max_steps_range=args.max_steps_range,
        step_penalty=args.step_penalty,
        grid_size=grid_sz,
        seed=args.seed,
        use_subproc=not args.no_subproc,
        level_generator=level_gen,
    )

    run_label = args.wandb_run_name or (
        f"ma-ippo-{'handicap' if args.partner_noise_eps > 0 else 'selfplay'}-"
        f"{args.target_steps // 1_000_000}M"
    )

    obs_ch = 10
    cfg = dict(
        num_envs          = args.num_envs,
        obs_channels      = obs_ch,
        num_actions       = 4,
        hidden_channels   = args.hidden_channels,
        num_layers        = args.num_layers,
        num_ticks         = args.num_ticks,
        H                 = grid_sz,
        W                 = grid_sz,
        skip_connections  = not args.no_skip,
        pool_and_inject   = not args.no_pool_inject,
        concat_encoder    = not args.no_concat_encoder,
        learning_rate     = args.learning_rate,
        lr_decay          = not args.no_lr_decay,
        gamma             = args.gamma,
        gae_lambda        = args.gae_lambda,
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
        if args.additional_steps is not None:
            base = trainer.global_step
            trainer.cfg["target_steps"] = base + args.additional_steps
            print(f"Resume: train until {trainer.cfg['target_steps']:,} "
                  f"(ckpt {base:,} + {args.additional_steps:,})", flush=True)
        elif trainer.global_step >= args.target_steps:
            print(f"ERROR: ckpt step ({trainer.global_step:,}) >= target ({args.target_steps:,}). "
                  "Use --additional-steps or a higher --target-steps.",
                  file=sys.stderr, flush=True)
            sys.exit(1)

    n_params = sum(p.numel() for p in trainer.agent.parameters())
    print(f"DRC({args.num_layers},{args.num_ticks}) {args.hidden_channels}ch on {trainer.device} "
          f"| {n_params/1e3:.0f}K params | grid {grid_sz}x{grid_sz}", flush=True)
    print(f"  LR {args.learning_rate} | step_penalty={args.step_penalty} | "
          f"noise_eps={args.partner_noise_eps}", flush=True)
    print(f"  Target: {trainer.cfg['target_steps']:,} steps", flush=True)

    try:
        trainer.train(save_path=args.save_path, log_every=500_000)
    finally:
        env.close()
    print("Done.")


if __name__ == "__main__":
    main()
