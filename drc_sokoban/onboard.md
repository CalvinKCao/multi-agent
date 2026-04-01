# DRC Sokoban — Onboarding

## Purpose
Train a Deep Repeated ConvLSTM (DRC) agent on cooperative multi-agent Boxoban, then
probe Agent A's hidden state for emergent **Theory of Mind** (ToM): does it encode
information about Partner B's future intentions?

Based on Bush et al. (2025) — "Interpreting Emergent Planning in Model-Free RL."

## Quick Start

```bash
# 1a. Single-agent on generated 6x6 levels (sanity check)
python -m drc_sokoban.scripts.train --use-generator --grid-size 6 --n-boxes 1 \
    --target-steps 5000000 --save-path checkpoints/sa_tiny

# 1b. Single-agent on standard 8x8 boxoban
python -m drc_sokoban.scripts.train --data-dir data/boxoban_levels

# 2. Multi-agent IPPO on generated levels
python -m drc_sokoban.scripts.train_ma --use-generator --grid-size 6 --n-boxes 2 \
    --internal-walls 2 --target-steps 10000000 --save-path checkpoints/ma_tiny

# 3. Visualize generated levels
python -m drc_sokoban.scripts.visualize_levels --grid-size 6 --n-boxes 2 --count 5

# 4. ToM probing (after training)
python -m drc_sokoban.scripts.run_tom_experiment \
    --checkpoint checkpoints/ma_selfplay_final.pt \
    --data-dir data/boxoban_levels --quick
```

## File Tree

```
drc_sokoban/
  envs/
    boxoban_env.py          Single-agent Boxoban env (configurable grid, step penalty, episode cap)
    make_env.py             SubprocVecEnv factory for single-agent
    ma_boxoban_env.py       Two-agent Boxoban env (10-ch egocentric obs)
    ma_make_env.py          SubprocMAVecEnv factory for MA env
    level_generator.py      Procedural level generation (reverse-pull, configurable size/boxes/walls)

  models/
    conv_lstm.py            ConvLSTMCell + DRCStack + PoolAndInject (skip connections, pool-inject)
    agent.py                DRCAgent: encoder -> DRC -> cat(h,enc) -> MLP -> policy/value

  training/
    rollout_buffer.py       PPO rollout buffer (hidden state-aware)
    ppo.py                  Single-agent PPO trainer (LR decay, enhanced logging)
    ippo.py                 IPPO trainer (parameter sharing, LR decay, WandB logging)

  probing/
    concept_labeler.py      CA/CB labels for single-agent (approach dir / box push)
    train_probes.py         1x1 spatial logistic probes for single-agent
    evaluate_probes.py      Metrics, tables, heatmaps for single-agent probes
    hook_manager.py         Forward-hook alternative to return_all_ticks
    kill_tests.py           Single-agent kill tests (window, random-net, OOD)
    causal_intervention.py  Activation steering experiments
    visualize.py            ASCII overlays and matplotlib plots

    tom_concept_labeler.py  TA/TB/TC labels for multi-agent ToM
    tom_train_probes.py     Spatial probes for TA/TB/TC on Agent A's state
    tom_kill_tests.py       ToM kill tests: cross-policy, ambiguity, random-net

  scripts/
    train.py                Single-agent training CLI (dataset or generator mode)
    train_ma.py             Multi-agent IPPO training CLI (dataset or generator mode)
    visualize_levels.py     ASCII + PNG visualization of generated levels
    run_full_experiment.py  Single-agent full probe pipeline
    run_probes.py           (secondary entry point)
    generate_report.py      Single-agent Markdown report
    run_tom_experiment.py   Full ToM probing pipeline
    generate_tom_report.py  TOM_RESULTS.md generator

slurm_ma_tom.sh             Killarney Slurm script (all phases)
arch.md                     Current architecture snapshot
arch_log.md                 Append-only change history
```

## Key Architecture Facts

- **Obs channels**: 7 (single-agent) / 10 (multi-agent, egocentric)
- **Spatial invariant**: ALL Conv2d use kernel=3, padding=1 -- HxW preserved at any grid size
- **DRC**: D=3 layers, N=3 ticks, 32ch (paper defaults); skip connections + pool-and-inject
- **Readout**: cat(flatten(h^D), flatten(encoder)) -> 256-dim MLP -> policy/value
- **Training**: PPO with LR linear decay 4e-4->0, GAE lambda=0.97, step penalty -0.01, ep cap 120
- **IPPO batch trick**: stack obs_A and obs_B into 2N-size forward pass
- **Probes**: 1x1 logistic regression on 32-dim activation at each (y,x) cell
- **Level generator**: reverse-pull generation for 6x6-10x10 grids, 1-4 boxes, optional walls

## ToM Concepts

| Concept | Description | Classes |
|---------|-------------|---------|
| TA | Partner B's next approach direction at cell (x,y) | 5 (UP/DN/L/R/NEVER) |
| TB | Partner B's next box-push direction at cell (x,y) | 5 |
| TC | Is cell (x,y) B's next box-delivery target? | 2 (binary) |

## W&B credentials (one place)

- Copy `wandb.local.example` → `wandb.local` at the **repo root** (file is gitignored).
- Python entrypoints call `load_wandb_local_env()` so `WANDB_API_KEY` / `WANDB_ENTITY` apply before `wandb.init`.
- Slurm: `slurm_ma_tom.sh` sources `${PROJECT_ROOT}/wandb.local` after `cd` logic (same file).
- Override file location: `export WANDB_LOCAL_FILE=/path/to/file`.
- Shell exports still win over `wandb.local` (missing keys only are filled in).

## Gotchas

- MA env stores ONLY {WALL,FLOOR,TARGET,BOX} in grid; agent positions are separate tuples.
  Don't mix up with single-agent env which embeds AGENT tiles in the grid.
- `train_ma.py` uses `PYTHONUNBUFFERED=1` / `-u` flag in Slurm for log flushing.
- On Killarney, GPU jobs MUST run from **`$SCRATCH/drc-sokoban-ma`**, not `$HOME`.
  **`$SCRATCH` is already `/scratch/<username>`** — do **not** `cd $SCRATCH/$USER` (that
  becomes `/scratch/user/user` and fails). Clone with: `cd $SCRATCH && git clone … drc-sokoban-ma`.
- Slurm **`--account=`** must be your real CCDB Group Name (e.g. `aip-boyuwang`). Never
  replace it in scripts with a placeholder string via `sed`.
- See **`.ai/cluster-paths.md`** and **`slurm_ma_tom.sh`** (paths match `ts-sandbox` style).
- WandB run ID is saved alongside each checkpoint as `<ckpt_base>_wandb_run_id.txt`.
  Delete this file if you want a fresh WandB run on resume.
- The `partner_noise_eps` flag creates the handicapped v2 partner — this is NOT a
  separate model, just epsilon-greedy noise applied to B's actions at training time.
  For the cross-policy kill test you need a truly independently trained checkpoint:
  train with `--partner-noise-eps 0.5` and save → that's `ma_handicap_final.pt`.
