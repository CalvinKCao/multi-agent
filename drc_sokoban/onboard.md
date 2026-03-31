# DRC Sokoban — Onboarding

## Purpose
Train a Deep Repeated ConvLSTM (DRC) agent on cooperative multi-agent Boxoban, then
probe Agent A's hidden state for emergent **Theory of Mind** (ToM): does it encode
information about Partner B's future intentions?

Based on Bush et al. (2025) — "Interpreting Emergent Planning in Model-Free RL."

## Quick Start

```bash
# 1. Single-agent baseline (existing, working)
python -m drc_sokoban.scripts.train  --data-dir data/boxoban_levels --smoke-test

# 2. Multi-agent IPPO training
python -m drc_sokoban.scripts.train_ma  --data-dir data/boxoban_levels --smoke-test --no-subproc

# 3. ToM probing (after training)
python -m drc_sokoban.scripts.run_tom_experiment \
    --checkpoint checkpoints/ma_selfplay_final.pt \
    --data-dir data/boxoban_levels --quick

# 4. Generate report
python -m drc_sokoban.scripts.generate_tom_report --results-dir results/tom/
```

## File Tree

```
drc_sokoban/
  envs/
    boxoban_env.py          Single-agent Boxoban env (7-ch obs, 8x8)
    make_env.py             SubprocVecEnv factory for single-agent
    ma_boxoban_env.py       Two-agent Boxoban env (10-ch egocentric obs)
    ma_make_env.py          SubprocMAVecEnv factory for MA env

  models/
    conv_lstm.py            ConvLSTMCell + DRCStack (all same-padding, no pooling)
    agent.py                DRCAgent: encoder -> DRC -> policy+value heads

  training/
    rollout_buffer.py       PPO rollout buffer (hidden state-aware)
    ppo.py                  Single-agent PPO trainer
    ippo.py                 IPPO trainer (parameter sharing, 2x batch trick)

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
    train.py                Single-agent training CLI
    run_full_experiment.py  Single-agent full probe pipeline
    run_probes.py           (secondary entry point)
    generate_report.py      Single-agent Markdown report

    train_ma.py             Multi-agent IPPO training CLI
    run_tom_experiment.py   Full ToM probing pipeline
    generate_tom_report.py  TOM_RESULTS.md generator

slurm_ma_tom.sh             Killarney Slurm script (all phases)
arch.md                     Current architecture snapshot
arch_log.md                 Append-only change history
```

## Key Architecture Facts

- **Obs channels**: 7 (single-agent) / 10 (multi-agent, egocentric)
- **Spatial invariant**: ALL Conv2d use kernel=3, padding=1 — H=W=8 is preserved
- **DRC**: D=2 layers, N=3 ticks, 32 hidden channels (PoC defaults)
- **IPPO batch trick**: stack obs_A and obs_B → single 2N-size forward pass
- **Probes**: 1x1 logistic regression on 32-dim activation at each (x,y) cell

## ToM Concepts

| Concept | Description | Classes |
|---------|-------------|---------|
| TA | Partner B's next approach direction at cell (x,y) | 5 (UP/DN/L/R/NEVER) |
| TB | Partner B's next box-push direction at cell (x,y) | 5 |
| TC | Is cell (x,y) B's next box-delivery target? | 2 (binary) |

## Gotchas

- MA env stores ONLY {WALL,FLOOR,TARGET,BOX} in grid; agent positions are separate tuples.
  Don't mix up with single-agent env which embeds AGENT tiles in the grid.
- `train_ma.py` uses `PYTHONUNBUFFERED=1` / `-u` flag in Slurm for log flushing.
- On Killarney, code MUST run from `$SCRATCH`, not `$HOME`.
- WandB run ID is saved alongside each checkpoint as `<ckpt_base>_wandb_run_id.txt`.
  Delete this file if you want a fresh WandB run on resume.
- The `partner_noise_eps` flag creates the handicapped v2 partner — this is NOT a
  separate model, just epsilon-greedy noise applied to B's actions at training time.
  For the cross-policy kill test you need a truly independently trained checkpoint:
  train with `--partner-noise-eps 0.5` and save → that's `ma_handicap_final.pt`.
