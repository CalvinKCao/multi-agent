# Architecture Snapshot

## System Overview
DRC Sokoban probing project.  Two modes: (1) single-agent planning analysis
following Bush et al. 2025; (2) multi-agent ToM experiment extending it to
two-agent cooperative Boxoban.  Supports both standard 8x8 Boxoban dataset
and procedurally generated levels at arbitrary grid sizes (e.g. 6x6 for fast
curriculum experiments).

## Data Flow

### Single-Agent
```
Boxoban levels (or LevelGenerator) -> BoxobanEnv (HxW, 7-ch obs, -0.01/step)
                                    -> SubprocVecEnv (N=32 parallel)
                                    -> DRCAgent (skips + pool-and-inject + concat readout)
                                    -> PPO (LR decay 4e-4->0, GAE lambda=0.97)
                                    -> Spatial probes on hidden states -> CA/CB F1
```

### Multi-Agent (ToM)
```
Levels (or generator) -> MABoxobanEnv (HxW, 10-ch per agent, 2 agents)
                       -> SubprocMAVecEnv (N=64 parallel)
                       -> IPPO (param-sharing, 2N-batch, shared reward)
                       -> Collect trajectories -> TA/TB/TC labels
                       -> Spatial probes on Agent A's hidden state
                       -> Kill tests -> TOM_RESULTS.md
```

## Key Modules

### Models (`models/`)
- `PoolAndInject`: mean+max pool h -> linear -> add back (global context injection)
- `ConvLSTMCell`: (B, C_in+C_h, H, W) -> (h, c) via 4-gate conv, kernel=3, pad=1
- `DRCStack`: D layers x N ticks; bottom-up skip (encoder to all layers),
  top-down skip (final h at tick n-1 to layer 0 at tick n), pool-and-inject
- `DRCAgent`: encoder -> DRCStack -> cat(h^D, encoder) -> MLP -> policy/value
  - Configurable: obs_channels, grid size (H, W), skip_connections, pool_and_inject, concat_encoder

### Environments (`envs/`)
- `BoxobanEnv`: configurable grid size, step_penalty (-0.01 default), episode cap (120+U[0,5])
- `MABoxobanEnv`: 10 channels, agent positions stored separately, A resolves first
- `LevelGenerator`: reverse-pull procedural generation, configurable grid size,
  n_boxes, internal walls; `make_sa_generator` and `make_ma_generator` helpers
- Both envs accept a `level_generator` callable or file-based dataset

### Training (`training/`)
- `RolloutBuffer`: stores TxN transitions + ConvLSTM hidden states as numpy
- `PPOTrainer`: clipped PPO, GAE (lambda=0.97), LR linear decay, hidden-state masking
- `IPPOTrainer`: parameter-sharing IPPO, same PPO core, optional frozen partner
- Both trainers log: solve_rate, mean_reward, ep_length, timeout_rate, entropy,
  value_loss, policy_loss, clip_frac, grad_norm, learning_rate, SPS

### Probing (`probing/`)
- 1x1 logistic regression probes on ConvLSTM h-tensor
- SA: CA (approach dir), CB (push dir) -- 5-class
- MA: TA/TB (partner B) + TC (box-on-target landing)
- Kill tests validate genuine planning/ToM signal

## Spatial Invariant (critical)
Every Conv2d uses kernel_size=3, padding=1 (same-padding).
Grid maps 1:1 to ConvLSTM hidden state at any HxW size.
No pooling, no striding -- this is what makes 1x1 probes meaningful.

## WandB
Project: `drc-sokoban-ma-tom`
Run naming: `sa-tiny-6x6-5M`, `ma-selfplay-50M`, `ma-handicap-10M`, etc.
Run ID persisted next to checkpoint for resume support.
