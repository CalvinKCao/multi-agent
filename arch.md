# Architecture Snapshot

## System Overview
DRC Sokoban probing project.  Two modes: (1) single-agent planning analysis
following Bush et al. 2025; (2) multi-agent ToM experiment extending it to
two-agent cooperative Boxoban.

## Data Flow

### Single-Agent
```
Boxoban levels (900k, .txt) → BoxobanEnv (8x8, 7-ch obs)
                             → SubprocVecEnv (N=32 parallel)
                             → DRCAgent forward pass
                             → PPO update
                             → Spatial probes on hidden states → CA/CB F1
```

### Multi-Agent (ToM)
```
Boxoban levels → MABoxobanEnv (8x8, 10-ch obs per agent, 2 agents)
               → SubprocMAVecEnv (N=64 parallel MA envs)
               → IPPO (param-sharing, 2N-batch trick, shared reward)
               → Collect trajectories → TA/TB/TC labels
               → Spatial probes on Agent A's hidden state
               → Kill tests → TOM_RESULTS.md
```

## Key Modules

### Models (`models/`)
- `ConvLSTMCell`: (B, C_in+C_h, H, W) → (h, c) via 4-gate conv, kernel=3, pad=1
- `DRCStack`: D cells × N ticks, returns all_tick_hiddens for probing
- `DRCAgent`: encoder (1-layer conv) → DRCStack → flatten → policy + value heads
  - `obs_channels` is configurable: 7 (single) or 10 (MA)

### Environments (`envs/`)
- `BoxobanEnv`: 8×8 playable interior, AGENT tile in grid, 7 one-hot channels
- `MABoxobanEnv`: 10 channels, agent positions stored separately from grid,
  _step_agent() resolves A first then B, shared cooperative reward
- Both backed by the official Boxoban levels dataset

### Training (`training/`)
- `RolloutBuffer`: stores T×N transitions + ConvLSTM hidden states as numpy
- `PPOTrainer`: standard clipped PPO, GAE, hidden-state masking at episode ends
- `IPPOTrainer`: wraps PPOTrainer logic; stacks A/B into 2N-batch for efficiency,
  supports fixed partner_ckpt for v2 (handicapped) condition

### Probing (`probing/`)
- Probes are 1×1 logistic regression classifiers — they see only the 32-dim
  activation at position (y, x) in the ConvLSTM h-tensor
- Single-agent: CA (approach dir) and CB (push dir) — 5-class
- Multi-agent: TA/TB (partner B) + TC (next box-on-target landing; uses env `onto_target` on `box_push_b`)
- ToM training: probe train/val split by **episode** (`GroupShuffleSplit`); cross-policy reuses **fitted** probes on v2
- Kill tests validate the signal is genuine planning / ToM, not artefact

## Spatial Invariant (critical)
Every Conv2d in the entire pipeline uses kernel_size=3, padding=1 (same-padding).
The 8×8 grid maps 1:1 to the 8×8 ConvLSTM hidden state.
No pooling, no striding.  This is what makes 1×1 probes meaningful.

## WandB
Project: `drc-sokoban-ma-tom`
Run naming: `ma-selfplay-50M`, `ma-handicap-10M`, `tom-probes-step50M`, etc.
Run ID persisted next to checkpoint for resume support.
