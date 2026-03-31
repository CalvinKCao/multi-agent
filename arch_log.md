# Architecture Log (append-only)

---

## 2026-03-31 — Multi-Agent ToM Extension

### Added
- `envs/ma_boxoban_env.py`: Two-agent Boxoban.  Grid stores WALL/FLOOR/TARGET/BOX;
  agent positions separate.  A resolves first, B second.  10-ch egocentric obs.
- `envs/ma_make_env.py`: SubprocMAVecEnv + DummyMAVecEnv for MA environments.
- `training/ippo.py`: IPPO with parameter sharing.  obs_A and obs_B stacked into
  2N-batch for one forward pass per step.  Optional fixed partner_ckpt for v2 condition.
  WandB integrated with run-ID persistence for resume.
- `probing/tom_concept_labeler.py`: TA (partner approach dir), TB (partner push dir),
  TC (partner goal target, binary) labels via backward scan, O(T) per concept.
- `probing/tom_train_probes.py`: 1×1 spatial logistic probes for TA/TB (5-class)
  and TC (binary) on Agent A's hidden state.
- `probing/tom_kill_tests.py`: cross-policy generalization, ambiguity, random-weights.
- `scripts/train_ma.py`: IPPO training CLI with self-play / handicapped modes.
- `scripts/run_tom_experiment.py`: Full ToM probing pipeline (collect → label → probe
  → kill tests → heatmaps → summary JSON).
- `scripts/generate_tom_report.py`: TOM_RESULTS.md generator.
- `slurm_ma_tom.sh`: Killarney H100 job script (gpubase_h100_b4) with phase dispatch.
- `onboard.md`, `arch.md`, `arch_log.md`: project docs.

### Components (current full list)
- DRCAgent (obs_channels=7 or 10, D=2 layers, N=3 ticks, 32 ch) — unchanged
- ConvLSTMCell / DRCStack — unchanged (same-padding everywhere)
- BoxobanEnv / SubprocVecEnv — unchanged
- MABoxobanEnv / SubprocMAVecEnv — NEW
- PPOTrainer — unchanged
- IPPOTrainer — NEW (extends PPO logic for 2-agent shared-weight setup)
- ProbeTrainer (single-agent CA/CB) — unchanged
- TomProbeTrainer (TA/TB/TC) — NEW

### Data flow changes vs single-agent
- Env step input: (action_a, action_b) joint tuple
- Env step output: (obs_a, obs_b), shared_reward, done, info
- Buffer: 2N entries per step (A in slots 0..N-1, B in N..2N-1)
- Probing target: Agent A's hidden state only (observer); B is the probed agent

---

## (prior entry placeholder — add single-agent initial build date here)
