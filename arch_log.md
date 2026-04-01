# Architecture Log (append-only)

---

## 2026-04-01 -- LevelGenerator reverse-pull bugfix + BFS check

- `_try_pull`: agent cell for a legal undo was `box + d`; correct forward geometry is
  agent at `box - 2*d` (two steps behind the box along the push direction). Wrong
  geometry broke the solvability guarantee; some printed levels were unsolvable.
- After placing the agent, try up to 80 random floor spawns and keep the first layout
  that passes `_grid_solvable_bfs` (full-grid BFS using `_apply_action`, capped states).

---

## 2026-04-01 -- Paper-matching DRC + tiny experiment infrastructure

### Architecture (models/)
- `PoolAndInject` module: mean+max pool h -> linear -> broadcast-add back to h
- `DRCStack`: bottom-up skip (encoder cat'd to all layers), top-down skip
  (final h at tick n-1 feeds layer 0 at tick n), pool-and-inject per layer
- `DRCAgent`: concat(h^D, encoder) -> 256-dim MLP -> policy/value heads
- All new features default=True; set False for ablation vs old minimal stack
- DRC(3,3) 32ch is now ~932K params (was ~180K without skips/MLP)

### Environment (envs/)
- `BoxobanEnv` + `MABoxobanEnv`: step_penalty (-0.01 default), configurable
  episode cap (120+U[0,5]), configurable grid_size, `level_generator` callable
- `LevelGenerator`: reverse-pull procedural generation, 5x5-10x10 grids,
  1-4 boxes, optional internal walls, connectivity-checked
- `make_sa_generator` / `make_ma_generator` convenience helpers

### Training (training/)
- LR linear decay 4e-4 -> 0 over target_steps (both PPO and IPPO)
- GAE lambda 0.97 (was 0.95)
- Enhanced logging: ep_length, timeout_rate, clip_frac, grad_norm, learning_rate
- WandB logging expanded with all new metrics

### Scripts
- `train.py` / `train_ma.py`: --use-generator, --grid-size, --n-boxes,
  --internal-walls, --step-penalty, --max-steps, --no-skip, --no-lr-decay, etc.
- `visualize_levels.py`: ASCII + PNG rendering of generated or dataset levels

---

## 2026-03-31 — ToM / MA env probe fixes

- `ma_boxoban_env`: push info dict `box_push_{a,b}` (`from_xy`, `to_xy`, `onto_target`); `box_pushed_by_*` = box source cell (was wrong: agent cell).
- `tom_concept_labeler`: TB uses normalized push `from_xy`; TC uses `onto_target` when present, else legacy “any push destination”; `count_valid_moves` adds box push-through check (still heuristic).
- `tom_train_probes`: episode-level train/val split (`GroupShuffleSplit`); `prepare_tom_dataset(..., return_episode_ids=True)`; `return_probes` + `evaluate_fitted_tom_probes` for v1→v2 transfer.
- `tom_kill_tests`: cross-policy uses same fitted probes on v2 data; random-weights rollout samples actions (not argmax).
- `run_tom_experiment`: one `reset` per env at collect start; caches `probe_models_v1.pkl` for transfer; JSON summary includes TB ambiguity metrics.
- `generate_tom_report`: ambiguity gap cell avoids `:+` on missing values.

---

## 2026-03-31 — `slurm_ma_tom.sh` default GPU: L40S

- Default Slurm request is **`--gres=gpu:l40s:1`**, **50G** RAM, **8** CPUs (no H100 partition).
- H100 remains available via explicit `sbatch` overrides in script comments.

---

## 2026-03-31 — `.ai` manifest, Alliance paths, Slurm script fix

- `.ai/.lnai-manifest.json`: added `metadata` (cluster path guide, skills source dir, sync note for symlinked tools).
- `.ai/cluster-paths.md`: documents Killarney `$SCRATCH` vs `$USER` (no doubled path), Slurm account rules (no placeholder `sed`).
- `.ai/AGENTS.md`: Alliance/Killarney section pointing at cluster-paths + alliancecan skill.
- `.ai/skills/alliancecan/SKILL.md`: expanded pitfalls (account, SCRATCH, PROJECT auto-detect, ts-sandbox alignment).
- `slurm_ma_tom.sh`: `PROJECT_ROOT` = `$SCRATCH/drc-sokoban-ma` or `$HOME/...`; `STORAGE_ROOT` = `$PROJECT/$USER/drc-sokoban-ma`; removed broken `REPO_DIR=$SCRATCH/$USER/...`; auto-detect `PROJECT`; `set -e`, `PYTHONUNBUFFERED`.
- `drc_sokoban/onboard.md`: gotchas for SCRATCH clone path and Slurm account.

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
