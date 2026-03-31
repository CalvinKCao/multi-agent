# DRC Sokoban — Planning Probes in Model-Free RL

PyTorch reimplementation of the DRC (Deep Repeated ConvLSTM) agent from
[Bush et al. 2025 "Interpreting Emergent Planning in Model-Free Reinforcement
Learning" (ICLR 2025)](https://arxiv.org/abs/2411.07137), trained on the
[Boxoban](https://github.com/google-deepmind/boxoban-levels) dataset.

**Goal:** Train a DRC agent on Boxoban and verify that its ConvLSTM hidden state
develops spatially-grounded linear representations of the agent's future plan,
detectable by simple 1×1 logistic-regression probes.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the Boxoban dataset

```bash
git clone https://github.com/google-deepmind/boxoban-levels data/boxoban_levels
```

Optionally, build the fast C extension for the environment:

```bash
git clone https://github.com/google-deepmind/boxoban-environment
cd boxoban-environment && pip install -e .
```

If the C extension fails, `gym-sokoban` is used as fallback automatically.

### 3. Smoke test (< 5 minutes)

```bash
cd drc_sokoban

python scripts/train.py \
    --data-dir data/boxoban_levels \
    --num-envs 4 \
    --target-steps 100000 \
    --num-layers 2 \
    --hidden-channels 16 \
    --save-path checkpoints/smoke_test \
    --smoke-test \
    --no-subproc

python scripts/collect_for_probing.py \
    --checkpoint checkpoints/smoke_test_final.pt \
    --n-episodes 10 \
    --save-path data/probe_data/smoke_test.pkl

python scripts/run_probes.py \
    --data data/probe_data/smoke_test.pkl \
    --n-positions 5 \
    --quick-check
```

### 4. Full PoC training (50M steps, ~hours on a single GPU)

```bash
python scripts/train.py \
    --data-dir data/boxoban_levels \
    --num-envs 32 \
    --target-steps 50000000 \
    --num-layers 2 \
    --save-path checkpoints/agent

python scripts/collect_for_probing.py \
    --checkpoint checkpoints/agent_final.pt \
    --n-episodes 4000 \
    --save-path data/probe_data/full.pkl

python scripts/run_probes.py \
    --data data/probe_data/full.pkl \
    --results-dir results/
```

---

## Architecture

```
obs (7, 8, 8) ──► Encoder ──────────────► (32, 8, 8)
                  Conv2d(7→32, k=3, p=1)     │
                  ReLU                        │
                                              ▼
                                    DRC Stack (D layers, N ticks/step)
                                    ┌─────────────────────────────────┐
                                    │  Tick 1                         │
                                    │   ConvLSTMCell layer 0          │
                                    │   ConvLSTMCell layer 1          │
                                    │   ConvLSTMCell layer 2          │
                                    │  Tick 2 (same cells, new h/c)   │
                                    │  Tick 3                         │
                                    └─────────────────────────────────┘
                                              │
                                    final_h (32, 8, 8) ──► flatten (2048)
                                              │
                                    ┌─────────┴──────────┐
                                    ▼                    ▼
                              policy_head            value_head
                              Linear(2048→4)         Linear(2048→1)
```

**Critical constraint:** every Conv2d uses `kernel_size=3, padding=1` — NO striding,
NO pooling — so the spatial dimensions stay at 8×8 throughout.  Position (x,y) in the
hidden state corresponds exactly to board cell (x,y), enabling 1×1 spatial probes.

---

## Probe Methodology (Bush et al. 2025)

After training, we collect rollout episodes and label each timestep t with two concepts
for every grid cell (x,y):

| Concept | Label | Meaning |
|---------|-------|---------|
| **CA** | UP/DOWN/LEFT/RIGHT/NEVER | Direction FROM WHICH the agent will next step onto (x,y) |
| **CB** | UP/DOWN/LEFT/RIGHT/NEVER | Direction a box will next be pushed OFF (x,y) |

A **1×1 logistic regression probe** is then trained on the 32-dim activation at (x,y)
to predict the label.  If the probe beats the observation-only baseline by >0.05 macro F1,
the hidden state encodes planning information beyond what the current observation contains.

Expected results at 250M steps:

|            | Layer 0 | Layer 1 | Layer 2 | Obs-baseline |
|------------|---------|---------|---------|-------------|
| CA probe   |  0.55   |  0.60   |  0.62   |     0.35    |
| CB probe   |  0.45   |  0.52   |  0.55   |     0.28    |

At 50M steps (PoC) expect ~60–70% of these values.

---

## Hyperparameters

| Parameter          | Value      | Source           |
|--------------------|-----------|-----------------|
| `hidden_channels`  | 32         | Bush et al. 2025 |
| `num_layers` D     | 2 (PoC), 3 (full) | Bush et al. 2025 |
| `num_ticks` N      | 3          | Guez et al. 2019 |
| `gamma`            | 0.97       | Bush et al. 2025 |
| `learning_rate`    | 3e-4       | —                |
| `clip_eps`         | 0.1        | —                |
| `max_grad_norm`    | 10.0       | higher for LSTM  |
| `rollout_steps`    | 20         | —                |
| `num_envs`         | 32         | —                |

---

## Project Structure

```
drc_sokoban/
├── envs/
│   ├── boxoban_env.py       Env wrapper (boxoban-env C ext → gym-sokoban → internal)
│   └── make_env.py          SubprocVecEnv / DummyVecEnv factory
├── models/
│   ├── conv_lstm.py         ConvLSTMCell + DRCStack — THE critical file
│   └── agent.py             Full DRCAgent (encoder + DRC + policy/value heads)
├── training/
│   ├── ppo.py               PPO trainer with ConvLSTM hidden-state management
│   └── rollout_buffer.py    Rollout buffer storing hidden states per step
├── probing/
│   ├── hook_manager.py      Forward-hook extraction of hidden states
│   ├── concept_labeler.py   CA and CB concept labeling from episode trajectories
│   ├── train_probes.py      1×1 logistic-regression probe training
│   └── evaluate_probes.py   F1 evaluation, tables, and plots
├── scripts/
│   ├── train.py             Main training entry point
│   ├── collect_for_probing.py  Run agent, save hidden states per episode
│   └── run_probes.py        Full probe pipeline
├── checkpoints/
├── data/
│   ├── boxoban_levels/      Symlink or copy of Boxoban dataset
│   └── probe_data/          Collected trajectories
└── results/                 Probe metrics and figures
```

---

## Common Failure Modes

| Symptom | Likely cause |
|---------|-------------|
| Solve rate < 0.02 at 20M steps | Hidden state not reset at episode boundaries |
| Probe F1 ≈ baseline | Agent hasn't learned (train more) or wrong tick/layer indexing |
| NaN loss | Gradient explosion — lower `max_grad_norm` or reduce `learning_rate` |
| OOM error | Reduce `num_envs` or `minibatch_size` |

---

## Citation

```bibtex
@inproceedings{bush2025interpreting,
  title     = {Interpreting Emergent Planning in Model-Free Reinforcement Learning},
  author    = {Bush, Thomas and others},
  booktitle = {ICLR},
  year      = {2025},
}

@article{guez2019investigation,
  title   = {An Investigation of Model-Free Planning},
  author  = {Guez, Arthur and others},
  journal = {ICML},
  year    = {2019},
}
```
