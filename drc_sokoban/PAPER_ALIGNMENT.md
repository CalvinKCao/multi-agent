# Bush et al. (ICLR 2025, arXiv:2504.01871) vs this repo

Official release: **[github.com/tuphs28/emergent-planning](https://github.com/tuphs28/emergent-planning)**
(interpretability + checkpoints; training builds on **Thinker**).
Related: **[github.com/AlignmentResearch/learned-planner](https://github.com/AlignmentResearch/learned-planner)**.

---

## DRC architecture (Appendix E.3)

| Paper (Guez-style DRC) | This repo |
|------------------------|-----------|
| Encoder i_t fed to every ConvLSTM layer (bottom-up skip) | **Implemented** (`skip_connections=True` default) |
| Top-down skip: final-layer h at tick n-1 feeds layer 0 at tick n | **Implemented** (same flag) |
| Pool-and-inject (mean+max pool h, linear, add to h) | **Implemented** (`pool_and_inject=True` default) |
| Policy/value: concat(flatten(h^D), flatten(i_t)) -> MLP/ReLU -> heads | **Implemented** (`concat_encoder=True`, 256-dim MLP) |
| DRC(3,3), 32 ch, 3x3 conv + padding | **Matches** (num_layers=3, num_ticks=3, hidden_channels=32) |
| Probes on **cell state g** (LSTM c) | Collection uses **h** -- TODO switch to c |

Set `skip_connections=False`, `pool_and_inject=False`, `concat_encoder=False`
to recover the old minimal stack for ablation.

---

## Training (Appendix E.4)

| Paper | This repo |
|-------|-----------|
| **IMPALA** + V-trace, lambda=0.97 | **PPO** + **GAE** lambda=**0.97** (was 0.95, now fixed) |
| Adam batch 16, LR **linear decay 4e-4 -> 0** | LR **4e-4 linear decay** (was 3e-4 const, now fixed) |
| L2 on logits 1e-3, on heads 1e-5 | No logit L2; no explicit head L2 |
| Entropy coef 1e-2 | 0.01 -- aligned |
| Discount gamma=0.97 | 0.97 -- aligned |
| Unroll / rollout length 20 | rollout_steps=20 -- aligned |
| 250M transitions | Default 50M in scripts (PoC); raise for paper-scale |

IPPO / `train_ma.py` is **not** in the paper; it is this project's MA extension.

---

## Environment / reward (Appendix E.2)

| Paper | This repo |
|-------|-----------|
| -0.01 per step | **-0.01** per step (was 0, now fixed) |
| Episode cap **uniform random 115-120** | **120 + U[0,5]** (configurable via max_steps/max_steps_range) |
| +1 on target, -1 off, +10 all on targets | Same event structure |
| 7-way symbolic tile one-hot | Present; channel ordering may differ |
| Optional no-op fifth action | 4 actions only (no no-op) |

---

## Level generator (new, not in paper)

For curriculum/tiny experiments, `LevelGenerator` produces guaranteed-solvable
levels at arbitrary grid sizes (6x6 to 10x10) via reverse-pull generation.
Supports configurable n_boxes and internal walls.  Standard 8x8 Boxoban dataset
is still supported for paper-comparable runs.

---

## Probing methodology (main text + Appendix)

| Paper | This repo |
|-------|-----------|
| 1x1 (and 3x3) probes on **cell state** | 1x1 probes on **hidden output h** |
| Macro F1, obs baseline | Aligned in spirit (`train_probes.py`) |
| CA / CB definitions | Closely aligned (`concept_labeler.py`) |

---

## Practical takeaway

- To **replicate paper numbers** (solve rate ~97%, probe F1s): use their
  **checkpoints + Thinker/IMPALA stack**, or use this repo with paper-matching
  defaults and 250M steps.
- Architecture now matches Appendix E.3 with default settings.
- Main remaining gap: PPO vs IMPALA/V-trace, and probes on h vs c.
