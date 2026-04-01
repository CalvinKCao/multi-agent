# Bush et al. (ICLR 2025, arXiv:2504.01871) vs this repo

Official release: **[github.com/tuphs28/emergent-planning](https://github.com/tuphs28/emergent-planning)**  
(interpretability + checkpoints; training builds on **Thinker** — see that repo’s `train.py` / README).  
Related follow-up code: **[github.com/AlignmentResearch/learned-planner](https://github.com/AlignmentResearch/learned-planner)**.

This file flags **intentional PoC shortcuts** and **real mismatches** vs Appendix E of the paper.

---

## DRC architecture (Appendix E.3)

| Paper (Guez-style DRC) | This repo |
|------------------------|-----------|
| Encoder output **i_t fed to every ConvLSTM layer** (bottom-up skip) | Encoder only feeds **layer 0**; higher layers see previous layer’s **h** only |
| **Top-down skip**: final-layer **h** at tick *n−1* → extra input to **bottom** layer at tick *n* | **Not implemented** |
| **Pool-and-inject** (mean+max pool **h**, linear, reshape, add to cell input) | **Not implemented** |
| Policy/value: **concat(flatten(h^D), flatten(i_t))** → MLP/ReLU → heads | **flatten(h^D) only** — no concat with encoder |
| DRC(3,3), 32 ch, 3×3 conv + padding | **Matches** when `num_layers=3`, `num_ticks=3`, `hidden_channels=32` |
| Probes on **cell state g** (LSTM **c**) | Collection uses **h** (`all_ticks[..][..][0]`), not **c** |

The docstring on `DRCAgent` that says it “matches” the paper is **overstated**; this is a **minimal** repeated ConvLSTM stack suitable for spatial 1×1 probes, not the full Guez DRC block.

---

## Training (Appendix E.4)

| Paper | This repo (`PPOTrainer` / `train.py`) |
|-------|----------------------------------------|
| **IMPALA** + V-trace, λ = 0.97 | **PPO** + **GAE** λ = 0.95 |
| Adam **batch 16**, LR **linear decay 4e-4 → 0** | Minibatch **256**, LR **3e-4 constant** |
| **L2** on logits **1e-3**, on heads **1e-5** | **No** logit L2; no explicit head L2 |
| Entropy coef **1e-2** | **0.01** — aligned |
| Discount **γ = 0.97** | **0.97** — aligned |
| Unroll / rollout length **20** | `rollout_steps=20` — aligned |
| **250M** transitions | Default **50M** in scripts (PoC); raise for paper-scale |

`IPPO` / `train_ma.py` is **not** in the paper; it is this project’s multi-agent extension.

---

## Environment / reward (Appendix E.2)

| Paper | This repo (`boxoban_env.py`) |
|-------|------------------------------|
| **−0.01** per step | **0** per step (only box / win shaping) |
| Episode cap **uniform random 115–120** | **400** steps (common Boxoban RL default) |
| +1 on target, −1 off, +10 when all on targets | **Same** event structure (generalised to “all boxes”, not only four) |
| 7-way symbolic tile one-hot | **Present**; **channel ordering** may differ from the paper’s enumeration (still a valid one-hot) |
| Optional “no-op” fifth action in appendix text | **4** actions only (no no-op) |

---

## Probing methodology (main text + Appendix)

| Paper | This repo |
|-------|-----------|
| 1×1 (and 3×3) probes on **cell state** | 1×1 probes on **hidden output h** |
| Macro F1, obs baseline | **Aligned in spirit** (`train_probes.py`) |
| CA / CB definitions | **Closely aligned** (`concept_labeler.py`); verify against official feature names `agent_onto_after` / `tracked_box_next_push_onto_with` in emergent-planning |

---

## Cloning the official repo

```bash
git clone --recursive https://github.com/tuphs28/emergent-planning.git
# Then follow README: sokoban pip install, thinker pip install, experiments/requirements.txt
```

A shallow clone without submodules may only show `README.md` + `checkpoints/`.

---

## Practical takeaway

- To **replicate paper numbers** (solve rate ~97%, probe F1s): use their **checkpoints + Thinker/IMPALA stack**, or port the **full DRC** (skips + pool-inject + readout) and **IMPALA + rewards + horizon** from Appendix E.  
- This repo is best read as a **self-contained PoC**: same *ideas* (spatial DRC, 1×1 probes, CA/CB), **not** bit-identical to the published agent.
