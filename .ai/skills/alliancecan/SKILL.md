---
name: alliancecan
description: Alliance Canada (Compute Canada) Slurm clusters — jobs, storage, modules, Killarney GPUs, and per-user paths. Use when editing Slurm scripts, cluster setup, or HPC workflows for docs.alliancecan.ca systems.
---

# Alliance Canada HPC

Apply when working on Slurm job scripts, cluster setup, paths, or GPU requests for Alliance national systems (Killarney, Fir, Narval, Nibi, Rorqual, Trillium, etc.).

## Accounts and Slurm basics

- **Scheduler:** Slurm only. No compute on login nodes except tiny tasks (~≤10 CPU-minutes, ~≤4 GB RAM). Everything else: `sbatch`, `salloc`, `srun`.
- **Account:** Every job needs a **Resource Allocation Project** via `#SBATCH --account=<group-name>`. If you have multiple allocations, Slurm will error until you pick one. The string is the **Group Name** from [CCDB](https://ccdb.alliancecan.ca/) → My Projects → My Resources and Allocations (not necessarily the same as a “project title”).
- **Optional convenience:** `export SLURM_ACCOUNT=def-user` and `export SBATCH_ACCOUNT=$SLURM_ACCOUNT` (and `SALLOC_ACCOUNT`) in `~/.bashrc` so you do not repeat `--account`—but CLI `sbatch --account=...` overrides when needed.
- **Minimum directives:** Always set **`#SBATCH --time=...`**. Add **`--mem`** or **`--mem-per-cpu`** on general-purpose clusters (default can be very small per core). `#SBATCH` lines must come **before** any shell commands in the script.
- **Do not** hammer Slurm with `squeue`/`sq` in tight loops; use mail notifications or reasonable polling.
- **Alliance guidance:** Prefer **not** pinning `--partition` unless software forces it; if something insists on a partition, `default` is treated like “let the scheduler decide.” This repo sometimes sets Killarney H100 partitions explicitly—verify with `sinfo` on the target system.

## Where to put files (storage hygiene)

- **HOME (`~`):** Small quota; keep **source, job scripts, tiny configs**. Not for large datasets or heavy I/O.
- **SCRATCH:** Large, fast for big sequential I/O; **not backed up**; old files may be **purged** (e.g. 60-day policy—check current docs). Use for **checkpoints during runs**, bulk output, datasets you can re-fetch, and **clone/run the repo** where policy requires (see Killarney below).
- **PROJECT (`$PROJECT` / `~/projects/...`):** Shared by the allocation group, larger quota, backed up; intended for **relatively static** shared data—frequent churn hurts tape backup. For **your** artifacts, use a **per-user subdirectory**: `$PROJECT/$USER/<app>/` (venv, checkpoints, wandb, copied datasets), **not** the group root as a personal scratch pad.
- **`$SLURM_TMPDIR`:** Per-job local disk on the compute node; great for **many small files** and ephemeral shuffles; **deleted when the job ends**.
- **Python/R packages:** Often **not** full Lmod modules; use Alliance **Python + pip/wheel** docs, or install into **your** space (venv under `$PROJECT/$USER/...` or similar).
- **Modules:** Start job scripts with **`module purge`** then load what you need; prerequisites matter—use **`module spider <name>/<version>`** to see the load chain. Avoid relying on whatever was loaded in the interactive shell.

**Per-user working space (important):** Treat **group project space as shared infrastructure**. Keep **your** working data, venvs, and experiment outputs under **`$PROJECT/$USER/...`** and/or **`$SCRATCH/$USER/...`**. Do **not** stash personal experiments or venvs directly under `$PROJECT/` without a `/$USER/` (or agreed team) path—avoids quota fights and policy issues.

## Killarney (GPUs and layout)

This cluster is common for this repo. **Verify live** with `sinfo -o "%P %G %l"` and `scontrol show node | grep -i gres`—names and partitions change.

From repo/onboarding patterns and `setup/alliance_setup_killarney.sh`:

- **Performance tier (H100):** Dell XE9680-class nodes with **8× NVIDIA H100 SXM 80GB** per node (docs/setup: 48 cores, ~2 TB RAM class). Submit with something like:
  - `#SBATCH --partition=gpubase_h100_b4` (example: **b4** = up to **4 days** wall time—match partition letter to desired max time: `b1`…`b5` for shorter→longer caps per local policy)
  - `#SBATCH --gpus-per-node=h100:1` (or `:4`, `:8` for multi-GPU)
- **Standard tier (L40S):** Often **shorter queue** for smaller jobs. Example pattern: `#SBATCH --gres=gpu:l40s:1` (no `gpubase_h100_*` mix-up—those partitions are **H100-specific**).
- **Choosing GPU vs partition:**
  - Need **80 GB** and heavy training → **H100** + appropriate `gpubase_h100_*` and wall-time partition.
  - Smoke tests, smaller memory → **L40S** via `--gres=gpu:l40s:1` is often faster to start.
  - **Never** combine incompatible partition/GRES (e.g. H100 partition with L40S `gres`—follow working examples in-repo and cluster `sinfo`).
- **Code location:** Killarney **must not run GPU work from `/home`**; keep a checkout under **`$SCRATCH/...`** (e.g. `$SCRATCH/$USER/ts-sandbox`) and `cd` there in the job. Scripts in this repo prefer `$SCRATCH/ts-sandbox` then fall back to `$HOME/ts-sandbox`.
- **Account prefix:** Allocations may show as **`aip-...`** on Killarney vs **`def-...`** elsewhere—use the CCDB **Group Name** for `--account`.

## Other clusters (quick reference)

Repo comments (`slurm_pipeline.sh`): **Narval** → e.g. A100; **Fir / Nibi / Rorqual** → e.g. H100-style requests. Always match **`--account`** to an allocation **valid on that cluster** (RAPs are not always portable).

## Software modules (typical ML stack)

Example stack used in this repo’s Slurm scripts:

```bash
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9
```

Load **CUDA/cuDNN** versions compatible with your PyTorch build. If a module fails, use `module spider` to resolve prerequisites. **Docker** is not available; **Apptainer** is (`module load apptainer`).

## Patterns from this repository

- **`#SBATCH --account=aip-boyuwang`** — replace with the user’s real CCDB group name.
- **Venv and bulky data:** `$PROJECT/$USER/diffusion-tsf/venv` (and checkpoints/results under the same tree)—**per-user under project**, not the bare group directory.
- **Killarney smoke vs full:** `slurm_unet_fullvar.sh` submits **L40S** for `--smoke-test`, **H100** + `gpubase_h100_b4` for full runs.
- **Latent experiments:** `slurm_latent_experiment.sh` uses `--gres=gpu:l40s:1` by default; comments document switching to H100 partitions for long runs.
- **`PYTHONUNBUFFERED=1`** and **`python -u`** help with Slurm log latency (buffering).

## Gotchas (see also project `onboard.md`)

- **Empty/broken venv on cluster:** If jobs fail with missing `torch`, recreate or repair **`$PROJECT/$USER/.../venv`** (or delete and let the Slurm script reinstall).
- **Imports:** Run Python as **`python -m package.module`** from the repo root.
- **Slurm output buffering** can make logs look “stuck”; use unbuffered Python or interactive `salloc` to debug.
- **`module purge` in jobs** avoids surprise inherited environments from the submission shell.
- **`sbatch` from scripts:** Do not submit thousands of jobs at once; prefer **arrays** or spacing submissions—Alliance warns this can harm Slurm.

## Official docs

Prefer [docs.alliancecan.ca](https://docs.alliancecan.ca/) for authoritative quotas, partition names, and policy updates—this skill is a condensed assistant checklist, not a substitute for current site documentation.
