#!/bin/bash
# ============================================================
# Multi-Agent ToM Experiment — Killarney Slurm submission
#
# Paths (same idea as ts-sandbox Slurm scripts):
#   - Clone / run the repo from $SCRATCH/drc-sokoban-ma  (NOT $SCRATCH/$USER/...
#     on Killarney $SCRATCH is usually already /scratch/<user>, so doubling
#     $USER breaks with "No such file or directory").
#   - Checkpoints, venv, wandb: $PROJECT/$USER/drc-sokoban-ma/...
#
# Account:
#   - #SBATCH --account= must be your exact CCDB Group Name (e.g. aip-boyuwang).
#   - Do NOT sed-replace the account line to a placeholder string — Slurm will
#     reject the job. Override at submit time if needed:
#       sbatch --account=aip-boyuwang --export=PHASE=train slurm_ma_tom.sh
#
# Phases (PHASE env):
#   train   : IPPO self-play (L40S by default — shorter queue than H100)
#   train_v2: Handicapped-partner fine-tune
#   probe   : ToM probing pipeline
#   smoke   : Quick sanity check (~5 min)
#
# GPU: default is **L40S** (`--gres=gpu:l40s:1`). Do not mix H100 partitions with L40S gres.
# For H100 / long runs, override at submit time, e.g.:
#   sbatch --partition=gpubase_h100_b4 --gpus-per-node=h100:1 \
#          --cpus-per-task=16 --mem=64G --export=PHASE=train slurm_ma_tom.sh
#
# Submit:
#   sbatch --export=PHASE=train slurm_ma_tom.sh
#
# Wall time (short requests queue faster than multi-day asks):
#   Slurm uses the #SBATCH --time below unless you override at submit time:
#     sbatch --time=0-09:00:00 --export=PHASE=train slurm_ma_tom.sh
#   Rough guide (L40S, 64 envs, Python env — tune after your first run):
#     train    ~4–6 h typical for 50M steps → ask ~6–9 h (estimate + ~50%)
#     train_v2 ~0.5–1.5 h for 10M → ask ~2–3 h
#     probe    ~1–3 h collect + sklearn → ask ~4–5 h
#     smoke    ~10 min → ask 0-00:30:00
#   If a job hits TIMEOUT, resubmit the same PHASE; train/train_v2 resume from ckpt.
# ============================================================

#SBATCH --job-name=ma-tom
#SBATCH --account=aip-boyuwang
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:l40s:1
# Logs: Slurm resolves paths at submit time. A *relative* --output goes to the
# directory where you ran `sbatch`, NOT where this script later `cd`s — easy to
# "lose" ma-tom-JOBID.out.  We pin logs under /scratch/<user>/slurm_logs/ instead.
# Create once on Killarney:   mkdir -p /scratch/$USER/slurm_logs
#SBATCH --output=/scratch/%u/slurm_logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
export PYTHONUNBUFFERED=1

# Weights & Biases — one file on the cluster (gitignored): copy wandb.local.example
# to ${PROJECT_ROOT}/wandb.local after clone.  Python loads it automatically; we also
# source it here so the key exists before any subprocess.  Do not commit wandb.local.
# Alternative: wandb login on the login node (~/.netrc).  Offline: export WANDB_MODE=offline

PHASE="${PHASE:-train}"
echo "Phase: ${PHASE}"

# ============================================================
# Paths — mirror ts-sandbox: PROJECT_ROOT under SCRATCH, storage under PROJECT
# ============================================================
if [[ -n "${SCRATCH}" && -d "${SCRATCH}/drc-sokoban-ma" ]]; then
    export PROJECT_ROOT="${SCRATCH}/drc-sokoban-ma"
elif [[ -d "${HOME}/drc-sokoban-ma" ]]; then
    export PROJECT_ROOT="${HOME}/drc-sokoban-ma"
else
    echo "ERROR: drc-sokoban-ma not found."
    echo "  Clone on Killarney:  cd \"\${SCRATCH}\" && git clone <url> drc-sokoban-ma"
    echo "  (do not use cd \"\\\$SCRATCH/\\\$USER\" unless your site uses that layout)"
    exit 1
fi

if [[ -f "${PROJECT_ROOT}/wandb.local" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${PROJECT_ROOT}/wandb.local"
    set +a
fi

if [[ -z "${PROJECT}" && -d "${HOME}/projects" ]]; then
    shopt -s nullglob
    _proj_candidates=("${HOME}"/projects/def-* "${HOME}"/projects/aip-*)
    shopt -u nullglob
    if [[ ${#_proj_candidates[@]} -gt 0 ]]; then
        PROJECT=$(readlink -f "${_proj_candidates[0]}")
        export PROJECT
    fi
fi

if [[ -z "${PROJECT}" ]]; then
    echo "ERROR: PROJECT not set. Example:"
    echo "  export PROJECT=\$(readlink -f ~/projects/aip-boyuwang)"
    exit 1
fi

STORAGE_ROOT="${PROJECT}/${USER}/drc-sokoban-ma"
DATA_DIR="${STORAGE_ROOT}/data/boxoban_levels"
VENV_DIR="${STORAGE_ROOT}/venv"
CKPT_DIR="${STORAGE_ROOT}/checkpoints"
RES_DIR="${STORAGE_ROOT}/results"

mkdir -p "${STORAGE_ROOT}" "${CKPT_DIR}" "${RES_DIR}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "STORAGE_ROOT=${STORAGE_ROOT}"
echo "PROJECT=${PROJECT}"

# ============================================================
# Environment setup
# ============================================================
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Create venv on first run (subsequent runs reuse it)
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r "${PROJECT_ROOT}/drc_sokoban/requirements.txt"
    pip install wandb tqdm
else
    source "${VENV_DIR}/bin/activate"
fi

cd "${PROJECT_ROOT}"

# Log GPU info for debugging
echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "================"

# ============================================================
# Phase dispatch
# ============================================================

if [[ "$PHASE" == "train" ]]; then
    # ---------------------------------------------------------
    # Phase 1: Self-play IPPO, 50M steps
    # ---------------------------------------------------------
    CKPT_BASE="${CKPT_DIR}/ma_selfplay"

    # Resume if a checkpoint already exists (robust to job preemption)
    RESUME_FLAG=""
    LATEST=$(ls -t "${CKPT_BASE}"_*.pt 2>/dev/null | head -1)
    if [[ -n "$LATEST" ]]; then
        echo "Resuming from ${LATEST}"
        RESUME_FLAG="--resume ${LATEST}"
    fi

    python -u -m drc_sokoban.scripts.train_ma \
        --data-dir    "${DATA_DIR}" \
        --num-envs    64 \
        --num-layers  2 \
        --hidden-channels 32 \
        --target-steps 50000000 \
        --save-every   5000000 \
        --save-path   "${CKPT_BASE}" \
        --wandb-project drc-sokoban-ma-tom \
        --wandb-run-name "ma-selfplay-50M-$(hostname)" \
        ${RESUME_FLAG}

elif [[ "$PHASE" == "train_v2" ]]; then
    # ---------------------------------------------------------
    # Phase 1b: Handicapped partner (v2 condition)
    # Fine-tune from self-play checkpoint with 50% random actions for B.
    # ---------------------------------------------------------
    SELFPLAY_CKPT="${CKPT_DIR}/ma_selfplay_final.pt"
    if [[ ! -f "$SELFPLAY_CKPT" ]]; then
        echo "ERROR: self-play checkpoint not found at ${SELFPLAY_CKPT}"
        echo "Run Phase train first."
        exit 1
    fi

    CKPT_BASE="${CKPT_DIR}/ma_handicap"
    RESUME_FLAG=""
    LATEST=$(ls -t "${CKPT_BASE}"_*.pt 2>/dev/null | head -1)
    if [[ -n "$LATEST" ]]; then
        RESUME_FLAG="--resume ${LATEST}"
    else
        RESUME_FLAG="--resume ${SELFPLAY_CKPT}"
    fi

    python -u -m drc_sokoban.scripts.train_ma \
        --data-dir         "${DATA_DIR}" \
        --num-envs         64 \
        --target-steps     10000000 \
        --save-every       2000000 \
        --save-path        "${CKPT_BASE}" \
        --partner-noise-eps 0.5 \
        --entropy-coef     0.05 \
        --wandb-project    drc-sokoban-ma-tom \
        --wandb-run-name   "ma-handicap-v2-10M" \
        ${RESUME_FLAG}

elif [[ "$PHASE" == "probe" ]]; then
    # ---------------------------------------------------------
    # Phase 2: ToM probing pipeline
    # ---------------------------------------------------------
    SELFPLAY_CKPT="${CKPT_DIR}/ma_selfplay_final.pt"
    HANDICAP_CKPT="${CKPT_DIR}/ma_handicap_final.pt"

    if [[ ! -f "$SELFPLAY_CKPT" ]]; then
        echo "ERROR: ${SELFPLAY_CKPT} not found.  Run Phase train first."
        exit 1
    fi

    PARTNER_V2_FLAG=""
    if [[ -f "$HANDICAP_CKPT" ]]; then
        PARTNER_V2_FLAG="--partner-v2-ckpt ${HANDICAP_CKPT}"
    fi

    python -u -m drc_sokoban.scripts.run_tom_experiment \
        --checkpoint      "${SELFPLAY_CKPT}" \
        --data-dir        "${DATA_DIR}" \
        --results-dir     "${RES_DIR}/tom_$(date +%Y%m%d_%H%M)" \
        --n-train-eps     3000 \
        --n-val-eps       1000 \
        --n-positions     64 \
        --wandb-project   drc-sokoban-ma-tom \
        ${PARTNER_V2_FLAG}

    # Generate final report
    LATEST_RES=$(ls -td "${RES_DIR}"/tom_* 2>/dev/null | head -1)
    if [[ -n "$LATEST_RES" ]]; then
        python -u -m drc_sokoban.scripts.generate_tom_report \
            --results-dir "${LATEST_RES}" \
            --output      "${LATEST_RES}/TOM_RESULTS.md"
        echo "Report: ${LATEST_RES}/TOM_RESULTS.md"
    fi

elif [[ "$PHASE" == "smoke" ]]; then
    # ---------------------------------------------------------
    # Quick smoke test (no GPU needed beyond tiny allocation)
    # ---------------------------------------------------------
    python -u -m drc_sokoban.scripts.train_ma \
        --smoke-test \
        --no-subproc \
        --num-envs 4 \
        --target-steps 500 \
        --save-path /tmp/ma_smoke_test \
        --data-dir "${DATA_DIR}"
    echo "Smoke test passed."

else
    echo "Unknown PHASE='${PHASE}'.  Use: train | train_v2 | probe | smoke"
    exit 1
fi

echo "Job finished: $(date)"
