#!/bin/bash
# ============================================================
# Multi-Agent ToM Experiment — Killarney Slurm submission
#
# IMPORTANT: change --account below to your CCDB group name.
#
# This script handles all phases via the PHASE env variable:
#   train   : IPPO self-play (H100, up to 3 days)
#   train_v2: Handicapped-partner fine-tune (H100, ~6h)
#   probe   : ToM probing pipeline (~2-4h)
#   smoke   : Quick sanity check (~5 min)
#
# Submit examples:
#   # Training phase (H100, default):
#   sbatch --export=PHASE=train slurm_ma_tom.sh
#
#   # Probing phase on L40S (faster queue, probe is CPU-bound):
#   sbatch --export=PHASE=probe \
#          --partition=default --gres=gpu:l40s:1 --time=0-06:00:00 \
#          slurm_ma_tom.sh
#
# Adjust REPO_DIR / DATA_DIR / VENV_DIR to match your cluster layout.
# ============================================================

#SBATCH --job-name=ma-tom
#SBATCH --account=aip-boyuwang          # <-- change to your CCDB group name
#SBATCH --time=3-00:00:00               # 3 days wall (gpubase_h100_b4 allows up to 4)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpubase_h100_b4
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

PHASE="${PHASE:-train}"
echo "Phase: ${PHASE}"

# ============================================================
# Paths (edit these to match your cluster layout)
# ============================================================
REPO_DIR="${SCRATCH}/${USER}/drc-sokoban-ma"
DATA_DIR="${PROJECT}/${USER}/drc-sokoban-ma/data/boxoban_levels"
VENV_DIR="${PROJECT}/${USER}/drc-sokoban-ma/venv"
CKPT_DIR="${PROJECT}/${USER}/drc-sokoban-ma/checkpoints"
RES_DIR="${PROJECT}/${USER}/drc-sokoban-ma/results"

# Fallback to HOME if SCRATCH/PROJECT not set (e.g. interactive testing)
if [[ -z "$SCRATCH" ]]; then
    REPO_DIR="${HOME}/drc-sokoban-ma"
    DATA_DIR="${HOME}/data/boxoban_levels"
    VENV_DIR="${HOME}/drc-sokoban-ma/venv"
    CKPT_DIR="${HOME}/drc-sokoban-ma/checkpoints"
    RES_DIR="${HOME}/drc-sokoban-ma/results"
fi

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
    pip install -r "${REPO_DIR}/drc_sokoban/requirements.txt"
    pip install wandb tqdm
else
    source "${VENV_DIR}/bin/activate"
fi

cd "${REPO_DIR}"

mkdir -p "${CKPT_DIR}" "${RES_DIR}"

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
