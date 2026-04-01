#!/bin/bash
# ============================================================
# Single-agent DRC on Boxoban — 50M env steps (sanity / baseline).
#
# Default hyperparams match train.py: 50M steps, save every 5M, 32 envs in CLI
# but we use 64 envs here for throughput (same as slurm_ma_tom.sh).
#
# Submit:
#   mkdir -p /scratch/$USER/slurm_logs
#   sbatch --account=aip-boyuwang slurm_sa_boxoban_50m.sh
#
# Override account / wall / GPU at submit time if needed:
#   sbatch --account=aip-boyuwang --time=0-12:00:00 --gres=gpu:l40s:1 slurm_sa_boxoban_50m.sh
#
# Logs: /scratch/<user>/slurm_logs/sa-boxoban-<jobid>.out
# Checkpoints: ${STORAGE_ROOT}/checkpoints/sa_agent_{5,10,...}M.pt and sa_agent_final.pt
# ============================================================

#SBATCH --job-name=sa-boxoban
#SBATCH --account=aip-boyuwang
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/scratch/%u/slurm_logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
export PYTHONUNBUFFERED=1

# ---- Paths (same logic as slurm_ma_tom.sh) ----
if [[ -n "${SCRATCH}" && -d "${SCRATCH}/drc-sokoban-ma" ]]; then
    export PROJECT_ROOT="${SCRATCH}/drc-sokoban-ma"
elif [[ -d "${HOME}/drc-sokoban-ma" ]]; then
    export PROJECT_ROOT="${HOME}/drc-sokoban-ma"
else
    echo "ERROR: drc-sokoban-ma not found under \$SCRATCH or \$HOME"
    exit 1
fi

if [[ -f "${PROJECT_ROOT}/wandb.local" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${PROJECT_ROOT}/wandb.local"
    set +a
fi

if [[ -z "${PROJECT:-}" && -d "${HOME}/projects" ]]; then
    shopt -s nullglob
    _proj_candidates=("${HOME}"/projects/def-* "${HOME}"/projects/aip-*)
    shopt -u nullglob
    if [[ ${#_proj_candidates[@]} -gt 0 ]]; then
        PROJECT="$(readlink -f "${_proj_candidates[0]}")"
        export PROJECT
    fi
fi

if [[ -z "${PROJECT:-}" ]]; then
    echo "ERROR: PROJECT not set. Example:"
    echo "  export PROJECT=\$(readlink -f ~/projects/aip-boyuwang)"
    echo "Then: sbatch --export=ALL slurm_sa_boxoban_50m.sh"
    exit 1
fi

STORAGE_ROOT="${PROJECT}/${USER}/drc-sokoban-ma"
DATA_DIR="${STORAGE_ROOT}/data/boxoban_levels"
VENV_DIR="${STORAGE_ROOT}/venv"
CKPT_DIR="${STORAGE_ROOT}/checkpoints"

mkdir -p "${STORAGE_ROOT}" "${CKPT_DIR}" "/scratch/${USER}/slurm_logs"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "STORAGE_ROOT=${STORAGE_ROOT}"
echo "DATA_DIR=${DATA_DIR}"

# ---- Modules + venv ----
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: venv missing at ${VENV_DIR} — run slurm_ma_tom.sh once or create venv there."
    exit 1
fi
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Resume ----
CKPT_BASE="${CKPT_DIR}/sa_agent"
RESUME_FLAG=()
LATEST=$(ls -t "${CKPT_BASE}"_*.pt 2>/dev/null | head -1 || true)
if [[ -n "${LATEST}" ]]; then
    echo "Resuming from ${LATEST}"
    RESUME_FLAG=(--resume "${LATEST}")
fi

# ---- Train (50M is also the default in train.py if you omit --target-steps) ----
python -u -m drc_sokoban.scripts.train \
    --data-dir       "${DATA_DIR}" \
    --num-envs       64 \
    --target-steps   50000000 \
    --save-every     5000000 \
    --save-path      "${CKPT_BASE}" \
    --num-layers     2 \
    --hidden-channels 32 \
    "${RESUME_FLAG[@]}"

echo "Done $(date)"
