#!/bin/bash
# ============================================================
# Single-agent easy sanity: procedural 6x6, 1 box, short budget.
#
# Wall-time estimate (L40S, 64 SubprocVecEnv workers, generator 6x6):
#   Env steps are cheap; bottleneck is DRC forward + PPO. Empirical band on
#   similar boxes is roughly 1.5e3–3e3 env-steps/s. For TARGET_STEPS=5e6:
#     5e6 / 3e3 ≈ 1670 s (~28 min) optimistic
#     5e6 / 1.5e3 ≈ 3333 s (~56 min) pessimistic
#   Mid ~42 min × (1 + 50% slack) ≈ 63 min → #SBATCH --time=01:30:00
#
# Submit:
#   mkdir -p /scratch/$USER/slurm_logs
#   sbatch --account=aip-boyuwang slurm_sa_tiny_6x6.sh
#
# Overrides (examples):
#   export SBATCH_TARGET_STEPS=2000000   # shorter run before sbatch
#   sbatch --time=02:00:00 slurm_sa_tiny_6x6.sh
# ============================================================

#SBATCH --job-name=sa-tiny-6x6
#SBATCH --account=aip-boyuwang
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/scratch/%u/slurm_logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
export PYTHONUNBUFFERED=1

# ---- Budget (override before sbatch if you like) ----
: "${SBATCH_TARGET_STEPS:=5000000}"
: "${SBATCH_SAVE_EVERY:=2500000}"
: "${SBATCH_NUM_ENVS:=64}"

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
    echo "Then: sbatch --export=ALL slurm_sa_tiny_6x6.sh"
    exit 1
fi

STORAGE_ROOT="${PROJECT}/${USER}/drc-sokoban-ma"
VENV_DIR="${STORAGE_ROOT}/venv"
CKPT_DIR="${STORAGE_ROOT}/checkpoints"
mkdir -p "${STORAGE_ROOT}" "${CKPT_DIR}" "/scratch/${USER}/slurm_logs"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "STORAGE_ROOT=${STORAGE_ROOT}"
echo "TARGET_STEPS=${SBATCH_TARGET_STEPS} SAVE_EVERY=${SBATCH_SAVE_EVERY} NUM_ENVS=${SBATCH_NUM_ENVS}"

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: venv missing at ${VENV_DIR}"
    exit 1
fi
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CKPT_BASE="${CKPT_DIR}/sa_tiny_6x6"
RESUME_FLAG=()
LATEST=$(ls -t "${CKPT_BASE}"_*.pt "${CKPT_BASE}"_final.pt 2>/dev/null | head -1 || true)
if [[ -n "${LATEST}" ]]; then
    echo "Resuming from ${LATEST}"
    RESUME_FLAG=(--resume "${LATEST}")
fi

python -u -m drc_sokoban.scripts.train \
    --use-generator \
    --grid-size       6 \
    --n-boxes         1 \
    --internal-walls  0 \
    --num-envs        "${SBATCH_NUM_ENVS}" \
    --target-steps    "${SBATCH_TARGET_STEPS}" \
    --save-every      "${SBATCH_SAVE_EVERY}" \
    --save-path       "${CKPT_BASE}" \
    "${RESUME_FLAG[@]}"

echo "Done $(date)"
