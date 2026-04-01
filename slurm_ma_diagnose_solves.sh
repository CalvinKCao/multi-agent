#!/bin/bash
# ============================================================
# Run ma_diagnose_solves on a *compute node* (not klogin).
#
# Why: Login nodes often lack CPU features (AVX-512, etc.) that pip wheels
# (numpy, torch) were built for on compute — you get "Illegal instruction".
#
# Usage — use your real CCDB Group Name (not the literal "YOUR_CCDB_GROUP").
# If unsure, run: sacctmgr show user $USER withassoc  |  or submit once and read Slurm's error.
#   sbatch --account=aip-boyuwang slurm_ma_diagnose_solves.sh
#
# If ~/projects/* is missing, Slurm still has no $PROJECT — pass storage explicitly:
#   sbatch --account=aip-boyuwang \
#     --export=ALL,VENV_DIR=/project/6101823/$USER/drc-sokoban-ma/venv,\
#DATA_DIR=/project/6101823/$USER/drc-sokoban-ma/data/boxoban_levels \
#     slurm_ma_diagnose_solves.sh
# ============================================================

#SBATCH --job-name=ma-diagnose
#SBATCH --account=aip-boyuwang
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=ma-diagnose-%j.out

# CPU-only is enough; use default partition (override if your site requires GPU partition)
#SBATCH --partition=default

set -eo pipefail

# Slurm rarely exports PROJECT / SCRATCH the way an interactive shell does.
SCRATCH="${SCRATCH:-}"
if [[ -n "${SCRATCH}" && -d "${SCRATCH}/drc-sokoban-ma" ]]; then
  REPO="${REPO_DIR:-${SCRATCH}/drc-sokoban-ma}"
elif [[ -d "${HOME}/drc-sokoban-ma" ]]; then
  REPO="${REPO_DIR:-${HOME}/drc-sokoban-ma}"
else
  REPO="${REPO_DIR:-${SCRATCH}/drc-sokoban-ma}"
fi

# Same discovery as slurm_ma_tom.sh (~/projects/def-* or aip-*)
if [[ -z "${PROJECT:-}" && -d "${HOME}/projects" ]]; then
  shopt -s nullglob
  _proj_candidates=("${HOME}"/projects/def-* "${HOME}"/projects/aip-*)
  shopt -u nullglob
  if [[ ${#_proj_candidates[@]} -gt 0 ]]; then
    PROJECT="$(readlink -f "${_proj_candidates[0]}")"
    export PROJECT
  fi
fi

STORAGE_ROOT="${PROJECT:-}/${USER}/drc-sokoban-ma"
# Avoid leading "//..." if PROJECT still empty
if [[ -z "${PROJECT:-}" ]]; then
  STORAGE_ROOT=""
fi

VENV="${VENV_DIR:-}"
DATA="${DATA_DIR:-}"
if [[ -z "${VENV}" && -n "${STORAGE_ROOT}" ]]; then
  VENV="${STORAGE_ROOT}/venv"
fi
if [[ -z "${DATA}" && -n "${STORAGE_ROOT}" ]]; then
  DATA="${STORAGE_ROOT}/data/boxoban_levels"
fi

# Last resort: venv colocated with repo (unusual but valid)
if [[ -z "${VENV}" && -d "${REPO}/venv" ]]; then
  VENV="${REPO}/venv"
fi
if [[ -z "${DATA}" && -n "${VENV}" ]]; then
  _parent="$(dirname "${VENV}")"
  if [[ -d "${_parent}/data/boxoban_levels" ]]; then
    DATA="${_parent}/data/boxoban_levels"
  fi
fi

EPISODES="${DIAG_EPISODES:-5000}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "ERROR: Could not find venv. Slurm jobs often have no \$PROJECT set."
  echo "  Fix: pass explicit paths, e.g."
  echo "    sbatch --export=ALL,VENV_DIR=/project/6101823/\$USER/drc-sokoban-ma/venv,\\"
  echo "DATA_DIR=/project/6101823/\$USER/drc-sokoban-ma/data/boxoban_levels \\"
  echo "    slurm_ma_diagnose_solves.sh"
  echo "  Or: export PROJECT=\$(readlink -f ~/projects/aip-boyuwang)  # then sbatch"
  exit 1
fi

module purge
module load StdEnv/2023
module load python/3.11

if [[ ! -d "$REPO" ]]; then
  echo "ERROR: REPO not found: $REPO  (set REPO_DIR or clone to \$SCRATCH/drc-sokoban-ma)"
  exit 1
fi
if [[ -z "${DATA}" || ! -d "${DATA}" ]]; then
  echo "ERROR: DATA_DIR not set or not a directory: ${DATA:-<empty>}"
  echo "  Pass:  --export=ALL,DATA_DIR=/project/.../${USER}/drc-sokoban-ma/data/boxoban_levels"
  exit 1
fi

cd "$REPO"
source "$VENV/bin/activate"

echo "Host: $(hostname)  episodes=$EPISODES  data=$DATA"
python -u -m drc_sokoban.scripts.ma_diagnose_solves \
  --data-dir "$DATA" \
  --episodes "$EPISODES" \
  --max-steps 400

echo "Done $(date)"
