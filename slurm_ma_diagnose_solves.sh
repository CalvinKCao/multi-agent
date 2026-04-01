#!/bin/bash
# ============================================================
# Run ma_diagnose_solves on a *compute node* (not klogin).
#
# Why: Login nodes often lack CPU features (AVX-512, etc.) that pip wheels
# (numpy, torch) were built for on compute — you get "Illegal instruction".
#
# Usage (edit ACCOUNT / paths if needed):
#   sbatch --account=YOUR_CCDB_GROUP slurm_ma_diagnose_solves.sh
#
# Or pass env vars:
#   sbatch --account=YOUR_CCDB_GROUP \
#     --export=ALL,DIAG_EPISODES=5000,DATA_DIR=/project/6101823/$USER/drc-sokoban-ma/data/boxoban_levels \
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

set -euo pipefail

REPO="${REPO_DIR:-${SCRATCH}/drc-sokoban-ma}"
VENV="${VENV_DIR:-${PROJECT}/${USER}/drc-sokoban-ma/venv}"
DATA="${DATA_DIR:-${PROJECT}/${USER}/drc-sokoban-ma/data/boxoban_levels}"
EPISODES="${DIAG_EPISODES:-5000}"

module purge
module load StdEnv/2023
module load python/3.11

if [[ ! -d "$REPO" ]]; then
  echo "ERROR: REPO not found: $REPO  (set REPO_DIR or clone to \$SCRATCH/drc-sokoban-ma)"
  exit 1
fi
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "ERROR: venv not found: $VENV/bin/activate"
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
