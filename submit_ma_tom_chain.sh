#!/bin/bash
# Submit train -> train_v2 -> probe as three jobs with Slurm dependencies (afterok).
#
# Each job gets its own wall limit (estimate × ~1.5). Override with env vars:
#   TIME_TRAIN=0-12:00:00 TIME_V2=0-04:00:00 TIME_PROBE=0-06:00:00 ./submit_ma_tom_chain.sh
#
# Pass-through sbatch flags (account, partition, gres, etc.):
#   ./submit_ma_tom_chain.sh --account=aip-boyuwang
#
# Do not pass --time in the arguments — use TIME_TRAIN / TIME_V2 / TIME_PROBE instead
# (otherwise the same limit applies to all three and the last flag may win unpredictably).
#
# Slurm: afterok — next job runs only if the previous exited 0.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SH="${HERE}/slurm_ma_tom.sh"
[[ -f "$SLURM_SH" ]] || { echo "Missing ${SLURM_SH}"; exit 1; }

# Defaults: median wall estimate × ~1.5 (L40S, 64 envs; tune with sacct after a run)
TIME_TRAIN="${TIME_TRAIN:-0-08:00:00}"     # 50M steps (~5 h typical → ×1.5)
TIME_V2="${TIME_V2:-0-02:00:00}"           # 10M steps (~1 h → ×1.5)
TIME_PROBE="${TIME_PROBE:-0-03:30:00}"    # 4k eps + probes (~2.3 h → ×1.5)

J1=$(sbatch --parsable --time="${TIME_TRAIN}"  "$@" --export=ALL,PHASE=train     "$SLURM_SH")
J2=$(sbatch --parsable --time="${TIME_V2}"    "$@" --export=ALL,PHASE=train_v2 --dependency=afterok:"${J1}" "$SLURM_SH")
J3=$(sbatch --parsable --time="${TIME_PROBE}" "$@" --export=ALL,PHASE=probe    --dependency=afterok:"${J2}" "$SLURM_SH")

echo "Chained jobs (afterok):  train=${J1} (${TIME_TRAIN})  ->  train_v2=${J2} (${TIME_V2})  ->  probe=${J3} (${TIME_PROBE})"
echo "Check:  squeue -u \$USER"
