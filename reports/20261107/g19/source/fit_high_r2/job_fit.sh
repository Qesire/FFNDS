#!/bin/bash
#SBATCH --job-name=g19_quality_fit
#SBATCH --partition=P107-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=anode21
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [ "$#" -ne 2 ]; then echo 'usage: job_fit.sh ABS_STAGE SCOPE_SHA' >&2; exit 2; fi
stage="$1"
scope_sha="$2"
case "$stage" in /*) ;; *) exit 2;; esac
cd "$stage"
exec /public/home/scc/pb23061276/miniconda3/envs/hif4/bin/python3.11 "$stage/run_fit_guard.py" --stage "$stage" --scope-sha "$scope_sha"
