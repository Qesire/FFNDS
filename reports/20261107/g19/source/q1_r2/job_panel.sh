#!/bin/bash
#SBATCH --partition=P107-A100
#SBATCH --account=competition
#SBATCH --qos=qos_p107-a100
#SBATCH --no-requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$(dirname "$2")"
exec "$1" runner_panel.py --scope "$2" --scope-sha "$3" --output "$4"
