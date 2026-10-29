#!/usr/bin/env bash
# One physical GPU UUID plus one of two global V3 native group leases; no retry.
set -euo pipefail
[[ "$#" -ge 4 ]] || exit 2
gpu="$1"; uuid="$2"; group_slot="$3"; shift 3
[[ "$gpu" =~ ^[0-3]$ && "$uuid" =~ ^GPU-[A-Za-z0-9-]+$ && "$group_slot" =~ ^[01]$ ]] || exit 2
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$root/.venv/bin/python" -B "$root/tools/worker_identity.py" --check
exec 8>"/tmp/ffnds-${USER}-teal-v3-native-${group_slot}.lock"
flock -n 8 || exit 6
exec 9>"/tmp/ffnds-${USER}-gpu${gpu}.lock"
flock -n 9 || exit 3
actual="$(/usr/bin/nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits)"
[[ "$actual" == "$uuid" ]] || exit 5
occupants="$(/usr/bin/nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$occupants" ]] || exit 4
export FFNDS_PHYSICAL_GPU="$gpu" FFNDS_TEAL_V3_GROUP_SLOT="$group_slot" CUDA_VISIBLE_DEVICES="$uuid"
unset PYTHONPATH PYTHONHOME
exec "$@"
