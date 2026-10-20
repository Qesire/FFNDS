#!/usr/bin/env bash
# Run at most one project CPU diagnostic alongside the two controlled GPU jobs.
set -euo pipefail
[[ "$(hostname -s)" == worker ]] || { echo 'CPU diagnostics must run on worker.' >&2; exit 2; }
FFNDS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export FFNDS_ROOT
unset PYTHONPATH PYTHONHOME
source "$FFNDS_ROOT/config/worker.env"
export CUDA_VISIBLE_DEVICES=''
exec 9>"/tmp/ffnds-${USER}-cpu.lock"
# Serialize independently scheduled diagnostics without failing a completed
# GPU workflow simply because its CPU audit arrives during a resumable fit.
flock -w "${FFNDS_CPU_LOCK_WAIT_SECONDS:-1800}" 9 || {
  echo 'Timed out waiting for the FFNDS CPU diagnostic lock.' >&2; exit 3;
}
cd "$FFNDS_ROOT"
exec "$@"
