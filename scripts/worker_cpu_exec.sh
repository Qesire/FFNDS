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
flock -n 9 || { echo 'Another FFNDS CPU diagnostic is running.' >&2; exit 3; }
cd "$FFNDS_ROOT"
exec "$@"
