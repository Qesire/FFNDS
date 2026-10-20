#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname -s)" == worker ]] || { echo 'Build on worker.' >&2; exit 2; }
: "${FFNDS_ROOT:?Run through tools/run_on_worker.py}"
# This source includes the audited current-stream/device-guard patches.
cd "$FFNDS_ROOT/.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/cusrc"
python setup.py build_ext --inplace
