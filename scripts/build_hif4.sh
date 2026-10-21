#!/usr/bin/env bash
set -euo pipefail
: "${FFNDS_ROOT:?Run through tools/run_on_worker.py}"
"$FFNDS_ROOT/.venv/bin/python" "$FFNDS_ROOT/tools/worker_identity.py" --check
# This source includes the audited current-stream/device-guard patches.
cd "$FFNDS_ROOT/.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/cusrc"
python setup.py build_ext --inplace
