#!/usr/bin/env bash
# Install only into the FFNDS environment. Does not install upstream requirements.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
export UV_LINK_MODE=hardlink
if [[ ! -x "$project_root/.venv/bin/python" ]]; then
    "$uv_bin" venv --python 3.11 "$project_root/.venv"
fi
"$uv_bin" pip install --python "$project_root/.venv/bin/python" \
    'torch==2.10.0' 'torchvision==0.25.0' 'torchaudio==2.10.0' \
    --index-url https://download.pytorch.org/whl/cu128
"$uv_bin" pip install --python "$project_root/.venv/bin/python" \
    -r "$project_root/environment/worker-runtime.txt"
