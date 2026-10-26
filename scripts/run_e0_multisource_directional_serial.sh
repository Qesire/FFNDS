#!/usr/bin/env bash
# Four fixed sources; each fresh process inherits one idle-GPU UUID lock.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_hf='/home/xiang/.cache/huggingface'
task_study="$task_hf/ffnds-derived/e0_multisource_directional_20261026/attempt1"
task_queue="$task_hf/ffnds-derived/e0_multisource_directional_20261026/launch1"
mkdir -p "$task_queue"
for task_source in 04 08 16 20; do
    bash "$task_root/scripts/worker_wait_uuid_idle_exec.sh" "$task_queue/source${task_source}_queue.json" \
        "$task_root/.venv/bin/python" "$task_root/tools/run_e0_multisource_directional.py" \
        --protocol "$task_root/config/e0_multisource_directional_native_20261026_v1.json" \
        --bank "$task_hf/ffnds-derived/e0_function_basis_20261026/prototype_banks_attempt1" \
        --hf-root "$task_hf" --study-root "$task_study" --pair "dc_source${task_source}" --execute-native
done
