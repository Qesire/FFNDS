"""Export completed R14N0 scalar evidence without private paths or tensors."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

from r14_stripe_contract import CAPS, ORDER, ROOT, execution_root, read, sha, write_new
from worker_identity import require_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    require_worker()
    protocol_path = ROOT / 'research/20261022/R14N0_registered_protocol.json'
    protocol = read(protocol_path)
    root = execution_root(protocol)
    result, capture, budget = [read(root / name) for name in ('VALUE.json', 'CAPTURE.json', 'BUDGET.json')]
    if (result['status'] != 'COMPLETE_FIXED_NATIVE_STRIPE_VALUE'
            or capture['status'] != 'PASS_FOUR_ORIGINAL_UB1_OBSERVATIONS'
            or result['protocol_sha256'] != capture['protocol_sha256']
            or result['protocol_sha256'] != sha(protocol_path)
            or budget != result['budget'] or budget['used'] != CAPS):
        raise RuntimeError('Fixed native result/capture/budget incomplete')
    for name, expected in protocol['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Executed source changed before export')
    base = ROOT / 'research/20261022'
    receipt = {'status': capture['status'], 'id': protocol['id'],
               'protocol_sha256': sha(protocol_path), 'raw_capture_receipt_sha256': sha(root / 'CAPTURE.json'),
               'completed': [], 'new_candidate_rollouts': 0, 'new_fitted_parameters': 0,
               'pipeline_peak_memory_logged': False}
    if [entry['index'] for entry in capture['completed']] != list(ORDER):
        raise RuntimeError('Original four source coverage changed')
    for entry in capture['completed']:
        if sha(entry['report']) != entry['sha256']:
            raise RuntimeError('Original observation receipt changed')
        case = read(entry['report'])
        scope = case['scope']
        if sha(scope['path']) != scope['sha256']:
            raise RuntimeError('New complete-stripe tensor file changed')
        scope = {key: value for key, value in scope.items() if key != 'path'}
        receipt['completed'].append({**case, 'scope': scope, 'private_input_receipt_sha256': entry['sha256']})
    receipt_path = base / 'R14N0_capture_receipt.json'
    write_new(receipt_path, receipt)
    for original, name in (('BUDGET.json', 'R14N0_budget_debit.json'),
                           ('TRAINING_ONLY_FIXED_MASK.json', 'R14N0_training_fixed_mask.json')):
        target = base / name
        with target.open('xb') as stream:
            stream.write((root / original).read_bytes())
    result.update(raw_private_result_sha256=sha(root / 'VALUE.json'),
                  source_sha256=protocol['source_sha256'],
                  protocol={'path': str(protocol_path.relative_to(ROOT)), 'sha256': sha(protocol_path)},
                  capture_receipt={'path': str(receipt_path.relative_to(ROOT)), 'sha256': sha(receipt_path)})
    write_new(base / 'R14N0_native_results.json', result)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    used = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
    import shutil
    cache = read(ROOT / '.runtime/hif4_rebuilt_cache.json')
    resources = {'status': 'COMPLETE_FIXED_R14N0_RESOURCES', 'actual_model_generations': 4,
                 'actual_standalone_FFN_forwards': 32, 'extra_linear_calls_including_capture_teacher': 104,
                 'standalone_actual_linear_calls': sum(d.get('actual_linear_calls', 0) for d in budget['debits']),
                 'model_native_runtime_linear_and_QDQ_calls': 4 * 1600,
                 'original_UB1_R256_positions': 4 * 160,
                 'capture_extra_teacher_linears_and_QDQ_calls': 8,
                 'max_GPUs': 1, 'GPU_slot': 0, 'post_run_GPU_readback': gpu.stdout.strip().splitlines(),
                 'incremental_HF_output_bytes': used,
                 'free_HF_output_bytes_after': shutil.disk_usage(root).free,
                 'free_node_cache_bytes_after': shutil.disk_usage(Path(cache['path']).parent).free,
                 'standalone_max_allocated_bytes': result['max_allocated_bytes'],
                 'standalone_max_reserved_bytes': result['max_reserved_bytes'],
                 'pipeline_peak_memory_logged': False,
                 'post_run_installed_torch': importlib.metadata.version('torch'),
                 'post_run_installed_triton': importlib.metadata.version('triton'),
                 'new_fitted_parameters': 0, 'new_candidate_rollouts': 0,
                 'native_model_speed_or_video_quality_acceptance': False}
    if used > 1 << 29 or resources['standalone_actual_linear_calls'] != 96:
        raise RuntimeError('Completed fixed resource accounting differs')
    write_new(base / 'R14N0_completed_resources.json', resources)
    print('R14N0 public scalar/capture/budget/resource export complete; no operand tensors or private paths.', flush=True)


if __name__ == '__main__':
    main()
