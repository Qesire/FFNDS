"""Export the single finished C2 attempt, including failure/partial evidence."""
import os
from pathlib import Path
from archive_public_json import archive, numeric_values
from c2_contract import ROOT, atomic, read, sha


def neutral(value, hf):
    if isinstance(value, dict): return {k: neutral(v, hf) for k, v in value.items()}
    if isinstance(value, list): return [neutral(v, hf) for v in value]
    if isinstance(value, str):
        for prefix, label in [(str(hf), 'HF_CACHE'), (str(hf.resolve()), 'HF_CACHE'), (str(ROOT), 'REPOSITORY')]:
            value = value.replace(prefix, label)
    return value


def main():
    from worker_identity import require_worker
    require_worker()
    hf = Path(os.environ['HF_HOME']); root = hf / 'derived/ffnds/C2N0_norm_scaled_smoke'
    job = read(ROOT / 'runs/c2_execution/JOB.json')
    if job['status'] == 'RUNNING' or job['child_exit_code'] is None: raise RuntimeError('Native attempt still running')
    run = read(root / 'RUN.json')
    if run['status'] in ('PREPARING_C2_FIXED_SCOPE', 'RUNNING_C2_FIXED_NATIVE_SMOKE'): raise RuntimeError('No final native receipt')
    scratch = root / 'public_numeric_exports'; scratch.mkdir(exist_ok=False)
    originals = [('RUN.json', 'C2N0_native_results.json'), ('BUDGET.json', 'C2N0_budget.json')]
    if (root / 'VIDEO_REVIEW.json').exists(): originals.append(('VIDEO_REVIEW.json', 'C2N0_video_review.json'))
    bindings = []
    for source_name, public_name in originals:
        source = root / source_name; raw = read(source); public = neutral(raw, hf)
        if numeric_values(raw) != numeric_values(public): raise RuntimeError('Normalization changed numbers')
        temporary = scratch / public_name; atomic(temporary, public)
        destination = ROOT / 'research/20261023' / public_name
        receipt = archive(temporary, destination)
        bindings.append({'original_HF_relative': str(source.relative_to(hf)), 'original_sha256': sha(source),
                         'public_path': str(destination.relative_to(ROOT)), 'public_sha256': receipt['public_sha256'],
                         'numeric_values_unchanged': True})
    view_path = ROOT / 'runs/c2_execution/SOURCE_VIEW_RECEIPT_PRIVATE.json'
    resource = {'status': 'SINGLE_C2_ATTEMPT_EXPORTED_PENDING_VIDEO_REVIEW',
                'actual_budget': run['budget']['used'], 'actual_completed_videos': len(run['requests']),
                'actual_native_child_exit_code': job['child_exit_code'],
                'native_supervisor_receipt_sha256': sha(ROOT / 'runs/c2_execution/JOB.json'),
                'dependency_receipt_sha256': sha(view_path) if view_path.exists() else None,
                'dependency_receipt_status': read(view_path)['status'] if view_path.exists() else None,
                'max_concurrent_GPUs': 1, 'GPU_isolation_violations': run.get('resources', {}).get('violations'),
                'peak_allocated_bytes': max((r['peak_allocated_bytes'] for r in run['requests']), default=None),
                'logical_output_bytes_at_export': sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
                'source_sha256': sha(__file__), 'path_transform_receipts': bindings,
                'raw_videos_routing_and_logs_retained': True, 'automatic_visual_PASS': False,
                'human_full_video_review_completed': False, 'official_quality_or_repeated_system_speed_result': False}
    if resource['logical_output_bytes_at_export'] > 1 << 29: raise RuntimeError('C2 output cap exceeded')
    target = ROOT / 'research/20261023/C2N0_completed_resources.json'
    if target.exists(): raise RuntimeError('Preserve first public C2 resource receipt')
    atomic(target, resource)
    print('C2 actual numbers exported; human full-video verdict remains pending', flush=True)


if __name__ == '__main__': main()
