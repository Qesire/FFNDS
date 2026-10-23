"""Export completed C1 numeric evidence with HF path/identifier receipts."""
import os
from pathlib import Path

from archive_public_json import archive, numeric_values
from c1_contract import CAPS, ROOT, atomic, read, sha
from export_fresh_quality_results import normalize_paths


def main():
    hf = Path(os.environ['HF_HOME'])
    root = hf / 'derived/ffnds/C1N0_native_information'
    run = read(root / 'RUN.json')
    if run['status'] != 'COMPLETE_NATIVE_C1_FIXED_SCOPE' or run['budget']['used'] != CAPS:
        raise RuntimeError('Require completed finite C1 diagnostic for this export')
    scratch = root / 'public_numeric_exports'
    scratch.mkdir(exist_ok=False)
    bindings = []
    for original, public in [('RUN.json', 'C1N0_native_results.json'),
                             ('ANALYSIS.json', 'C1N0_analysis.json'),
                             ('GEOMETRY.json', 'C1N0_geometry.json'),
                             ('DENSE_VALUE_CPU.json', 'C1N0_dense_value_CPU.json'),
                             ('BASELINE_MEDIA.json', 'C1N0_baseline_media.json'),
                             ('ACTION_PARAMETERS.json', 'C1N0_action_parameters.json'),
                             ('BUDGET.json', 'C1N0_budget.json')]:
        raw_path = root / original
        raw = read(raw_path); neutral = normalize_paths(raw, hf)
        if numeric_values(raw) != numeric_values(neutral):
            raise RuntimeError('Path normalization changed numeric evidence')
        intermediate = scratch / public
        atomic(intermediate, neutral)
        destination = ROOT / 'research/20261023' / public
        receipt = archive(intermediate, destination)
        bindings.append({'original_HF_relative': str(raw_path.relative_to(hf)),
                         'original_sha256': sha(raw_path), 'normalized_sha256': sha(intermediate),
                         'public_path': str(destination.relative_to(ROOT)),
                         'public_sha256': receipt['public_sha256'], 'numeric_values_unchanged': True})
    job = read(ROOT / 'runs/c1_execution/JOB.json')
    view = read(ROOT / 'runs/c1_execution/SOURCE_VIEW_RECEIPT_PRIVATE.json')
    if job['child_exit_code'] != 0 or job['status'] != 'COMPLETE':
        raise RuntimeError('Actual detached supervisor exit receipt required')
    if view['status'] != 'PASS_CHILD_WITH_VERIFIED_SOURCE_VIEW':
        raise RuntimeError('Verified dependency view did not complete')
    resources = {'status': 'COMPLETED_ONE_DETACHED_FINITE_C1_ATTEMPT', 'actual_budget': run['budget']['used'],
                 'actual_child_exit_code': job['child_exit_code'], 'source_view_completed': True,
                 'source_view_manifest_sha256': view['view_manifest_sha256'],
                 'actual_source_view_receipt_sha256': sha(ROOT / 'runs/c1_execution/SOURCE_VIEW_RECEIPT_PRIVATE.json'),
                 'actual_supervisor_receipt_sha256': sha(ROOT / 'runs/c1_execution/JOB.json'),
                 'max_concurrent_GPUs': 1, 'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in run['requests']),
                 'GPU_isolation_violations': run['resources']['violations'],
                 'logical_output_bytes_including_public_intermediates': sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
                 'source': 'tools/export_c1_native_results.py', 'source_sha256': sha(__file__),
                 'all72_numeric_feature_payloads_and8_baseline_videos_retained_in_HF': True,
                 'candidate_video_or_official_quality_or_speed_acceptance': False,
                 'path_transform_receipts': bindings}
    if resources['logical_output_bytes_including_public_intermediates'] > 1 << 30:
        raise RuntimeError('C1 actual output cap exceeded')
    destination = ROOT / 'research/20261023/C1N0_completed_resources.json'
    if destination.exists():
        raise RuntimeError('Preserve first C1 resource export')
    atomic(destination, resources)
    print('C1 native scalar/action/feature-summary evidence exported; raw payloads and videos retained in HF')


if __name__ == '__main__':
    main()
