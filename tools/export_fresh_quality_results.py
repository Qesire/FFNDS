"""Publish compact finite results; retain weights/videos/per-boundary traces in HF."""
import json
import os
from pathlib import Path

from archive_public_json import archive, numeric_values
from fresh_quality_contract import CAPS, METHOD, ROOT, atomic, read, sha


def normalize_paths(value, hf):
    if isinstance(value, dict):
        return {k: normalize_paths(v, hf) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_paths(v, hf) for v in value]
    if isinstance(value, str):
        for prefix, public in [(str(hf), 'HF_CACHE'), (str(hf.resolve()), 'HF_CACHE'),
                               (str(ROOT), 'REPOSITORY')]:
            if value == prefix or value.startswith(prefix + '/'):
                return public + value[len(prefix):]
    return value


def main():
    from worker_identity import require_worker
    require_worker()
    hf = Path(os.environ['HF_HOME'])
    root = hf / 'derived/ffnds/R17E0_fresh_quality'
    analysis = read(root / 'ANALYSIS.json')
    if (analysis['status'] != 'COMPLETE_FIXED_SCOPE_ACTUAL_QUALITY_AND_REPEATED_NATIVE_TIMING' or
            analysis['actual_budget'] != CAPS or analysis['automatic_quality_or_Pareto_or_paper_acceptance'] is not False):
        raise RuntimeError('Incomplete finite scope or unsupported promotion')
    scratch = root / 'public_numeric_exports'
    scratch.mkdir(exist_ok=False)
    bindings = []
    exports = [('ANALYSIS.json', 'R17E0_analysis.json'),
               ('GENERATION_RECEIPTS_COMPACT.json', 'R17E0_generation_receipts.json'),
               ('OFFICIAL_RECORDS_COMPLETE.json', 'R17E0_official_metric_records.json'),
               ('BUDGET.json', 'R17E0_budget_debit.json'),
               ('VIDEO_DECODE.json', 'R17E0_video_decode.json')]
    for original_name, public_name in exports:
        original = root / original_name
        raw = read(original)
        neutral = normalize_paths(raw, hf)
        if numeric_values(raw) != numeric_values(neutral):
            raise RuntimeError('Public path normalization changed a numeric result')
        intermediate = scratch / public_name
        atomic(intermediate, neutral)
        destination = ROOT / 'research/20261023' / public_name
        receipt = archive(intermediate, destination)
        bindings.append({'original_HF_relative': str(original.relative_to(hf)),
                         'original_raw_sha256': sha(original), 'public_repository_path': str(destination.relative_to(ROOT)),
                         'normalized_intermediate_sha256': sha(intermediate),
                         'public_sha256': receipt['public_sha256'], 'numeric_values_unchanged': True,
                         'only_string_paths_and_private_worker_identifiers_normalized': True})
    metric_run = read(root / 'metrics/RUN.json')
    generation_run = read(root / 'GENERATIONS.json')
    metric_peak = metric_run['last_resource_check']['peak_allocated_bytes']
    resources = {'id': METHOD, 'status': 'FIXED145_GENERATIONS_45_OFFICIAL_CALLS_AND_CPU_ANALYSIS_COMPLETE',
                 'actual_budget': analysis['actual_budget'], 'actual_fresh_independent_images': 23,
                 'actual_fresh_quality_videos': 115, 'old_source_warmup_or_timing_videos': 30,
                 'official_evaluator_calls': 45, 'max_concurrent_GPUs': 1,
                 'peak_allocated_bytes': max(metric_peak, max(r['peak_allocated_bytes'] for r in read(root / 'GENERATION_RECEIPTS_COMPACT.json')['records'])),
                 'metric_phase_peak_allocated_bytes': metric_peak,
                 'generation_GPU_isolation_violations': generation_run['resources']['violations'],
                 'metric_GPU_isolation_violations': metric_run['resources']['violations'],
                 'original_generation_receipt_sha256': sha(root / 'GENERATIONS.json'),
                 'original_metric_receipt_sha256': sha(root / 'metrics/RUN.json'),
                 'logical_experiment_output_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink()),
                 'full_per_boundary_records_and_all_media_retained_in_HF': True,
                 'path_transform_receipts': bindings, 'official_full_dataset_benchmark': False,
                 'automatic_quality_or_Pareto_or_paper_acceptance': False, 'source_sha256': sha(__file__)}
    if resources['logical_experiment_output_bytes'] > 4 << 30:
        raise RuntimeError('Actual experimental output cap exceeded')
    target = ROOT / 'research/20261023/R17E0_completed_resources.json'
    if target.exists():
        raise RuntimeError('Preserve first public numeric export')
    atomic(target, resources)
    print('Finite fresh quality/timing numbers exported with exact numeric preservation; no weights/videos inGit.', flush=True)


if __name__ == '__main__':
    main()
