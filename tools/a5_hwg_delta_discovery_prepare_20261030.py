"""Bind actual closed parents and exact finite E0 jobs; never authorize/run.

Only metadata and code SHA reads occur. No model/tensor value is loaded.
Private locators, assigned UUIDs and machine names never enter public JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a5_hwg_matched_capture_20261030 as guard
from a5_hwg_matched_capture_prepare_20261030 import write_new
from a5_hwg_matched_capture_namespace_v2_20261030 import namespace_bindings

FILES = [
    'tools/a5_hwg_delta_math_20261030.py',
    'tools/a5_hwg_support_regularity_math_20261030.py',
    'tools/a5_hwg_e0_native_replay_20261030.py',
    'tools/a5_hwg_static_signed_math_20261030.py',
    'tools/a5_hwg_evidence_io_20261030.py',
    'tools/a5_hwg_registered_static_fit_20261030.py',
    'tools/a5_hwg_registered_delta_discovery_20261030.py',
    'tools/a5_hwg_delta_discovery_prepare_20261030.py',
    'tests/test_a5_hwg_delta_math_20261030.py',
    'tests/test_a5_hwg_support_regularity_math_20261030.py',
    'tests/test_a5_hwg_e0_native_replay_20261030.py',
    'tests/test_a5_hwg_static_signed_math_20261030.py',
    'tests/test_a5_hwg_evidence_io_20261030.py',
    'tests/test_a5_hwg_registered_discovery_contract_20261030.py',
]


def closed_source(table_path):
    table = guard.read(table_path)
    run_job = next(v for v in table if v['kind'] == 'run')
    verify_job = next(v for v in table if v['kind'] == 'verify')
    p = guard.read(guard.check_binding(ROOT, run_job['protocol']))
    rjob = guard.read(Path(run_job['job_path']))
    vjob = guard.read(Path(verify_job['job_path']))
    hf = Path(rjob['hf_root']).resolve()
    run_path, verify_path = Path(rjob['run_json']), Path(vjob['run_json'])
    run_exit, verify_exit = Path(rjob['output_dir']) / 'EXIT_RECEIPT.json', Path(vjob['output_dir']) / 'EXIT_RECEIPT.json'
    result = {'producer_id': p['id'], 'producer_protocol': run_job['protocol'],
              'producer_job_spec_sha256': run_job['job_spec_sha256'],
              'verifier_job_spec_sha256': verify_job['job_spec_sha256'],
              'producer_RUN': guard.bind(run_path, hf), 'producer_EXIT': guard.bind(run_exit, hf),
              'verifier_RUN': guard.bind(verify_path, hf), 'verifier_EXIT': guard.bind(verify_exit, hf),
              'case': p['case'], 'results_HF_relative': p['output_HF_relative']}
    run = guard.read(run_path)
    manifest = guard.read(guard.check_binding(hf, run['phases'][1]['capture_manifest']))
    result['metadata'] = {f'step{v["step"]}_block{v["block"]:02d}': v['metadata'] for v in manifest['scopes']}
    # Reuse the execution validator with metadata-only charged reads. It loads
    # no observed payload or model values, and requires both actual EXIT0s.
    from a5_hwg_registered_static_fit_20261030 import validate_parent
    budget = guard.ReadBudget(32 << 20)
    validate_parent(result, hf, budget, p['cache'])
    return result, p, hf


def source_closure(template):
    rows = {r['path']: r for r in template['repo_sources']}
    for row in rows.values(): guard.check_binding(ROOT, row)
    for rel in FILES: rows[rel] = guard.bind(ROOT / rel, ROOT)
    return [rows[k] for k in sorted(rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['static', 'discovery'])
    ap.add_argument('--parent-table', type=Path, action='append', required=True)
    ap.add_argument('--worker', type=Path, required=True)
    ap.add_argument('--private-output', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--static-table', type=Path)
    a = ap.parse_args()
    parents = [closed_source(path) for path in a.parent_table]
    sources, template, hf = [v[0] for v in parents], parents[0][1], parents[0][2]
    guard.need(all(v[1]['cache'] == template['cache'] and v[2] == hf for v in parents), 'shared actual cache/HF')
    worker = guard.read(a.worker)
    guard.need(Path(worker['hf_root']).resolve() == hf and len(worker['cpu_affinity']) == 4, 'actual worker layout')
    a.private_output.mkdir(parents=True, exist_ok=False)
    locator = a.private_output / 'assets.json'
    private = {k: worker[k] for k in ['hf_root', 'cache_path', 'worker_profile', 'local_package_root']}
    private['worker_profile_sha256'] = guard.sha(Path(private['worker_profile']))
    write_new(locator, private, True)
    if a.kind == 'static':
        import a5_hwg_registered_static_fit_20261030 as entry
        guard.need([s['case']['native_source_ordinal'] for s in sources] == entry.CALIBRATION_SOURCES, 'exact CAL1 only')
        sid = 'E0_A5_HWG_SIGNED_STATIC_CAL1_1040_20261030_V3'
        output = 'ffnds-derived/e0_a5_hwg_signed_static_cal1_1040_v3_20261030/attempt1/worker/results'
        extra = {'calibration_source_ordinals': entry.CALIBRATION_SOURCES, 'sources': sources,
                 'keep_groups': 80, 'hidden_group': 64, 'source_role': 'EXPOSED_CALIBRATION_NO_DEV_READ',
                 'kind': entry.CALIBRATION_KIND}
    else:
        import a5_hwg_registered_delta_discovery_20261030 as entry
        from a5_hwg_e0_native_replay_20261030 import arm_specs
        guard.need(len(sources) == 1 and a.static_table is not None, 'one exact discovery source and frozen CAL1')
        st = guard.read(a.static_table)
        sj = guard.read(Path(st[0]['job_path']))
        sr = guard.read(Path(sj['run_json']))
        static = {'id': st[0]['id'], 'protocol': st[0]['protocol'],
                  'job_spec_sha256': st[0]['job_spec_sha256'],
                  'RUN': guard.bind(Path(sj['run_json']), hf),
                  'EXIT': guard.bind(Path(sj['output_dir']) / 'EXIT_RECEIPT.json', hf),
                  'artifact': sr['static_support'], 'schema': sr['static_support_schema']}
        ordinal = sources[0]['case']['native_source_ordinal']
        sid = f'E0_A5_HWG_FULLM_DISCOVERY_20261030_SOURCE{ordinal}_V1'
        output = f'ffnds-derived/e0_a5_hwg_fullm_discovery_20261030/source{ordinal}/attempt1/worker/results'
        view = guard.read(ROOT / '.runtime/model_view.json')
        bias = {}
        for expert in ['high_noise_model', 'low_noise_model']:
            row = next(v for v in view['files'] if ('/' + expert + '/') in v['view_path'] and v['view_path'].endswith('.safetensors'))
            bias[expert] = {k: row[k] for k in ['bytes', 'sha256', 'repo', 'revision', 'filename']}
        extra = {'kind': 'FULLM_ALL80_58ARM_HIF4_TEACHER_DISCOVERY_V1',
                 'source': sources[0], 'source_role': 'EXPOSED_CALIBRATION' if ordinal == 1040 else 'EXPOSED_DEVELOPMENT_NOT_BLIND',
                 'static_calibration': static, 'keep_channels': 5120, 'grid': [21, 29, 52],
                 'P64_seed': 42, 'shuffle_seed': 101, 'random_group_seed_base': 42000,
                 'arms': arm_specs(), 'bias_checkpoints': bias, 'model_view': template['model_view']}
        # Check closed CAL1 identity here without reading its tensor payload.
        receipt = guard.read(Path(sj['output_dir']) / 'EXIT_RECEIPT.json')
        guard.need(receipt['process_exited'] is True and receipt['child_returncode'] == 0
                   and receipt['timeout'] is False and receipt['automatic_retry'] is False
                   and receipt['run_sha256'] == static['RUN']['sha256']
                   and receipt['spec_sha256'] == static['job_spec_sha256']
                   and receipt['protocol_sha256'] == static['protocol']['sha256']
                   and sr['completed_sites'] == 80 and sr['dev_tensor_files_read'] == 0
                   and sr['status'] == 'COMPLETE_CAL1_SIGNED_GREEDY_FULL80_EXIT_PENDING', 'actual closed CAL1')
    rows = source_closure(template)
    p = {'id': sid, 'author_agent_id': '/root', 'status': 'FROZEN_PENDING_EXACT_INDEPENDENT_RECHECK_NOT_AUTHORIZED',
         'resources': entry.RES, 'call_limits': entry.CALL_LIMITS,
         'cpu_affinity': worker['cpu_affinity'],
         'runtime_versions': {k: template['runtime_versions'][k] for k in ['torch', 'numpy', 'safetensors', 'triton']},
         'software_scope': 'NUMERICAL_REPLAY_AND_RAW_TENSOR_IO_ONLY; ORIGINAL_PIPELINE_VERSIONS_REMAIN_BOUND_IN_PARENT_PROTOCOL; NO_TEXT_ENCODER_VAE_OR_TRANSFORMERS_CALLS',
         'private_assets_sha256': guard.sha(locator), 'private_worker_profile_sha256': private['worker_profile_sha256'],
         'cache': template['cache'], 'repo_sources': rows, 'namespace_bindings': namespace_bindings(rows),
         'output_HF_relative': output, 'automatic_retry': False, **extra}
    write_new(a.protocol, p)
    job_path = a.private_output / 'run.json'
    out = hf / output
    argv = ['env', 'FFNDS_WORKER_CONFIG=' + worker['worker_profile'],
            'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'PATH=/usr/bin:/bin',
            'bash', str(ROOT / 'scripts/worker_teal_assigned_uuid_exec.sh'),
            str(worker['physical_gpu']), worker['GPU_UUID'], 'taskset', '-c', ','.join(map(str, worker['cpu_affinity'])),
            str(ROOT / '.venv/bin/python'), '-I', '-B', str(ROOT / entry.SELF),
            '--protocol', str(a.protocol.resolve()), '--job-spec', str(job_path.resolve()), '--private-assets', str(locator.resolve())]
    job = {'id': sid, 'cwd': str(ROOT), 'hf_root': str(hf), 'output_dir': str(out.parent),
           'run_json': str(out / 'RUN.json'), 'protocol_path': str(a.protocol.resolve()),
           'protocol_sha256': guard.sha(a.protocol), 'argv': argv,
           'external_timeout_seconds': entry.RES['outer_seconds']}
    guard.need(not out.parent.exists(), 'fresh finite output required')
    write_new(job_path, job, True)
    table = [{'id': sid, 'kind': 'run', 'protocol': guard.bind(a.protocol.resolve(), ROOT),
              'job_path': str(job_path.resolve()), 'job_spec_sha256': guard.sha(job_path), 'fingerprint': guard.canonical(p)}]
    write_new(a.private_output / 'BOUND_JOBS.json', table, True)
    print(json.dumps({'id': sid, 'source_files': len(rows), 'protocol_sha256': table[0]['protocol']['sha256'],
                      'job_spec_sha256': table[0]['job_spec_sha256'], 'status': 'NOT_REGISTERED_NOT_RUN'}, indent=2))


if __name__ == '__main__': main()
