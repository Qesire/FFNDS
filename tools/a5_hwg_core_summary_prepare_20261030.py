"""Metadata-only preparation of the finite CPU closing comparison."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a5_hwg_matched_capture_20261030 as guard
from a5_hwg_matched_capture_prepare_20261030 import write_new
import a5_hwg_registered_core_summary_20261030 as entry


def parent(table_path, *, successful):
    q = next(v for v in guard.read(table_path) if v['kind'] == 'run')
    protocol = guard.read(guard.check_binding(ROOT, q['protocol']))
    job = guard.read(Path(q['job_path']))
    hf = Path(job['hf_root']).resolve()
    rp, ep = Path(job['run_json']), Path(job['output_dir']) / 'EXIT_RECEIPT.json'
    r, e = guard.read(rp), guard.read(ep)
    ref = guard.bind(rp, hf)
    guard.need(e['process_exited'] is True and e['child_returncode'] == (0 if successful else 1)
               and e['timeout'] is False and e['automatic_retry'] is False
               and e['run_sha256'] == ref['sha256'] and e['spec_sha256'] == q['job_spec_sha256']
               and e['protocol_sha256'] == r['protocol_sha256'] == q['protocol']['sha256'], 'actual parent EXIT and immutable binding')
    if successful:
        ordinal = protocol['source']['case']['native_source_ordinal']
        guard.need(r['status'] == 'COMPLETE_FULLM_ALL80_58ARM_DISCOVERY_EXIT_PENDING' and r['completed_pairs'] == 80,
                   'actual complete E0')
    else:
        ordinal = protocol['case']['native_source_ordinal']
        guard.need(ordinal == 96 and r['phases'] == [], 'actual failed source96')
    return {'id': q['id'], 'protocol': q['protocol'], 'job_spec_sha256': q['job_spec_sha256'],
            'RUN': ref, 'EXIT': guard.bind(ep, hf), 'source_ordinal': ordinal,
            'role': 'EXPOSED_CALIBRATION' if ordinal in [898, 1040] else 'EXPOSED_DEVELOPMENT_NOT_BLIND'}, protocol, hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent-table', action='append', type=Path, required=True)
    ap.add_argument('--failed-source-table', type=Path, required=True)
    ap.add_argument('--worker', type=Path, required=True)
    ap.add_argument('--private-output', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    a = ap.parse_args()
    parents = [parent(v, successful=True) for v in a.parent_table]
    sources = [v[0] for v in parents]
    guard.need([s['source_ordinal'] for s in sources] == [898, 1040, 979], 'three actual E0 sources')
    failed, _, failed_hf = parent(a.failed_source_table, successful=False)
    hf = parents[0][2]
    guard.need(all(v[2] == hf for v in parents) and failed_hf == hf, 'shared source HF')
    worker = guard.read(a.worker)
    guard.need(Path(worker['hf_root']).resolve() == hf, 'actual private worker HF')
    a.private_output.mkdir(parents=True, exist_ok=False)
    locator = a.private_output / 'assets.json'
    private = {k: worker[k] for k in ['hf_root', 'cache_path', 'worker_profile', 'local_package_root']}
    private['worker_profile_sha256'] = guard.sha(Path(private['worker_profile']))
    write_new(locator, private, True)
    rows = {r['path']: r for r in parents[0][1]['repo_sources']}
    for r in rows.values(): guard.check_binding(ROOT, r)
    for name in [entry.SELF, 'tools/a5_hwg_core_summary_math_20261030.py',
                 'tools/a5_hwg_core_summary_prepare_20261030.py', 'tests/test_a5_hwg_core_summary_20261030.py']:
        rows[name] = guard.bind(ROOT / name, ROOT)
    sid = 'E0_A5_HWG_CORE3_CLOSING_COMPARISON_20261030_V1'
    output = 'ffnds-derived/e0_a5_hwg_core3_closing_comparison_20261030/attempt1/worker/results'
    p = {'id': sid, 'author_agent_id': '/root', 'status': 'FROZEN_PENDING_EXACT_RECHECK_NOT_AUTHORIZED',
         'kind': entry.KIND, 'resources': entry.RES, 'call_limits': {}, 'cpu_affinity': [worker['cpu_affinity'][0]],
         'repo_sources': [rows[k] for k in sorted(rows)], 'private_assets_sha256': guard.sha(locator),
         'private_worker_profile_sha256': private['worker_profile_sha256'], 'sources': sources,
         'failed_source96': failed, 'output_HF_relative': output, 'automatic_retry': False}
    write_new(a.protocol, p)
    dest = hf / output
    guard.need(not dest.parent.exists(), 'fresh CPU closing output')
    job_path = a.private_output / 'run.json'
    job = {'id': sid, 'cwd': str(ROOT), 'hf_root': str(hf), 'output_dir': str(dest.parent),
           'run_json': str(dest / 'RUN.json'), 'protocol_path': str(a.protocol.resolve()), 'protocol_sha256': guard.sha(a.protocol),
           'external_timeout_seconds': entry.RES['outer_seconds'], 'argv': [
               'env', 'FFNDS_WORKER_CONFIG=' + worker['worker_profile'], 'CUDA_VISIBLE_DEVICES=',
               'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'PATH=/usr/bin:/bin',
               'taskset', '-c', str(p['cpu_affinity'][0]), str(ROOT / '.venv/bin/python'), '-I', '-B', str(ROOT / entry.SELF),
               '--protocol', str(a.protocol.resolve()), '--job-spec', str(job_path.resolve()), '--private-assets', str(locator.resolve())]}
    write_new(job_path, job, True)
    write_new(a.private_output / 'BOUND_JOBS.json', [{'id': sid, 'kind': 'run',
        'protocol': guard.bind(a.protocol.resolve(), ROOT), 'job_path': str(job_path.resolve()),
        'job_spec_sha256': guard.sha(job_path), 'fingerprint': guard.canonical(p)}], True)
    print(sid, 'NOT_REGISTERED_NOT_RUN')


if __name__ == '__main__': main()
