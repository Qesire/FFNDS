"""Prepare one independently reviewed relocation after actual foreign compute.

Metadata only. Preserves the failed full-M attempt; never runs or authorizes.
The numerical entry point, 80 pairs, 58 arms and CAL1 artifact stay frozen.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a5_hwg_matched_capture_20261030 as guard
from a5_hwg_matched_capture_prepare_20261030 import write_new

SELF = 'tools/a5_hwg_relocated_discovery_prepare_20261030.py'


def main():
    ap = argparse.ArgumentParser()
    for name in ['previous-table', 'worker', 'private-output', 'protocol']:
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    old = guard.read(a.previous_table)
    guard.need(len(old) == 1, 'one original finite scope')
    q = old[0]
    p = guard.read(guard.check_binding(ROOT, q['protocol']))
    job = guard.read(Path(q['job_path']))
    guard.need(guard.sha(Path(q['job_path'])) == q['job_spec_sha256'], 'old exact job')
    hf = Path(job['hf_root']).resolve()
    rp, ep = Path(job['run_json']), Path(job['output_dir']) / 'EXIT_RECEIPT.json'
    r, e = guard.read(rp), guard.read(ep)
    rr, er = guard.bind(rp, hf), guard.bind(ep, hf)
    guard.need(q['id'] == p['id'] == r['id'] == e['id']
               == 'E0_A5_HWG_FULLM_DISCOVERY_20261030_SOURCE1040_V1'
               and e['process_exited'] is True and e['child_returncode'] == 1
               and e['timeout'] is False and e['automatic_retry'] is False
               and e['spec_sha256'] == q['job_spec_sha256']
               and e['protocol_sha256'] == r['protocol_sha256'] == q['protocol']['sha256']
               and e['run_sha256'] == rr['sha256']
               and r['status'] == 'FAILED_ASSIGNED_GPU_MONITOR_PRESERVED_NO_RETRY'
               and r['error'] == 'Foreign compute appeared on the assigned physical GPU'
               and r['completed_pairs'] == len(r['site_manifest']) == 10,
               'actual known resource failure; no generic or automatic retry')
    for row in p['repo_sources']: guard.check_binding(ROOT, row)
    worker = guard.read(a.worker)
    guard.need(Path(worker['hf_root']).resolve() == hf and len(worker['cpu_affinity']) == 4,
               'exact ready worker geometry')
    a.private_output.mkdir(parents=True, exist_ok=False)
    locator = a.private_output / 'assets.json'
    private = {k: worker[k] for k in ['hf_root', 'cache_path', 'worker_profile', 'local_package_root']}
    private['worker_profile_sha256'] = guard.sha(Path(private['worker_profile']))
    write_new(locator, private, True)
    p.update(id='E0_A5_HWG_FULLM_DISCOVERY_20261030_SOURCE1040_V2_RELOCATED',
             status='FROZEN_PENDING_EXACT_INDEPENDENT_RECHECK_NOT_AUTHORIZED',
             cpu_affinity=worker['cpu_affinity'], private_assets_sha256=guard.sha(locator),
             private_worker_profile_sha256=private['worker_profile_sha256'],
             output_HF_relative='ffnds-derived/e0_a5_hwg_fullm_discovery_20261030/source1040/attempt2_relocated/worker/results',
             preceding_failed_attempt={'id': q['id'], 'protocol': q['protocol'],
                 'job_spec_sha256': q['job_spec_sha256'], 'RUN': rr, 'EXIT': er,
                 'completed_pairs': 10, 'included_in_new_numerical_mean': False,
                 'correction': 'NEW_ASSIGNED_DEVICE_AND_WORKER_ONLY; ALL80_PAIRS_REDO; NUMERICAL_CODE_UNCHANGED'})
    p['repo_sources'].append(guard.bind(ROOT / SELF, ROOT))
    p['repo_sources'].sort(key=lambda row: row['path'])
    write_new(a.protocol, p)
    out = hf / p['output_HF_relative']
    guard.need(not out.parent.exists(), 'fresh relocated finite output')
    jp = a.private_output / 'run.json'
    guard.need(job['argv'][6] == 'bash'
               and Path(job['argv'][7]).resolve().parent == ROOT / 'scripts',
               'original assigned-UUID wrapper shape')
    argv = ['env', 'FFNDS_WORKER_CONFIG=' + worker['worker_profile'],
            'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'PATH=/usr/bin:/bin',
            'bash', job['argv'][7],
            str(worker['physical_gpu']), worker['GPU_UUID'], 'taskset', '-c',
            ','.join(map(str, worker['cpu_affinity'])), str(ROOT / '.venv/bin/python'), '-I', '-B',
            str(ROOT / 'tools/a5_hwg_registered_delta_discovery_20261030.py'),
            '--protocol', str(a.protocol.resolve()), '--job-spec', str(jp.resolve()),
            '--private-assets', str(locator.resolve())]
    new_job = {'id': p['id'], 'cwd': str(ROOT), 'hf_root': str(hf),
               'output_dir': str(out.parent), 'run_json': str(out / 'RUN.json'),
               'protocol_path': str(a.protocol.resolve()), 'protocol_sha256': guard.sha(a.protocol),
               'external_timeout_seconds': p['resources']['outer_seconds'], 'argv': argv}
    write_new(jp, new_job, True)
    row = {'id': p['id'], 'kind': 'run', 'protocol': guard.bind(a.protocol.resolve(), ROOT),
           'job_path': str(jp.resolve()), 'job_spec_sha256': guard.sha(jp),
           'fingerprint': guard.canonical(p)}
    write_new(a.private_output / 'BOUND_JOBS.json', [row], True)
    print(p['id'], row['protocol']['sha256'], row['job_spec_sha256'], 'NOT_REGISTERED_NOT_RUN')


if __name__ == '__main__': main()
