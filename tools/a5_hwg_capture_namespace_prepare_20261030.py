"""Freeze a separately reviewed namespace-repair scope; never register/start."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a5_hwg_matched_capture_namespace_v2_20261030 as repair
import a5_hwg_matched_capture_prepare_20261030 as original

owner = repair.owner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, required=True, choices=range(4))
    ap.add_argument('--parent-jobs', type=Path, required=True)
    ap.add_argument('--private-output', type=Path, required=True)
    a = ap.parse_args()
    parents = json.loads(a.parent_jobs.read_text())
    parent = next(j for j in parents if j['kind'] == 'run' and f'PART{a.part}_' in j['id'])
    job = json.loads(Path(parent['job_path']).read_text())
    hf = Path(job['hf_root'])
    out = Path(job['output_dir'])
    receipt_path = out / 'EXIT_RECEIPT.json'
    receipt = owner.read(receipt_path)
    report = owner.read(Path(job['run_json']))
    owner.need(receipt['process_exited'] is True and receipt['child_returncode'] == 1
               and receipt['timeout'] is False and not report['phases']
               and report.get('error') == 'native namespace module lacks bound source',
               'repair only follows preserved actual namespace failure before generation')
    a.private_output.mkdir(parents=True, exist_ok=False)
    p = copy.deepcopy(owner.read(Path(job['protocol_path'])))
    p['id'] = f'E0_A5_HWG_MATCHED_CAPTURE_20261030_PART{a.part}_NS_V2'
    p.pop('no_child_failure_parent', None)
    p['namespace_failure_parent'] = {
        'id': parent['id'], 'receipt': owner.bind(receipt_path, hf),
        'RUN': owner.bind(Path(job['run_json']), hf),
        'actual_child_returncode': 1, 'actual_completed_phases': 0,
        'same_scientific_capture_and_numerical_operations': True,
    }
    p['output_HF_relative'] = f'ffnds-derived/e0_a5_hwg_matched_capture_namespace_v2_20261030/part{a.part}/attempt1/worker/results'
    sources = {r['path']: r for r in p['repo_sources']}
    for path in [repair.SELF, 'tools/a5_hwg_capture_namespace_prepare_20261030.py',
                 'tests/test_a5_hwg_namespace_v2_20261030.py']:
        sources[path] = owner.bind(ROOT / path, ROOT)
    p['repo_sources'] = [sources[k] for k in sorted(sources)]
    p['namespace_bindings'] = repair.namespace_bindings(p['repo_sources'])
    protocol = ROOT / f'config/e0_a5_hwg_matched_capture_namespace_20261030_part{a.part}_frozen_v2.json'
    original.write_new(protocol, p)
    summary = []
    old_script = str(ROOT / 'tools/a5_hwg_matched_capture_20261030.py')
    for kind in ['run', 'verify']:
        old = next(j for j in parents if j['kind'] == kind and f'PART{a.part}_' in j['id'])
        j = owner.read(Path(old['job_path']))
        jpath = a.private_output / f'{kind}_part{a.part}.json'
        argv = j['argv']
        argv[argv.index(old_script)] = str(ROOT / repair.SELF)
        argv[argv.index('--protocol') + 1] = str(protocol)
        argv[argv.index('--job-spec') + 1] = str(jpath)
        dest = hf / p['output_HF_relative']
        if kind == 'verify': dest = dest.parent.parent / 'verification/worker/results'
        j.update(id=p['id'] + ('_VERIFY' if kind == 'verify' else ''),
                 protocol_path=str(protocol), protocol_sha256=owner.sha(protocol),
                 output_dir=str(dest.parent), run_json=str(dest / 'RUN.json'))
        original.write_new(jpath, j, True)
        summary.append({'id': j['id'], 'kind': kind, 'protocol': owner.bind(protocol, ROOT),
                        'job_path': str(jpath), 'job_spec_sha256': owner.sha(jpath),
                        'fingerprint': owner.canonical(p)})
    original.write_new(a.private_output / 'BOUND_JOBS.json', summary, True)
    print(json.dumps([{k: v for k, v in j.items() if k != 'job_path'} for j in summary], indent=2))


if __name__ == '__main__': main()
