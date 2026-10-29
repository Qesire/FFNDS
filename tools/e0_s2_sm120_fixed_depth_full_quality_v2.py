"""Immutable V2 launch repair of the SM120 four-arm eight-axis producer.

V1 shard0 taskset failed before Python startup. V2 retains all V1 metric code,
limits and resources, adds a fresh identity/output and explicit CPU affinity.
Draft is not authorization; no failed scope is retried or overwritten.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('_sm120_full_quality_v1_for_v2', ROOT / 'tools/e0_s2_sm120_fixed_depth_full_quality.py')
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
base = q.base
need = base.require
OLD_DRAFT, OLD_VALIDATE, OLD_BINDINGS, OLD_CPU_EXPECTED = q.draft, q.validate, q.all_bindings, q.cpu_expected
SELF = 'tools/e0_s2_sm120_fixed_depth_full_quality_v2.py'
TEST = 'tests/test_e0_s2_sm120_fixed_depth_full_quality_v2.py'
V1_CONFIG = 'config/e0_s2_sm120_fixed_depth_full_quality_shard0_20261029_v1.json'
V1_CONFIG_SHA = '786e2dcaec345898977ddfd58573fd3530125791470242c81502db2e04ffc61b'
V1_EXIT = 'ffnds-derived/e0_s2_sm120_fixed_depth_full_quality_20261029/shard0/cpu/attempt1/worker/EXIT_RECEIPT.json'
V1_STDERR = 'ffnds-derived/e0_s2_sm120_fixed_depth_full_quality_20261029/shard0/cpu/attempt1/worker/child.stderr.log'
EXIT_SHA = '46bd810c41fa3c61aedca525b88bf11d8b1ff3531b5759c9f249ef8c4864a8e3'
STDERR_SHA = '91799b6be9c7c469adf4b3d0b7f2d26ca790e26c6559c76cd19fa1d56b654ff7'
EXTRA_PATHS = {SELF, TEST, V1_CONFIG}
REPAIR = {'cause':'V1_SHARD0_CPU_TASKSET_24_25_OUTSIDE_TARGET_0_TO_23',
          'effect':'TASKSET_CHILD_EXIT1_BEFORE_PYTHON_NO_RUN_NO_MODEL_NO_GPU',
          'automatic_retry':False, 'prior_failed_evidence_preserved':True}


def identity(shard):
    need(type(shard) is int and shard in range(4), 'Invalid shard')
    return 'E0_S2_SM120_FIXED_DEPTH_EIGHT_AXIS_20261029_SHARD{}_V2'.format(shard)


def output(shard):
    return 'ffnds-derived/e0_s2_sm120_fixed_depth_full_quality_v2_20261029/shard{}/attempt1/worker/results'.format(shard)


def cpus(shard):
    identity(shard)
    return [16 + 2 * shard, 17 + 2 * shard]


def verify_failure(plan, hf):
    proof = plan['v1_launch_failure']
    need(proof['protocol']['path'] == V1_CONFIG and proof['protocol']['sha256'] == V1_CONFIG_SHA
         and proof['exit']['path'] == V1_EXIT and proof['exit']['sha256'] == EXIT_SHA
         and proof['stderr']['path'] == V1_STDERR and proof['stderr']['sha256'] == STDERR_SHA,
         'Exact V1 launch failure binding differs')
    protocol = base.read_bound(ROOT, proof['protocol']); receipt = base.read_bound(hf, proof['exit'])
    stderr = base.bound(hf, proof['stderr']).read_bytes()
    need(protocol['id'] == 'E0_S2_SM120_FIXED_DEPTH_EIGHT_AXIS_20261029_SHARD0'
         and protocol['status'] == 'FROZEN_REGISTERED'
         and receipt['id'] == protocol['id'] and receipt['protocol_sha256'] == V1_CONFIG_SHA
         and receipt['process_exited'] is True and type(receipt['child_returncode']) is int
         and receipt['child_returncode'] == 1 and receipt['run_sha256'] is None
         and receipt['timeout'] is False and receipt['automatic_retry'] is False
         and 0 <= receipt['outer_seconds'] < 1 and stderr.startswith(b'taskset:'),
         'V1 taskset failure evidence differs')
    return proof


def draft(hf, shard):
    plan = OLD_DRAFT(hf, shard)
    proof = {'protocol':base.binding(ROOT / V1_CONFIG, ROOT), 'exit':base.binding(hf / V1_EXIT, hf),
             'stderr':base.binding(hf / V1_STDERR, hf)}
    plan['v1_launch_failure'] = proof; verify_failure(plan, hf)
    paths = {b['path'] for b in plan['repo_sources']} | EXTRA_PATHS
    plan.update(id=identity(shard), output_HF_relative=output(shard), cpu_affinity=cpus(shard),
                launch_repair=copy.deepcopy(REPAIR), repo_sources=[base.binding(ROOT / p, ROOT) for p in sorted(paths)])
    return plan


def validate(plan, frozen=False):
    shard = plan['shard']
    need(plan.get('id') == identity(shard) and plan.get('cpu_affinity') == cpus(shard)
         and plan.get('output_HF_relative') == output(shard) and plan.get('launch_repair') == REPAIR,
         'V2 identity, CPU affinity, output or repair differs')
    source_paths = [b['path'] for b in plan['repo_sources']]
    need(len(source_paths) == len(set(source_paths)) and EXTRA_PATHS <= set(source_paths), 'V2 source closure incomplete')
    normalized = copy.deepcopy(plan)
    normalized['output_HF_relative'] = 'ffnds-derived/e0_s2_sm120_fixed_depth_full_quality_20261029/shard{}/attempt1/worker/results'.format(shard)
    normalized['repo_sources'] = [b for b in normalized['repo_sources'] if b['path'] not in EXTRA_PATHS]
    OLD_VALIDATE(normalized, frozen)
    need(plan['v1_launch_failure']['protocol']['path'] == V1_CONFIG
         and plan['v1_launch_failure']['protocol']['sha256'] == V1_CONFIG_SHA
         and plan['v1_launch_failure']['exit']['path'] == V1_EXIT
         and plan['v1_launch_failure']['exit']['sha256'] == EXIT_SHA
         and plan['v1_launch_failure']['stderr']['path'] == V1_STDERR
         and plan['v1_launch_failure']['stderr']['sha256'] == STDERR_SHA, 'V1 failure binding differs')


def all_bindings(plan):
    rows = OLD_BINDINGS(plan)
    rows.extend(('HF', plan['v1_launch_failure'][key]) for key in ('exit', 'stderr'))
    seen = {}
    for root, b in rows:
        key = (root, b['path']); item = {k:b[k] for k in ('path','bytes','sha256')}
        need(key not in seen or seen[key] == item, 'Conflicting V2 binding')
        seen[key] = item
    return [(root,b) for (root,_), b in sorted(seen.items())]


def cpu_expected(plan, digest):
    result = OLD_CPU_EXPECTED(plan, digest)
    result.update(actual_CPU_affinity=cpus(plan['shard']), v1_launch_failure=plan['v1_launch_failure'],
                  launch_repair=REPAIR)
    return result


def check_affinity(plan):
    actual = sorted(os.sched_getaffinity(0))
    need(actual == cpus(plan['shard']), 'Actual CPU affinity differs from fixed V2 shard allocation')
    return actual


def preflight(args):
    plan = base.read(args.protocol); verify_failure(plan, args.hf_root); check_affinity(plan)
    return q.preflight(args)


def run(args):
    plan = base.read(args.protocol); verify_failure(plan, args.hf_root); affinity = check_affinity(plan)
    original = base.write
    def write(path, value):
        if Path(path).name == 'RUN.json' and isinstance(value, dict) and value.get('id') == plan['id']:
            value['actual_CPU_affinity'] = affinity
            value['v1_launch_failure'] = plan['v1_launch_failure']; value['launch_repair'] = REPAIR
        return original(path, value)
    base.write = write
    try: return q.run(args)
    finally: base.write = original


def verify_closed(args):
    plan = base.read(args.protocol); verify_failure(plan, args.hf_root)
    report = base.read(args.run)
    need(report.get('actual_CPU_affinity') == cpus(plan['shard']) and report.get('launch_repair') == REPAIR
         and report.get('v1_launch_failure') == plan['v1_launch_failure'], 'V2 run affinity/launch lineage differs')
    return q.verify_closed(args)


# Isolated imported V1 namespace: numerical implementation and cost constants stay unchanged.
q.identity, q.validate, q.all_bindings, q.cpu_expected = identity, validate, all_bindings, cpu_expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('draft','inspect','preflight','run','verify-closed'))
    p.add_argument('--shard', type=int, choices=range(4)); p.add_argument('--protocol', type=Path)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    p.add_argument('--out', type=Path); p.add_argument('--run', type=Path); p.add_argument('--exit', type=Path)
    args = p.parse_args()
    if args.action == 'draft':
        need(args.shard is not None and args.out is not None and not args.out.exists(), 'Fresh shard draft required')
        args.out.parent.mkdir(parents=True, exist_ok=True); base.write(args.out, draft(args.hf_root, args.shard))
    elif args.action == 'inspect':
        plan = base.read(args.protocol); validate(plan); print(json.dumps({'id':plan['id'], 'cpu_affinity':plan['cpu_affinity'],
             'limits':q.LIMITS, 'resources':q.RESOURCES}, indent=2))
    elif args.action == 'preflight': preflight(args)
    elif args.action == 'run': run(args)
    else: print(json.dumps(verify_closed(args), indent=2))

if __name__ == '__main__': main()
