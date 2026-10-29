"""Prepare one exact local view/package and prove four masters on an SM120 worker.

Draft is not authorization. This CPU-only scope never downloads or loads models.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('_sm120_private_v5', ROOT / 'tools/e0_s2_causal_worker_proof_v5.py')
v5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v5)
import local_python_package_mirror as package
native, mirror = v5.native, v5.mirror
ID = 'E0_S2_SM120_BF16_WORKER_ENV_MASTER_CPU_20261029'
OUTPUT = 'ffnds-derived/e0_s2_sm120_bf16_worker_master_cpu_20261029/attempt1/worker/results'
TEMPLATE = 'config/e0_s2_new12_dense_baseline_part0_20261029_v2.json'
ALIAS = Path('/var/tmp/ffnds-xiang/huggingface')
LOCAL = Path('/var/tmp/ffnds-xiang/python-packages/b0_20261028_v2')
SOURCE = ROOT / '.venv/lib/python3.11/site-packages/transformers'
RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'hash_threads': 2, 'wall_seconds': 2100,
             'outer_seconds': 2400, 'new_HF_bytes': 8 << 20, 'new_local_bytes': 64 << 20,
             'automatic_retry': False}
LIMITS = {'full_payload_bytes': mirror.TOTAL, 'models': 4, 'native_headers': 2,
          'package_files': 2302, 'package_bytes_copied': 42053803, 'metadata_bytes_verified': 21454628,
          'cache_aliases': 1, 'model_download_bytes': 0, 'GPU_calls': 0, 'model_calls': 0}
SOURCES = ('tools/e0_s2_sm120_bf16_worker_proof.py', 'tests/test_e0_s2_sm120_bf16_worker_proof.py',
           'tools/e0_s2_causal_worker_proof_v5.py', 'tools/raw_bf16_worker_mirror_proof.py',
           'tools/e0_original_bf16_videos.py', 'tools/local_python_package_mirror.py',
           'tools/worker_identity.py', TEMPLATE, mirror.PARENT, '.runtime/model_view.json')
v5.ID = mirror.ID = ID
mirror.RESOURCES, mirror.LIMITS = RESOURCES, LIMITS
v5.RESOURCES = {**RESOURCES, 'wall_seconds': 1800}


def draft():
    template = native.read(ROOT / TEMPLATE)
    plan = mirror.draft()
    native.need(plan['models'] == template['models'], 'Original and template model identity differs')
    plan.update(id=ID, output_HF_relative=OUTPUT, resources=RESOURCES, limits=LIMITS,
                model_view_root=template['model_view_root'], model_view_metadata=template['model_view_metadata'],
                local_package=template['local_transformers'], local_root=str(LOCAL), alias=str(ALIAS),
                package_source=str(SOURCE), sha_wall_seconds=1800, environment_wall_seconds=300,
                repo_sources=[native.binding(ROOT / p, ROOT) for p in sorted(set(SOURCES) | native.TOOLS)],
                execution_identity='Fresh local view and package; four complete hashes on actual worker, no model/GPU/download')
    return plan


def validate(plan):
    expected = draft()
    expected['status'] = plan.get('status'); expected['private_worker_config_sha256'] = plan.get('private_worker_config_sha256')
    native.need(plan == expected and plan['status'] in ('DRAFT_NOT_REGISTERED', 'FROZEN_REGISTERED'), 'Exact SM120 scope differs')
    value = plan['private_worker_config_sha256']
    native.need(value is None or isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef'), 'Worker SHA format')
    if plan['status'] == 'FROZEN_REGISTERED': native.need(value is not None, 'Worker not bound')


def strict_package(root, rows):
    root = Path(root)
    native.need(root.is_dir() and not root.is_symlink(), 'Package root must be a directory')
    found = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        native.need(all(not (base / n).is_symlink() for n in dirs), 'Package directory symlink')
        for name in files:
            p = base / name
            native.need(not p.is_symlink() and p.is_file(), 'Package nonregular or symlink')
            found.append(p.relative_to(root).as_posix())
    native.need(sorted(found) == sorted(x['path'] for x in rows), 'Package path inventory differs')
    for row in rows: native.bound(root, row)


def prepare_view(view, source_view):
    view, source_view = Path(view), Path(source_view)
    native.need(source_view.is_dir() and not view.exists() and not view.is_symlink(), 'Existing view or missing HF view')
    view.parent.mkdir(parents=True, exist_ok=True)
    view.symlink_to(source_view.resolve(), target_is_directory=True)
    native.need(view.is_symlink() and view.resolve() == source_view.resolve(), 'View link differs')


def environment(plan, hf, out):
    info = plan['local_package']; inventory = native.read(native.bound(hf, info['inventory'])); rows = inventory['files']
    native.need(len(rows) == LIMITS['package_files'] and sum(r['bytes'] for r in rows) == LIMITS['package_bytes_copied']
                and native.canonical(rows) == info['inventory_sha256'], 'Frozen package inventory differs')
    view = Path(plan['model_view_root']); source_view = Path(hf) / view.relative_to(ALIAS)
    native.need(not (LOCAL/'transformers').exists() and not (LOCAL/'transformers').is_symlink(), 'Local package must be absent')
    report = {'id': ID, 'status': 'STARTED_ENVIRONMENT', 'view': str(view), 'source_view': str(source_view),
              'predebited_cache_aliases': 1, 'completed_cache_aliases': 0,
              'predebited_package_bytes': LIMITS['package_bytes_copied'], 'copied_package_bytes': 0,
              'GPU_calls': 0, 'model_calls': 0, 'model_download_bytes': 0}
    native.write(out/'ENVIRONMENT.json', report)
    try:
        prepare_view(view, source_view); report['completed_cache_aliases'] = 1
        native.write(out/'ENVIRONMENT.json', report)
        native.model_metadata(plan)
        report['copied_package_bytes'] = None  # A failed copy has no measured partial byte count.
        native.write(out/'ENVIRONMENT.json', report)
        result = package.build(SOURCE, LOCAL/'transformers', info['inventory_sha256'], info['version'])
        native.need(result['files'] == rows, 'Copied package differs from bound inventory')
        strict_package(LOCAL/'transformers', rows)
        report.update(status='PASS_BYTE_IDENTICAL_LOCAL_PACKAGE_AND_HF_VIEW', copied_package_bytes=sum(x['bytes'] for x in rows),
                      package_files=len(rows), package_inventory_sha256=info['inventory_sha256'],
                      metadata_files=7, metadata_bytes=LIMITS['metadata_bytes_verified'])
    except BaseException as e:
        report.update(status='FAILED_ENVIRONMENT_NO_RETRY', error_type=type(e).__name__, error=str(e)); raise
    finally: native.write(out/'ENVIRONMENT.json', report)
    return report


def verify_environment(plan, hf, env):
    info = plan['local_package']; rows = native.read(native.bound(hf, info['inventory']))['files']
    view = Path(plan['model_view_root']); source = Path(hf) / view.relative_to(ALIAS)
    native.need(view.is_symlink() and view.resolve() == source.resolve() and
                env.get('id') == ID and env.get('status') == 'PASS_BYTE_IDENTICAL_LOCAL_PACKAGE_AND_HF_VIEW'
                and env.get('predebited_cache_aliases') == env.get('completed_cache_aliases') == 1
                and env.get('predebited_package_bytes') == env.get('copied_package_bytes') == LIMITS['package_bytes_copied']
                and env.get('package_files') == len(rows) == LIMITS['package_files']
                and env.get('package_inventory_sha256') == native.canonical(rows) == info['inventory_sha256']
                and env.get('metadata_files') == 7 and env.get('metadata_bytes') == LIMITS['metadata_bytes_verified']
                and all(env.get(k) == 0 for k in ('GPU_calls', 'model_calls', 'model_download_bytes')), 'Environment proof differs')
    native.model_metadata(plan); strict_package(LOCAL/'transformers', rows)


def validate_copy_accounting(run):
    native.need(run.get('copied_bytes_scope') == 'SHA_STAGE_ONLY'
                and run.get('sha_stage_copied_bytes') == run.get('copied_bytes') == 0
                and run.get('environment_copied_package_bytes') == run.get('scope_copied_package_bytes')
                == LIMITS['package_bytes_copied'], 'Scope copy accounting differs')


def require_closed_proof(plan, hf):
    proof = plan['execution_raw_master_CPU_proof']; protocol = native.read(native.bound(ROOT, proof['protocol']))
    validate(protocol)
    native.need(protocol['status'] == 'FROZEN_REGISTERED' and protocol['models'] == plan['models']
                and protocol['model_view_root'] == plan['model_view_root']
                and protocol['model_view_metadata'] == plan['model_view_metadata']
                and protocol['local_package'] == plan['local_transformers']
                and protocol['private_worker_config_sha256'] == plan['execution_private_worker_config_sha256'], 'Worker/model identity differs')
    run = native.read(native.bound(hf, proof['RUN']))
    receipt = native.read(native.bound(hf, proof['launch_exit']))
    native.need(receipt.get('id') == ID and receipt.get('protocol_sha256') == proof['protocol']['sha256']
                and receipt.get('run_sha256') == proof['RUN']['sha256'] and receipt.get('process_exited') is True
                and type(receipt.get('child_returncode')) is int and receipt['child_returncode'] == 0
                and receipt.get('timeout') is False and receipt.get('automatic_retry') is False, 'Actual exit0 required')
    mirror.require_worker(); native.need(native.sha(mirror.private_config_path()) == plan['execution_private_worker_config_sha256'], 'Different execution worker')
    v5.verify_report(plan, proof['protocol'], run)
    validate_copy_accounting(run)
    native.need(set(run['native_headers']) == {'high', 'low'} and
                all(type(h.get('header_bytes')) is int and 0 < h['header_bytes'] <= 4 << 20
                    and h.get('tensor_count') == 1095 for h in run['native_headers'].values()), 'Native header closure differs')
    native.need(0 <= run.get('total_wall_seconds', -1) <= RESOURCES['wall_seconds'], 'Total wall differs')
    verify_environment(protocol, hf, native.read(native.bound(hf, run['environment'])))
    return proof


def run(args):
    start = time.monotonic(); plan = native.read(args.protocol); validate(plan); digest = native.sha(args.protocol)
    scope = mirror.authorize(plan, digest)
    native.need(plan['status'] == 'FROZEN_REGISTERED' and scope.get('protocol') == str(args.protocol.resolve().relative_to(ROOT))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'numpy', 'torch'} & set(sys.modules)
                and sys.dont_write_bytecode and len(os.sched_getaffinity(0)) <= 4, 'Exact CPU/-B scope required')
    mirror.require_worker(); native.need(native.sha(mirror.private_config_path()) == plan['private_worker_config_sha256'], 'Worker identity differs')
    for row in plan['repo_sources']: native.bound(ROOT, row)
    hf = args.hf_root.resolve(); out = (hf / OUTPUT).resolve()
    native.need(out.is_relative_to(hf) and not out.exists(), 'Fresh confined output required'); out.mkdir(parents=True)
    report = {'id': ID, 'protocol_sha256': digest, 'status': 'STARTED_ENVIRONMENT_NO_RETRY', 'automatic_retry': False,
              'GPU_calls': 0, 'model_calls': 0, 'actual_full_payload_read_bytes': 0}
    native.write(out/'RUN.json', report)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Environment 300s budget exhausted'))); signal.alarm(300)
    try:
        environment(plan, hf, out); signal.alarm(0)
        report = v5.run_proof(plan, out, digest)
        native.need(time.monotonic()-start <= RESOURCES['wall_seconds'], 'Total wall exceeded')
        report['environment'] = native.binding(out/'ENVIRONMENT.json', hf)
        report.update(copied_bytes_scope='SHA_STAGE_ONLY', sha_stage_copied_bytes=report['copied_bytes'],
                      environment_copied_package_bytes=LIMITS['package_bytes_copied'],
                      scope_copied_package_bytes=LIMITS['package_bytes_copied'])
        report['total_wall_seconds'] = time.monotonic()-start
        native.need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink()) <= RESOURCES['new_HF_bytes'], 'HF output budget exceeded')
    except BaseException as e:
        if (out/'RUN.json').exists(): report = native.read(out/'RUN.json')
        report.update(status='FAILED_SM120_PROOF_PRESERVED_NO_RETRY', error_type=type(e).__name__, error=str(e), total_wall_seconds=time.monotonic()-start)
        raise
    finally:
        signal.alarm(0); native.write(out/'RUN.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('draft','run'))
    p.add_argument('--protocol', type=Path, required=True); p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    args = p.parse_args()
    if args.action == 'draft':
        native.need(not args.protocol.exists(), 'Fresh draft required'); args.protocol.parent.mkdir(parents=True, exist_ok=True); native.write(args.protocol, draft())
    else: run(args)

if __name__ == '__main__': main()
