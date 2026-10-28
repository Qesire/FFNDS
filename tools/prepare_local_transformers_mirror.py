"""Registered CPU-only extraction and import check for a frozen local package mirror."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time

from local_python_package_mirror import canonical_sha, extract, verify_origin

ROOT = Path(__file__).resolve().parents[1]
ID = 'PREPARE_LOCAL_TRANSFORMERS_MIRROR_20261028'
LIMITS = {'cpu_threads': 2, 'wall_seconds': 600, 'outer_wall_seconds': 900,
          'local_bytes': 256 << 20, 'hf_bytes': 8 << 20, 'gpu': 0, 'model_calls': 0}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(root, item):
    p = (root / item['path']).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file() or p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
        raise ValueError('Frozen asset mismatch')
    return p


def authorize(plan, digest, active):
    if (plan.get('id') != ID or plan.get('assets_finalized') is not True or plan.get('limits') != LIMITS
            or plan.get('transformers_version') != '5.6.2' or plan.get('automatic_retry') is not False):
        raise ValueError('Unfrozen scope')
    user = active.get('user_continuation_authorization_20261027', {})
    if user.get('id') != 'USER_RESUME_OFFLINE_IFC_20261027' or user.get('status') != 'ACTIVE_CURRENT_USER_DIRECTED_SESSION':
        raise ValueError('Current user authority inactive')
    scopes = [x for x in active.get('experiment_authority', {}).get('current_user_authorized_scopes', []) if x.get('id') == ID]
    if len(scopes) != 1 or scopes[0].get('protocol_sha256') != digest or scopes[0].get('execution_status') not in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE'):
        raise ValueError('Exact unique scope not authorized')


def write(path, obj):
    data = json.dumps(obj, indent=2, allow_nan=False) + '\n'
    if len(data.encode()) > LIMITS['hf_bytes']:
        raise ValueError('Receipt exceeds HF budget')
    tmp = path.with_suffix('.tmp'); tmp.write_text(data); os.replace(tmp, path)


def run(args):
    plan = json.loads(args.protocol.read_text()); protocol_sha = sha(args.protocol); hf = args.hf_root.resolve()
    authorize(plan, protocol_sha, json.loads((ROOT / 'research/ACTIVE_RESEARCH.json').read_text()))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(x == 'torch' or x.startswith('torch.') or x == 'transformers' or x.startswith('transformers.') for x in sys.modules):
        raise ValueError('Fresh CPU-only process required')
    from worker_identity import private_config_path, require_worker
    require_worker()
    if sha(private_config_path()) not in plan['worker_profile_sha256']:
        raise ValueError('Worker profile mismatch')
    if importlib.metadata.version('transformers') != plan['transformers_version']:
        raise ValueError('Distribution version mismatch')
    out = (hf / plan['output_HF_relative']).resolve()
    if out == hf or not out.is_relative_to(hf): raise ValueError('Invalid output root')
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out / 'ATTEMPT.json', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as f: json.dump({'id': ID, 'protocol_sha256': protocol_sha}, f)
    report = {'id': ID, 'protocol_sha256': protocol_sha, 'status': 'STARTED', 'GPU_model_calls': 0}
    started = time.monotonic()
    def timeout(*_): raise TimeoutError('CPU mirror wall limit')
    old = signal.signal(signal.SIGALRM, timeout); signal.alarm(LIMITS['wall_seconds'])
    try:
        required = {'tools/prepare_local_transformers_mirror.py', 'tools/local_python_package_mirror.py'}
        if not required <= {item['path'] for item in plan['repo_sources']}:
            raise ValueError('Required mirror sources not bound')
        for item in plan['repo_sources']: bind(ROOT, item)
        manifest = json.loads(bind(hf, plan['inventory']).read_text()); rows = manifest['files']
        if canonical_sha(rows) != plan['inventory_sha256']: raise ValueError('Inventory SHA mismatch')
        if sha(args.archive) != plan['archive_sha256'] or args.archive.stat().st_size != plan['archive_bytes']:
            raise ValueError('Archive identity mismatch')
        if args.archive.stat().st_size + sum(row['bytes'] for row in rows) > LIMITS['local_bytes']:
            raise ValueError('Archive and extracted bytes exceed local budget')
        local = args.local_root.resolve()
        if hashlib.sha256(str(local).encode()).hexdigest() != plan['local_root_path_sha256']:
            raise ValueError('Local root identity mismatch')
        try: os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
        except AttributeError: pass
        extracted = extract(args.archive, local / 'transformers', rows, plan['inventory_sha256'], plan['transformers_version'], max_bytes=LIMITS['local_bytes'])
        receipt = {'inventory_sha256': plan['inventory_sha256'], 'archive_sha256': extracted['archive_sha256'], 'files': rows}
        write(out / 'MIRROR_INVENTORY.json', receipt)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(local))
        import transformers
        from transformers import AutoTokenizer
        if transformers.__version__ != plan['transformers_version'] or not isinstance(AutoTokenizer, type):
            raise ValueError('Actual package version or tokenizer class differs')
        origins = {}
        for name, module in sorted(sys.modules.items()):
            if name == 'transformers' or name.startswith('transformers.'):
                path = getattr(module, '__file__', None)
                if path is not None:
                    origins[name] = verify_origin(path, local / 'transformers', rows)
        if 'transformers' not in origins: raise ValueError('Package origin missing')
        import torch
        if torch.cuda.is_initialized() or torch.cuda.device_count() != 0:
            raise ValueError('CUDA not isolated')
        report.update(status='PASS_LOCAL_MIRROR_IMPORT_NO_MODEL', imported_origins=origins,
                      inventory_sha256=extracted['inventory_sha256'], archive_sha256=extracted['archive_sha256'],
                      mirror_file_count=len(rows), cuda_initialized=False, cuda_device_count=0,
                      tokenizer_class_resolved=True, transformers_version=transformers.__version__)
        if sum(x.stat().st_size for x in out.rglob('*') if x.is_file()) > LIMITS['hf_bytes']:
            raise ValueError('Total HF output exceeds budget')
    except BaseException as e:
        report.update(status='FAILED_NO_RETRY', error_type=type(e).__name__, error='See private launcher stderr')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        try: write(out / 'RUN.json', report)
        finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True, type=Path); p.add_argument('--hf-root', required=True, type=Path)
    p.add_argument('--archive', required=True, type=Path); p.add_argument('--local-root', required=True, type=Path)
    run(p.parse_args())
