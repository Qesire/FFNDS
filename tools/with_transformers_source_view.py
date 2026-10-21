"""Run one Python script with an opt-in, exact installed-source Transformers view."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import sys
import time

from worker_identity import require_worker


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    require_worker()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if len(command) < 2 or command[0] != 'python' or not command[1].endswith('.py'):
        parser.error('Supply -- python path/to/script.py [arguments]')
    if args.receipt.exists():
        raise RuntimeError('Preserve prior source-view execution receipts')
    started = time.monotonic()
    view = json.loads(args.manifest.read_text())
    package = distribution('transformers')
    if (view['status'] != 'PASS_EXACT_INSTALLED_SOURCE_VIEW_NOT_ACTIVATED'
            or view['package'] != 'transformers' or view['version'] != package.version
            or view['wheel_RECORD_sha256'] != sha(package.read_text('RECORD').encode())):
        raise RuntimeError('Source view and the installed dependency identity differ')
    root = Path(view['view_root']).resolve()
    if args.manifest.resolve() != root / 'VIEW.json':
        raise RuntimeError('View manifest does not belong to its declared root')
    expected = {}
    for entry in package.files:
        if str(entry).startswith('transformers/') and '__pycache__' not in entry.parts and entry.suffix != '.pyc':
            if entry.hash is None or entry.hash.mode != 'sha256' or '..' in entry.parts:
                raise RuntimeError('Unsupported source identity in installed wheel')
            expected[str(entry)] = {'bytes': entry.size,
                'sha256': base64.urlsafe_b64decode(entry.hash.value + '=' * (-len(entry.hash.value) % 4)).hex()}
    inventory = view['source_inventory']
    actual = {item['relative_path']: {k: item[k] for k in ['bytes', 'sha256']} for item in inventory}
    if actual != expected or len(actual) != len(inventory) or len(actual) != view['files_verified']:
        raise RuntimeError('Source view inventory differs from the original wheel files')

    def verify(item):
        path = root / item['relative_path']
        if path.is_symlink() or path.resolve().parent != (root / item['relative_path']).absolute().parent:
            raise RuntimeError('Source view file resolves through an unexpected link')
        data = path.read_bytes()
        if len(data) != item['bytes'] or sha(data) != item['sha256']:
            raise RuntimeError('Source-view package bytes changed')

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, inventory))
    receipt = {'status': 'SOURCE_VIEW_VERIFIED_CHILD_NOT_STARTED', 'package': 'transformers',
        'version': package.version, 'view_manifest_sha256': sha(args.manifest.read_bytes()),
        'wheel_RECORD_sha256': view['wheel_RECORD_sha256'], 'files_verified': len(inventory),
        'source_inventory_sha256': sha(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()),
        'source_sha256': sha(Path(__file__).read_bytes()), 'command': command,
        'validation_seconds': time.monotonic() - started, 'quality_or_speed_acceptance': False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(root) + (':' + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    bootstrap = '''import json,runpy,sys,time
from pathlib import Path
receipt_path=Path(sys.argv[1]);view=Path(sys.argv[2]);script=sys.argv[3];arguments=sys.argv[4:]
meta=json.loads(receipt_path.read_text());started=time.monotonic()
try:
 import transformers
 module=Path(transformers.__file__).resolve()
 if not module.is_relative_to(view) or transformers.__version__!=meta['version']:
  raise RuntimeError('Child did not load the verified source view')
 meta.update(status='RUNNING_CHILD_WITH_VERIFIED_SOURCE_VIEW',loaded_package_path=str(module))
 receipt_path.write_text(json.dumps(meta,indent=2)+'\\n')
 sys.path.insert(0,str(Path(script).resolve().parent));sys.argv=[script,*arguments]
 runpy.run_path(script,run_name='__main__')
 meta.update(status='PASS_CHILD_WITH_VERIFIED_SOURCE_VIEW',child_elapsed_seconds=time.monotonic()-started)
except BaseException as error:
 meta.update(status='FAILED_CHILD_SOURCE_VIEW_RECEIPT',error=str(error),child_elapsed_seconds=time.monotonic()-started)
 raise
finally:
 receipt_path.write_text(json.dumps(meta,indent=2)+'\\n')
'''
    os.execvpe(sys.executable, [sys.executable, '-c', bootstrap, str(args.receipt.resolve()), str(root),
                              command[1], *command[2:]], environment)


if __name__ == '__main__':
    main()
