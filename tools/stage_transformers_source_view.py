"""Mirror exact installed Transformers source bytes into worker-local HF cache.

This does not install, upgrade, patch, or activate a dependency. A caller must
explicitly select the mirror and prove its native controls before using it.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import shutil
import time

from worker_identity import require_worker


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived/python_source_views'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require_worker()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Source mirroring is CPU only')
    os.umask(0o077)
    package = distribution('transformers')
    record = package.read_text('RECORD')
    if record is None:
        raise RuntimeError('Installed package has no wheel RECORD identity')
    identity = sha(record.encode())
    files = sorted((p for p in package.files if str(p).startswith('transformers/')
                    and '__pycache__' not in p.parts and p.suffix != '.pyc'), key=str)
    if not files or any('..' in p.parts or p.hash is None or p.hash.mode != 'sha256' or p.size is None for p in files):
        raise RuntimeError('Every mirrored package file requires its original wheel SHA256 and size')
    args.root.mkdir(parents=True, exist_ok=True)
    source_root = Path(package.locate_file('transformers')).resolve()
    if args.root.stat().st_dev == source_root.stat().st_dev:
        raise RuntimeError('Source view must use a separate worker-local filesystem')
    required = sum(p.size for p in files)
    if shutil.disk_usage(args.root).free < (80 << 30) + required + (1 << 20):
        raise RuntimeError('Preserve80GiB worker reserve while copying package source')
    target = args.root / f'transformers-{package.version}-{identity[:16]}'
    partial = target.with_name(target.name + '.partial')
    if target.exists() or partial.exists() or args.report.exists():
        raise RuntimeError('Preserve an existing source view or partial build; choose a new root')
    partial.mkdir()
    started = time.perf_counter()

    def copy(entry):
        import base64
        source = Path(package.locate_file(entry))
        if source.is_symlink():
            raise RuntimeError('Mirroring package symlinks is unsupported')
        data = source.read_bytes()
        expected = base64.urlsafe_b64decode(entry.hash.value + '=' * (-len(entry.hash.value) % 4)).hex()
        if len(data) != entry.size or sha(data) != expected:
            raise RuntimeError('Installed source differs from its original wheel RECORD')
        destination = partial / str(entry)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        if sha(destination.read_bytes()) != expected:
            raise RuntimeError('Worker-local package source copy differs')
        return {'relative_path': str(entry), 'sha256': expected, 'bytes': len(data)}

    meta = {'status': 'COPYING_EXACT_SOURCE_VIEW_ONLY', 'package': 'transformers', 'version': package.version,
            'wheel_RECORD_sha256': identity, 'files_expected': len(files), 'source_bytes': required,
            'dependency_activated': False, 'quality_or_speed_acceptance': False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            inventory = list(pool.map(copy, files))
        if shutil.disk_usage(args.root).free < 80 << 30:
            raise RuntimeError('Worker reserve fell below80GiB during source mirror')
        meta.update(status='PASS_EXACT_INSTALLED_SOURCE_VIEW_NOT_ACTIVATED', files_verified=len(inventory),
            view_root=str(target), source_inventory=inventory, source_sha256=sha(Path(__file__).read_bytes()),
            elapsed_seconds=time.perf_counter() - started,
            scope='Exact installed source-byte mirror only; original environment unchanged, native equivalence not yet tested')
        (partial / 'VIEW.json').write_text(json.dumps(meta, indent=2) + '\n')
        partial.rename(target)
    except BaseException as error:
        meta.update(status='FAILED_PRESERVED_PARTIAL_SOURCE_VIEW', error=str(error))
        raise
    finally:
        args.report.write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({k: v for k, v in meta.items() if k != 'source_inventory'}, indent=2))


if __name__ == '__main__':
    main()
