"""Copy a frozen Python package to a local directory without importing it."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Package root must be a real directory')
    rows = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in dirs:
            if (base / name).is_symlink():
                raise ValueError('Directory symlink forbidden')
        dirs[:] = sorted(name for name in dirs if name != '__pycache__')
        for name in sorted(files):
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Nonregular file or symlink forbidden')
            if path.suffix in ('.pyc', '.pyo'):
                continue
            rows.append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    return sorted(rows, key=lambda x: x['path'])


def canonical_sha(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build(source, destination, expected_inventory_sha, expected_version, package='transformers'):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if source.is_symlink():
        raise ValueError('Package root symlink forbidden')
    source = source.resolve()
    if source.name != package or destination.name != package or destination.exists() or destination.is_symlink():
        raise ValueError('Expected matching package names and absent destination')
    if destination.parent.resolve().is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Source and destination overlap')
    if importlib.metadata.version(package) != expected_version:
        raise ValueError('Installed distribution version mismatch')
    rows = inventory(source)
    if canonical_sha(rows) != expected_inventory_sha:
        raise ValueError('Source inventory differs from frozen identity')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + package + '-', dir=destination.parent))
    try:
        for row in rows:
            src, dst = source / row['path'], staging / row['path']
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst, follow_symlinks=False)
        if inventory(staging) != rows or inventory(source) != rows:
            raise ValueError('Source or copied bytes changed during mirror')
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Destination appeared during copy')
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'package': package, 'version': expected_version, 'inventory_sha256': expected_inventory_sha,
            'files': rows, 'destination': str(destination), 'source_and_copy_verified': True}


def verify_origin(module_file, destination, rows):
    origin, root = Path(module_file).resolve(strict=True), Path(destination).resolve(strict=True)
    if not origin.is_relative_to(root) or not origin.is_file():
        raise ValueError('Actual module origin is outside mirrored package')
    relative = origin.relative_to(root).as_posix()
    expected = next((row for row in rows if row['path'] == relative), None)
    if expected is None or origin.stat().st_size != expected['bytes'] or digest(origin) != expected['sha256']:
        raise ValueError('Actual module bytes are not in the frozen inventory')
    return relative


def archive(source, archive_path, expected_inventory_sha, expected_version, package='transformers'):
    source, archive_path = Path(source), Path(archive_path)
    if source.is_symlink() or source.name != package or importlib.metadata.version(package) != expected_version:
        raise ValueError('Source package/version mismatch')
    rows = inventory(source)
    if canonical_sha(rows) != expected_inventory_sha:
        raise ValueError('Source inventory differs')
    with archive_path.open('xb') as stream, tarfile.open(fileobj=stream, mode='w') as tar:
        for row in rows:
            path = source / row['path']
            info = tarfile.TarInfo(row['path']); info.size = row['bytes']; info.mode = 0o644
            with path.open('rb') as f:
                tar.addfile(info, f)
    if inventory(source) != rows:
        raise ValueError('Source changed while archiving')
    return rows


def extract(archive_path, destination, rows, expected_inventory_sha, expected_version, package='transformers', max_bytes=256 << 20):
    destination = Path(destination).absolute()
    if destination.name != package or destination.exists() or destination.is_symlink():
        raise ValueError('Absent package destination required')
    if importlib.metadata.version(package) != expected_version or canonical_sha(rows) != expected_inventory_sha:
        raise ValueError('Version or frozen inventory mismatch')
    expected = {}
    for row in rows:
        name = row['path']; p = Path(name)
        if (not name or p.is_absolute() or '..' in p.parts or p.as_posix() != name or name in expected
                or type(row['bytes']) is not int or row['bytes'] < 0):
            raise ValueError('Invalid inventory member')
        expected[name] = row
    if sum(x['bytes'] for x in rows) > max_bytes:
        raise ValueError('Inventory exceeds byte budget')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + package + '-', dir=destination.parent))
    try:
        seen = set()
        with tarfile.open(archive_path, mode='r:') as tar:
            for member in tar:
                name = member.name
                if (name in seen or name not in expected or not member.isfile() or member.size != expected[name]['bytes']):
                    raise ValueError('Unexpected, duplicate or invalid archive member')
                seen.add(name)
                dst = staging / name; dst.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dst.open('xb') as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                if digest(dst) != expected[name]['sha256']:
                    raise ValueError('Archive member bytes differ')
        if seen != set(expected) or inventory(staging) != rows:
            raise ValueError('Archive incomplete or inventory order differs')
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Destination appeared during extraction')
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'package': package, 'version': expected_version, 'inventory_sha256': expected_inventory_sha,
            'files': rows, 'destination': str(destination), 'archive_sha256': digest(Path(archive_path))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--inventory-sha256', required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    result = build(a.source, a.destination, a.inventory_sha256, a.version)
    with a.manifest.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')


if __name__ == '__main__':
    main()
