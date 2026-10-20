"""Check pinned source hashes, then optionally overlay audited text deltas.

Default is a read-only plan. Never installs dependencies or compiles kernels.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repos-root', type=Path, required=True)
    ap.add_argument('--repo', action='append', help='Repository directory; repeat to select multiple.')
    ap.add_argument('--apply', action='store_true', help='Write deltas after every selected hash has passed.')
    ap.add_argument('--verify-only', action='store_true', help='Require the completed handoff source hashes.')
    args = ap.parse_args()
    if args.apply and args.verify_only:
        ap.error('--apply and --verify-only cannot be combined')
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    names = args.repo or ['LightX2V', 'Wan2.2-I2V-A14B-W4A4', 'HiF4_Sp']
    root = args.repos_root.resolve(strict=True)
    plans, errors = [], []
    checked = 0
    for name in names:
        if name not in manifest['repositories']:
            ap.error('Unknown repository: ' + name)
        repo = root / name
        if not repo.is_dir() or repo.is_symlink():
            errors.append(str(repo) + ': missing directory or symlink root')
            continue
        for rel, info in manifest['repositories'][name]['files'].items():
            target = repo / rel
            if not target.resolve().is_relative_to(repo.resolve()):
                errors.append(str(target) + ': path escapes repository')
                continue
            if any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
                errors.append(str(target) + ': symlink in path')
                continue
            actual = digest(target)
            checked += 1
            if actual == info['handoff_sha256']:
                continue
            if args.verify_only or info['status'] == 'unchanged':
                errors.append(str(target) + ': expected handoff hash ' + info['handoff_sha256'])
                continue
            if target.exists() and not target.is_file():
                errors.append(str(target) + ': target is not a file')
                continue
            allowed = actual == info['upstream_sha256'] if info['status'] == 'modified' else actual is None
            if not allowed:
                errors.append(str(target) + ': differs from both pinned upstream and handoff')
                continue
            source = here / 'source_deltas' / name / 'files' / rel
            if digest(source) != info['handoff_sha256']:
                errors.append(str(source) + ': delta payload hash mismatch')
                continue
            plans.append((source, target))
    result = {'checked_files': checked, 'planned_writes': len(plans), 'errors': errors,
              'mode': 'verify' if args.verify_only else ('apply' if args.apply else 'read_only_plan')}
    if errors:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    if args.apply:
        for source, target in plans:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        result['written_files'] = len(plans)
    result['status'] = 'PASS'
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
