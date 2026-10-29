"""Scan the entire Git index with one cat-file process; preserve legacy results."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess

from check_public_metadata import inspect_entry, private_identifiers, scan as legacy_scan


def index_entries():
    raw = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], stderr=subprocess.DEVNULL)
    entries = {}
    for record in raw.split(b'\0'):
        if not record:
            continue
        try:
            metadata, path = record.split(b'\t', 1)
            mode, oid, stage = metadata.split(b' ')
        except ValueError:
            raise RuntimeError('Cannot parse Git index') from None
        if not path or stage not in (b'0', b'1', b'2', b'3') or not oid or any(c not in b'0123456789abcdef' for c in oid):
            raise RuntimeError('Invalid Git index entry')
        entries.setdefault(path, []).append((mode, oid, stage))
    result = []
    for path in sorted(entries):
        records = entries[path]
        stage_zero = [entry for entry in records if entry[2] == b'0']
        # Legacy `git show :path` fails on unresolved index entries.
        if len(stage_zero) != 1 or len(records) != 1:
            raise RuntimeError('Unmerged Git index cannot be scanned')
        mode, oid, _ = stage_zero[0]
        if mode not in (b'100644', b'100755', b'120000', b'160000'):
            raise RuntimeError('Unsupported Git index entry')
        result.append((path, mode, oid))
    return result


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(min(1 << 20, size - len(data)))
        if not chunk:
            raise RuntimeError('Incomplete Git object stream')
        data.extend(chunk)
    return bytes(data)


def scan_index():
    forbidden = private_identifiers()
    if not forbidden:
        raise RuntimeError('Configure the private worker identifiers before checking publication')
    entries = index_entries()
    violations = []
    process = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        for raw, mode, oid in entries:
            if mode == b'160000':
                # `git show :path` formats a gitlink commit instead of returning
                # its raw object bytes; preserve that rare legacy behavior.
                result = subprocess.run(['git', 'show', ':' + os.fsdecode(raw)],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                if result.returncode:
                    raise RuntimeError('Unable to inspect staged gitlink')
                contents = result.stdout
            else:
                # One request, then consume its complete response before the
                # next request: neither pipe can fill while waiting on the other.
                process.stdin.write(oid + b'\n')
                process.stdin.flush()
                header = process.stdout.readline()
                parts = header.rstrip(b'\n').split(b' ')
                if len(parts) != 3 or parts[0] != oid or parts[1] != b'blob' or not parts[2].isdigit():
                    raise RuntimeError('Invalid staged object response')
                contents = read_exact(process.stdout, int(parts[2]))
                if process.stdout.read(1) != b'\n':
                    raise RuntimeError('Invalid staged object terminator')
            matches = inspect_entry(os.fsdecode(raw), contents, forbidden)
            if matches:
                violations.append({'path_id_sha256': hashlib.sha256(raw).hexdigest(), 'private_matches': matches})
    finally:
        process.stdin.close()
        process.stdout.close()
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
            raise RuntimeError('Git object reader did not terminate') from None
    if return_code:
        raise RuntimeError('Git object reader failed')
    return {'status': 'PASS_PUBLIC_METADATA_SCAN' if not violations else 'REJECTED_PRIVATE_METADATA',
            'scope': 'git index', 'files_checked': len(entries), 'private_rules': len(forbidden),
            'violations': violations}


def scan(staged=False):
    return scan_index() if staged else legacy_scan(staged=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = scan(args.staged)
    value = json.dumps(result, indent=2) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(value)
    print(value, end='')
    raise SystemExit(bool(result['violations']))


if __name__ == '__main__':
    main()
