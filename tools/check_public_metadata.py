"""Reject private worker identifiers in publishable paths and file contents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from worker_identity import private_config_path


def private_identifiers():
    identifiers = set()
    config = private_config_path()
    if config.exists():
        private = json.loads(config.read_text())
        for host in [private.get('expected_hostname', ''), private.get('execution_hostname', ''),
                     *private.get('additional_private_hostnames', [])]:
            if host:
                identifiers.update((host.lower(), host.split('.')[0].lower()))
    ssh_config = Path(os.environ.get('FFNDS_SSH_CONFIG', str(config.with_name('ssh_config'))))
    if ssh_config.exists():
        for line in ssh_config.read_text().splitlines():
            fields = shlex.split(line, comments=True)
            if len(fields) == 2 and fields[0].lower() == 'hostname':
                identifiers.add(fields[1].lower())
    return sorted(value.encode() for value in identifiers if value)


def inspect_entry(path, contents, forbidden):
    """Return only counts; never echo a private identifier or leaking path."""
    path_bytes = os.fsencode(path).lower()
    content_bytes = contents.lower()
    return sum(value in path_bytes or value in content_bytes for value in forbidden)


def scan(staged=False):
    forbidden = private_identifiers()
    if not forbidden:
        raise RuntimeError('Configure the private worker identifiers before checking publication')
    arguments = ['git', 'ls-files', '-z']
    if not staged:
        arguments += ['--cached', '--others', '--exclude-standard']
    paths = sorted(set(p for p in subprocess.check_output(arguments).split(b'\0') if p))
    violations = []
    checked = 0
    for raw in paths:
        path = os.fsdecode(raw)
        if staged:
            contents = subprocess.check_output(['git', 'show', ':' + path])
        else:
            file = Path(path)
            if not file.is_file():
                continue
            contents = file.read_bytes()
        checked += 1
        matches = inspect_entry(path, contents, forbidden)
        if matches:
            violations.append({'path_id_sha256': hashlib.sha256(raw).hexdigest(), 'private_matches': matches})
    return {'status': 'PASS_PUBLIC_METADATA_SCAN' if not violations else 'REJECTED_PRIVATE_METADATA',
            'scope': 'git index' if staged else 'publishable worktree',
            'files_checked': checked, 'private_rules': len(forbidden), 'violations': violations}


if __name__ == '__main__':
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
