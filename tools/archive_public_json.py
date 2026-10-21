"""Publish an explicitly redacted JSON copy without altering original evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from check_public_metadata import private_identifiers


def numeric_values(value):
    if isinstance(value, dict):
        return [v for item in value.values() for v in numeric_values(item)]
    if isinstance(value, list):
        return [v for item in value for v in numeric_values(item)]
    return [(type(value).__name__, value)] if type(value) in (int, float, bool) else []


def archive(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.with_suffix(destination.suffix + '.redaction.json').exists():
        raise RuntimeError('Public evidence copies are immutable; choose a new destination')
    forbidden = private_identifiers()
    if not forbidden:
        raise RuntimeError('Private identifier configuration is required for archive redaction')
    original = source.read_bytes()
    redacted = original.decode('utf8')
    for host in sorted(forbidden, key=len, reverse=True):
        redacted = re.sub(re.escape(host.decode()), lambda m: 'WORKER' if m[0].isupper()
                          else 'Worker' if m[0][0].isupper() else 'worker', redacted, flags=re.I)
    public = redacted.encode('utf8')
    if numeric_values(json.loads(original)) != numeric_values(json.loads(public)):
        raise RuntimeError('Identifier redaction changed numerical fields')
    if any(host in public.lower() or host in str(destination).lower().encode() for host in forbidden):
        raise RuntimeError('Private identifier remains in public copy or destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(public)
    receipt = {'status': 'PASS_PUBLIC_JSON_REDACTION_NUMERIC_VALUES_UNCHANGED',
        'original_sha256': hashlib.sha256(original).hexdigest(), 'public_sha256': hashlib.sha256(public).hexdigest(),
        'original_path_sha256': hashlib.sha256(str(source.resolve()).encode()).hexdigest(),
        'content_redacted': original != public, 'numeric_values_exact': True,
        'identity_semantics': 'Embedded hashes/fingerprints retain original private byte identities; this is a public copy, not a new native execution.',
        'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination.with_suffix(destination.suffix + '.redaction.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(archive(args.source, args.destination), indent=2))
