"""Additional provenance checks for future native paired receipts.

Historical v1 metrics/gates/receipts remain unchanged. This strict manifest
cross-checks the independent native output inventory and actual model assets.
"""
import argparse
import hashlib
import json
from pathlib import Path

from ffnds_video_metric_gate import inspect as core_inspect
from ffnds_video_metrics import digest


def model_digest(receipt, kind):
    binding = {'checkpoint_sha256': receipt['assets']['checkpoint_sha256'],
               'cache_manifest_sha256': receipt['assets']['cache_manifest_sha256'],
               'prototype_family': 'dense' if kind == 'dense' else 'affine_residual',
               'bank_RUN_sha256': None if kind == 'dense' else receipt['bank_RUN_sha256']}
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _identity_from_actual_receipt(receipt_path, video_path, video_sha):
    """Only two explicitly supported source schemas, never copy report identity."""
    raw = json.loads(receipt_path.read_text())
    if isinstance(raw.get('videos'), dict) and 'pair' in raw:
        matches = [branch for branch, clip in raw['videos'].items()
                   if (receipt_path.parent / clip['path']).resolve() == video_path.resolve()
                   and clip['sha256'] == video_sha]
        if len(matches) != 1:
            raise ValueError('Video not uniquely named in actual paired generation receipt')
        branch = matches[0]
        kind = raw['pair']['modes'][['teacher', 'student'].index(branch)]
        if kind not in ['dense', 'candidate']:
            raise ValueError('Unsupported native operator identity')
        return {'model_digest': model_digest(raw, kind), 'source_id': raw['source_id'], 'seed': raw['seed'], 'kind': kind}
    if raw.get('mode') == 'dense' and isinstance(raw.get('phases'), list):
        matches = [phase for phase in raw['phases']
                   if (receipt_path.parent / phase['video']).resolve() == video_path.resolve()
                   and phase['video_sha256'] == video_sha]
        if len(matches) != 1:
            raise ValueError('Reference absent or ambiguous in original dense generation receipt')
        return {'model_digest': model_digest(raw, 'dense'), 'seed': matches[0]['seed'], 'kind': 'dense'}
    raise ValueError('Unsupported generation receipt schema; strict identity is unverified')


def inspect(strict_path):
    strict_path = Path(strict_path).resolve(); base = strict_path.parent
    strict = json.loads(strict_path.read_text())
    if strict.get('schema') != 'ffnds.paired.video.strict.v1':
        raise ValueError('Wrong strict paired evidence schema')
    core_path = (base / strict['core_bundle_path']).resolve()
    if digest(core_path) != strict.get('core_bundle_sha256'):
        return ['Core bundle digest drift']
    failures = core_inspect(core_path)
    core = json.loads(core_path.read_text()); core_base = core_path.parent
    receipts = strict.get('native_generation_receipts')
    if not isinstance(receipts, list) or not receipts:
        return failures + ['Missing independent native generation inventory']
    expected = set()
    try:
        for item in receipts:
            path = (base / item['path']).resolve()
            if digest(path) != item['sha256']:
                raise ValueError('Native generation receipt digest drift')
            native = json.loads(path.read_text())
            if set(native['videos']) != {'teacher', 'student'} or len(native['pair']['modes']) != 2:
                raise ValueError('Paired generation receipt must enumerate both native clips')
            paths = set()
            for branch, clip in native['videos'].items():
                video = (path.parent / clip['path']).resolve()
                if digest(video) != clip['sha256']:
                    raise ValueError('Native output inventory artifact digest mismatch')
                key = (item['sha256'], str(video))
                if key in expected:
                    raise ValueError('Repeated authoritative generated artifact')
                expected.add(key); paths.add(video)
            if {v.resolve() for v in path.parent.glob('*.mp4')} != paths:
                raise ValueError('Actual native directory contains unregistered or missing clips')
        entries = core['videos']
        actual = [(entry['run_sha256'], str((core_base / entry['video_path']).resolve())) for entry in entries]
        if set(actual) != expected or len(actual) != len(expected):
            failures.append('Core bundle omits, duplicates or invents an actual generated clip')
        for entry in entries:
            report = json.loads((core_base / entry['metric_report_path']).read_text())
            if not report.get('reference') or not report.get('reference_identity'):
                failures.append('Every paired native clip requires an actual dense reference identity')
            actual_identity = _identity_from_actual_receipt(
                (core_base / entry['run_receipt_path']).resolve(),
                (core_base / entry['video_path']).resolve(), entry['video_sha256'])
            for key in ['model_digest', 'source_id', 'seed']:
                if key in actual_identity and report['identity'].get(key) != actual_identity[key]:
                    failures.append('Candidate/report identity differs from actual native receipt: ' + key)
            if report.get('reference'):
                identity = _identity_from_actual_receipt(
                    (core_base / entry['reference_run_receipt_path']).resolve(),
                    (core_base / entry['reference_video_path']).resolve(), entry['reference_video_sha256'])
                if identity['kind'] != 'dense':
                    failures.append('Reference is not the actual dense operator')
                for key in ['model_digest', 'source_id', 'seed']:
                    if key in identity and report['reference_identity'].get(key) != identity[key]:
                        failures.append('Dense reference/report identity differs from actual generation receipt: ' + key)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        failures.append('Strict native provenance incomplete: ' + str(exc))
    return failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-bundle', type=Path, required=True)
    errors = inspect(parser.parse_args().strict_bundle)
    if errors:
        raise SystemExit('STRICT_PAIRED_EVIDENCE_INCOMPLETE:\n' + '\n'.join(errors))
    print('STRICT_PAIRED_INVENTORY_AND_MODEL_IDENTITY_PASS_NOT_QUALITY_ACCEPTANCE')
