"""Registered CPU-only full archive header audit; never exports or decodes images."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
import raw_bf16_fresh_source_extract_v2 as base
import raw_bf16_s2_source18_extract as source

ID = 'E0_S2_SOURCE18_TAR_HEADER_AUDIT_CPU_20261029'
OUTPUT = 'ffnds-derived/e0_s2_source18_tar_header_audit_cpu_20261029/attempt1/worker/results'
OLD_PROTOCOL = 'config/e0_s2_source18_archive_cpu_20261029_v1.json'
OLD_WORKER = 'ffnds-derived/e0_s2_source18_provisional_archive_cpu_20261029/attempt1/worker'
RESOURCES = {'GPU': 0, 'CPU_threads': 2, 'wall_seconds': 300, 'outer_seconds': 600,
             'archive_stream_bytes_max': 3 << 30, 'new_HF_bytes': 8 << 20, 'automatic_retry': False}
LIMITS = {'archive_streams': 1, 'target_members': 18, 'exported_images': 0,
          'image_decode_calls': 0, 'model_calls': 0, 'GPU_calls': 0}
REPO_SOURCES = ('tools/raw_bf16_s2_source18_tar_header_audit.py',
                'tests/test_raw_bf16_s2_source18_tar_header_audit.py',
                'tools/raw_bf16_fresh_source_extract_v2.py', 'tools/raw_bf16_s2_source18_extract.py',
                'tools/worker_identity.py')
need = base.need


def parent_proof(hf):
    hf = Path(hf).resolve(); protocol_path = ROOT/OLD_PROTOCOL; worker = hf/OLD_WORKER
    proof = {'protocol': base.bind(protocol_path, ROOT), 'RUN': base.bind(worker/'results/RUN.json', hf),
             'exit': base.bind(worker/'EXIT_RECEIPT.json', hf)}
    verify_parent(proof, hf)
    return proof


def verify_parent(proof, hf):
    hf = Path(hf).resolve()
    need(proof['protocol']['path'] == OLD_PROTOCOL, 'Unexpected parent protocol')
    parent = json.loads(base.bound(ROOT, proof['protocol']).read_text())
    run = json.loads(base.bound(hf, proof['RUN']).read_text())
    receipt = json.loads(base.bound(hf, proof['exit']).read_text())
    need(parent['id'] == run['id'] == receipt['id'] == source.ID
         and run['protocol_sha256'] == receipt['protocol_sha256'] == proof['protocol']['sha256']
         and receipt['run_sha256'] == proof['RUN']['sha256'] and receipt['process_exited'] is True
         and receipt['child_returncode'] == 1 and receipt['timeout'] is False and receipt['automatic_retry'] is False
         and run['status'] == 'FAILED_SOURCE_EXTRACTION_PRESERVED_NO_RETRY'
         and run['error_type'] == 'ValueError' and run['error'] == 'duplicate, unsafe or oversized selected member'
         and 0 < run['actual_archive_bytes'] <= RESOURCES['archive_stream_bytes_max']
         and run['actual_image_bytes'] == run['image_files_observed'] == 0 and run['sources'] == []
         and run['model_calls'] == run['GPU_calls'] == 0, 'Parent failed execution identity differs')
    cases = parent['source_candidates']
    need([c['native_source_ordinal'] for c in cases] == list(source.ORDINALS)
         and len({c['archive_member'] for c in cases}) == LIMITS['target_members'], 'Parent targets differ')
    return parent


def audit_headers(fileobj, targets, max_image_bytes=4 << 20):
    """Scan the full tar stream. Never call extractfile; record exact targets only."""
    wanted = set(targets); need(len(wanted) == len(targets), 'Duplicate target definition')
    occurrences = {name: [] for name in targets}; header_count = 0
    with tarfile.open(fileobj=fileobj, mode='r|gz') as tar:
        for member in tar:
            header_count += 1
            if member.name not in wanted: continue
            name = member.name
            unsafe = name.startswith('/') or '..' in PurePosixPath(name).parts
            reasons = []
            if unsafe: reasons.append('unsafe')
            if not member.isfile(): reasons.append('nonregular')
            if member.size > max_image_bytes: reasons.append('size_gt_4MiB')
            if member.size <= 0: reasons.append('size_nonpositive')
            record = {'name': name, 'type_hex': member.type.hex(), 'is_regular': member.isfile(),
                      'size': member.size, 'linkname': member.linkname, 'rejection_reasons': reasons}
            occurrences[name].append(record)
            need(sum(map(len, occurrences.values())) <= 4096, 'Target occurrence budget exceeded')
    result = []
    for name in targets:
        entries = occurrences[name]
        if len(entries) > 1:
            for entry in entries: entry['rejection_reasons'].append('duplicate')
        result.append({'target_name': name, 'occurrences': entries, 'missing': not entries,
                       'original_extractor_would_accept_unique_header': len(entries) == 1 and not entries[0]['rejection_reasons']})
    return {'total_tar_headers': header_count, 'targets': result}


def make_draft(hf):
    hf = Path(hf).resolve(); proof = parent_proof(hf); parent = verify_parent(proof, hf)
    archive = (hf/base.ARCHIVE).resolve(); st = archive.stat()
    need(archive.is_relative_to(hf) and stat.S_ISREG(st.st_mode)
         and st.st_size == parent['archive']['bytes'] <= RESOURCES['archive_stream_bytes_max']
         and parent['archive']['sha256'] == base.ARCHIVE_SHA, 'Archive stat or parent differs')
    return {'id': ID, 'status': 'DRAFT', 'assets_finalized': False, 'automatic_retry': False,
        'resources': RESOURCES, 'limits': LIMITS, 'output_HF_relative': OUTPUT,
        'archive': parent['archive'], 'parent_failure': proof,
        'targets': [{'native_source_ordinal': c['native_source_ordinal'], 'archive_member': c['archive_member']}
                    for c in parent['source_candidates']],
        'repo_sources': [base.bind(ROOT/p, ROOT) for p in REPO_SOURCES],
        'independent_review': None, 'action': 'FULL_TAR_HEADERS_ONLY_NO_EXTRACTION_NO_DECODE'}


def authorize(plan, sha):
    need(plan['id'] == ID and plan['status'] == 'FROZEN_REGISTERED' and plan['assets_finalized'] is True
         and plan['resources'] == RESOURCES and plan['limits'] == LIMITS and plan['output_HF_relative'] == OUTPUT
         and plan['automatic_retry'] is False and bool(plan.get('independent_review')), 'Frozen scope differs')
    active = json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text())
    scopes = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    need(active['user_continuation_authorization_20261027'].get('new_scope_registration_allowed') is True
         and len(scopes) == 1 and scopes[0].get('protocol_sha256') == sha
         and scopes[0].get('resources') == RESOURCES and scopes[0].get('limits') == LIMITS
         and scopes[0].get('max_gpu_cards') == 0
         and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE'), 'ACTIVE scope differs')


def run(args):
    started = time.monotonic(); hf = args.hf_root.resolve(); raw = args.protocol.read_bytes(); sha = base.sha(raw); plan = json.loads(raw)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0)) <= 2
         and not {'torch', 'numpy', 'PIL'} & set(sys.modules), 'Fresh bounded CPU-only worker required')
    from worker_identity import require_worker
    require_worker(); authorize(plan, sha)
    out = (hf/OUTPUT).resolve(); need(out.is_relative_to(hf) and not out.exists(), 'Fresh confined output required'); out.mkdir(parents=True)
    report = {'id': ID, 'protocol_sha256': sha, 'status': 'STARTED_TAR_HEADER_AUDIT',
              'archive_predebited_bytes': 0, 'actual_archive_bytes': 0, 'archive_streams_predebited': 0,
              'archive_streams_observed': 0, 'archive_payload_verified': False,
              'exported_images': 0, 'image_decode_calls': 0, 'model_calls': 0, 'GPU_calls': 0,
              'automatic_retry': False, 'targets': []}
    def save():
        size = len((json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
        need(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) + size <= RESOURCES['new_HF_bytes'], 'HF output bound exceeded')
        base.write(out/'RUN.json', report)
    def timeout(*_): raise TimeoutError('CPU tar-header audit wall exhausted')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(RESOURCES['wall_seconds']); save()
    try:
        for item in plan['repo_sources']: base.bound(ROOT, item)
        need({b['path'] for b in plan['repo_sources']} == set(REPO_SOURCES), 'Source closure differs')
        parent = verify_parent(plan['parent_failure'], hf)
        targets = [{'native_source_ordinal': c['native_source_ordinal'], 'archive_member': c['archive_member']}
                   for c in parent['source_candidates']]
        need(plan['targets'] == targets and plan['archive'] == parent['archive'] and plan['archive']['sha256'] == base.ARCHIVE_SHA,
             'Parent targets/archive differ')
        archive = (hf/plan['archive']['path']).resolve(); before = archive.stat()
        need(archive.is_relative_to(hf) and stat.S_ISREG(before.st_mode)
             and before.st_size == plan['archive']['bytes'] <= RESOURCES['archive_stream_bytes_max'], 'Archive stat invalid')
        report['archive_predebited_bytes'] = before.st_size; report['archive_streams_predebited'] = 1; save()
        with archive.open('rb', buffering=0) as stream:
            reader = base.HashReader(stream, report)
            audit = audit_headers(reader, [c['archive_member'] for c in targets])
            while reader.read(8 << 20): pass
        after = archive.stat(); fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        need(tuple(getattr(before, k) for k in fields) == tuple(getattr(after, k) for k in fields)
             and report['actual_archive_bytes'] == before.st_size and reader.hash.hexdigest() == base.ARCHIVE_SHA,
             'Archive full SHA/stat differs')
        report.update(archive_streams_observed=1, archive_payload_verified=True,
                      total_tar_headers=audit['total_tar_headers'], targets=audit['targets'],
                      status='COMPLETE_TAR_HEADER_AUDIT_NOT_SOURCE_EXTRACTION')
    except BaseException as error:
        report.update(status='FAILED_TAR_HEADER_AUDIT_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-started; save()


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('action', choices=('draft', 'run'))
    ap.add_argument('--hf-root', type=Path, default=base.HF); ap.add_argument('--protocol', type=Path); ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.action == 'draft':
        need(args.out is not None and not args.out.exists(), 'Fresh draft required')
        args.out.parent.mkdir(parents=True, exist_ok=True); base.write(args.out, make_draft(args.hf_root))
    else: run(args)


if __name__ == '__main__': main()
