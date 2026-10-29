"""S2 fixed 18-candidate CPU archive extraction; provisional metadata, no model.

Draft scans metadata only. Run requires separately frozen ACTIVE registration.
Historical and reserved images are never opened. Conflicts remain in results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:
    sys.path.insert(0, str(ROOT/'tools'))
import raw_bf16_fresh_source_extract_v2 as base

ID = 'E0_S2_SOURCE18_PROVISIONAL_ARCHIVE_CPU_20261029'
OUTPUT = 'ffnds-derived/e0_s2_source18_provisional_archive_cpu_20261029/attempt1/worker/results'
ORDINALS = (910, 911, 912, 977, 978, 979, 1039, 1040, 1041, 922, 923, 924, 616, 624, 632, 456, 464, 472)
TYPES = ('animal',)*3 + ('single-human',)*3 + ('multiple-human',)*3 + ('transportation',)*3 + ('scenery',)*3 + ('indoor',)*3
RESOURCES = {'GPU': 0, 'CPU_threads': 2, 'wall_seconds': 300, 'outer_seconds': 600,
             'archive_stream_bytes_max': 3 << 30, 'new_HF_bytes': 128 << 20,
             'single_image_bytes_max': 4 << 20, 'automatic_retry': False}
LIMITS = {'archive_streams': 1, 'selected_original_images': 18, 'model_calls': 0,
          'GPU_calls': 0, 'new_crops': 0}
REPO_SOURCES = ('tools/raw_bf16_s2_source18_extract.py', 'tests/test_raw_bf16_s2_source18_extract.py',
                'tools/raw_bf16_fresh_source_extract_v2.py', 'tools/worker_identity.py',
                'research/method_discussion/discoveries/E0/s2-source-expansion-metadata-independent-20261029.md')
need = base.need


def source_candidates(rows):
    result = []
    for i, (ordinal, typ) in enumerate(zip(ORDINALS, TYPES)):
        row = rows[ordinal]; name = row['image_name']; prompt = row['prompt_en']
        need(row['image_type'] == typ and isinstance(name, str) and name not in ('.', '..')
             and '/' not in name and '\\' not in name and isinstance(prompt, str), 'Fixed metadata candidate changed')
        result.append({'native_source_ordinal': ordinal, 'cohort_index': i,
            'source_id': f'lxl-158/vbench-i2v@{base.REV}:row{ordinal}',
            'role': 'PROVISIONAL_NEW_DEVELOPMENT', 'metadata_image_type': typ,
            'original_prompt': prompt, 'original_prompt_sha256': base.sha(prompt.encode()),
            'image_name': name, 'archive_member': 'data/crop/16-9/'+name})
    need(len(result) == 18 and len({x['image_name'] for x in result}) == 18, 'Candidate identity collision')
    return result


def history_paths(root=ROOT):
    """Freeze all existing config JSON plus seven banks and task metadata."""
    root = Path(root)
    banks = [root/'.runtime/inputs'/b/'canonical.json' for b in base.BANKS]
    tasks = sorted((root/'.runtime/inputs').glob('task*.json'))
    configs = sorted((root/'config').rglob('*.json'))
    need(len(banks) == 7 and all(p.is_file() for p in banks) and tasks and configs, 'History manifest incomplete')
    return sorted(set(banks+tasks+configs))


def history_index(paths):
    """Only documented image identity fields; never inspect image contents."""
    hashes, names, ordinals = {}, {}, {}
    def add(mapping, key, origin):
        mapping.setdefault(key, set()).add(origin)
    def walk(obj, origin, parent_key=''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    if ((key in ('image_sha256', 'legacy_image_fingerprint', 'image_id') or
                         (key == 'sha256' and parent_key == 'image')) and
                        len(value) == 64 and all(c in '0123456789abcdefABCDEF' for c in value)):
                        add(hashes, value.lower(), origin)
                    if key == 'image_name': add(names, value, origin)
                    if key in ('image_path', 'archive_member'): add(names, Path(value).name, origin)
                if key in ('native_source_ordinal', 'source_index') and type(value) is int:
                    add(ordinals, value, origin)
                walk(value, origin, key)
        elif isinstance(obj, list):
            for value in obj:
                if parent_key == 'img_paths' and isinstance(value, str):
                    add(names, Path(value).name, origin)
                walk(value, origin, parent_key)
    for path in paths:
        path = Path(path)
        walk(json.loads(path.read_text(encoding='utf-8')), str(path))
    return hashes, names, ordinals


def make_draft(hf):
    hf = Path(hf).resolve()
    metadata = base.bound(hf, base.bind(hf/base.METADATA, hf)).read_bytes()
    need(base.sha(metadata) == base.META_SHA, 'Metadata SHA mismatch')
    candidates = source_candidates(json.loads(metadata))
    paths = history_paths(); bindings = [base.bind(p, ROOT) for p in paths]
    hashes, names, indices = history_index(paths)
    archive = (hf/base.ARCHIVE).resolve(); st = archive.stat()
    need(archive.is_relative_to(hf) and stat.S_ISREG(st.st_mode) and 0 < st.st_size <= RESOURCES['archive_stream_bytes_max'], 'Archive stat invalid')
    prior = ROOT/'.runtime/inputs/formal20/recovery_report.json'
    need(json.loads(prior.read_text())['archive_sha256'] == base.ARCHIVE_SHA, 'Archive provenance mismatch')
    return {'id': ID, 'status': 'DRAFT', 'assets_finalized': False, 'automatic_retry': False,
        'resources': RESOURCES, 'limits': LIMITS, 'output_HF_relative': OUTPUT,
        'archive': {'path': base.ARCHIVE, 'bytes': st.st_size, 'sha256': base.ARCHIVE_SHA},
        'metadata': base.bind(hf/base.METADATA, hf), 'archive_provenance': base.bind(prior, ROOT),
        'source_candidates': candidates, 'history_sources': bindings,
        'history_scan_scope': 'ALL_EXISTING_CONFIG_JSON_RECURSIVE_PLUS_SEVEN_CANONICAL_AND_TASK_JSON_AT_DRAFT',
        'history_counts': {'image_sha256': len(hashes), 'image_names': len(names), 'ordinals': len(indices)},
        'repo_sources': [base.bind(ROOT/p, ROOT) for p in REPO_SOURCES],
        'independent_review': None, 'source_readiness': 'PROVISIONAL_NOT_A_COHORT_NO_VIDEO_BEHAVIOR'}


def authorize(plan, protocol_sha):
    need(plan['id'] == ID and plan['status'] == 'FROZEN_REGISTERED' and plan['assets_finalized'] is True
         and plan['resources'] == RESOURCES and plan['limits'] == LIMITS and plan['output_HF_relative'] == OUTPUT
         and plan['automatic_retry'] is False and bool(plan.get('independent_review')), 'Frozen scope differs')
    active = json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text())
    scopes = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    need(active['user_continuation_authorization_20261027'].get('new_scope_registration_allowed') is True
         and len(scopes) == 1 and scopes[0].get('protocol_sha256') == protocol_sha
         and scopes[0].get('resources') == RESOURCES and scopes[0].get('limits') == LIMITS
         and scopes[0].get('max_gpu_cards') == 0
         and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE'), 'ACTIVE scope differs')


def validate(plan, hf, protocol_sha, protocol_path):
    authorize(plan, protocol_sha)
    need(plan['history_scan_scope'] == 'ALL_EXISTING_CONFIG_JSON_RECURSIVE_PLUS_SEVEN_CANONICAL_AND_TASK_JSON_AT_DRAFT', 'History scope differs')
    for b in plan['repo_sources'] + plan['history_sources'] + [plan['archive_provenance']]: base.bound(ROOT, b)
    need({b['path'] for b in plan['repo_sources']} == set(REPO_SOURCES), 'Source closure differs')
    history_set = {b['path'] for b in plan['history_sources']}
    current_history = {str(p.resolve().relative_to(ROOT.resolve())) for p in history_paths()
                       if p.resolve() != protocol_path.resolve()}
    need(current_history == history_set, 'History file inventory changed since draft; freeze a new scope')
    need(all(f'.runtime/inputs/{b}/canonical.json' in history_set for b in base.BANKS)
         and any(p.startswith('.runtime/inputs/task') for p in history_set)
         and any(p.startswith('config/') for p in history_set), 'History source list incomplete')
    metadata = base.bound(hf, plan['metadata']).read_bytes()
    need(base.sha(metadata) == base.META_SHA and plan['source_candidates'] == source_candidates(json.loads(metadata)), 'Metadata/candidate changed')
    need(plan['archive']['path'] == base.ARCHIVE and plan['archive']['sha256'] == base.ARCHIVE_SHA,
         'Archive identity differs')
    paths = [base.bound(ROOT, b) for b in plan['history_sources']]
    hashes, names, ordinals = history_index(paths)
    need(plan['history_counts'] == {'image_sha256': len(hashes), 'image_names': len(names), 'ordinals': len(ordinals)}, 'History counts differ')
    return hashes, names, ordinals


def candidate_results(candidates, payload, hashes, names, ordinals, image_info):
    result = []
    for case in candidates:
        data = payload[case['archive_member']]; digest = base.sha(data); info = image_info(data)
        name_hits = sorted(names.get(case['image_name'], set()))
        sha_hits = sorted(hashes.get(digest, set()))
        # Ordinals can refer to unrelated namespaces: retain as possible, not a definitive exclusion.
        ordinal_hits = sorted(ordinals.get(case['native_source_ordinal'], set()))
        result.append({**case, 'image_sha256': digest, 'image_bytes': len(data), 'image_info': info,
                       'historical_exact_sha_hits': sha_hits, 'historical_name_hits': name_hits,
                       'historical_ordinal_possible_hits': ordinal_hits})
    by_sha = {}
    for row in result: by_sha.setdefault(row['image_sha256'], []).append(row['native_source_ordinal'])
    for row in result:
        row['candidate_exact_duplicate_ordinals'] = [i for i in by_sha[row['image_sha256']] if i != row['native_source_ordinal']]
        row['candidate_dhash_distances'] = [{'ordinal': other['native_source_ordinal'],
            'hamming': (int(row['image_info']['dhash64'], 16) ^ int(other['image_info']['dhash64'], 16)).bit_count()}
            for other in result if other is not row]
        row['exclusion'] = bool(row['historical_exact_sha_hits'] or row['historical_name_hits'] or row['candidate_exact_duplicate_ordinals'])
        row['historical_near_duplicate'] = 'UNKNOWN_NO_HISTORICAL_IMAGES_OPENED'
        row['scene_motion'] = 'NOT_OBSERVED'; row['cohort_eligibility'] = 'NOT_ESTABLISHED'
    return result


def run(args):
    started = time.monotonic(); hf = args.hf_root.resolve(); plan_bytes = args.protocol.read_bytes()
    plan = json.loads(plan_bytes); protocol_sha = base.sha(plan_bytes)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0)) <= 2
         and not {'torch', 'numpy'} & set(sys.modules), 'Fresh bounded CPU-only worker required')
    from worker_identity import require_worker
    require_worker()
    authorize(plan, protocol_sha)
    out = (hf/OUTPUT).resolve(); need(out.is_relative_to(hf) and not out.exists(), 'Fresh confined output required')
    out.mkdir(parents=True)
    report = {'id': ID, 'protocol_sha256': protocol_sha, 'status': 'STARTED_CPU_SOURCE_EXTRACTION_NO_RETRY',
              'predebited_archive_bytes': 0, 'actual_archive_bytes': 0, 'archive_streams_predebited': 0,
              'archive_streams_observed': 0, 'predebited_image_bytes': 0, 'actual_image_bytes': 0,
              'image_files_predebited': 0, 'image_files_observed': 0, 'sources': [], 'model_calls': 0, 'GPU_calls': 0,
              'automatic_retry': False, 'archive_payload_verified': False}
    def save():
        encoded = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
        existing = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        need(existing + len(encoded) <= RESOURCES['new_HF_bytes'], 'HF output bound exceeded')
        base.write(out/'RUN.json', report)
    def timeout(*_): raise TimeoutError('CPU source extraction wall exhausted')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(RESOURCES['wall_seconds']); save()
    try:
        hashes, names, ordinals = validate(plan, hf, protocol_sha, args.protocol)
        archive = (hf/plan['archive']['path']).resolve(); before = archive.stat()
        need(archive.is_relative_to(hf) and stat.S_ISREG(before.st_mode)
             and before.st_size == plan['archive']['bytes'] <= RESOURCES['archive_stream_bytes_max'], 'Archive stat invalid')
        report['predebited_archive_bytes'] = before.st_size; report['archive_streams_predebited'] = 1; save()
        with archive.open('rb', buffering=0) as stream:
            reader = base.HashReader(stream, report)
            payload = base.extract_members(reader, {x['archive_member'] for x in plan['source_candidates']}, RESOURCES['single_image_bytes_max'])
            while reader.read(8 << 20): pass
        after = archive.stat(); fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        need(tuple(getattr(before, k) for k in fields) == tuple(getattr(after, k) for k in fields)
             and report['actual_archive_bytes'] == before.st_size and reader.hash.hexdigest() == base.ARCHIVE_SHA,
             'Archive full SHA/stat differs')
        report['archive_streams_observed'] = 1; report['archive_payload_verified'] = True; save()
        rows = candidate_results(plan['source_candidates'], payload, hashes, names, ordinals, base.image_info)
        total = sum(len(v) for v in payload.values())
        need(total + (4 << 20) < RESOURCES['new_HF_bytes'], 'Payload output reserve exceeded')
        report['predebited_image_bytes'] = total; report['image_files_predebited'] = len(rows); save()
        for row in rows:
            path = out/f"source_{row['cohort_index']:02}.image"
            data = payload[row['archive_member']]; need(not path.exists(), 'Fresh candidate file required')
            with path.open('xb') as stream:
                written = stream.write(data)
                report['actual_image_bytes'] += written
                need(written == len(data), 'Short candidate write')
            report['image_files_observed'] += 1; row['image'] = base.bind(path, hf)
            need(row['image']['sha256'] == row['image_sha256'], 'Written image identity differs')
            report['sources'].append(row); save()
        need(len(report['sources']) == LIMITS['selected_original_images']
             and report['actual_image_bytes'] == report['predebited_image_bytes']
             and report['image_files_observed'] == report['image_files_predebited'], 'Candidate count differs')
        report.update(status='COMPLETE_PROVISIONAL_SOURCE_BYTES_PENDING_COHORT_REVIEW',
                      source_bank_status='PROVISIONAL_NOT_AUTHORIZED_FOR_MODEL')
    except BaseException as error:
        report.update(status='FAILED_SOURCE_EXTRACTION_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-started
        report['new_output_bytes_before_final_RUN'] = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        save()


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('action', choices=('draft', 'run'))
    ap.add_argument('--hf-root', type=Path, default=base.HF); ap.add_argument('--protocol', type=Path); ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.action == 'draft':
        need(args.out is not None and not args.out.exists(), 'Fresh draft required')
        args.out.parent.mkdir(parents=True, exist_ok=True); base.write(args.out, make_draft(args.hf_root))
    else: run(args)


if __name__ == '__main__': main()
