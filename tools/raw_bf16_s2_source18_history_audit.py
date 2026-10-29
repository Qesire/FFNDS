"""Bounded CPU perceptual screening of known, non-confirmation image bytes.

New candidate fingerprints are reused from the exited extraction. Reserved
confirmation SHA identities are filtered before any image file is opened.
This screening cannot establish complete scene or crop independence.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ID = 'E0_S2_SOURCE18_NONCONFIRMATION_HISTORY_DHASH_CPU_20261029'
OUTPUT = 'ffnds-derived/e0_s2_source18_nonconfirmation_history_dhash_cpu_20261029/attempt1/worker/results'
RESOURCES = {'GPU': 0, 'CPU_threads': 2, 'wall_seconds': 300, 'outer_seconds': 360,
             'image_bytes_max': 512 << 20, 'single_image_bytes_max': 8 << 20,
             'new_HF_bytes': 8 << 20, 'automatic_retry': False}
RECOMMENDED = (911, 912, 978, 979, 1039, 1040, 922, 923, 616, 624, 456, 472)
SOURCES = ('tools/raw_bf16_s2_source18_history_audit.py',
           'tests/test_raw_bf16_s2_source18_history_audit.py', 'tools/worker_identity.py')


def need(ok, message):
    if not ok: raise ValueError(message)


def read(p): return json.loads(Path(p).read_text())
def sha(raw): return hashlib.sha256(raw).hexdigest()


def binding(p, root):
    p, root = Path(p).resolve(), Path(root).resolve()
    need(p.is_relative_to(root), 'Binding escaped root')
    raw = p.read_bytes()
    return {'path': str(p.relative_to(root)), 'bytes': len(raw), 'sha256': sha(raw)}


def bound(root, item):
    p = (Path(root)/item['path']).resolve()
    need(binding(p, root) == item, 'Binding changed')
    return p


def write(p, value):
    tmp = p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(p)


def image_hashes(obj):
    out = set()
    def walk(x, parent=''):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, str) and len(v) == 64 and (k in
                        ('image_sha256', 'legacy_image_fingerprint', 'image_id') or
                        (k == 'sha256' and parent == 'image')):
                    out.add(v)
                walk(v, k)
        elif isinstance(x, list):
            for v in x: walk(v, parent)
    walk(obj)
    return out


def eligible_history(inventory, fresh_bank, reserved, hf):
    """Metadata/stat only; all reserved identities are rejected before opening."""
    chosen = {}
    for item in inventory['cached_images']:
        if item['sha256'] in reserved or 'diverse_confirmation20' in item['known_exposure_banks']:
            continue
        need(item['known_exposure_banks'], 'Unknown cache exposure is not an allowed historical control')
        chosen[item['sha256']] = {'path': item['path'], 'bytes': item['bytes'],
            'sha256': item['sha256'], 'exposure': item['known_exposure_banks']}
    for item in fresh_bank['sources']:
        digest = item['image_id']
        need(digest not in reserved, 'Fresh-fit bank conflicts with reserved confirmation')
        path = Path(item['image_path']).resolve()
        need(path.is_relative_to(hf.resolve()), 'Historical image escaped HF')
        chosen.setdefault(digest, {'path': str(path.relative_to(hf.resolve())),
            'bytes': path.stat().st_size, 'sha256': digest, 'exposure': ['EXPOSED_FRESH10_NOW_FITTED']})
    need(all(x['sha256'] not in reserved for x in chosen.values()), 'Reserved filter failed')
    return sorted(chosen.values(), key=lambda x: x['sha256'])


def draft(hf):
    inv = ROOT/'research/20261029/RAW_BF16_FRESH_SOURCE_INVENTORY_20261029.json'
    bank = ROOT/'config/e0_raw_bf16_fresh10_discovery_source_bank_20261029_v1.json'
    reserve = ROOT/'.runtime/inputs/diverse_confirmation20/canonical.json'
    parent = hf/'ffnds-derived/e0_s2_source18_provisional_archive_cpu_v2_20261029/attempt1/worker'
    run, receipt = read(parent/'results/RUN.json'), read(parent/'EXIT_RECEIPT.json')
    need(run['status'] == 'COMPLETE_PROVISIONAL_SOURCE_BYTES_PENDING_COHORT_REVIEW'
         and receipt['child_returncode'] == 0 and receipt['process_exited'] is True
         and receipt['run_sha256'] == binding(parent/'results/RUN.json', hf)['sha256'], 'Extraction is not exited and complete')
    reserved = image_hashes(read(reserve))
    history = eligible_history(read(inv), read(bank), reserved, hf)
    need(len(history) == 74 and sum(x['bytes'] for x in history) < RESOURCES['image_bytes_max'], 'Expected 64 cached plus ten fitted images')
    return {'id': ID, 'status': 'DRAFT', 'assets_finalized': False, 'automatic_retry': False,
        'resources': RESOURCES, 'limits': {'image_decodes': len(history), 'candidate_fingerprints_reused': 18,
            'pairwise_distances': len(history)*18, 'GPU_calls': 0, 'model_calls': 0, 'reserved_image_opens': 0},
        'output_HF_relative': OUTPUT, 'metadata': [binding(p, ROOT) for p in (inv, bank, reserve)],
        'parent': {'RUN': binding(parent/'results/RUN.json', hf), 'exit': binding(parent/'EXIT_RECEIPT.json', hf)},
        'history_images': history, 'reserved_sha256': sorted(reserved), 'recommended_ordinals': list(RECOMMENDED),
        'algorithm': 'PIL_DEFAULT_RESIZE_GRAYSCALE_9x8_DHASH64_HAMMING_FLAG_LE6_NO_SEMANTIC_INDEPENDENCE_CLAIM',
        'screen_threshold': 6, 'repo_sources': [binding(ROOT/p, ROOT) for p in SOURCES],
        'independent_input_review': binding(ROOT/'research/method_discussion/discoveries/E0/s2-source18-independent-visual-review-20261029.md', ROOT)}


def authorize(plan, digest):
    active = read(ROOT/'research/ACTIVE_RESEARCH.json')
    scopes = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    need(plan['id'] == ID and plan['resources'] == RESOURCES and plan['status'] == 'FROZEN_REGISTERED'
         and plan['assets_finalized'] is True and plan['automatic_retry'] is False
         and plan['output_HF_relative'] == OUTPUT and active['user_continuation_authorization_20261027']['new_scope_registration_allowed']
         and len(scopes) == 1 and scopes[0]['protocol_sha256'] == digest
         and scopes[0]['resources'] == RESOURCES and scopes[0]['limits'] == plan['limits']
         and scopes[0]['max_gpu_cards'] == 0 and scopes[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE'), 'Exact CPU scope not authorized')


def run(args):
    started = time.monotonic(); hf = args.hf_root.resolve(); plan = read(args.protocol)
    digest = sha(args.protocol.read_bytes()); authorize(plan, digest)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0)) <= 2
         and not {'torch', 'numpy'} & set(sys.modules), 'Fresh bounded CPU process required')
    from worker_identity import require_worker
    require_worker()
    fresh = draft(hf)
    need(all(plan[k] == fresh[k] for k in fresh if k not in ('status', 'assets_finalized')), 'Metadata or source closure changed')
    for item in plan['repo_sources']+plan['metadata']+[plan['independent_input_review']]: bound(ROOT, item)
    out = (hf/OUTPUT).resolve(); need(out.is_relative_to(hf) and not out.exists(), 'Fresh confined output required')
    out.mkdir(parents=True)
    report = {'id': ID, 'protocol_sha256': digest, 'status': 'STARTED_NONCONFIRMATION_CPU_SCREEN',
        'image_bytes_predebited': 0, 'image_bytes_observed': 0, 'image_decodes_predebited': 0,
        'image_decodes_observed': 0, 'pairwise_distances': 0, 'GPU_calls': 0, 'model_calls': 0,
        'reserved_image_opens': 0, 'history': [], 'candidates': [], 'automatic_retry': False}
    save = lambda: write(out/'RUN.json', report)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU screen wall exhausted')))
    signal.alarm(RESOURCES['wall_seconds']); save()
    try:
        from PIL import Image
        reserved = set(plan['reserved_sha256'])
        for item in plan['history_images']:
            need(item['sha256'] not in reserved and item['bytes'] <= RESOURCES['single_image_bytes_max'], 'Reserved/oversized image before open')
            report['image_bytes_predebited'] += item['bytes']; report['image_decodes_predebited'] += 1; save()
            need(report['image_bytes_predebited'] <= RESOURCES['image_bytes_max'], 'Image stream budget exhausted')
            path = (hf/item['path']).resolve(); need(path.is_relative_to(hf), 'Image escaped HF')
            before = path.stat(); need(before.st_size == item['bytes'], 'Image size changed')
            raw = path.read_bytes(); report['image_bytes_observed'] += len(raw)
            after = path.stat()
            fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
            need(sha(raw) == item['sha256'] and all(getattr(before, k) == getattr(after, k) for k in fields), 'Image SHA/stat changed')
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                pixels = list(image.convert('L').resize((9, 8)).getdata())
                code = sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
            report['image_decodes_observed'] += 1
            report['history'].append({**item, 'dhash64': f'{code:016x}'})
        parent = read(bound(hf, plan['parent']['RUN']))
        for source in parent['sources']:
            code = int(source['image_info']['dhash64'], 16)
            nearest = sorted(({'image': {k: old[k] for k in ('path', 'sha256', 'bytes')},
                'hamming': (code ^ int(old['dhash64'], 16)).bit_count()} for old in report['history']), key=lambda x: (x['hamming'], x['image']['sha256']))
            report['pairwise_distances'] += len(nearest)
            report['candidates'].append({'native_source_ordinal': source['native_source_ordinal'],
                'recommended': source['native_source_ordinal'] in RECOMMENDED, 'nearest_three': nearest[:3],
                'flagged_pairs': [x for x in nearest if x['hamming'] <= plan['screen_threshold']]})
        need(report['image_decodes_observed'] == report['image_decodes_predebited'] == plan['limits']['image_decodes']
             and report['image_bytes_observed'] == report['image_bytes_predebited']
             and report['pairwise_distances'] == plan['limits']['pairwise_distances'], 'CPU counts did not close')
        report.update(status='COMPLETE_BOUNDED_NONCONFIRMATION_HISTORY_SCREEN', coverage='74_PREDECLARED_EXPOSED_IMAGES_ONLY_RESERVED20_NOT_OPENED',
            scene_independence='PERCEPTUAL_SCREEN_AND_INPUT_REVIEW_ONLY_NOT_EXHAUSTIVE_CROP_OR_SCENE_PROOF')
    except BaseException as error:
        report.update(status='FAILED_HISTORY_SCREEN_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-started
        need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) < RESOURCES['new_HF_bytes']//2, 'Output budget exhausted')
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('draft', 'run'))
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    p.add_argument('--protocol', type=Path); p.add_argument('--out', type=Path); args = p.parse_args()
    if args.action == 'draft':
        need(args.out is not None and not args.out.exists(), 'Fresh draft required'); write(args.out, draft(args.hf_root))
    else: run(args)


if __name__ == '__main__': main()
