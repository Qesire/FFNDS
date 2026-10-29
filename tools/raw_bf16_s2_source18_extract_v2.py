"""Fresh source18 extraction namespace with verified header audit; no old code mutation."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
import raw_bf16_fresh_source_extract_v2 as base
import raw_bf16_s2_source18_tar_header_audit as header

spec = importlib.util.spec_from_file_location('_ffnds_s2_source18_v2_private', ROOT/'tools/raw_bf16_s2_source18_extract.py')
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
ID = 'E0_S2_SOURCE18_PROVISIONAL_ARCHIVE_CPU_V2_20261029'
OUTPUT = 'ffnds-derived/e0_s2_source18_provisional_archive_cpu_v2_20261029/attempt1/worker/results'
AUDIT_PROTOCOL = 'config/e0_s2_source18_tar_header_audit_cpu_20261029_v1.json'
AUDIT_WORKER = 'ffnds-derived/e0_s2_source18_tar_header_audit_cpu_20261029/attempt1/worker'
EXCLUDED = (header.OLD_PROTOCOL, AUDIT_PROTOCOL)
REASON = 'ONLY_TWO_BOUND_QUERY_PROTOCOLS_WITH_VERIFIED_ZERO_EXPORTED_IMAGES; ALL_OTHER_HISTORY_RETAINED'
core.ID = ID; core.OUTPUT = OUTPUT
core.RESOURCES = {**core.RESOURCES, 'single_image_bytes_max': 8 << 20}
core.REPO_SOURCES = core.REPO_SOURCES + ('tools/raw_bf16_s2_source18_extract_v2.py',
    'tests/test_raw_bf16_s2_source18_extract_v2.py', 'tools/raw_bf16_s2_source18_tar_header_audit.py')
original_history_index = core.history_index
original_validate = core.validate
need = base.need


def filter_history(paths, excluded=EXCLUDED, root=ROOT):
    root = Path(root).resolve(); blocked = {(root/x).resolve() for x in excluded}
    return [p for p in paths if Path(p).resolve() not in blocked]


def history_index(paths):
    return original_history_index(filter_history(paths))


core.history_index = history_index


def make_audit_proof(hf):
    worker = Path(hf)/AUDIT_WORKER
    return {'protocol': base.bind(ROOT/AUDIT_PROTOCOL, ROOT), 'RUN': base.bind(worker/'results/RUN.json', hf),
            'exit': base.bind(worker/'EXIT_RECEIPT.json', hf)}


def verify_audit(proof, hf):
    need(proof['protocol']['path'] == AUDIT_PROTOCOL, 'Unexpected audit protocol')
    protocol = json.loads(base.bound(ROOT, proof['protocol']).read_text())
    report = json.loads(base.bound(hf, proof['RUN']).read_text())
    receipt = json.loads(base.bound(hf, proof['exit']).read_text())
    need(protocol['id'] == report['id'] == receipt['id'] == header.ID
         and report['protocol_sha256'] == receipt['protocol_sha256'] == proof['protocol']['sha256']
         and receipt['run_sha256'] == proof['RUN']['sha256'] and receipt['process_exited'] is True
         and receipt['child_returncode'] == 0 and receipt['timeout'] is False and receipt['automatic_retry'] is False
         and report['status'] == 'COMPLETE_TAR_HEADER_AUDIT_NOT_SOURCE_EXTRACTION'
         and report['archive_payload_verified'] is True and report['archive_streams_observed'] == 1
         and report['actual_archive_bytes'] == protocol['archive']['bytes']
         and protocol['archive']['sha256'] == base.ARCHIVE_SHA
         and report['exported_images'] == report['image_decode_calls'] == report['model_calls'] == report['GPU_calls'] == 0,
         'Actual audit completion differs')
    parent = header.verify_parent(protocol['parent_failure'], hf)
    parent_targets = [{'native_source_ordinal': c['native_source_ordinal'], 'archive_member': c['archive_member']}
                      for c in parent['source_candidates']]
    need(protocol['targets'] == parent_targets and len(report['targets']) == 18, 'Audit target closure differs')
    total = 0
    for case, target in zip(parent_targets, report['targets']):
        need(target['target_name'] == case['archive_member'] and target['missing'] is False
             and len(target['occurrences']) == 1, 'Audit member missing or duplicate')
        item = target['occurrences'][0]
        need(item['name'] == case['archive_member'] and item['is_regular'] is True
             and type(item['size']) is int and 0 < item['size'] <= core.RESOURCES['single_image_bytes_max']
             and item['rejection_reasons'] in ([], ['size_gt_4MiB']), 'Unsafe or oversized audited target')
        total += item['size']
    need(total + (4 << 20) < core.RESOURCES['new_HF_bytes'], 'Audited image bytes exceed output budget')
    return parent, protocol, total


def make_draft(hf):
    hf = Path(hf).resolve(); audit_proof = make_audit_proof(hf); parent, audit, total = verify_audit(audit_proof, hf)
    plan = core.make_draft(hf)
    need(plan['source_candidates'] == parent['source_candidates'] and plan['archive'] == parent['archive'], 'Candidates or archive changed')
    need(set(EXCLUDED) <= {b['path'] for b in plan['history_sources']}, 'Excluded parents must remain bound in history')
    plan['lineage'] = {'failed_extraction': audit['parent_failure'], 'header_audit': audit_proof,
        'excluded_from_exposure_index': list(EXCLUDED), 'exclusion_reason': REASON,
        'audited_18_member_bytes': total, 'old_single_image_max_bytes': 4 << 20,
        'new_single_image_max_bytes': 8 << 20, 'decision': 'HEADER_AUDIT_ONLY_NOT_A_NEW_COHORT'}
    return plan


def validate(plan, hf, protocol_sha, protocol_path):
    lineage = plan['lineage']
    need(lineage['excluded_from_exposure_index'] == list(EXCLUDED) and lineage['exclusion_reason'] == REASON
         and lineage['old_single_image_max_bytes'] == 4 << 20 and lineage['new_single_image_max_bytes'] == 8 << 20,
         'Lineage exposure exception differs')
    parent, audit, total = verify_audit(lineage['header_audit'], hf)
    need(lineage['failed_extraction'] == audit['parent_failure'] and lineage['audited_18_member_bytes'] == total
         and plan['source_candidates'] == parent['source_candidates'] and plan['archive'] == parent['archive']
         and set(EXCLUDED) <= {b['path'] for b in plan['history_sources']}, 'Bound zero-export parent closure differs')
    return original_validate(plan, hf, protocol_sha, protocol_path)


core.validate = validate


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('action', choices=('draft', 'run'))
    ap.add_argument('--hf-root', type=Path, default=base.HF); ap.add_argument('--protocol', type=Path); ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.action == 'draft':
        need(args.out is not None and not args.out.exists(), 'Fresh draft required')
        args.out.parent.mkdir(parents=True, exist_ok=True); base.write(args.out, make_draft(args.hf_root))
    else: core.run(args)


if __name__ == '__main__': main()
