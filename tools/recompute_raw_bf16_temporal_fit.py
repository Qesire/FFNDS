"""Assemble two actually exited expert fits and publish the full80 factor bank."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import e0_raw_bf16_temporal_balanced_fit as fit


def recompute(hf):
    rows, banks, parents = [], [], []
    models = None
    for part in (0, 1):
        path = ROOT/f'config/e0_raw_bf16_new10_temporal_balanced_fit_part{part}_20261029_v1.json'
        plan = fit.data.read(path)
        fit.validate(plan, True)
        out = hf/plan['output_HF_relative']
        rb, eb = fit.data.binding(out/'RUN.json', hf), fit.data.binding(out.parent/'EXIT_RECEIPT.json', hf)
        report, exited = fit.data.read(out/'RUN.json'), fit.data.read(out.parent/'EXIT_RECEIPT.json')
        fit.prior.video.exited(exited, rb, fit.native.sha(path), plan['id'])
        fit.data.need(report['status'] == 'COMPLETE_EXPERT40_MATCHED_VALUE_DC_AC_R960_BANKS'
                      and len(report['functions']) == 40 and report['exported_factors'] == 80
                      and not report['assigned_GPU_resources']['violations']
                      and all(report['operation_ledger'][k] == plan['limits'] for k in ('predebited', 'observed')),
                      'Complete actual expert shard required')
        fit.data.need(models is None or models == plan['models'], 'Expert numeric checkpoints differ')
        models = plan['models']
        parents.append({'part': part, 'protocol': fit.data.binding(path, ROOT), 'RUN': rb, 'exit': eb,
                        'actual_exit': exited['child_returncode'], 'actual_outer_seconds': exited['outer_seconds']})
        for item in report['functions']:
            record = json.loads(fit.data.bound(hf, item['record']))
            fit.data.need(record['expert'] == ('high' if part == 0 else 'low') and record['block'] == item['block']
                          and {v['family'] for v in record['fits']} == set(fit.FAMILIES), 'Actual function/families differ')
            for result in record['fits']:
                banks.append({'expert': part, 'block': record['block'], 'family': result['family'],
                              'training_source_ids': result['training_source_ids'], **result['export']})
                rows.extend({'family': result['family'], 'block': record['block'], **v} for v in result['evaluations'])
    fit.data.need(len(banks) == 160 and {(r['expert'], r['block'], r['family']) for r in banks}
                  == {(e, b, f) for e in (0, 1) for b in range(40) for f in fit.FAMILIES}, 'Whole80 two-family bank incomplete')
    tables = {}
    for role, n in (('TRAIN_NEW10', 10), ('IMAGE_DISJOINT_EXPOSED_DIAGNOSTIC_OLD8', 8)):
        tables[role] = {}
        for family in fit.FAMILIES:
            chosen = [r for r in rows if r['role'] == role and r['family'] == family]
            sources = defaultdict(list)
            for r in chosen:
                sources[r['source_id']].append(r)
            fit.data.need(len(sources) == n and all(len(v) == 160 for v in sources.values()), 'Complete source-equal full160 evidence required')
            keys = ('gated_Y_relative_SSE', 'post_u_relative_SSE', 'normalized_u_relative_SSE')
            if role == 'TRAIN_NEW10':
                keys += ('tau1_to20_DC_relative_SSE', 'tau1_to20_AC_relative_SSE',
                         'tau1_to20_AC_energy_ratio', 'tau1_to20_AC_aligned_gain', 'tau0_relative_SSE')
            per_source = []
            for sid, values in sorted(sources.items()):
                means = {k: statistics.mean(v[k] for v in values if v.get(k) is not None)
                         if any(v.get(k) is not None for v in values) else None for k in keys}
                per_source.append({'source_id': sid, 'native_source_ordinal': values[0]['native_source_ordinal'], **means})
            tables[role][family] = {'image_clusters': n,
                'source_equal_means': {k: statistics.mean(r[k] for r in per_source if r[k] is not None) for k in keys},
                'per_source_all160_means': per_source}
    return {'schema': 'ffnds.raw_bf16.temporal_representation.full80.bank.v1',
            'id': 'RAW_BF16_NEW10_VALUE_DC_AC_FULL80_R960_BANK_20261029',
            'status': 'COMPLETE_ACTUAL_TWO_EXPERT_EXIT0_FULL80_BANKS_VIDEO_NOT_RUN',
            'parents': parents, 'models': models, 'design': fit.DESIGN, 'rank': 960,
            'fitting_image_clusters': 10, 'diagnostic_image_clusters': 8, 'families': list(fit.FAMILIES),
            'factor_files': 160, 'factor_payload_bytes': 160*fit.FACTOR_BYTES,
            'exported_factors': banks, 'source_equal_metrics': tables,
            'formal_confirmation': 'NOT_RUN', 'new_student_video': 'NOT_RUN',
            'quality_or_speed_accepted': False, 'publication_new_tensor_model_GPU_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = recompute(args.hf_root.resolve())
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'factor_files': result['factor_files'], 'metrics': result['source_equal_metrics']}))


if __name__ == '__main__':
    main()
