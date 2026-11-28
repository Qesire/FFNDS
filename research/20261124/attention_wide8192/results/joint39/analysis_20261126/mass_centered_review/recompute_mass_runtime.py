"""Independent cost and execution-identity audit of archived job 201500.

Uses the raw receipt for every calculation. No model, GPU or video decode runs.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, geometric_mean


def data_bytes(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def load(path):
    return json.loads(data_bytes(path))


def same_trace(a, b):
    return {key: a[key] == b[key] for key in ('initial', 'flows', 'latents', 'video', 'experts')}


def measurements(record):
    step = [float(t) / 1000 for t in record['step_ms']]
    return dict(anchor=step[2], hit=step[3], cycle=sum(step[2:]),
                gpu=sum(step), wall=record['wall'],
                high=sum(step[:2]), nongpu=record['wall'] - sum(step),
                peak_gib=record['peak_allocated']/2**30,
                reserved_gib=record['peak_reserved']/2**30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--historical-quality', type=Path)
    parser.add_argument('--frozen-code', type=Path)
    parser.add_argument('--frozen-recipe', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('mass_runtime_verified.json'))
    args = parser.parse_args()
    src = args.source
    payload = data_bytes(src/'MASS_CENTERED_201500.json.gz')
    receipt = json.loads(payload)
    archived = load(src/'MASS_CENTERED_SUMMARY.json')
    assert receipt['status'] == 'complete' and receipt['job'] == '201500'
    assert receipt['plan_commit'] == 'ff8fd9dcbed3812b1da7f589f3d2e817f48c26c9'
    assert receipt['source_sha256'] == hashlib.sha256((src/'attention_mass_centered_low.py').read_bytes()).hexdigest()
    assert hashlib.sha256(payload).hexdigest() == archived['receipt_sha256']
    assert receipt['scope'] == dict(anchor_step=2, hit_step=3, audit_layers=[0,13,26,39],
                                   frozen_high=True, sources=['vbi_0392','vbi_1013'],
                                   timing_repeats=2, threshold=1e-12, probes=64)
    sources = receipt['scope']['sources']
    assert list(receipt['fixed']) == sources
    assert len(receipt['pairs']) == 4
    checks = {}
    if args.frozen_code:
        digest = hashlib.sha256(args.frozen_code.read_bytes()).hexdigest()
        assert digest == '8781a7983461e5c0042e1e0f4d58e545c2cab5b7508d35dc450e2de87b07dcee'
        checks['frozen_code_sha256'] = digest
    if args.frozen_recipe:
        recipe = load(args.frozen_recipe)
        assert recipe['attention_source_sha256'] == '8781a7983461e5c0042e1e0f4d58e545c2cab5b7508d35dc450e2de87b07dcee'
        assert recipe['calibration']['low']['lambda']['OUT-D'] == .1
        checks['original_low_lambda'] = .1
        checks['fixed_probe_ridge_added_diagonal'] = 64*.1
    references = {s: receipt['fixed'][s]['track']['record'] for s in sources}
    if args.historical_quality:
        old = load(args.historical_quality)
        c = {row['source']:row['record'] for row in old['renders'] if row['arm'] == 'C'}
        checks['historical_quality_sha256'] = hashlib.sha256(data_bytes(args.historical_quality)).hexdigest()
        checks['fixed_vs_historical_C'] = {s:same_trace(references[s],c[s]) for s in sources}
        assert all(all(v.values()) for v in checks['fixed_vs_historical_C'].values())
    pair_details = []
    all_records = []
    for sid, fixed in receipt['fixed'].items():
        all_records.append((sid, 'fixed', fixed['track']))
    for mode, warmup in receipt['warmup'].items():
        all_records.append((warmup['record']['source'], 'warm_'+mode, warmup))
    for pair in receipt['pairs']:
        sid, rep = pair['source'], pair['repeat']
        assert pair['order'] == (['raw','centered'] if (sources.index(sid)+rep)%2 == 0 else ['centered','raw'])
        assert pair['prefix_exact']
        r, c = [pair['arms'][a]['record'] for a in ('raw','centered')]
        prefix = {'initial':r['initial']==c['initial'],
                  'flows_first_three':r['flows'][:3]==c['flows'][:3],
                  'latents_first_three':r['latents'][:3]==c['latents'][:3],
                  'experts':r['experts']==c['experts']}
        assert all(prefix.values())
        assert all(same_trace(r,references[sid]).values())
        assert r['flows'][3] != c['flows'][3] and r['latents'][3] != c['latents'][3]
        rm, cm = measurements(r), measurements(c)
        difference = {k:cm[k]-rm[k] for k in rm}
        pair_details.append(dict(source=sid,repeat=rep,order=pair['order'],prefix_verified=prefix,
            raw_matches_fixed_C=True,step3_flow_and_latent_changed=True,
            raw=rm,centered=cm,centered_minus_raw=difference,
            step_delta_seconds=[(y-x)/1000 for x,y in zip(r['step_ms'],c['step_ms'])]))
        for arm in ('raw','centered'):
            all_records.append((sid,f'pair_{rep}_{arm}',pair['arms'][arm]))
    assert len(all_records) == 12
    for sid, group, row in all_records:
        attn, ff = row['attention'], row['record']['counters']
        assert tuple(attn[k] for k in ('attention_sites','anchors','hits','native_probe_queries')) == (160,80,80,5120)
        assert tuple(ff[k] for k in ('sites','anchors','hits','tile_rows')) == (160,80,80,8192)
        assert ff['variant'] == 'wide8192' and ff['anchor_gpu_capacity_bytes'] == 32428032000
        assert ff['anchor_d2h'] == ff['anchor_h2d'] == 0
        assert attn['low_anchor_query_heads_with_direct_omitted_mass'] == (8776 if sid=='vbi_0392' else 16485)
        video = row['record']['video']
        assert (video['frames'],video['width'],video['height'],video['fps']) == (81,832,464,16.)
        assert row['record']['experts'] == [0,0,1,1]

    per_source, repeated = {}, {}
    for sid in sources:
        pairs = [p for p in receipt['pairs'] if p['source']==sid]
        per_source[sid], repeated[sid] = {}, {}
        for arm in ('raw','centered'):
            records = [p['arms'][arm]['record'] for p in pairs]
            repeated[sid][arm] = same_trace(*records)
            assert all(repeated[sid][arm].values())
            metrics = [measurements(r) for r in records]
            per_source[sid][arm] = {key:(max if key in ('peak_gib','reserved_gib') else mean)(m[key] for m in metrics)
                                  for key in metrics[0]}
            for key in archived['timing']['sources'][sid][arm]:
                assert math.isclose(per_source[sid][arm][key],archived['timing']['sources'][sid][arm][key],rel_tol=1e-12,abs_tol=1e-12)
    aggregate = {}
    for key in ('anchor','hit','cycle','gpu','wall','high','nongpu'):
        row = {arm:mean(v[arm][key] for v in per_source.values()) for arm in ('raw','centered')}
        row['ratio'] = geometric_mean(v['centered'][key]/v['raw'][key] for v in per_source.values())
        row['saved_seconds'] = row['raw'] - row['centered']
        row['fraction_saved'] = 1-row['ratio']
        aggregate[key] = row
        if key in archived['timing']:
            for k,v in archived['timing'][key].items():
                assert math.isclose(row[k],v,rel_tol=1e-12,abs_tol=1e-12)
    order = {}
    for first in ('raw','centered'):
        pairs = [p for p in pair_details if p['order'][0]==first]
        order[first+'_first'] = {'pairs':len(pairs), **{key+'_delta_seconds':mean(p['centered_minus_raw'][key] for p in pairs)
                                                       for key in ('cycle','gpu','wall','nongpu')}}
    rpeak = receipt['pairs'][0]['arms']['raw']['record']['peak_allocated']
    cpeak = receipt['pairs'][0]['arms']['centered']['record']['peak_allocated']
    memory = dict(raw_peak_bytes=rpeak,centered_peak_bytes=cpeak,
                  saved_peak_bytes=rpeak-cpeak,saved_peak_GiB=(rpeak-cpeak)/2**30,
                  two_full_FP32_fields_bytes=2*31668*40*128*4,
                  peak_difference_minus_two_fields_bytes=rpeak-cpeak-2*31668*40*128*4,
                  scope='Measured max allocated, shared process; persistent full field shape unchanged. No allocation timeline for exact temporary attribution.')
    warm_checks = {}
    for arm in ('raw','centered'):
        formal = next(p['arms'][arm]['record'] for p in receipt['pairs'] if p['source']==sources[0])
        warm_checks[arm] = same_trace(receipt['warmup'][arm]['record'],formal)
        assert all(warm_checks[arm].values())
    diagnostics = [row for s in receipt['fixed'].values() for row in s['sites'].values()]
    diag = dict(sites=len(diagnostics),
                max_raw_replay_to_live_relative_l2=max(r['tail']['live_relative_l2'] for r in diagnostics),
                max_probe_to_live_relative_l2=max(r['tail']['shared_probe_vs_live_relative_l2'] for r in diagnostics),
                max_linear_update_identity_relative_l2=max(r['difference_identity_relative_l2'] for r in diagnostics),
                inactive_anchor_channels=sum(r['inactive_anchor_channels'] for r in diagnostics),
                inactive_current_channels=sum(r['inactive_current_channels'] for r in diagnostics),
                rho_min=min(r['rho_min'] for r in diagnostics),
                extra_H448_attention_queries=448*len(diagnostics),extra_Dense_FFN_rows=448*len(diagnostics))
    result = dict(status='VERIFIED',evidence_commit='ec8a39b8a6c8c6a6cb25badfe11e8b6a38b09f63',
                  raw_receipt_sha256=hashlib.sha256(payload).hexdigest(),source_sha256=receipt['source_sha256'],
                  scope='CPU recomputation of archived two-source repeated matched experiment; no new model/GPU/video execution',
                  checks=checks,formal_paired_renders=8,fixed_renders=2,warmup_renders=2,
                  sources=per_source,aggregate=aggregate,pairs=pair_details,order_strata=order,
                  repeat_trace_fields=repeated,warmup_trace_matches=warm_checks,memory=memory,diagnostics=diag,
                  faster_pair_counts={key:sum(p['centered_minus_raw'][key]<0 for p in pair_details) for key in ('anchor','hit','cycle','gpu','wall')},
                  implementation_scope={
                      'raw_formal':'Original OnlineAttention low_cache/approximate; layer=None bypasses diagnostic branch',
                      'centered_formal':'Compiled current moments, centered BF16 encode, and FP32 decode; original selected/probes/features/CAL/lambda',
                      'speed_attribution':'Complete candidate including fusion, compared with old raw implementation; no fusion-only raw ablation',
                      'fixed_panel':'Compiled anchor encode, eager panel decode, same full-key short-query P and native H448 teachers',
                      'live_reference':'LowTail.live_arm=raw; live_relative_l2 compares raw replay with actual C block exit',
                      'local_quality_reference':'native attention at the same C-trajectory input followed by shared theta wide or native Dense FFN',
                      'production_decode_equivalence':'No archived same-input bitwise check between eager panel decode and fullfield compiled decode',
                      'persistent_memory':'Same full history slot shape/dtype; observed peak reduction is working storage/execution evidence, not a history compression ratio'})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','aggregate','order_strata','memory','diagnostics','faster_pair_counts')},indent=2))


if __name__ == '__main__':
    main()
