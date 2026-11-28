"""Recompute matched real-Wan transport evidence from the archived receipts.

This performs CPU arithmetic on recorded measurement values. It does not load
Wan tensors, run inference, infer visual quality, or add independent sources.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gm(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def recompute(src, historical=None):
    receipt = json.loads((src / 'wan_runtime_receipt_198062.json').read_text())
    summary = json.loads((src / 'wan_runtime_summary_198062.json').read_text())
    assert receipt['status'] == 'complete' and receipt['job'] == '198062'
    fingerprint = {
        'attention_defect_transport.py': digest(src / 'attention_defect_transport.py') == receipt['method_sha256'],
        'run_attention_defect_transport.py': digest(src / 'run_attention_defect_transport.py') == receipt['source_sha256'],
        'wan_runtime_receipt_198062.json': digest(src / 'wan_runtime_receipt_198062.json') == summary['receipt_sha256'],
    }
    assert all(fingerprint.values())
    observations = receipt['replay']['observations']
    assert set(observations) == {f'{s}:{l}' for s in (1, 3) for l in (0, 13, 26, 39)}
    rows = []
    for site, obs in observations.items():
        step, layer = map(int, site.split(':'))
        base, fit = obs['base_energy'], obs['fit_energy']
        row = dict(site=site, step=step, layer=layer,
            raw_base_energy=base['raw'], transported_base_energy=base['transport'],
            raw_fit_energy=fit['raw'], transported_fit_energy=fit['transport'],
            base_energy_ratio=base['transport']/base['raw'],
            fit_energy_ratio=fit['transport']/fit['raw'],
            fit_l2_ratio=math.sqrt(fit['transport']/fit['raw']),
            raw_fit_over_base=fit['raw']/base['raw'],
            transported_fit_over_base=fit['transport']/base['transport'],
            bf16_cache_fit_energy_ratio=fit['transport_bf16']/fit['transport'],
            short_teacher_relative_l2=obs['short_teacher_relative_l2'])
        if 'wan_wide8192_tail' in obs:
            tail=obs['wan_wide8192_tail']; e=tail['held_block_energy']
            raw_gain=e['raw']/fit['raw']; transported_gain=e['transport']/fit['transport']
            ratio=e['transport']/e['raw']
            closure=row['fit_energy_ratio']*transported_gain/raw_gain
            assert abs(closure-ratio)<1e-12
            row['tail']=dict(energy_ratio=ratio,l2_ratio=math.sqrt(ratio),
                raw_realized_finite_gain_squared=raw_gain,
                transported_realized_finite_gain_squared=transported_gain,
                gain_squared_ratio=transported_gain/raw_gain,
                ratio_factorization_error=abs(closure-ratio),
                bf16_over_fp32_tail_energy=e['transport_bf16']/e['transport'],
                probe_ffn_input_exact=tail['probe_ffn_input_exact'],
                native_live_block_relative_l2=tail['native_live_block_relative_l2'])
            assert all(tail['probe_ffn_input_exact'].values())
        rows.append(row)
    aggregate={}
    for step in (1,3):
        group=[row for row in rows if row['step']==step]
        out={}
        for name in ('base','fit'):
            values=[row[name+'_energy_ratio'] for row in group]
            raw=sum(row['raw_'+name+'_energy'] for row in group)
            transported=sum(row['transported_'+name+'_energy'] for row in group)
            out[name]=dict(site_equal_weight_geomean_energy_ratio=gm(values),
                site_equal_weight_geomean_l2_ratio=math.sqrt(gm(values)),
                pooled_energy_ratio=transported/raw,
                raw_energy_sum=raw,transported_energy_sum=transported,
                improved_sites=sum(x<1 for x in values),total_sites=len(values))
        out['fit_excluding_layer0_geomean_energy_ratio']=gm([row['fit_energy_ratio'] for row in group if row['layer']!=0])
        first=next(row for row in group if row['layer']==0)
        out['layer0_raw_fit_energy_fraction']=first['raw_fit_energy']/out['fit']['raw_energy_sum']
        out['cache_geomean_energy_ratio']=gm([row['bf16_cache_fit_energy_ratio'] for row in group])
        expected=summary['fit_replay'][str(step)]
        for observed,key in [(out['base']['site_equal_weight_geomean_energy_ratio'],'base'),
                             (out['fit']['site_equal_weight_geomean_energy_ratio'],'fitted'),
                             (out['cache_geomean_energy_ratio'],'cache')]:
            assert abs(observed-expected[key])<1e-12
        aggregate[str(step)]=out
    candidates={}
    for sid,candidate in receipt['candidates'].items():
        record=candidate['record']; counts=candidate['counters']
        assert counts==dict(sites=160,anchors=80,peak_slots=40,hits=80,native_probe_queries=5120)
        candidates[sid]=dict(gpu_step_seconds=sum(record['step_ms'])/1000,
            wall_seconds=record['wall'],anchor_step_seconds=sum(record['step_ms'][::2])/1000,
            hit_step_seconds=sum(record['step_ms'][1::2])/1000,
            peak_allocated_gib=record['peak_allocated']/2**30,
            peak_reserved_gib=record['peak_reserved']/2**30,
            counters=counts,video=record['video'])
    trace=None
    if historical is not None:
        old=json.loads(historical.read_text())
        replay=receipt['replay']['record']; original=old['control']['vbi_0392']['wide8192']
        match={key: replay[key]==original[key] for key in ('initial','flows','latents','experts','video')}
        assert all(match.values())
        candidate_match={sid:{'initial':c['record']['initial']==old['control'][sid]['wide8192']['initial'],
            'first_flow':c['record']['flows'][0]==old['control'][sid]['wide8192']['flows'][0],
            'first_latent':c['record']['latents'][0]==old['control'][sid]['wide8192']['latents'][0]}
            for sid,c in receipt['candidates'].items()}
        assert all(v for row in candidate_match.values() for v in row.values())
        trace=dict(historical_job=old['job'],historical_sha256=digest(historical),
                   replay_equal=match,candidate_initial_and_first_step=candidate_match)
    return dict(evidence_commit='253d2b72569289469ef191086e81643c90c37cfb',
        operation='CPU recomputation of archived real measurements; no new inference',
        statistical_units={'replay_sources':1,'replay_source':'vbi_0392','replay_sites':8,
            'tail_sites':2,'candidate_video_sources':2,'independent_timing_repeats_per_candidate_source':1},
        metric='energy = squared Frobenius error, not L2 error',
        fingerprints=fingerprint,sites=rows,aggregate=aggregate,candidates=candidates,
        maximum_short_teacher_relative_l2=max(row['short_teacher_relative_l2'] for row in rows),
        max_abs_bf16_cache_relative_energy_change=max(abs(row['bf16_cache_fit_energy_ratio']-1) for row in rows),
        replay_reference_trace=trace,
        comparison_scope='Same current G/features/full-query P teacher/fixed ridge; FP32 raw versus FP32 normalized historical panel. Actual BF16 cache is a separate arm. Tail reference is native self-attention plus wide8192 FFN.',
        inference_limits='Eight sites are nested in one reused DEV trajectory; end-to-end two videos use the transported candidate only. Their quality judgment is pending human review. Finite gain is direction-specific, not an operator norm.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parent.parent/'method_transport')
    parser.add_argument('--historical',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    result=recompute(args.source,args.historical)
    output=args.output or root/'transport_result_recomputed.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(output),'aggregate':result['aggregate'],'fingerprints':result['fingerprints']},ensure_ascii=False,indent=2))
