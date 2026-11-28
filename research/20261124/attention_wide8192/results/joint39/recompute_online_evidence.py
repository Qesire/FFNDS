"""Recompute paired arithmetic from the published online attention receipts.

No model, video decode, hardware timer, or pixel/latent distance is executed.
Hashes verify that timing renders reproduce the archived quality renders.
Existing quality metrics are postprocessed, with human review attributed only
to the user's supplied sentence.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import statistics as st

HEAD = '832e9eae827364d49900232f439b9d74daabb624'
USER_OBSERVATION = '结果已同步，三个方法的视频对比，肉眼几乎没有差别'


def payload(path):
    b=path.read_bytes()
    return gzip.decompress(b) if path.suffix=='.gz' else b


def read(root, name):
    return json.loads(payload(root/name))


def summary(vals):
    vals=list(vals)
    return {'n':len(vals),'median':st.median(vals),'mean':st.mean(vals),
            'min':min(vals),'max':max(vals),
            'population_sd':st.pstdev(vals)}


def run(root):
    t=read(root,'ONLINE_TIMING_194228.json.gz')
    q=read(root,'ONLINE_QUALITY_194188.json.gz')
    ts=read(root,'ONLINE_TIMING_SUMMARY.json')
    qs=read(root,'ONLINE_QUALITY_SUMMARY.json')
    traj=read(root,'ONLINE_TRAJECTORY_194188.json')['records']
    assert t['implementation_sha256']==q['implementation_sha256']
    assert len(t['renders'])==18 and len(q['renders'])==4
    assert ts['full_timing_receipt_sha256']==hashlib.sha256(payload(root/'ONLINE_TIMING_194228.json.gz')).hexdigest()
    assert qs['full_quality_receipt_sha256']==hashlib.sha256(payload(root/'ONLINE_QUALITY_194188.json.gz')).hexdigest()
    lookup={(r['source'],r['repeat'],r['arm']):r for r in t['renders']}
    qlookup={(r['source'],r['arm']):r for r in q['renders']}
    sources=t['registration']['sources']
    timing_rows=[]
    for r in t['renders']:
        rec=r['record']; gpu=sum(rec['step_ms'])/1000
        if r['arm']=='A':
            reference=t['control'][r['source']]['wide8192']
        else:
            reference=qlookup[r['source'],r['arm']]['record']
        for key in ('initial','flows','latents','video'):
            assert rec[key]==reference[key],(r['source'],r['repeat'],r['arm'],key)
        assert r['trace_tensors']=={}
        if r['arm']!='A':
            assert r['attention']['anchors']==80 and r['attention']['hits']==80
            assert r['attention']['native_probe_queries']==5120
        timing_rows.append({'source':r['source'],'repeat':r['repeat'],'arm':r['arm'],
            'wall_s':rec['wall'],'four_step_gpu_s':gpu,
            'wall_minus_four_step_gpu_s':rec['wall']-gpu,
            'step_s':[v/1000 for v in rec['step_ms']],
            'peak_allocated_GiB':rec['peak_allocated']/2**30,
            'output_matches_quality_or_A_archive':True})
    arms={}
    for arm in 'ABC':
        rows=[r for r in timing_rows if r['arm']==arm]
        arms[arm]={key:summary(r[key] for r in rows) for key in
            ('wall_s','four_step_gpu_s','wall_minus_four_step_gpu_s','peak_allocated_GiB')}
        arms[arm]['step_median_s']=[st.median(r['step_s'][i] for r in rows) for i in range(4)]
        assert abs(arms[arm]['wall_s']['median']-ts['timing'][arm]['wall_median_s'])<1e-12
        assert abs(arms[arm]['four_step_gpu_s']['median']-ts['timing'][arm]['gpu_four_steps_median_s'])<1e-12
        assert abs(arms[arm]['peak_allocated_GiB']['max']-ts['timing'][arm]['peak_gib'])<1e-12
    comparisons={}
    paired_rows=[]
    for cand,ref in [('B','A'),('C','A'),('B','C')]:
        pairs=[]
        for source in sources:
            for repeat in range(3):
                c=lookup[source,repeat,cand]['record']; a=lookup[source,repeat,ref]['record']
                cg,ag=sum(c['step_ms'])/1000,sum(a['step_ms'])/1000
                row={'source':source,'repeat':repeat,'candidate':cand,'reference':ref,
                    'wall_ratio':c['wall']/a['wall'],'wall_delta_s':c['wall']-a['wall'],
                    'gpu_ratio':cg/ag,'gpu_delta_s':cg-ag,
                    'step_delta_s':[(u-v)/1000 for u,v in zip(c['step_ms'],a['step_ms'])]}
                pairs.append(row);paired_rows.append(row)
        comp={k:summary(r[k] for r in pairs) for k in ('wall_ratio','wall_delta_s','gpu_ratio','gpu_delta_s')}
        comp['wall_faster_pairs']=sum(r['wall_ratio']<1 for r in pairs)
        comp['gpu_faster_pairs']=sum(r['gpu_ratio']<1 for r in pairs)
        comp['step_delta_median_s']=[st.median(r['step_delta_s'][i] for r in pairs) for i in range(4)]
        comp['step_delta_mean_s']=[st.mean(r['step_delta_s'][i] for r in pairs) for i in range(4)]
        assert abs(sum(comp['step_delta_mean_s'])-comp['gpu_delta_s']['mean'])<1e-12
        comp['by_source']={s:{k:summary(r[k] for r in pairs if r['source']==s) for k in
                                 ('wall_ratio','wall_delta_s','gpu_ratio','gpu_delta_s')} for s in sources}
        comparisons[cand+'/'+ref]=comp
        if ref=='A':
            assert abs(comp['wall_ratio']['median']-ts['timing'][cand]['paired_wall_ratio'])<1e-12
            for s in sources:
                assert abs(comp['by_source'][s]['wall_ratio']['median']-ts['timing'][cand]['source_wall_ratios'][s])<1e-12
    shared_prefix={}
    for source in sources:
        b,c=(qlookup[source,a] for a in ('B','C'))
        assert b['record']['initial']==c['record']['initial']
        assert b['record']['flows'][:3]==c['record']['flows'][:3]
        assert b['record']['latents'][:3]==c['record']['latents'][:3]
        route=lambda r:[(x['step'],x['layer'],x['support_sha256']) for x in r['attention_anchor_identity']]
        assert route(b)==route(c) and len(route(b))==80
        shared_prefix[source]={'initial':True,'flows_steps_0_to_2':True,
            'latents_steps_0_to_2':True,'support_fingerprints_equal':80,
            'anchor_tensor_values_hashed':False,
            'interpretation':'B/C share the exact incoming last-step trajectory and supports. Their low recovery features, coupled solver, augmented arithmetic and un-hashed anchor tensor values are not all established as identical.'}
    profile=ts['profile']
    components={}
    for arm in ('B','C'):
        assert profile[arm]['profile_not_used_for_formal_timing']
        a=profile[arm]['components']; low_total=a['selected_probe_recovery_total']['3']
        components[arm]={
            'context':profile[arm]['metric'],
            'components_overlap':True,
            'low_total_ms':low_total,
            'low_selected_statistics_ms':a['selected_and_statistics']['3'],
            'low_selected_statistics_fraction_of_total':a['selected_and_statistics']['3']/low_total,
            'low_probe_ms':a['native_P64_full_KV']['3'],
            'low_probe_fraction_of_total':a['native_P64_full_KV']['3']/low_total,
            'low_schur_ms':a['probe_schur_fit']['3'],
            'low_schur_fraction_of_total':a['probe_schur_fit']['3']/low_total,
            'low_anchor_selected_statistics_ms':a['selected_and_statistics']['2'],
        }
    b,c=profile['B']['components'],profile['C']['components']
    component_contrast={
        'low_selected_ratio_B_C':b['selected_and_statistics']['3']/c['selected_and_statistics']['3'],
        'low_total_ratio_B_C':b['selected_probe_recovery_total']['3']/c['selected_probe_recovery_total']['3'],
        'low_selected_difference_ms':b['selected_and_statistics']['3']-c['selected_and_statistics']['3'],
        'low_total_difference_ms':b['selected_probe_recovery_total']['3']-c['selected_probe_recovery_total']['3'],
        'selected_difference_over_total_difference':(b['selected_and_statistics']['3']-c['selected_and_statistics']['3'])/(b['selected_probe_recovery_total']['3']-c['selected_probe_recovery_total']['3']),
        'counterfactual_40_times_selected_anchor_hit_difference_s':40*((b['selected_and_statistics']['2']-c['selected_and_statistics']['2'])+(b['selected_and_statistics']['3']-c['selected_and_statistics']['3']))/1000,
        'counterfactual_scope':'Multiplying per-layer medians is descriptive budget arithmetic, not an achieved backend change or an additive sum of nested events.'}
    distance=[]
    for r in qs['comparisons']:
        assert abs(20*math.log10(255/r['rgb_rmse'])-r['rgb_psnr_db'])<1e-12
        distance.append(r|{'rgb_rmse_fraction_of_255':r['rgb_rmse']/255})
    return {'pinned_head':HEAD,'execution_scope':'Published receipt/hash and arithmetic recomputation only; no model, video decode, tensor refit, or new hardware timing.',
        'human_review':{'quote':USER_OBSERVATION,'source':'Direct user message in current conversation','assistant_independent_visual_review':False,
                        'reviewed_source_ids':'Not separately enumerated in the user statement; retain the reported three-method comparison without inventing per-source scores.',
                        'scope':'User reports three-method comparison nearly indistinguishable. Does not establish a blind randomized equivalence test or a Dense quality judgment.'},
        'statistical_unit':'Two reused DEV sources, seed42; three timing repeats per arm/source are repeated measures, not six independent sources.',
        'identity':{'quality_job':q['job'],'timing_job':t['job'],'gpu':t['gpu'],'implementation_sha256':t['implementation_sha256'],
                    'timing_outputs_verified_against_quality_or_A_archive':18,'same_implementation_quality_timing':True,
                    'paired_same_repeat':True,'shared_prefix_B_C':shared_prefix},
        'raw_sha256':{n:hashlib.sha256(payload(root/n)).hexdigest() for n in
            ('ONLINE_TIMING_194228.json.gz','ONLINE_QUALITY_194188.json.gz','ONLINE_TRAJECTORY_194188.json')},
        'arms':arms,'comparisons':comparisons,'paired_rows':paired_rows,'timing_rows':timing_rows,
        'profile':components,'profile_B_C_contrast':component_contrast,
        'quality_metrics':distance,'trajectory_metrics':traj,
        'omitted_mass_fallback':[r|{'fraction':r['count']/r['denominator']} for r in qs['omitted_mass_recomputations']],
        'quality_capture_wall_s':[{k:r[k] for k in ('arm','source')}|{'wall_s':r['record']['wall']} for r in q['renders']],
        'quality_capture_timing_use':'Do not mix with the separate formal timing job.',
        'setup':ts['setup'],
        'interpretation_limits':['Profile components overlap and cannot be naively added.',
            'wall minus four-step GPU events combines all unpartitioned work; the receipt does not assign it specifically to decode, I/O, synchronization or CPU.',
            'The measured B/C selected-statistics difference bundles the actual implementations; it is not an isolated universal cost of the finite mathematical identity.',
            'Human perceptual acceptance is retained as the task outcome; paired pixel/latent distances describe changed trajectories.']}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'online_evidence_recomputed.json')
    a=ap.parse_args();result=run(a.source)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'identity':result['identity'],'comparisons':result['comparisons'],
        'profile_B_C_contrast':result['profile_B_C_contrast']},ensure_ascii=False,indent=2))
