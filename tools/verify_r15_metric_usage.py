#!/usr/bin/env python3
"""Use R15 metrics on frozen scalar results and synthetic mathematical fixtures.

No Wan execution, no actual video, no official evaluator, no GPU/network.
"""
import argparse
import json
import math
from pathlib import Path
from audit_r15_metric_evidence import audit
from check_metric_contract import catalog, validate_catalog, validate_record

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'research/20261023/R16_metric_usage_acceptance.json'

def require(condition, message):
    if not condition: raise AssertionError(message)

def paired_mse(a,b):
    require(len(a)==len(b) and len(a)>0,'Nonempty aligned frame pixels required')
    require(all(math.isfinite(float(x)) for x in (*a,*b)),'Nonfinite pixels')
    return math.fsum((float(x)-float(y))**2 for x,y in zip(a,b))/len(a)

def paired_psnr(mse,peak):
    require(math.isfinite(mse) and mse>=0 and math.isfinite(peak) and peak>0,'Bad PSNR definition')
    return None if mse==0 else 10*math.log10(peak**2/mse)

def spatial_mean_contrast(errors):
    require(bool(errors) and all(math.isfinite(float(x)) for x in errors),'Bad stripe')
    mu=math.fsum(errors)/len(errors)
    total=math.fsum(x*x for x in errors)
    shared=len(errors)*mu**2
    centered=math.fsum((x-mu)**2 for x in errors)
    require(math.isclose(total,shared+centered,rel_tol=1e-12,abs_tol=1e-12),'Mean/contrast identity failed')
    return {'SSE':total,'mean_response_part':shared,'within_stripe_part':centered}

def static_roi_temporal_mae(frames,mask):
    require(len(frames)>=2 and bool(mask) and any(mask),'Stable ROI and >=2 frames required')
    require(all(len(f)==len(mask) for f in frames),'Misaligned ROI')
    require(all(math.isfinite(float(v)) for f in frames for v in f),'Nonfinite ROI')
    n=sum(bool(x) for x in mask)
    return math.fsum(math.fsum(abs(float(b)-float(a)) for a,b,valid in
                      zip(frames[i],frames[i+1],mask) if valid)/n
                     for i in range(len(frames)-1))/(len(frames)-1)

def source_aggregation(reference,candidate):
    require(len(reference)==len(candidate) and bool(reference),'Paired source strata required')
    require(all(math.isfinite(float(a)) and math.isfinite(float(b)) and a>0 and b>=0 for a,b in zip(reference,candidate)),'Bad source losses')
    return {'ratio_of_sums':math.fsum(candidate)/math.fsum(reference),
            'equal_source_mean_ratio':math.fsum(b/a for a,b in zip(reference,candidate))/len(reference)}

def run():
    c=catalog()
    require(not validate_catalog(c),'Invalid operational catalog')
    academic=json.loads((ROOT/'research/metric_system/academic_metric_families.json').read_text(encoding='utf8'))
    require((len(academic['families']),len(academic['metrics']))==(10,93),'93/10 academic families changed')
    require(all(m['measurement_status']=='DEFINITION_ONLY' for m in academic['metrics']), 'Definition misrepresented as measured')
    require(len({m['id'] for m in academic['metrics']})==93,'Duplicate academic ID')
    a=audit()
    frozen=json.loads((ROOT/'research/20261023/metric_evidence_recalculation_R15.json').read_text(encoding='utf8'))
    for key in ('per_source','pooled','relative_local_changes','decomposition','aggregation','backend'):
        require(a[key]==frozen[key], 'R15 scalar audit changed: '+key)
    require(math.isclose(a['relative_local_changes']['dynamic'],.31164172660792255,rel_tol=1e-12),'R14N0 local +31.16% drift')
    directory=ROOT/'research/metric_system/records'
    records={}
    for file in ('R15_R14N0_local_mapping.json','R15_R14_backend_mapping.json'):
        r=json.loads((directory/file).read_text(encoding='utf8'))
        require(not validate_record(r,c), 'Invalid mapped record: '+file)
        records[file]=r
    sources={s['source_index']:s for s in a['per_source']}
    checked=0
    for m in records['R15_R14N0_local_mapping.json']['observations']:
        if m['metric_id'] not in ('post_writeback_sse','post_writeback_mse'): continue
        source,arm=m['observation_id'].split('/')
        idx=int(source.removeprefix('source'))
        column='SSE' if m['metric_id']=='post_writeback_sse' else 'MSE'
        expected=sources[idx]['statistics'][arm][column]
        require(math.isclose(m['value'],expected,rel_tol=1e-12,abs_tol=1e-12),'L1 mapping changed')
        checked+=1
    require(checked>=10, 'Too few local observations actually checked')
    back=records['R15_R14_backend_mapping.json']['observations'][0]
    times=a['backend']['synthetic_same_run_ms']
    require(back['metric_id']=='synthetic_backend_speedup' and back['execution_mode']=='synthetic','Synthetic speed misclassified')
    require(math.isclose(back['value'],times['dense']/times['three_fused_dynamic'],rel_tol=1e-12),'Speed fraction wrong')
    require(a['backend']['native_model_speedup'] is None,'Invented native speed')
    fixture_mse=[paired_mse([0,0],[0,1]),paired_mse([0,0],[1,1])]
    frame_psnr=sum(paired_psnr(e,1.) for e in fixture_mse)/2
    pooled_psnr=paired_psnr(sum(fixture_mse)/2,1.)
    require(fixture_mse==[.5,1] and frame_psnr>pooled_psnr,'Wrong PSNR aggregation')
    require(paired_psnr(0,1.) is None,'Perfect pixel match infinity not represented')
    shared=spatial_mean_contrast([1,1,1,1]); scattered=spatial_mean_contrast([1,-1,1,-1])
    require(shared['SSE']==scattered['SSE']==4 and shared['mean_response_part']==4 and scattered['within_stripe_part']==4,'SSE hides spatial structure')
    base=[[.4,0,.4]]*4
    flicker=[[.4,0,.4],[.6,0,.6],[.4,0,.4],[.6,0,.6]]
    mask=[True,False,True]
    base_diff=static_roi_temporal_mae(base,mask)
    flicker_diff=static_roi_temporal_mae(flicker,mask)
    require(base_diff==0 and math.isclose(flicker_diff,.2,abs_tol=1e-12),'Stable background flicker fixture failed')
    agg=source_aggregation([1,100],[2,90])
    require(math.isclose(agg['ratio_of_sums'],92/101) and math.isclose(agg['equal_source_mean_ratio'],1.45),'Pooled and cluster aggregation confused')
    official={'i2v_subject','i2v_background','i2v_camera_motion','vbench_subject_consistency','vbench_background_consistency','vbench_motion_smoothness','vbench_dynamic_degree','vbench_aesthetic','vbench_imaging'}
    require(official.issubset({m['id'] for m in academic['metrics']}),'Official I2V dimension missing')
    observed_levels=sorted({m['level'] for r in records.values() for m in r['observations']})
    require(observed_levels==['L1','L5'],'Unexpected new video evidence requires separate review')
    return {
      'id':'R16_R15_METRIC_USAGE_ACCEPTANCE','status':'OFFLINE_METRIC_USAGE_PASS',
      'scope':'Frozen R14N0/R14 scalar evidence and explicitly synthetic math fixture only; no actual video, official VBench or native model execution',
      'academic_definitions':93,'academic_families':10,'operational_metric_ids':len(c['metrics']),
      'observed_levels':observed_levels,
      'frozen_native_local':{'development_sources':[0,10],'numeric_observations_checked':checked,
          'dynamic_to_UB1_SSE_ratio':a['pooled']['dynamic']/a['pooled']['UB1'],
          'equal_source_mean_ratio':a['aggregation']['equal_source_mean_ratio']},
      'synthetic_backend':{'same_run_speedup':back['value'],'actual_wan_model_speedup':None},
      'synthetic_math_fixtures':{'per_frame_mse':fixture_mse,'mean_frame_psnr_db':frame_psnr,
          'psnr_of_pooled_mse_db':pooled_psnr,'perfect_match_psnr':'POSITIVE_INFINITY',
          'shared_stripe':shared,'alternating_stripe':scattered,'stable_roi_baseline_mae':base_diff,
          'stable_roi_flicker_mae':flicker_diff,'source_aggregation':agg,'not_official_vbench':True},
      'capabilities':{'existing_native_scalar_reaggregation':'REPRODUCED_SAVED_MEASUREMENT',
          'paired_pixel_math':'SYNTHETIC_MATH_ONLY','temporal_roi_math':'SYNTHETIC_MATH_ONLY_NOT_OFFICIAL_FLICKER',
          'official_nine_dimensional_i2v':'REQUIRES_GENERATED_VIDEOS_OFFICIAL_EVALUATOR',
          'human_quality_collapse':'REQUIRES_PAIRED_FULL_FREE_RUNNING_VIDEO',
          'actual_native_wan_speed':'REQUIRES_MODEL_AND_HARDWARE_EXECUTION'},
      'independent_video_quality':None,'new_model_generations':0,'new_native_ffn_calls':0,
      'agent_next':'Select a distinct candidate; execute approved paired full four-step free-running I2V video and full-duration noncollapse review; then official nine-dimensional per-source-manifest benchmark, genuine native speed and same-setting quality-speed Pareto.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true',help='Require frozen committed deterministic acceptance report')
    p.add_argument('--output',type=Path,default=REPORT)
    args=p.parse_args()
    result=run()
    if args.check:
        require(result==json.loads(REPORT.read_text(encoding='utf8')),'Committed usage report does not match source recomputation')
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'checked':result['frozen_native_local']['numeric_observations_checked'],
                      'academic_definitions':93,'official_video_scores':0,'mode':'CHECK' if args.check else 'WRITE'}))

if __name__=='__main__': main()
