"""Check all160 complete FFN regions and compare sums of all measured repeats."""
import argparse
import json
from pathlib import Path
import statistics

from ar2_checkpoint import atomic_json,digest


def validate_candidate_runtime(packed,policy_sha,*,require_timing=True):
    expected={(s,b) for s in range(4) for b in range(40)}
    expected_calls={f'step{s}_block{b}_ffn{slot}' for s,b in expected for slot in [0,2]}
    residual=packed['residual_correction']
    if (packed['policy_sha256']!=policy_sha or set(packed['coverage'])!=expected_calls
            or not residual['active'] or residual['actual_positions']!=160 or residual['pending_predictions']
            or len(residual['coverage'])!=160
            or (require_timing and not residual['timing_includes_auxiliary_cost'])
            or any(r['auxiliary_linear_calls']!=2 for r in residual['coverage'].values())):
        raise RuntimeError('Candidate does not execute all main and auxiliary GEMMs')
    if packed['policy']['kind']=='STEP_BUDGET_PACKED_REFERENCE':
        from step_budget_runtime import validate_budget
        groups=validate_budget(packed['policy'])
    else:
        groups=[108]*4
    numerator=denominator=0
    for key,row in packed['coverage'].items():
        step=int(key.split('_')[0][4:]);width=groups[step]*64
        shape=[width,5120] if key.endswith('ffn0') else [5120,width]
        if (row['actual_weight_shape']!=shape or row['main_GEMM_density']!=width/13824
                or type(row['rows']) is not int or row['rows']<=0):
            raise RuntimeError('Actual main GEMM dimensions differ from frozen per-step widths')
        numerator+=row['rows']*shape[0]*shape[1]
        denominator+=row['rows']*5120*13824
    if numerator*2!=denominator:
        raise RuntimeError('Actual row-weighted total main GEMM MAC density is not50%')
    if (packed['policy']['kind']=='STEP_BUDGET_PACKED_REFERENCE'
            and packed.get('actual_global_main_MAC_density')!=.5):
        raise RuntimeError('Budget runtime does not report complete actual50% main compute')
    for s,b in expected:
        row=residual['coverage'][f'step{s}_block{b}']
        if (row['auxiliary_GEMM_shapes']!=[[256,5120],[5120,256]]
                or row['rows']!=packed['coverage'][f'step{s}_block{b}_ffn0']['rows']):
            raise RuntimeError('Auxiliary GEMM shapes or rows differ from frozen rank256')
    if packed['policy']['kind']=='OUTPUT_VARIANCE_CALIBRATED_REFERENCE':
        affine=packed.get('output_affine',{})
        rows=affine.get('coverage',[])
        if (affine.get('identity_control',True) or affine.get('actual_positions')!=160 or len(rows)!=160
                or {(r['step'],r['block']) for r in rows}!=expected
                or (require_timing and not affine.get('timing_includes_affine_cost',False))
                or any(r['pointwise_kernel_calls']!=1
                    or (require_timing and r['coefficient_H2D_bytes']!=0) for r in rows)
                or affine.get('kernel_source_sha256')!=packed['policy']['kernel_source_sha256']):
            raise RuntimeError('Output affine candidate lacks full160 frozen native kernel costs')


def load_run(path,candidate):
    meta=json.loads(path.read_text())
    if (meta['status']!='PASS_RESIDENT_FULL160_AND_PAIRED_LATENT_EQUIVALENCE'
            or meta['identity']['measured_repeats']!=3 or len(meta['requests'])!=4
            or not meta['resources']['isolation_pass'] or meta['resources']['violations']
            or meta['raw_placeholder_audit']['fallback_calls']):
        raise RuntimeError('Complete isolated warmup+3-repeat run required')
    times=[];peaks=[]
    for index,row in enumerate(meta['requests']):
        if digest(row['report'])!=row['sha256'] or row['warmup']!=(index==0):
            raise RuntimeError('Repeat record identity or warmup designation changed')
        request=json.loads(Path(row['report']).read_text())
        timing=request['FFN_timing']
        core=[r for r in timing['rows'] if r['kind']=='FFN_norm_through_gate_residual']
        expected={(s,b) for s in range(4) for b in range(40)}
        if (not request['four_latents_bitwise_equal'] or len(core)!=160
                or {(r['step'],r['block']) for r in core}!=expected
                or not timing['conservative_shared_preparation_and_all_prefetch_charged']
                or not abs(sum(r['wall_ms'] for r in timing['rows'])-timing['wall_ms'])<1e-6):
            raise RuntimeError('Repeat lacks complete native timing/equivalence')
        if (request['hif4_counter_delta']['cache_misses'] or request['hif4_counter_delta']['weight_transfers']
                or request['hif4_counter_delta']['linear_forward_calls']!=1600
                or request['hif4_counter_delta']['activation_qdq_calls']!=1600):
            raise RuntimeError('Incomplete native bridge coverage')
        if candidate:
            validate_candidate_runtime(request['hif4_runtime']['packed_reference'],meta['identity']['policy_sha256'])
        if index:
            times.append(timing['wall_ms']);peaks.append(request['peak_cuda_memory_bytes'])
    return meta,times,peaks


def describe(values):
    return {'repeats_ms':values,'sum_ms':sum(values),'mean_ms':statistics.mean(values),
        'sample_std_ms':statistics.stdev(values),'min_ms':min(values),'max_ms':max(values)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dense',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    dense,dt,dm=load_run(args.dense,False);candidate,ct,cm=load_run(args.candidate,True)
    ignored={'reference_meta_sha256','reference_raw_latents','policy_sha256'}
    if ({k:v for k,v in dense['identity'].items() if k not in ignored}
            !={k:v for k,v in candidate['identity'].items() if k not in ignored}
            or dense['identity']['policy_sha256'] is not None or not candidate['identity']['policy_sha256']):
        raise RuntimeError('Paired benchmark sources/model/input/device/precision/protocol differ')
    speed=sum(dt)/sum(ct)
    result={'status':'COMPLETED_FULL160_FFN_PAIRED_THREE_REPEAT_BENCHMARK',
        'dense':describe(dt),'candidate':describe(ct),'full_FFN_wall_speed':speed,
        'full_FFN_speed_pass':speed>=1.5,'minimum_speed':1.5,'primary_main_GEMM_density':.5,
        'auxiliary_GEMMs_per_position':2,'positions_per_repeat':160,'repeats_per_arm':3,
        'all_online_auxiliary_and_conservative_shared_prep_costs_included':True,
        'dense_peak_cuda_memory_bytes':max(dm),'candidate_peak_cuda_memory_bytes':max(cm),
        'candidate_policy_sha256':candidate['identity']['policy_sha256'],
        'dense_run_sha256':digest(args.dense),'candidate_run_sha256':digest(args.candidate),
        'source_sha256':digest(Path(__file__)),'formal20_quality_acceptance':False,'joint_project_acceptance':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
