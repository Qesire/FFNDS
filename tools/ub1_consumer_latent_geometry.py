"""Frozen saved-latent signed geometry; NumPy CPU only, no model or flow."""
import argparse, hashlib, json, math, os, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ID='UB1_CONSUMER_LATENT_GEOMETRY_20261027'
SHAPE=(16,21,58,104)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()

def stats(dense, neutral, candidate):
    import numpy as np
    arrays=[np.asarray(x,dtype=np.float64) for x in (dense,neutral,candidate)]
    if any(x.shape!=arrays[0].shape or not np.isfinite(x).all() for x in arrays): raise ValueError('shape or finite mismatch')
    z,n,c=arrays;u=n-z;v=c-z;delta=c-n
    inner=lambda a,b:float(np.dot(a.ravel(),b.ravel()))
    ue=inner(u,u);ce=inner(v,v);de=inner(delta,delta);cross=2*inner(u,delta);ze=inner(z,z)
    discrepancy=ce-(ue+de+cross)
    if not math.isclose(ce,ue+de+cross,rel_tol=1e-11,abs_tol=1e-10*max(1.,ue,de,ce)): raise ValueError('signed decomposition mismatch')
    return {'dense_energy':ze,'ub1_error_energy':ue,'candidate_error_energy':ce,'correction_energy':de,'twice_cross':cross,'cos_ub1_correction':cross/(2*math.sqrt(ue*de)) if ue and de else None,'ub1_error_over_dense_energy':ue/ze if ze else None,'candidate_error_over_dense_energy':ce/ze if ze else None,'candidate_over_ub1_error_norm':math.sqrt(ce/ue) if ue else None,'closure_error':discrepancy}

def check_identity(case,mode,run,phase,generation,core):
    if run['status']!='PASS_FIXED_FULL_DEPTH_CONSUMER_EXECUTION_CORE_METRICS_QUALITY_PENDING': raise ValueError('Run status mismatch')
    if run['case']['index']!=case['index'] or run['case']['seed']!=case['seed'] or run['case']['source_id']!=case['source_id']: raise ValueError('Run source/seed mismatch')
    if phase['mode']!=mode or generation['mode']!=mode or phase['source_id']!=case['source_id'] or generation['source_id']!=case['source_id'] or phase['seed']!=case['seed'] or generation['seed']!=case['seed']: raise ValueError('Phase source/seed mismatch')
    if phase['prompt_sha256']!=case['prompt_sha256'] or generation['prompt_sha256']!=case['prompt_sha256'] or phase['trajectory']!=generation['trajectory']: raise ValueError('Phase/generation mismatch')
    matched=[x for x in run['phases'] if x['mode']==mode]
    if len(matched)!=1 or matched[0]['trajectory']!=phase['trajectory']: raise ValueError('Run trajectory mismatch')
    identity=core['identity'];artifact=core['metrics']['artifact']
    if identity['source_id']!=case['source_id'] or identity['seed']!=case['seed'] or identity['input_prompt_digest']!=case['prompt_sha256'] or identity['expected_frames']!=81 or artifact['verified_full_decode_frames']!=81 or artifact['frame_shape']!=[464,832,3]: raise ValueError('Core identity/frame mismatch')
    if identity['protocol_sha256']!=run['protocol_sha256'] or artifact['candidate_sha256']!=phase['video']['sha256'] or generation['video']['sha256']!=phase['video']['sha256'] or phase['video']['frames']!=81: raise ValueError('Core video/protocol mismatch')
    if len(phase['trajectory'])!=4 or [x['step'] for x in phase['trajectory']]!=list(range(4)): raise ValueError('Step identity mismatch')
    return phase['trajectory']

def main():
    import numpy as np
    from safetensors.numpy import load_file
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--hf-root',type=Path,required=True);args=ap.parse_args();plan=json.loads(args.protocol.read_text());digest=sha(args.protocol);hf=args.hf_root.resolve()
    if plan['id']!=ID or sha(__file__)!=plan['tool_sha256'] or sha(ROOT/plan['test'])!=plan['test_sha256']: raise ValueError('Frozen tool/test identity mismatch')
    active=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()); scopes=active['experiment_authority']['current_user_authorized_scopes']
    if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in scopes):raise ValueError('Exact registered scope required')
    if plan['limits']!={'cpu_threads':1,'wall_seconds':60,'memory_bytes':536870912,'output_bytes':4194304,'attempts':1,'gpu':0,'model_calls':0}:raise ValueError('Limits mismatch')
    if len(plan['cases'])!=2 or {(x['index'],x['seed']) for x in plan['cases']}!={(0,42),(10,42)}:raise ValueError('Case scope mismatch')
    for item in plan['inputs']:
        path=(hf/item['path']).resolve()
        if not path.is_relative_to(hf) or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Input SHA mismatch')
    out=(hf/plan['output_HF_relative']).resolve()
    if not out.is_relative_to(hf) or out==hf:raise ValueError('Output scope mismatch')
    out.mkdir(parents=True,exist_ok=False)
    with (out/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest},f)
    start=time.monotonic();result={'id':ID,'status':'STARTED','protocol_sha256':digest,'cases':[],'scope':'saved post-step latent geometry only; no initial latent, flow, video causality or quality acceptance'}
    try:
        for case in plan['cases']:
            root=hf/case['HF_relative'];run=json.loads((root/'RUN.json').read_text()); trajectories={}
            for mode in ('dense','neutral','candidate'):
                read=lambda name:json.loads((root/mode/name).read_text())
                phase=read('PHASE.json');generation=read('GENERATION.json');core=read('CORE_METRICS.json')
                if core['identity']['run_sha256']!=sha(root/mode/'GENERATION.json'):raise ValueError('Core generation digest mismatch')
                trajectories[mode]=check_identity(case,mode,run,phase,generation,core)
            for step in range(4):
                values={}
                for mode in ('dense','neutral','candidate'):
                    info=trajectories[mode][step]
                    if info['shape']!=list(SHAPE) or info['dtype']!='float32' or info['path']!=f'latent_step{step}.safetensors':raise ValueError('Tensor metadata mismatch')
                    path=root/mode/info['path'];raw=load_file(str(path))
                    if set(raw)!={'latents'}:raise ValueError('Tensor key mismatch')
                    array=raw['latents']
                    if array.shape!=SHAPE or array.dtype!=np.float32 or not np.isfinite(array).all() or hashlib.sha256(array.tobytes(order='C')).hexdigest()!=info['raw_tensor_sha256']:raise ValueError('Raw tensor identity mismatch')
                    if sha(path)!=info['file_sha256'] or path.stat().st_size!=info['bytes']:raise ValueError('Tensor file identity mismatch')
                    values[mode]=array
                slots=[stats(*(values[k][:,t,:,:] for k in ('dense','neutral','candidate'))) for t in range(21)]
                # Whole-state energies and cross are sums of the disjoint time slices.
                totals={k:sum(x[k] for x in slots) for k in ('dense_energy','ub1_error_energy','candidate_error_energy','correction_energy','twice_cross','closure_error')}
                ue,ce,de,ze,cross=(totals[k] for k in ('ub1_error_energy','candidate_error_energy','correction_energy','dense_energy','twice_cross'))
                totals.update(cos_ub1_correction=cross/(2*math.sqrt(ue*de)) if ue and de else None,ub1_error_over_dense_energy=ue/ze if ze else None,candidate_error_over_dense_energy=ce/ze if ze else None,candidate_over_ub1_error_norm=math.sqrt(ce/ue) if ue else None)
                if not math.isclose(ce,ue+de+cross,rel_tol=1e-11,abs_tol=1e-10*max(1.,ue,de,ce)):raise ValueError('Whole-state closure mismatch')
                result['cases'].append({'index':case['index'],'source_id':case['source_id'],'seed':case['seed'],'step':step,'whole':totals,'latent_time_slots':slots})
                if time.monotonic()-start>plan['limits']['wall_seconds']:raise TimeoutError('Wall limit')
        result.update(status='COMPLETE_SAVED_LATENT_GEOMETRY',elapsed_seconds=time.monotonic()-start)
    except BaseException as e:
        result.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        (out/'RUN.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
