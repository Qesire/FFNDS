"""Single-group saved-h CPU diagnostic; registered scope only, no model forward."""
import argparse, hashlib, json, os, pathlib, resource, sys, time, traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]
ID='E0_CHIPMUNK_SAVED_H_DISCRIMINATION_20261030'
OUTPUT='ffnds-derived/e0_chipmunk_saved_h_discrimination_20261030/attempt1/worker/results'

def need(x,msg):
    if not x: raise ValueError(msg)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def fingerprint(p):
    s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def checked(root,ref):
    p=(root/ref['path']).resolve();need(p.is_relative_to(root.resolve()) and p.is_file(),'asset path');before=fingerprint(p);need(before[2]==ref['bytes'] and sha(p)==ref['sha256'] and fingerprint(p)==before,'asset hash or identity');return p
def atomic(p,obj):
    data=(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode();need(len(data)<=4194304,'output limit');tmp=p.with_suffix('.tmp');tmp.write_bytes(data);os.replace(tmp,p)
def authorize(p,digest):
    need(p['id']==ID and p['assets_finalized'] is True and p['output']==OUTPUT,'scope not finalized')
    active=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text())
    need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True,'authorization revoked')
    s=[a for a in active['experiment_authority']['current_user_authorized_scopes'] if a.get('id')==ID]
    need(len(s)==1 and s[0].get('protocol_sha256')==digest and s[0].get('execution_status')=='RUNNING_FIXED_SCOPE' and s[0].get('max_gpu_cards')==0,'scope not registered')
    review=checked(ROOT,s[0]['review']); verdict=json.loads(review.read_text()); need(verdict.get('state')=='REVIEWER_RECHECK_APPROVED' and verdict.get('scope_id')==ID and verdict.get('protocol_sha256')==digest and verdict.get('job_spec_sha256')==s[0].get('job_spec_sha256') and verdict.get('author_agent_id') and verdict.get('reviewer_agent_id') and verdict['author_agent_id']!=verdict['reviewer_agent_id'] and verdict.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['repo_sources']} and verdict.get('adversarial_counterexample') and verdict.get('responses_to_review'),'independent recheck missing')
    return s[0]
def output_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())

def main():
    soft,hard=resource.getrlimit(resource.RLIMIT_FSIZE); need(hard == resource.RLIM_INFINITY or hard >= 4194304,'file limit too small'); resource.setrlimit(resource.RLIMIT_FSIZE,(4194304,hard))
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=pathlib.Path,required=True);ap.add_argument('--hf-root',type=pathlib.Path,required=True);ap.add_argument('--checkpoint',type=pathlib.Path,required=True);ap.add_argument('--job-spec',type=pathlib.Path,required=True);a=ap.parse_args()
    plan=json.loads(a.protocol.read_text());scope=authorize(plan,sha(a.protocol));need(sys.flags.isolated==1,'isolated interpreter required');need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be hidden');need(sha(a.job_spec)==scope.get('job_spec_sha256'),'job SHA');job=json.loads(a.job_spec.read_text());need(job['protocol_sha256']==sha(a.protocol) and job['id']==ID and pathlib.Path(job['checkpoint']).resolve()==a.checkpoint.resolve() and pathlib.Path(job['hf_root']).resolve()==a.hf_root.resolve(),'job mismatch');need(pathlib.Path(job['python']).resolve()==pathlib.Path(sys.executable).resolve() and job['cwd']==os.getcwd() and job['driver_argv']==sys.argv and job['external_timeout_seconds']==1920,'job argv or interpreter mismatch');need(sorted(os.sched_getaffinity(0))==job['cpu_affinity'] and len(job['cpu_affinity'])==2 and os.environ.get('OPENBLAS_NUM_THREADS')=='2' and os.environ.get('OMP_NUM_THREADS')=='2','CPU limit')
    out=(a.hf_root/OUTPUT);need(not out.exists(),'attempt already exists');out.mkdir(parents=True);report={'id':ID,'status':'ATTEMPT','counts':{},'inflight':None};atomic(out/'ATTEMPT.json',report)
    started=time.monotonic()
    def charge(name):
        authorize(plan,sha(a.protocol));need(time.monotonic()-started<plan['wall_seconds'],'timeout');need(report['counts'].get(name,0)<plan['operation_limits'][name],'operation limit');report['counts'][name]=report['counts'].get(name,0)+1;report['inflight']=name;atomic(out/'RUN.json',report)
    def done():report['inflight']=None;atomic(out/'RUN.json',report)
    try:
        report['worker_output_bytes_start']=output_bytes(out.parent);need(report['worker_output_bytes_start']<=plan['resources']['output_bytes_max']-4194304,'worker output budget');atomic(out/'RUN.json',report)
        report['source_hash_read_bytes_predebited']=plan['read_budget']['repo_source_hash_bytes'];report['source_hash_read_bytes_completed']=0;report['source_hash_read_bytes_inflight']='UNKNOWN';atomic(out/'RUN.json',report)
        for x in plan['repo_sources']:
            charge('source_hash');checked(ROOT,x);report['source_hash_read_bytes_completed']+=x['bytes'];done()
        report['source_hash_read_bytes_inflight']='NONE';atomic(out/'RUN.json',report)
        paths=[]; stats=[]
        for x in plan['hf_assets']:
            charge('hf_hash');paths.append(checked(a.hf_root,x));stats.append(paths[-1].stat());done()
        w=a.checkpoint.resolve();need(w.is_file() and w.stat().st_size==plan['checkpoint']['bytes'],'checkpoint size');st=fingerprint(w);charge('checkpoint_hash');need(sha(w)==plan['checkpoint']['sha256'] and fingerprint(w)==st,'checkpoint SHA or identity');done()
        import numpy as np
        import torch
        from safetensors import safe_open
        import importlib.util
        spec=importlib.util.spec_from_file_location('chipmunk_diagnostic_bound',ROOT/'tools/chipmunk_proxy_diagnostic_math.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);diagnose=module.diagnose
        torch.set_num_threads(2)
        metas=[json.loads(paths[i].read_text()) for i in (0,1)];rows=metas[0]['selected_rows'];need(rows==metas[1]['selected_rows'] and len(rows)==1092 and len(set(rows))==1092,'row identity')
        prior=json.loads(paths[4].read_text());need(prior['row_binding']['selected_rows']==rows[:128],'old proxy rows differ'); scores=prior['proxy_scores']['values'];mask=prior['mask_bindings']['proxy']['indices'];need(len(scores)==13824 and len(mask)==5120,'old proxy identity')
        hs=[]; gate=None
        need(all(m.get('step')==i and m.get('block')==20 and m.get('native_source_ordinal')==898 and m.get('seed')==42 and m.get('expert')=='high' and m.get('payload',{}).get('sha256')==plan['hf_assets'][2+i]['sha256'] for i,m in enumerate(metas)),'metadata identity')
        for i in (0,1):
            charge('payload_tensor_read')
            need(paths[2+i].stat()==stats[2+i],'payload changed before read')
            with safe_open(str(paths[2+i]),framework='pt',device='cpu') as f:
                ids=f.get_tensor('rows');h=f.get_tensor('h');g=f.get_tensor('gate')
                need(ids.tolist()==rows and tuple(h.shape)==(1092,13824) and h.dtype==torch.bfloat16 and tuple(g.shape)==(5120,) and g.dtype==torch.bfloat16,'payload shape')
                hs.append(h[:128].float().numpy().astype(np.float64));gate=g.float().numpy().astype(np.float64)
            need(paths[2+i].stat()==stats[2+i],'payload changed during read');done()
        charge('weight_tensor_read')
        with safe_open(str(w),framework='pt',device='cpu') as f:
            t=f.get_tensor('blocks.20.ffn.2.weight');need(tuple(t.shape)==(5120,13824) and t.dtype==torch.bfloat16,'weight shape'); weight=t.float().numpy().astype(np.float64)
        need(fingerprint(w)==st,'checkpoint changed');done()
        charge('diagnostic');result=diagnose(hs[1]-hs[0],weight,gate,mask);done()
        need(report['counts']==plan['operation_limits'],'incomplete operation ledger');report['worker_output_bytes_end_before_final_RUN']=output_bytes(out.parent);need(report['worker_output_bytes_end_before_final_RUN']<=plan['resources']['output_bytes_max']-4194304,'worker output budget');result['proxy_scores']=scores;report.update(status='COMPLETE_DIAGNOSTIC',result=result,wall_seconds=time.monotonic()-started,source=898,steps=[0,1],rows=rows[:128],note='Nonraster teacher group; oracle rankings are retrospective, not online; no video or speed claim.')
    except BaseException as e:
        report['worker_output_bytes_at_failure']=output_bytes(out.parent);report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started);atomic(out/'RUN.json',report);raise
    atomic(out/'RUN.json',report)
if __name__=='__main__':main()
