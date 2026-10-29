"""Bounded TEAL-style post-GELU CPU local diagnostic; not TEAL kernel/video.
Draft is metadata-only. Run requires independent recheck and exact ACTIVE scope.
"""
import argparse, copy, hashlib, importlib.metadata, importlib.util, json, math, os, resource, signal, sys, time, threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('_teal_cpu_bridge',ROOT/'tools/e0_s2_whole_ffn_boundary_bridge.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
native,vm=bridge.native,bridge.vm
ID='E0_TEAL_WAN_ALL160_CPU_LOCAL_20261029'
SELF='tools/e0_teal_wan_all160_cpu_local.py';TEST='tests/test_e0_teal_wan_all160_cpu_local.py'
CORPUS='research/20261029/ORIGINAL_BF16_FULL_DEPTH_CORPUS_RESULTS_20261029.json'
MASTER_TEMPLATE='config/e0_s2_new12_fixed_depth_part0_20261029_v1.json'
BRIDGE_CLOSURE='config/e0_s2_whole_ffn_boundary_bridge_20261029_v1.json'
OUT='ffnds-derived/e0_teal_wan_all160_cpu_local_20261029/attempt1/worker/results'
SOURCES=(894,256);CAL=(897,896,240,899,288,304);D=13824;RATES=(0.25,0.5)
RESOURCES={'GPU':0,'CPU_threads':4,'wall_seconds':3600,'outer_seconds':3900,'host_RSS_max_bytes':8<<30,'new_HF_bytes':128<<20,'automatic_retry':False}
LIMITS={'packet_reads':1280,'W2_tensor_loads':80,'W2_FP32_dense_matrix_calls':320,'row_method_evaluations':491520,'output_site_reports':160,
        'GPU_calls':0,'native_FFN_calls':0,'new_videos':0,'training_updates':0,'decompositions':0,'candidate_exports':0}
DESIGN={'development_sources':list(SOURCES),'calibration_sources':list(CAL),'states':'historically exposed teacher Dense; seed42',
 'sites':'all 4 denoise steps x 40 blocks','row_count_per_source_site':256,'calibration_rows_per_site':1536,'target_sparsities':list(RATES),
 'arms':['layer_CDF_abs_threshold','per_row_equal_k_random','layer_static_h2_times_W2_column_l2_equal_k'],
 'threshold':'other six source BF16 absolute values pooled, nearest rank empirical quantile; delete abs(h)<=t',
 'ties':'inclusive threshold; random/static exactly match actual per-row k; static score ties ascending channel index',
 'random':'per-row CPU torch.randperm; injective mixed-radix source-index/step/block/row seed; nested same permutation across rates',
 'static_calibration':'six calibration sources 1536 h rows per site; mean squared h times W2 column squared norm',
 'reference':'gated native Y, actual original BF16 dense h and Y; delta = FP32 dropped-h @ original W2 transpose; bias cancels',
 'equal_weight':'valid rows within site, then sites within step, steps within source, sources equal; 2 independent sources',
 'cpu_affinity':[4,5,6,7], 'peak_intermediate_shape':[1536,13824],
 'RSS_enforcement':'soft sampled /proc RSS watchdog and post-operation peak; no cgroup hard limit claim',
 'claims':'local teacher-state FP32 diagnostic only; no original author kernel, sparse saving, student trajectory, video or speed claim',
 'zero_denominator':'null relative SSE, retained explicitly; no silent zero replacement'}


def bind(root,item):return native.bound(root,item)
def parent(hf):
    result=native.read(ROOT/CORPUS);meta=result['manifest'];manifest=native.read(bind(hf,meta))
    native.need(result['status']=='COMPLETE_ASSET_AUDIT_ACTUAL_CPU_EXIT0' and manifest['totals']['sources']==8,'Original corpus audit differs')
    audit=native.read(bind(hf,result['RUN']));vm.exited(native.read(bind(hf,result['exit'])),result['RUN'],result['protocol']['sha256'],audit['id'])
    bind(ROOT,result['protocol']);return result,manifest

def environment():
    py=(ROOT/'.venv/bin/python').resolve();native.need(Path(sys.executable).resolve()==py,'Use repository .venv Python for draft and run')
    packages={}
    for name in ('torch','numpy','safetensors'):
        d=importlib.metadata.distribution(name);metadata=d.read_text('METADATA');record=d.read_text('RECORD')
        native.need(metadata is not None and record is not None,'Package distribution identity unavailable')
        packages[name]={'version':d.version,'metadata_sha256':hashlib.sha256(metadata.encode()).hexdigest(),
                        'record_sha256':hashlib.sha256(record.encode()).hexdigest()}
    return {'python_path':str(ROOT/'.venv/bin/python'),'python_sha256':native.sha(py),'distributions':packages}


def fingerprint(p):
    return native.canonical({k:v for k,v in p.items() if k not in ('status','assets_finalized','review')})


def random_seed(source,step,block,row):
    native.need(source in SOURCES and 0<=step<4 and 0<=block<40 and 0<=row<256,'Random seed coordinate differs')
    return 42000000+(((SOURCES.index(source)*4+step)*40+block)*256+row)


def draft(hf):
    result,m=parent(hf);template=native.read(ROOT/MASTER_TEMPLATE);cases=[next(c for c in m['cases'] if c['native_source_ordinal']==o) for o in (*SOURCES,*CAL)]
    native.need(len({c['image_sha256'] for c in cases})==8 and all(c['source_state']=='teacher_input' and c['role']==('development' if c['native_source_ordinal'] in SOURCES else 'calibration') and len(c['packets'])==160 for c in cases),'Source identity differs')
    native.need(all({(p['step'],p['block']) for p in c['packets']}=={(s,b) for s in range(4) for b in range(40)} for c in cases),'All160 missing')
    sources={x['path'] for x in template['repo_sources']}|{x['path'] for x in native.read(ROOT/BRIDGE_CLOSURE)['repo_sources']}|{SELF,TEST,CORPUS,MASTER_TEMPLATE,BRIDGE_CLOSURE,bridge.SELF,bridge.TEST,'tools/durable_worker_job.py','tools/worker_identity.py','tools/e0_s2_causal_worker_proof_v5.py'}
    return {'id':ID,'status':'DRAFT','assets_finalized':False,'review':None,'environment':environment(),'corpus_report':native.binding(ROOT/CORPUS,ROOT),
      'manifest':result['manifest'],'audit_protocol':result['protocol'],'audit_RUN':result['RUN'],'audit_exit':result['exit'],
      'master_template':native.binding(ROOT/MASTER_TEMPLATE,ROOT),'models':template['models'],
      'worker_master_proof':template['execution_raw_master_CPU_proof'],'private_worker_sha256':template['execution_private_worker_config_sha256'],
      'cases':cases,'resources':RESOURCES,'limits':LIMITS,'design':DESIGN,'run_ids':[ID],'output_HF_relative':OUT,
      'automatic_retry':False,'repo_sources':[native.binding(ROOT/p,ROOT) for p in sorted(sources)]}

def validate(p,hf,frozen=False):
    x=draft(hf);x['status']=p.get('status');x['assets_finalized']=p.get('assets_finalized');x['review']=p.get('review')
    native.need(p==x,'Scope or source closure differs')
    if frozen:native.need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['assets_finalized'] is True and p['review'] is None,'Frozen pending-registration plan required')

def authorize(p,digest,hf):
    validate(p,hf,True)
    active=native.read(ROOT/'research/ACTIVE_RESEARCH.json');scopes=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID]
    native.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True and len(scopes)==1
      and scopes[0].get('protocol_sha256')==digest and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
      and scopes[0].get('resources')==RESOURCES and scopes[0].get('limits')==LIMITS and scopes[0].get('run_ids')==[ID]
      and scopes[0].get('max_gpu_cards')==0,'Exact ACTIVE registration required')
    scope=scopes[0];review=native.read(bind(ROOT,scope['review']))
    native.need(review.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and review.get('scope_id')==ID
      and review.get('state')=='REVIEWER_RECHECK_APPROVED' and review.get('recheck_disposition')=='APPROVE'
      and review.get('author_agent_id') and review.get('reviewer_agent_id') and review['author_agent_id']!=review['reviewer_agent_id']
      and review.get('responses_to_review') and review.get('adversarial_counterexample')
      and review.get('approval_scope_fingerprint')==fingerprint(p)
      and review.get('critical_source_hashes')=={b['path']:b['sha256'] for b in p['repo_sources']}
      and review.get('input_cohort_manifest_sha256')==p['manifest']['sha256'],'Independent recheck fingerprint/closure required')
    for key in ('author_review','independent_review','reviewer_recheck'):bind(ROOT,review[key])
    spec=scope.get('launch_spec_sha256');native.need(isinstance(spec,str) and len(spec)==64 and set(spec)<=set('0123456789abcdef'),'Supervisor spec SHA missing')
    return scope


def verify_job_spec(job,args,p,digest):
    hf=args.hf_root.resolve();out=(hf/OUT).resolve();argv=job['argv']
    native.need(len(argv)==17 and argv[0]=='env' and argv[1].startswith('FFNDS_WORKER_CONFIG='),'Private worker spec differs')
    profile=Path(argv[1].split('=',1)[1]);native.need(profile.is_absolute() and native.sha(profile)==p['private_worker_sha256'],'Worker profile differs')
    expected=['env',argv[1],'CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','taskset','-c','4-7',
      p['environment']['python_path'],'-B',str(ROOT/SELF),'run','--protocol',str(args.protocol.resolve()),'--job-spec',str(args.job_spec.resolve())]
    native.need(argv==expected and job['id']==ID and job['protocol_sha256']==digest
      and Path(job['protocol_path']).resolve()==args.protocol.resolve() and Path(job['cwd']).resolve()==ROOT
      and Path(job['hf_root']).resolve()==hf and Path(job['output_dir']).resolve()==out.parent
      and Path(job['run_json']).resolve()==out/'RUN.json' and job['external_timeout_seconds']==RESOURCES['outer_seconds'], 'Supervisor chain differs')


def write(path,value):
    data=json.dumps(value,indent=2,allow_nan=False).encode()+b'\n';native.need(len(data)<=8<<20,'JSON size bound')
    temp=path.with_name(path.name+'.tmp');temp.write_bytes(data);os.replace(temp,path)

def sha_tensor(t,torch):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()

def thresholds_from_hist(hist,torch):
    native.need(int(hist.sum())==len(CAL)*256*D,'Calibration histogram coverage differs')
    cumulative=hist.cumsum(0);n=int(hist.sum());result=[]
    for rate in RATES:
        code=int(torch.searchsorted(cumulative,math.ceil(rate*n)).item())
        value=float(torch.tensor([code],dtype=torch.int16).view(torch.bfloat16).float()[0])
        result.append({'target_rate':rate,'threshold':value,'less_count':int(cumulative[code-1]) if code else 0,
                       'equal_count':int(hist[code]),'count':n})
    return result


def deletion_counts(selected,threshold,torch):
    return {'total':int(selected.numel()),'zero':int((selected==0).sum()),
            'nonzero':int((selected!=0).sum()),'threshold_tie':int((selected.abs()==threshold).sum()),
            'finite':int(torch.isfinite(selected).sum())}


def source_summary(records):
    result={}
    for source in SOURCES:
        vals=[r for r in records if r['source_ordinal']==source];native.need(len(vals)==160,'Missing site')
        for rate in RATES:
            for name in DESIGN['arms']:
                valid=[r['rates'][str(rate)]['arms'][name]['mean_row_relative_sse'] for r in vals if r['rates'][str(rate)]['arms'][name]['mean_row_relative_sse'] is not None]
                result[f'{source}:{rate}:{name}']={'mean_site_relative_sse':sum(valid)/len(valid) if len(valid)==160 else None,'valid_sites':len(valid)}
    pooled={}
    for rate in RATES:
        for name in DESIGN['arms']:
            v=[result[f'{o}:{rate}:{name}']['mean_site_relative_sse'] for o in SOURCES]
            pooled[f'{rate}:{name}']=sum(v)/len(v) if all(x is not None for x in v) else None
    return {'sources':result,'equal_source_mean_site_relative_sse':pooled}


def run(args):
    start=time.monotonic();hf=args.hf_root.resolve();p=native.read(args.protocol);digest=native.sha(args.protocol);scope=authorize(p,digest,hf)
    native.need(args.job_spec is not None and native.sha(args.job_spec)==scope['launch_spec_sha256'],'Supervisor spec binding differs')
    verify_job_spec(native.read(args.job_spec),args,p,digest)
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sys.dont_write_bytecode and sorted(os.sched_getaffinity(0))==DESIGN['cpu_affinity']
        and not {'torch','numpy'}&set(sys.modules),'CPU-only -B with at most 4 CPUs required')
    out=(hf/OUT).resolve();native.need(out.is_relative_to(hf) and not out.exists(),'Fresh attempt required');out.mkdir(parents=True)
    report={'id':ID,'protocol_sha256':digest,'status':'STARTED_NO_RETRY','predebited':{},'observed':{},'site_reports':[],
            'weight_tensors':[],'automatic_retry':False,'limits':LIMITS,'design':DESIGN,
            'launch_spec_sha256':scope['launch_spec_sha256'],'review':scope['review'],'approval_scope_fingerprint':fingerprint(p)}
    def persist():
        total=sum(x.stat().st_size for x in out.parent.rglob('*') if x.is_file());native.need(total+(16<<20)<=RESOURCES['new_HF_bytes'],'Output budget exhausted');write(out/'RUN.json',report)
    def work(key,count,fn):
        report['predebited'][key]=report['predebited'].get(key,0)+count;native.need(report['predebited'][key]<=LIMITS[key],'Operation budget exceeded')
        persist();value=fn();report['observed'][key]=report['observed'].get(key,0)+count
        report['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        native.need(report['peak_RSS_bytes']<=RESOURCES['host_RSS_max_bytes'],'RSS budget exceeded');persist();return value
    monitor={'samples':0,'peak_bytes':0,'exceeded':False,'error':None};stop=threading.Event()
    def rss_watch():
        try:
            while not stop.is_set():
                line=next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
                value=int(line.split()[1])*1024;monitor['samples']+=1;monitor['peak_bytes']=max(monitor['peak_bytes'],value)
                if value>RESOURCES['host_RSS_max_bytes']:
                    monitor['exceeded']=True;os.kill(os.getpid(),signal.SIGUSR1);return
                stop.wait(.1)
        except BaseException as e:
            monitor['error']=type(e).__name__;os.kill(os.getpid(),signal.SIGUSR1)
    signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(MemoryError('RSS watchdog exceeded or failed')))
    watchdog=threading.Thread(target=rss_watch,daemon=True);watchdog.start()
    persist();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU wall budget')));signal.alarm(RESOURCES['wall_seconds'])
    try:
        for b in p['repo_sources']:bind(ROOT,b)
        template=native.read(bind(ROOT,p['master_template']));import e0_s2_causal_worker_proof_v5 as worker_proof
        worker_proof.require_closed_proof(template,hf)
        import torch
        from safetensors import safe_open
        torch.set_num_threads(4);torch.set_num_interop_threads(1);native.need(not torch.cuda.is_initialized(),'CUDA initialized')
        results=[]
        for expert in ('high','low'):
            model=next(x for x in p['models'] if x['kind']==expert);stat=native.stat_record(model['path']);header=native.header(model['path'])
            native.need(header['header_sha256']==model['header_sha256'],'Master header differs')
            for block in range(40):
                name=f'blocks.{block}.ffn.2.weight';info=header['rows'][name]
                native.need(info['dtype']=='BF16' and info['shape']==[5120,D],'W2 header mismatch')
                with safe_open(model['path'],framework='pt',device='cpu') as f:
                    raw=work('W2_tensor_loads',1,lambda:f.get_tensor(name));sha=sha_tensor(raw,torch);w=raw.float();del raw
                report['weight_tensors'].append({'expert':expert,'block':block,'key':name,'raw_sha256':sha,'master_sha256':model['sha256'],'stat':stat});persist()
                wnorm=w.square().sum(dim=0);native.need(bool(torch.isfinite(wnorm).all()),'Nonfinite W2 column norm')
                for step in ((0,1) if expert=='high' else (2,3)):
                    packets=[];hist=torch.zeros(32768,dtype=torch.int64);sumh2=torch.zeros(D,dtype=torch.float64)
                    for case in p['cases']:
                        item=next(z for z in case['packets'] if z['step']==step and z['block']==block)
                        meta=native.read(bind(hf,item['metadata']));path=bind(hf,item['payload'])
                        native.need(meta['native_source_ordinal']==case['native_source_ordinal'] and meta['source_id']==case['source_id']
                          and meta['image_sha256']==case['image_sha256'] and meta['prompt_sha256']==case['prompt_sha256']
                          and meta['step']==step and meta['block']==block and meta['expert']==expert and meta['source_state']=='teacher_input'
                          and meta['grid']==[21,29,52] and len(meta['selected_rows'])==256,'Packet metadata mismatch')
                        keys=('h','Y','gate','rows') if case['native_source_ordinal'] in SOURCES else ('h',)
                        with safe_open(str(path),framework='pt',device='cpu') as f:
                            data=work('packet_reads',1,lambda:{k:f.get_tensor(k) for k in keys})
                        for k,t in data.items():
                            native.need(sha_tensor(t,torch)==item['tensor_metadata'][k]['raw_tensor_sha256'],'Packet tensor SHA mismatch')
                        native.need(list(data['h'].shape)==[256,D] and data['h'].dtype==torch.bfloat16,'Packet h geometry differs')
                        if case['native_source_ordinal'] in CAL:
                            h=data['h'];bits=(h.view(torch.int16).to(torch.int32)&32767).reshape(-1).to(torch.int64)
                            native.need(bool(torch.isfinite(h).all()),'Nonfinite calibration activation')
                            hist+=torch.bincount(bits,minlength=32768);sumh2+=h.float().square().sum(dim=0).double()
                        else:
                            native.need(data['rows'].tolist()==meta['selected_rows'] and list(data['Y'].shape)==[256,5120]
                              and list(data['gate'].shape)==[5120] and bool(torch.isfinite(data['Y']).all())
                              and bool(torch.isfinite(data['gate']).all()),'Development packet geometry or finite precondition differs')
                            packets.append((case,meta,data,item))
                    ts=thresholds_from_hist(hist,torch);static_score=sumh2.float()*wnorm
                    native.need(bool(torch.isfinite(static_score).all()),'Nonfinite static channel score')
                    static_order=torch.argsort(static_score,stable=True)
                    site_records=[{'source_ordinal':case['native_source_ordinal'],'step':step,'block':block,'expert':expert,
                                   'packet':item['payload'],'rates':{}} for case,meta,data,item in packets]
                    for threshold in ts:
                        def construct_masks():
                            vectors=[];stats=[]
                            for case,meta,data,item in packets:
                                h=data['h'].float();native.need(bool(torch.isfinite(h).all()),'Nonfinite development activation')
                                for row in range(256):
                                    x=h[row];dropped=(x.abs()<=threshold['threshold']);k=int(dropped.sum());zero=int((x==0).sum())
                                    gen=torch.Generator(device='cpu');gen.manual_seed(random_seed(case['native_source_ordinal'],step,block,row))
                                    random=torch.randperm(D,generator=gen)[:k];static=static_order[:k]
                                    arm_counts=[]
                                    for mask in (dropped,random,static):
                                        selected=x[mask];arm_counts.append(deletion_counts(selected,threshold['threshold'],torch))
                                        v=torch.zeros(D,dtype=torch.float32);v[mask]=selected;vectors.append(v)
                                    stats.append({'k':k,'zero':zero,'ties':int((x.abs()==threshold['threshold']).sum()),'arms':arm_counts})
                            return vectors,stats
                        vectors,stats=work('row_method_evaluations',1536,construct_masks)
                        output=work('W2_FP32_dense_matrix_calls',1,lambda:torch.stack(vectors).mm(w.T))
                        native.need(bool(torch.isfinite(output).all()),'Nonfinite W2 product')
                        for ci,(case,meta,data,item) in enumerate(packets):
                            gate=data['gate'].float();targets=data['Y'].float()*gate
                            den=targets.double().square().sum(dim=1);arms={}
                            for mi,name in enumerate(DESIGN['arms']):
                                errors=output[ci*768+mi:ci*768+768:3]*gate
                                energy=errors.double().square().sum(dim=1);valid=den>0
                                arms[name]={'mean_row_relative_sse':float((energy/den).mean()) if bool(valid.all()) else None,
                                            'valid_rows':int(valid.sum()),'null_rows':int((~valid).sum()),'total_squared_error':float(energy.sum()),'total_reference_energy':float(den.sum())}
                            subset=stats[ci*256:(ci+1)*256]
                            site_records[ci]['rates'][str(threshold['target_rate'])]={'threshold_calibration':threshold,
                              'actual_drop_count_sum':sum(x['k'] for x in subset),'preexisting_zero_count_sum':sum(x['zero'] for x in subset),
                              'threshold_tie_count_sum':sum(x['ties'] for x in subset),'row_count':256,
                              'native_rows':meta['selected_rows'],'drop_counts':[x['k'] for x in subset],
                              'preexisting_zero_counts':[x['zero'] for x in subset],
                              'threshold_tie_counts':[x['ties'] for x in subset],
                              'arm_row_deletion_counts':{name:[x['arms'][mi] for x in subset] for mi,name in enumerate(DESIGN['arms'])},'arms':arms}
                        del vectors,output
                    results.extend(site_records);pathout=out/f'{expert}_step{step}_block{block:02d}.json'
                    work('output_site_reports',1,lambda:write(pathout,{'id':ID,'protocol_sha256':digest,'records':site_records}))
                    report['site_reports'].append(native.binding(pathout,hf));persist();del packets
                native.need(native.stat_record(model['path'])==stat,'Master changed while loading');del w,wnorm
        native.need(all(report['predebited'].get(k,0)==report['observed'].get(k,0)==v for k,v in LIMITS.items()),'Incomplete accounting')
        report.update(summary=source_summary(results),status='COMPLETE_TEAL_WAN_CPU_LOCAL_DIAGNOSTIC',wall_seconds=time.monotonic()-start)
        native.need(report['wall_seconds']<=RESOURCES['wall_seconds'],'Wall exceeded')
    except BaseException as e:report.update(status='FAILED_CPU_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e),wall_seconds=time.monotonic()-start);raise
    finally:
        signal.alarm(0);stop.set();watchdog.join();report['rss_watchdog']=monitor;persist()

def verify_closed(args):
    hf=args.hf_root.resolve();p=native.read(args.protocol);validate(p,hf,True);digest=native.sha(args.protocol);out=hf/OUT
    native.need(args.run.resolve()==(out/'RUN.json').resolve() and args.exit.resolve()==(out.parent/'EXIT_RECEIPT.json').resolve(),'Wrong run or exit')
    b=native.binding(args.run,hf);r=native.read(args.run);receipt=native.read(args.exit);vm.exited(receipt,b,digest,ID)
    native.need(receipt.get('spec_sha256')==r['launch_spec_sha256'] and r['approval_scope_fingerprint']==fingerprint(p), 'Exit/spec/fingerprint differs')
    review=native.read(bind(ROOT,r['review']));native.need(review.get('approval_scope_fingerprint')==fingerprint(p)
      and review.get('state')=='REVIEWER_RECHECK_APPROVED' and review.get('recheck_disposition')=='APPROVE','Review closure differs')
    native.need(r['id']==ID and r['protocol_sha256']==digest and r['status']=='COMPLETE_TEAL_WAN_CPU_LOCAL_DIAGNOSTIC'
      and r['design']==DESIGN and r['limits']==LIMITS and 0<=r['wall_seconds']<=RESOURCES['wall_seconds']
      and r['peak_RSS_bytes']<=RESOURCES['host_RSS_max_bytes'] and r['rss_watchdog']['samples']>0
      and r['rss_watchdog']['exceeded'] is False and r['rss_watchdog']['error'] is None
      and sum(x.stat().st_size for x in out.parent.rglob('*') if x.is_file())<=RESOURCES['new_HF_bytes'] and len(r['weight_tensors'])==80 and len(r['site_reports'])==160
      and all(r['predebited'].get(k,0)==r['observed'].get(k,0)==v for k,v in LIMITS.items()),'Closed report differs')
    records=[]
    for item in r['site_reports']:
        data=native.read(bind(hf,item));native.need(data['id']==ID and data['protocol_sha256']==digest and len(data['records'])==2
         ,'Site report differs');records+=data['records']
    native.need({(x['source_ordinal'],x['step'],x['block']) for x in records}=={(o,s,b) for o in SOURCES for s in range(4) for b in range(40)}
      and source_summary(records)==r['summary'],'Source/site coverage differs')
    return {'id':ID,'status':'PASS_CLOSED_TEAL_CPU_LOCAL','RUN':b,'exit':native.binding(args.exit,hf)}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('draft','inspect','run','verify-closed'))
    ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--exit',type=Path)
    ap.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));ap.add_argument('--job-spec',type=Path);a=ap.parse_args()
    if a.action=='draft':native.need(a.out is not None and not a.out.exists(),'Fresh draft required');a.out.parent.mkdir(parents=True,exist_ok=True);write(a.out,draft(a.hf_root))
    elif a.action=='inspect':validate(native.read(a.protocol),a.hf_root);print('STATIC_ONLY_NO_TENSOR_PAYLOAD')
    elif a.action=='run':run(a)
    else:print(json.dumps(verify_closed(a),indent=2))

if __name__=='__main__':main()
