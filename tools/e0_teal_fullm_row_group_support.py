"""CPU-only natural-vs-fixed-shuffle row-group support on eight captured BF16 Hs.

The frozen plan must bind an exited capture and all packet hashes. This is an
observational scan, not a sparse implementation or a GPU speed measurement.
"""
import argparse, hashlib, json, math, mmap, os, signal, struct, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ID='E0_TEAL_FULLM_ROW_GROUP_SUPPORT_20261030'
PRODUCER_ID='E0_TEAL_WAN_FULLM_H_CAPTURE_20261030'
SITES={(0,0),(0,20),(2,0),(2,20)}
SOURCES=(1040,979); M=31668; K=13824; D=5120; SEED=42
OUTPUT='ffnds-derived/e0_teal_fullm_row_group_support_20261030/attempt1/worker/results'
RESOURCES={'GPU':0,'CPU_threads':4,'wall_seconds':3600,'outer_seconds':3900,
           'HF_new_bytes':32<<20,'attempts':1,'automatic_retry':False}
THRESHOLDS='config/e0_teal_wan_all160_calibration_thresholds_20261029_v1.json'


def need(v, message):
    if not v: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()


def bound(root,item):
    path=(root/item['path']).resolve()
    need(path.is_relative_to(root.resolve()) and path.is_file() and path.stat().st_size==item['bytes']
         and sha(path)==item['sha256'],'bound asset identity differs')
    return path


def bf16_scalar_bits(value):
    """Round finite nonnegative Python scalar to BF16, ties to even."""
    need(type(value) in (int,float) and math.isfinite(value) and value>=0,'threshold differs')
    u=struct.unpack('<I',struct.pack('<f',value))[0]
    return ((u+0x7fff+((u>>16)&1))>>16)&0xffff


def file_identity(stream, path, expected=None):
    def identity(st): return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    try: fd_id=identity(os.fstat(stream.fileno()));path_id=identity(path.stat())
    except FileNotFoundError: raise ValueError('input path disappeared') from None
    need(fd_id==path_id and (expected is None or fd_id==expected),'input replaced or modified during read')
    return fd_id


def sha_stream(stream):
    stream.seek(0);h=hashlib.sha256()
    for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def safetensor_header(stream):
    size=os.fstat(stream.fileno()).st_size;stream.seek(0)
    raw=stream.read(8);need(len(raw)==8,'short safetensors header');n=struct.unpack('<Q',raw)[0]
    need(0<n<=16<<20 and n+8<=size,'header exceeds bound');header=json.loads(stream.read(n))
    need(set(header)-{'__metadata__'}=={'h','gate'},'unexpected tensor keys')
    intervals=[]
    for name in ('h','gate'):
        item=header[name];a,b=item['data_offsets']
        need(item['dtype']=='BF16' and type(a) is int and type(b) is int and 0<=a<=b<=size-8-n,'invalid operand offsets or dtype')
        need(math.prod(item['shape'])*2==b-a,'tensor size differs');intervals.append((a,b))
    intervals.sort();need(intervals[0][0]==0 and intervals[0][1]==intervals[1][0] and intervals[1][1]==size-8-n,'overlap or gap in payload')
    need(header['h']['shape']==[M,K],'full-M H shape differs')
    return header,8+n


def build_keep(raw, thresholds, np, *, block_rows=128):
    """Compare the real BF16 bit patterns exactly under PyTorch scalar casting."""
    limits=[bf16_scalar_bits(x) for x in thresholds]
    need(len(limits)==2 and limits[0]<=limits[1],'threshold pair differs')
    keep=[np.empty(raw.shape,dtype=np.bool_) for _ in limits]
    for start in range(0,raw.shape[0],block_rows):
        sl=slice(start,min(raw.shape[0],start+block_rows)); absolute=np.bitwise_and(raw[sl],np.uint16(0x7fff))
        need(bool(np.all(absolute<0x7f80)),'nonfinite BF16 H')
        for j,limit in enumerate(limits):keep[j][sl]=absolute>limit
    return keep


def group_stats(keep, permutation, group_size, np, out_dim=D):
    """Return all group-level columns and tile totals; no padded rows."""
    rows,width=keep.shape
    need(width%64==0 and group_size in (16,32) and (permutation is None or len(permutation)==rows),'group geometry differs')
    seen=[]; all_start=time.perf_counter(); gather_seconds=0.; scan_seconds=0.
    for start in range(0,rows,group_size):
        n=min(group_size,rows-start); t=time.perf_counter()
        block=keep[start:start+n] if permutation is None else keep[permutation[start:start+n]]
        gather_seconds+=time.perf_counter()-t; t=time.perf_counter()
        union=np.any(block,axis=0);live=int(np.count_nonzero(union)); empty16=int(np.count_nonzero(~np.any(union.reshape(-1,16),axis=1)))
        empty64=int(np.count_nonzero(~np.any(union.reshape(-1,64),axis=1)))
        scan_seconds+=time.perf_counter()-t;seen.append((n,live,empty16,empty64))
    total_mac=sum(n*live*out_dim for n,live,_,_ in seen);dense_mac=rows*width*out_dim
    lives=[x[1] for x in seen]; full=[x for x in seen if x[0]==group_size]; tail=[x for x in seen if x[0]!=group_size]
    def summary(items):
        if not items:return None
        n=len(items); active=sum(x[1] for x in items)
        return {'groups':n,'column_empty_fraction':1-active/(n*width),
                'K16_empty_fraction':sum(x[2] for x in items)/(n*(width//16)),
                'K64_empty_fraction':sum(x[3] for x in items)/(n*(width//64)),
                'union_min':min(x[1] for x in items),'union_median':float(np.median([x[1] for x in items])),
                'union_p95':float(np.quantile([x[1] for x in items],.95)),
                'union_max':max(x[1] for x in items)}
    return {'group_size':group_size,'groups':len(seen),'group_rows':[x[0] for x in seen],
            'union_columns':lives,'empty_K16_tiles':[x[2] for x in seen],'empty_K64_tiles':[x[3] for x in seen],
            'logical_dense_macs':dense_mac,'logical_union_macs':total_mac,'logical_mac_fraction':total_mac/dense_mac,
            'all_groups':summary(seen),'full_groups':summary(full),'tail_group':summary(tail),'tail_rows':tail[0][0] if tail else 0,
            'gather_cpu_wall_seconds':gather_seconds,'union_scan_cpu_wall_seconds':scan_seconds,
            'total_group_loop_wall_seconds':time.perf_counter()-all_start}


def validate_plan(plan):
    need(plan.get('id')==ID and plan.get('status')=='FROZEN_PENDING_REGISTRATION' and plan.get('assets_finalized') is True,
         'NOT_BOUND: actual exited capture and all eight assets are required')
    need(plan['resources']==RESOURCES and plan['output_HF_relative']==OUTPUT
         and plan['seed']==SEED and plan['sources']==list(SOURCES) and plan['sites']==sorted(map(list,SITES)), 'scope differs')
    captures=plan['capture_assets']
    need(len(captures)==8 and {(x['source_ordinal'],x['step'],x['block']) for x in captures}==
         {(s,step,block) for s in SOURCES for step,block in SITES},'eight real cases required')
    need(len({x['file']['path'] for x in captures})==8 and all(x['h_shape']==[M,K] and x['h_dtype']=='bfloat16'
         and x['expert']==(0 if x['step']<2 else 1) and len(x['actual_grid'])==3
         and math.prod(x['actual_grid'])==M for x in captures),'capture geometry or identity differs')
    need(plan['threshold_artifact']['path']==THRESHOLDS and plan['capture_protocol']['path']=='config/e0_teal_wan_fullm_capture_20261030_v1.json',
         'unrelated producer or threshold source')
    expected_file=sum(x['file']['bytes'] for x in captures)
    need(plan['read_budget']=={'file_sha_bytes':expected_file,'H_tensor_sha_bytes':8*M*K*2,
         'H_analysis_input_bytes':8*M*K*2,'gate_tensor_sha_bytes':sum(math.prod(x['gate_shape'])*2 for x in captures)},'read budget differs')


def limits(plan):
    reads=plan['read_budget']
    return {**reads,'source_snapshot_bytes':2*sum(x['bytes'] for x in plan['repo_sources']), 'shuffle_permutations':1,'threshold_keep_matrices':16,
            'row_groups':8*2*2*((M+15)//16+(M+31)//32)}


def authorize(plan,digest):
    validate_plan(plan);active=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text())
    need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True,'authorization revoked')
    scopes=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID]
    need(len(scopes)==1 and scopes[0].get('protocol_sha256')==digest and scopes[0].get('execution_status')=='RUNNING_FIXED_SCOPE'
         and scopes[0].get('resources')==RESOURCES and scopes[0].get('limits')==limits(plan) and scopes[0].get('max_gpu_cards')==0,'exact registered CPU scope required')
    review=json.loads(bound(ROOT,scopes[0]['review']).read_text())
    fingerprint=hashlib.sha256(json.dumps({k:v for k,v in plan.items() if k not in ('status','assets_finalized','review')},
                              sort_keys=True,separators=(',',':')).encode()).hexdigest()
    need(review.get('state')=='REVIEWER_RECHECK_APPROVED' and review.get('recheck_disposition')=='APPROVE'
         and review.get('scope_id')==ID and review.get('approval_scope_fingerprint')==fingerprint
         and review.get('author_agent_id') and review.get('reviewer_agent_id')
         and review['author_agent_id']!=review['reviewer_agent_id'] and review.get('adversarial_counterexample')
         and review.get('responses_to_review') and review.get('job_spec_sha256')==scopes[0].get('job_spec_sha256')
         and review.get('critical_source_hashes')=={x['path']:x['sha256'] for x in plan['repo_sources']},'independent review missing')
    for k in ('author_review','independent_review','reviewer_recheck'):bound(ROOT,review[k])
    return scopes[0]


def read_tensor_sha(stream,offset,length):
    h=hashlib.sha256();remaining=length;stream.seek(offset)
    while remaining:
        data=stream.read(min(remaining,8<<20));need(bool(data),'short operand read');h.update(data);remaining-=len(data)
    return h.hexdigest()


def load_keep_from_bound_file(path,hf,case,thresholds,np,charge,done,case_result):
    """One fd anchors SHA, tensor reads and analysis; check path/fd each stage."""
    need(path.resolve().is_relative_to(hf.resolve()),'input escaped HF')
    with path.open('rb') as stream:
        identity=file_identity(stream,path);need(identity[2]==case['file']['bytes'],'file size differs')
        charge('file_sha_bytes',case['file']['bytes']);t=time.perf_counter();digest=sha_stream(stream)
        case_result['file_sha_wall_seconds']=time.perf_counter()-t
        need(digest==case['file']['sha256'],'file SHA differs');file_identity(stream,path,identity);done('file_sha_bytes',case['file']['bytes'])
        header,base=safetensor_header(stream);h=header['h'];g=header['gate'];need(g['shape']==case['gate_shape'],'real gate shape differs')
        for name,entry in (('h',h),('gate',g)):
            a,b=entry['data_offsets'];count=b-a;key='H_tensor_sha_bytes' if name=='h' else 'gate_tensor_sha_bytes'
            charge(key,count);t=time.perf_counter();actual=read_tensor_sha(stream,base+a,count)
            case_result[name+'_tensor_sha_wall_seconds']=time.perf_counter()-t
            need(actual==case['h_raw_sha256' if name=='h' else 'gate_raw_sha256'],'tensor raw SHA differs')
            file_identity(stream,path,identity);done(key,count)
        a,b=h['data_offsets'];file_identity(stream,path,identity)
        raw=np.memmap(stream,dtype='<u2',mode='r',offset=base+a,shape=(M,K))
        charge('H_analysis_input_bytes',M*K*2);charge('threshold_keep_matrices',2);t=time.perf_counter()
        keep=build_keep(raw,thresholds,np);case_result['mask_cpu_wall_seconds']=time.perf_counter()-t
        file_identity(stream,path,identity);done('H_analysis_input_bytes',M*K*2);done('threshold_keep_matrices',2)
        del raw
        case_result['input_stat_identity_checked_before_after_each_stage']=True
        return keep


def verify_parent(plan,hf):
    producer=bound(ROOT,plan['capture_protocol']); producer_sha=sha(producer); parent=json.loads(producer.read_text())
    need(parent.get('id')==PRODUCER_ID and [c['native_source_ordinal'] for c in parent['cases']]==list(SOURCES), 'capture protocol sources differ')
    run=json.loads(bound(hf,plan['capture_RUN']).read_text()); exit=json.loads(bound(hf,plan['capture_EXIT']).read_text())
    need(run['id']==PRODUCER_ID and run['protocol_sha256']==producer_sha and run['status']=='COMPLETE_FULLM_OBSERVATIONAL_CAPTURE'
         and len(run['captures'])==8 and len(run['phases'])==2 and exit.get('id')==PRODUCER_ID
         and exit.get('protocol_sha256')==producer_sha and exit.get('run_sha256')==plan['capture_RUN']['sha256']
         and exit.get('process_exited') is True and exit.get('child_returncode')==0 and exit.get('timeout') is False
         and exit.get('automatic_retry') is False and exit.get('spec_sha256')==plan['capture_launcher_spec_sha256']
         and run['captures']==plan['capture_assets'], 'capture RUN/actual EXIT differs')
    need(not run['assigned_GPU_resources']['violations'] and run['operation_ledger']['predebited'].get('capture_packets')==8
         and run['operation_ledger']['observed'].get('capture_packets')==8
         and {x['native_source_ordinal'] for x in run['phases']}==set(SOURCES)
         and all(x.get('all160_FFN_covered') is True and x.get('completed_native_flow_steps')==4
                 and x.get('matched_olddev_all_ten_states') is True and x.get('seed')==42 and x.get('mode')=='dense' for x in run['phases'])
         and not run['operation_ledger']['forbidden_attempts']
         and all(run['operation_ledger'][side].get(k,0)==v for side in ('predebited','observed') for k,v in parent['limits'].items()),
         'capture incomplete or monitor failed')
    return {'id':PRODUCER_ID,'protocol_sha256':producer_sha,'RUN':plan['capture_RUN'],'EXIT':plan['capture_EXIT']}


def atomic_json(path,record):
    text=(json.dumps(record,indent=2,allow_nan=False)+'\n').encode()
    live=path.stat().st_size if path.exists() else 0
    tmp=path.with_suffix(path.suffix+'.tmp')
    other=sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file() and p not in (path,tmp))
    need(len(text)+live+other+(1<<20)<=RESOURCES['HF_new_bytes'],'atomic JSON output budget exceeded')
    with tmp.open('xb') as f:f.write(text)
    os.replace(tmp,path)


def run(args):
    plan=json.loads(args.protocol.read_text());digest=sha(args.protocol);scope=authorize(plan,digest)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==4,'CPU-only four-core assignment required')
    need(args.job_spec is not None and sha(args.job_spec)==scope['job_spec_sha256'],'job identity differs')
    job=json.loads(args.job_spec.read_text()); affinity=scope['cpu_affinity']
    need(len(affinity)==4 and len(set(affinity))==4 and sorted(os.sched_getaffinity(0))==sorted(affinity),'CPU affinity differs')
    expected=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PATH=/usr/bin:/bin',
      'taskset','-c',','.join(map(str,affinity)),str(ROOT/'.venv/bin/python'),'-I','-B',str(ROOT/'tools/e0_teal_fullm_row_group_support.py'),
      '--protocol',str(args.protocol.resolve()),'--hf-root',str(args.hf_root.resolve()),'--job-spec',str(args.job_spec.resolve())]
    need(job['argv']==expected and Path(job['cwd']).resolve()==ROOT and job['id']==ID and job['protocol_sha256']==digest
       and Path(job['protocol_path']).resolve()==args.protocol.resolve() and Path(job['hf_root']).resolve()==args.hf_root.resolve()
       and Path(job['output_dir']).resolve()==(args.hf_root.resolve()/OUTPUT).parent
       and Path(job['run_json']).resolve()==args.hf_root.resolve()/OUTPUT/'RUN.json'
       and job['external_timeout_seconds']==RESOURCES['outer_seconds'],'job scope differs')
    hf=args.hf_root.resolve();out=(hf/OUTPUT).resolve();need(out.is_relative_to(hf) and not out.exists(),'exclusive output required')
    out.mkdir(parents=True);(out/'ATTEMPT.json').write_text(json.dumps({'id':ID,'protocol_sha256':digest})+'\n')
    report={'id':ID,'protocol_sha256':digest,'status':'STARTED_NO_RETRY','resources':RESOURCES,
            'read_budget':plan['read_budget'],'work':{'predebited':{},'observed':{}},'cases':[],
            'scope':'teacher post-GELU support; no sparse kernel, weight arithmetic, video or latency claim'}
    def persist():atomic_json(out/'RUN.json',report)
    def charge(k,n):
        need(k in limits(plan) and n>0 and report['work']['predebited'].get(k,0)+n<=limits(plan)[k], 'paid-work ceiling exceeded')
        report['work']['predebited'][k]=report['work']['predebited'].get(k,0)+n;persist()
    def done(k,n):
        report['work']['observed'][k]=report['work']['observed'].get(k,0)+n;persist()
    start=time.monotonic();persist();old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall limit')));signal.alarm(RESOURCES['wall_seconds'])
    try:
        snapshot_bytes=sum(item['bytes'] for item in plan['repo_sources']); charge('source_snapshot_bytes',2*snapshot_bytes)
        need(snapshot_bytes+args.protocol.stat().st_size+(2<<20)<RESOURCES['HF_new_bytes'],'source snapshot exceeds HF budget')
        snapshot=out/'EXECUTED_SOURCES';snapshot.mkdir()
        for item in plan['repo_sources']:
            source=bound(ROOT,item);target=snapshot/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
            data=source.read_bytes();need(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'snapshot source changed')
            with target.open('xb') as f:need(f.write(data)==len(data),'source snapshot short write')
        protocol_data=args.protocol.read_bytes();need(hashlib.sha256(protocol_data).hexdigest()==digest,'protocol changed')
        with (snapshot/'protocol.json').open('xb') as f:need(f.write(protocol_data)==len(protocol_data),'protocol snapshot short write')
        report['snapshot']={'repo_files':len(plan['repo_sources']),'repo_bytes':snapshot_bytes,'protocol_bytes':len(protocol_data)}
        done('source_snapshot_bytes',2*snapshot_bytes)
        threshold_data=json.loads(bound(ROOT,plan['threshold_artifact']).read_text())
        need(threshold_data['schema']=='ffnds.teal.wan.calibration.thresholds.v1' and threshold_data['rates']==[.25,.5]
             and threshold_data['source']=='SIX_CALIBRATION_SOURCES_ONLY','threshold source differs')
        rows={(x['step'],x['block']):x['thresholds'] for x in threshold_data['threshold_rows']}
        need(len(rows)==160 and SITES<=set(rows),'threshold site coverage differs')
        report['producer']=verify_parent(plan,hf);persist()
        import importlib.metadata
        need(importlib.metadata.version('numpy')==plan['runtime_versions']['numpy'],'numpy version differs')
        import numpy as np
        report['runtime']={'numpy':np.__version__,'python':sys.version,'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS')}
        charge('shuffle_permutations',1);t=time.perf_counter();permutation=np.random.default_rng(SEED).permutation(M)
        report['shuffle']={'generator':'numpy.PCG64','seed':SEED,'permutation_raw_sha256':hashlib.sha256(permutation.astype('<i8',copy=False).tobytes()).hexdigest(),
                           'rng_cpu_wall_seconds':time.perf_counter()-t,'same_permutation_all_cases_and_group_sizes':True};done('shuffle_permutations',1)
        for case in plan['capture_assets']:
            case_result={'source_ordinal':case['source_ordinal'],'step':case['step'],'block':case['block'],
                         'actual_grid':case['actual_grid'],'input':case['file'],'status':'STARTED','results':{}}
            report['cases'].append(case_result);persist()
            path=(hf/case['file']['path']).resolve()
            keep=load_keep_from_bound_file(path,hf,case,rows[case['step'],case['block']],np,charge,done,case_result)
            for i,label in enumerate(('q25','q50')):
                per={};case_result['results'][label]=per
                for n in (16,32):
                    for name,indices in (('natural',None),('shuffle',permutation)):
                        count=(M+n-1)//n;charge('row_groups',count)
                        per[f'{name}_M{n}']=group_stats(keep[i],indices,n,np);done('row_groups',count)
            case_result.update(status='COMPLETE',thresholds=rows[case['step'],case['block']]);del keep;persist()
        need(len(report['cases'])==8 and all(x['status']=='COMPLETE' for x in report['cases'])
             and all(report['work'][kind].get(k,0)==v for kind in ('predebited','observed') for k,v in limits(plan).items()),'incomplete paid work')
        # Average sites within source, then weight the two image sources equally.
        aggregate={}
        for label in ('q25','q50'):
            for mode in ('natural_M16','shuffle_M16','natural_M32','shuffle_M32'):
                per_source=[]
                for source in SOURCES:
                    site=[x['results'][label][mode] for x in report['cases'] if x['source_ordinal']==source]
                    per_source.append({'source_ordinal':source,'mean_logical_mac_fraction':sum(x['logical_mac_fraction'] for x in site)/4,
                      'mean_full_group_column_empty_fraction':sum(x['full_groups']['column_empty_fraction'] for x in site)/4,
                      'mean_full_group_K16_empty_fraction':sum(x['full_groups']['K16_empty_fraction'] for x in site)/4,
                      'mean_full_group_K64_empty_fraction':sum(x['full_groups']['K64_empty_fraction'] for x in site)/4})
                aggregate[label+'/'+mode]={'source_means':per_source,'source_equal_mean_logical_mac_fraction':sum(x['mean_logical_mac_fraction'] for x in per_source)/2,
                    'source_equal_mean_full_group_column_empty_fraction':sum(x['mean_full_group_column_empty_fraction'] for x in per_source)/2,
                    'source_equal_mean_full_group_K16_empty_fraction':sum(x['mean_full_group_K16_empty_fraction'] for x in per_source)/2,
                    'source_equal_mean_full_group_K64_empty_fraction':sum(x['mean_full_group_K64_empty_fraction'] for x in per_source)/2}
        report.update(status='COMPLETE_OBSERVATIONAL_SUPPORT_ONLY',source_equal_aggregate=aggregate);persist()
    except BaseException as error:
        report.update(status='FAILED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old);report['wall_seconds']=time.monotonic()-start;persist()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--hf-root',type=Path,required=True);ap.add_argument('--job-spec',type=Path,required=True)
    run(ap.parse_args())
if __name__=='__main__':main()
