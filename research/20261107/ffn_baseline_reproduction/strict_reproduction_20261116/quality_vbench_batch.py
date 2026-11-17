"""One-dimension, source/arm-batched native VBenchI2V metric worker.

Called by an independently reviewed guard with concrete producer proof callback.
Every filename maps one-to-one to a real generated video. VBench mixed aggregate
is discarded; outputs are per-video custom comparison cells only.
"""
import hashlib,importlib,importlib.metadata,json,math,os,random,sys,time
from pathlib import Path
from quality_pair_report import DEV,ARMS,DIMS

DEV3=('vbi_1013','vbi_0886','vbi_0800')
COMPARISONS={'main':ARMS,'E4':('ICC','ICC 2R E4'),'E6':('Chipmunk Author Safe','Chipmunk Author Safe E6')}

CAMERAS=('camera pans left','camera pans right','camera tilts up','camera tilts down','camera zooms in','camera zooms out','camera static')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,data):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def run(scope,*,scope_sha256,authorize,verify_generation,check_resources):
    authorize(scope)
    dim,records=scope['dimension'],scope['records']
    comparison=scope['comparison']
    if comparison not in COMPARISONS: raise ValueError('unknown comparison')
    allowed_arms=COMPARISONS[comparison]
    allowed_sources=DEV if comparison=='main' else DEV3
    if comparison=='E6' and scope.get('e6_variable') not in ('tail','precision'):
        raise ValueError('E6 requires exactly one frozen eligible variable')
    if dim not in DIMS or not 1<=len(records)<=len(allowed_sources)*len(allowed_arms): raise ValueError('dimension or batch size')
    ordering=sorted(records,key=lambda r:(allowed_sources.index(r['source']) if r['source'] in allowed_sources else len(allowed_sources),
                                          allowed_arms.index(r['arm']) if r['arm'] in allowed_arms else len(allowed_arms)))
    if records!=ordering:raise ValueError('scope records must use frozen source/method order')
    batch_material={'dimension':dim,'metric_seed':scope['metric_seed'],
                    'videos':[(r['source'],r['arm'],r['video_sha256']) for r in records]}
    batch_identity=hashlib.sha256(json.dumps(batch_material,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()
    seen=set();paths=set();inodes=set();sources={}; proofs={}
    for r in records:
        key=(r['source'],r['arm']);path=Path(r['video_path']).resolve()
        if r['source'] not in allowed_sources or r['arm'] not in allowed_arms or r['seed']!=42 or key in seen or path in paths:
            raise ValueError('invalid/duplicate source arm or video path')
        if dim=='camera_motion' and r.get('camera_instruction') not in CAMERAS:
            raise ValueError('camera metric requires explicit applicable instruction')
        stat=path.stat(); inode=(stat.st_dev,stat.st_ino)
        if inode in inodes: raise ValueError('distinct source/arm videos share one inode')
        seen.add(key);paths.add(path);inodes.add(inode)
        if not r.get('input_identity') or not r.get('source_cluster_id'): raise ValueError('missing source identity')
        if sha(path)!=r['video_sha256'] or sha(r['image_path'])!=r['image_sha256']:
            raise ValueError('video/image hash mismatch')
        if hashlib.sha256(r['prompt'].encode('utf8')).hexdigest()!=r['prompt_sha256']:
            raise ValueError('prompt identity mismatch')
        if '/' in r['prompt'] or '\\' in r['prompt'] or len((r['prompt']+'-4.mp4').encode())>255:
            raise ValueError('unsafe VBench prompt-based name')
        identity=(r['prompt'],r['prompt_sha256'],r['image_path'],r['image_sha256'],r['source_cluster_id'],r['input_identity'])
        if r['source'] in sources and sources[r['source']]!=identity: raise ValueError('source inputs differ between arms')
        sources[r['source']]=identity
        proof=verify_generation(r,scope)
        if not isinstance(proof,dict) or proof.get('status')!='PASS' or proof.get('proof_sha256')!=r.get('generation_proof_sha256') or not proof.get('proof_sha256'):
            raise ValueError('generation receipt verification failed')
        proofs[key]=proof['proof_sha256']
    if len({v[0] for v in sources.values()})!=len(sources): raise ValueError('distinct sources share prompt-based filename')
    ref=Path(scope['quality_reference_dir']).resolve()
    for name in ('runner_metrics.py','metric_contract.py','metric_loaders.py'):
        if sha(ref/name)!=scope['quality_reference_sha256'][name]:raise ValueError('metric reference mismatch')
    sys.path.insert(0,str(ref))
    from runner_metrics import check_video,extract_video_records
    from metric_contract import verify_closure
    from metric_loaders import safe_loaders
    project=Path(scope['metric_project_root']).resolve()
    if Path(scope['vbench_working_directory']).resolve()!=project: raise ValueError('VBench working directory mismatch')
    for r in records:
        expected=(project/'vbench2_beta_i2v/data/crop/16-9'/Path(r['image_path']).name).resolve()
        if Path(r['image_path']).resolve()!=expected:raise ValueError('VBench source image path mismatch')
        check_video(r['video_path'],scope);check_resources()
    mapping=verify_closure(dim,scope,sha)
    out=Path(scope['output']);out.mkdir(parents=True,exist_ok=False);videos=out/'videos';videos.mkdir()
    write_json(out/'effective_loader.json',{dim:mapping})
    info=[];lookup={}
    for source in allowed_sources:
        group=[r for r in records if r['source']==source]
        if not group:continue
        for r in group:
            index=allowed_arms.index(r['arm']);link=videos/(r['prompt']+f'-{index}.mp4')
            link.symlink_to(Path(r['video_path']).resolve())
            lookup[str(Path(r['video_path']).resolve())]=r
        first=group[0];info.append({'prompt_en':first['prompt'],'image_name':Path(first['image_path']).name,'dimension':[dim]})
    metadata=out/'full_info.json';write_json(metadata,info)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.chdir(project)
    import numpy as np,torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):raise RuntimeError('one allocated A100 required')
    if torch.__version__!=scope['runtime_versions']['torch'] or np.__version__!=scope['runtime_versions']['numpy']:raise RuntimeError('torch/numpy version mismatch')
    for package,version in scope['metric_package_versions'].items():
        if importlib.metadata.version(package)!=version:raise RuntimeError('metric package mismatch: '+package)
    gpu_limit=scope['budget']['gpu_memory_gib'];total=torch.cuda.get_device_properties(0).total_memory
    if not 0<gpu_limit<=80 or gpu_limit*1024**3>total:raise ValueError('invalid GPU limit')
    torch.cuda.set_per_process_memory_fraction(min(.95,gpu_limit*1024**3/total),0);torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(scope['budget']['cpus']);seed=scope['metric_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cuda.matmul.allow_tf32=False
    package_by_dim={'i2v_background':['dreamsim','open_clip','clip','peft'],'background_consistency':['clip'],'aesthetic_quality':['clip'],'imaging_quality':['pyiqa']}
    for name in ['vbench','vbench2_beta_i2v']+package_by_dim.get(dim,[]):
        module=importlib.import_module(name);path=str(Path(module.__file__).resolve())
        if path not in scope['metric_code_sha256'] or sha(path)!=scope['metric_code_sha256'][path]:raise RuntimeError('metric source mismatch: '+name)
    import vbench2_beta_i2v as vbench_module
    observations=[];restore=None
    if dim=='dynamic_degree':
        import vbench.dynamic_degree as dynamic
        score_old,infer_old,forward_old=dynamic.DynamicDegree.get_score,dynamic.DynamicDegree.infer,dynamic.RAFT.forward; current=[None]
        def score(self,*args,**kwargs):
            if current[0] is None:raise RuntimeError('RAFT score without video')
            value=float(score_old(self,*args,**kwargs))
            if not math.isfinite(value) or value<0:raise RuntimeError('invalid RAFT score')
            current[0]['scores'].append(value);return value
        def infer(self,path,*args,**kwargs):
            item={'video_path':str(Path(path).resolve()),'scores':[],'forward_count':0};observations.append(item);current[0]=item
            try:
                value=infer_old(self,path,*args,**kwargs);item['native_dynamic_degree']=bool(value)
                if len(item['scores'])!=40 or item['forward_count']!=40:raise RuntimeError('expected 40 RAFT pairs')
                item['continuous_motion']=sum(item['scores'])/40
                item['motion_windows']=[sum(item['scores'][i:i+8])/8 for i in range(0,40,8)]
                return value
            finally:current[0]=None
        def forward(self,*args,**kwargs):
            if current[0] is None or kwargs.get('iters')!=20 or kwargs.get('test_mode') is not True:raise RuntimeError('RAFT protocol mismatch')
            current[0]['forward_count']+=1;result=forward_old(self,*args,**kwargs)
            if not isinstance(result,(tuple,list)) or len(result)<2 or not torch.isfinite(result[1]).all().item():raise RuntimeError('nonfinite RAFT flow')
            return result
        dynamic.DynamicDegree.get_score=score;dynamic.DynamicDegree.infer=infer;dynamic.RAFT.forward=forward
        def restore():dynamic.DynamicDegree.get_score=score_old;dynamic.DynamicDegree.infer=infer_old;dynamic.RAFT.forward=forward_old
    started=time.monotonic()
    try:
        check_resources()
        with safe_loaders(torch,vbench_module,dim,mapping,scope):
            bench=vbench_module.VBenchI2V('cuda',str(metadata),str(out))
            bench.evaluate(videos_path=str(videos),name='metric',dimension_list=[dim],local=True,resolution='16-9')
    finally:
        if restore:restore()
    check_resources()
    if torch.cuda.max_memory_allocated()>gpu_limit*1024**3:raise MemoryError('metric GPU limit')
    raw_path=out/'metric_eval_results.json';raw=json.loads(raw_path.read_text(encoding='utf8'));raw_rows=extract_video_records(raw)
    scored_paths=[str(Path(x['video_path']).resolve()) for x in raw_rows]
    if len(scored_paths)!=len(records) or len(set(scored_paths))!=len(scored_paths) or set(scored_paths)!=set(lookup):
        raise RuntimeError('missing or duplicate per-video metric records')
    obsmap={o['video_path']:o for o in observations}
    if dim=='dynamic_degree' and (len(observations)!=len(records) or set(obsmap)!=set(scored_paths)):
        raise RuntimeError('RAFT observations missing or duplicated')
    cells=[]
    for row,path in zip(raw_rows,scored_paths):
        r=lookup[path];value=row.get('video_results')
        if not isinstance(value,(int,float,bool)) or not math.isfinite(float(value)):raise ValueError('invalid metric value')
        cell={'status':'COMPLETE','source':r['source'],'arm':r['arm'],'dimension':dim,'value':float(value),
              'video_sha256':r['video_sha256'],'scope_sha256':scope_sha256,'input_identity':r['input_identity'],
              'metric_protocol_sha256':scope['metric_protocol_sha256'],'metric_batch_identity':batch_identity,
              'source_cluster_id':r['source_cluster_id'],'generation_proof_sha256':proofs[(r['source'],r['arm'])],
              'comparison':comparison,'e6_variable':scope.get('e6_variable') if comparison=='E6' else None,
              'camera_instruction':r.get('camera_instruction'),
              'camera_stratum':('CAMERA_STATIC_EXPLICIT' if r.get('camera_instruction')=='camera static' else
                  'CAMERA_DYNAMIC_INSTRUCTION' if r.get('camera_instruction') in CAMERAS else 'CAMERA_UNSPECIFIED')}
        if dim=='dynamic_degree':
            ob=obsmap[path];cell['continuous_motion']=ob['continuous_motion'];cell['motion_windows']=ob['motion_windows']
        cells.append(cell)
    if dim=='dynamic_degree':write_json(out/'continuous_raft.json',observations)
    write_json(out/'per_video_scores.json',cells)
    write_json(out/'METRIC_SUMMARY.json',{'status':'CUSTOM_I2V_DIMENSION_COMPLETE_NUMERICS_UNASSESSED',
        'dimension':dim,'comparison':comparison,'source_count':len(sources),'video_count':len(cells),'seed':seed,
        'metric_batch_identity':batch_identity,
        'official_five_sample_leaderboard':False,'raw_mixed_aggregate':'DO_NOT_USE',
        'metric_raw_sha256':sha(raw_path),'per_video_sha256':sha(out/'per_video_scores.json'),
        'loader_sha256':sha(out/'effective_loader.json'),'continuous_raft_sha256':sha(out/'continuous_raft.json') if dim=='dynamic_degree' else None,'elapsed_seconds':time.monotonic()-started,
        'gpu_peak_bytes':torch.cuda.max_memory_allocated()})
    return {'status':'COMPLETE','dimension':dim,'video_count':len(cells),'per_video_sha256':sha(out/'per_video_scores.json'),'metric_summary_sha256':sha(out/'METRIC_SUMMARY.json')}








