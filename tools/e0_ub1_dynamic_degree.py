"""Bounded original UB1 dynamic-degree evaluation; frozen v8 PASS and exact registration required."""
ID = 'E0_UB1_DYNAMIC_DEGREE_20261027'


def counted_forward(original, counts):
    """Charge before delegation and preserve the exact original return object."""
    def wrapped(self, image1, image2, *args, **kwargs):
        if counts.get('forwards', 0) >= 160:
            raise RuntimeError('RAFT forward budget exceeded')
        if args or kwargs.get('iters') != 20 or kwargs.get('test_mode') is not True:
            raise ValueError('Unexpected original RAFT invocation')
        if tuple(image1.shape) != (1, 3, 464, 832) or tuple(image2.shape) != (1, 3, 464, 832):
            raise ValueError('Unexpected frame shape')
        counts['forwards'] = counts.get('forwards', 0) + 1
        counts['iterations'] = counts.get('iterations', 0) + 20
        return original(self, image1, image2, *args, **kwargs)
    return wrapped


def charge_video(counts, expected_paths, path):
    index=counts['videos']
    if index>=len(expected_paths) or path!=expected_paths[index]: raise RuntimeError('Video order/budget differs')
    counts['videos']+=1
    return index


def observed_score(original, scores):
    """Record the original score without changing its value or arguments."""
    def wrapped(self, *args, **kwargs):
        if len(scores) >= 40:
            raise RuntimeError('Per-clip score budget exceeded')
        value = original(self, *args, **kwargs)
        scores.append(float(value))
        return value
    return wrapped

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {'model_initializations':1,'checkpoint_loads':1,'forwards':160,'iterations':3200,'videos':4}
PENDING = 'PREPARATION_PENDING_ROOT_FREEZE'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def authorize(plan, digest, active):
    if plan['id']!=ID or plan['limits']!=LIMITS: raise ValueError('Fixed budget mismatch')
    for field in ('helper_sha256','v8_protocol_sha256','v8_run_sha256'):
        if plan[field]==PENDING or len(plan[field])!=64: raise ValueError('Pending frozen prerequisite: '+field)
    if not any(s.get('id')==ID and s.get('protocol_sha256')==digest and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for s in active['experiment_authority']['current_user_authorized_scopes']): raise ValueError('Exact protocol registration required')


def verify_preflight(plan, hf):
    for field in ('tool','test','helper','v8_protocol'):
        p=(ROOT/plan[field]).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=plan[field+'_sha256']: raise ValueError('Source identity mismatch: '+field)
    run=(hf/plan['v8_run_HF_relative']).resolve()
    if not run.is_relative_to(hf) or sha(run)!=plan['v8_run_sha256']: raise ValueError('V8 receipt identity differs')
    receipt=json.loads(run.read_text())
    if receipt.get('status')!='PASS_IMPORTS_ONLY' or receipt.get('protocol_sha256')!=plan['v8_protocol_sha256']: raise ValueError('V8 import compatibility not accepted')
    for item in plan['repo_sources']:
        path=(ROOT/item['path']).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=item['sha256']: raise ValueError('Repository source mismatch')
    readiness=(ROOT/plan['readiness']).resolve()
    if not readiness.is_relative_to(ROOT): raise ValueError('Readiness path escaped repository')
    assets=plan['assets']
    if readiness.exists():
        if sha(readiness)!=plan['readiness_sha256']: raise ValueError('Readiness identity differs')
        archived=json.loads(readiness.read_text())
        if any(assets[k]!=archived[k] for k in ('videos','sources','checkpoint')): raise ValueError('Embedded asset manifest differs')
    if len(assets['videos'])!=4 or len(assets['sources'])!=14: raise ValueError('Wrong fixed cohort')
    for v in assets['videos']:
        p=Path(v['path']).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=v['sha256'] or v['frames']!=81 or v['fps']!=16.0 or v['size']!=[832,464]: raise ValueError('Video binding differs')
        q=(ROOT/v['comparison_manifest']).resolve()
        if not q.is_relative_to(ROOT) or sha(q)!=v['comparison_manifest_sha256']: raise ValueError('Historical manifest differs')
    for x in assets['sources']+[assets['checkpoint'],plan['vbench_init']]:
        p=(hf/x['HF_relative']).resolve()
        if not p.is_relative_to(hf) or not p.is_file() or sha(p)!=x['sha256'] or p.stat().st_size!=x['bytes']: raise ValueError('Official source/checkpoint mismatch')
    return assets


def write(path, obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def install_audit(scope, checkpoint, trusted_query_code, binary_path, gpu_index):
    """Application-level audit hook; it does not confine native-extension I/O."""
    scope=scope.resolve();checkpoint=checkpoint.resolve(); query_count=[0]
    allowed={('--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'),('--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu','--format=csv,noheader,nounits'),('--query-gpu=memory.used','--format=csv,noheader,nounits')}
    def audit(event,values):
        if event=='subprocess.Popen':
            executable, argv = values[:2]; frame=sys._getframe(); trusted=False
            while frame is not None:
                if frame.f_code is trusted_query_code: trusted=True; break
                frame=frame.f_back
            if (trusted and isinstance(argv,(list,tuple)) and len(argv)==5 and argv[0]=='nvidia-smi' and argv[1]=='-i' and argv[2]==gpu_index and tuple(argv[3:]) in allowed and Path(shutil.which(executable) or '').resolve()==binary_path and query_count[0]<1500):
                query_count[0]+=1;return
            raise RuntimeError('Untrusted subprocess denied')
        if event.startswith('socket.') or event in ('os.system','os.exec','os.posix_spawn'): raise RuntimeError('Network/subprocess denied: '+event)
        if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(values[0])).resolve();mode=values[1] or '';flags=values[2] or 0
            writing=any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and not p.is_relative_to(scope): raise RuntimeError('Write outside scope')
            if not writing and p.suffix.lower() in ('.pt','.pth','.ckpt','.safetensors','.bin','.npz') and p!=checkpoint: raise RuntimeError('Other checkpoint denied')
        if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.symlink','os.link','os.chmod','os.truncate'):
            for v in values[:2] if event in ('os.rename','os.link','os.symlink') else values[:1]:
                if isinstance(v,(str,bytes,os.PathLike)) and not Path(os.fsdecode(v)).resolve().is_relative_to(scope): raise RuntimeError('Mutation outside scope')
    return audit


def run(args):
    plan=json.loads(args.protocol.read_text());digest=sha(args.protocol)
    authorize(plan,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()))
    hf=args.hf_root.resolve();assets=verify_preflight(plan,hf)
    site=(hf/plan['site_HF_relative']).resolve();overlay=(hf/plan['overlay_HF_relative']).resolve()
    if not site.is_relative_to(hf) or not overlay.is_relative_to(hf): raise ValueError('Environment path escaped HF')
    sys.path.append(str(site));sys.path.insert(0,str(overlay));sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('ub1_v8_verification',ROOT/plan['helper']);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    v8=json.loads((ROOT/plan['v8_protocol']).read_text());lock=json.loads((ROOT/v8['lock_path']).read_text())
    if sha(ROOT/v8['lock_path']) != v8['lock_sha256'] or sha(Path(sys.executable)) != v8['installed_environment']['python_sha256']: raise ValueError('Interpreter or lock mismatch')
    helper.verify_original_sources(hf,lock);helper.verify_shared(lock['shared_runtime'])
    if helper.verify_installed_environment(hf,v8,lock).resolve()!=site or helper.verify_overlay(hf/plan['v8_directory_HF_relative'],v8,hf).resolve()!=overlay: raise ValueError('Environment site/overlay mismatch')
    helper.two_cpus()
    sys.path.append(str(ROOT/'tools'))
    from worker_identity import require_worker
    require_worker();scope=(hf/plan['output_HF_relative']).resolve()
    if scope==hf or not scope.is_relative_to(hf): raise ValueError('Output escaped HF')
    scope.mkdir(parents=True,exist_ok=True)
    with (scope/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest,'attempt':1},f)
    result={'id':ID,'status':'STARTED','protocol_sha256':digest,'counts':dict.fromkeys(LIMITS,0),'clips':[],'scope':'Four existing clips, original binary dynamic_degree only; not quality, prompt compliance or new video generation.'}
    monitor=None;started=time.monotonic()
    def check():
        if monitor:monitor.check()
        if shutil.disk_usage(scope).free < 10<<30: raise RuntimeError('Disk reserve')
        if sum(p.stat().st_size for p in scope.rglob('*') if p.is_file() and not p.is_symlink()) > 32<<20: raise RuntimeError('Output budget')
    def alarm(*_):raise TimeoutError('Wall budget exceeded')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
    try:
        for n in ('tmp','cache','torch','hf','mpl','xdg','inductor','compile_debug'): (scope/n).mkdir()
        for k in ('TMPDIR','TEMP','TMP'):os.environ[k]=str(scope/'tmp')
        os.environ.update(XDG_CACHE_HOME=str(scope/'xdg'),MPLCONFIGDIR=str(scope/'mpl'),TORCH_HOME=str(scope/'torch'),HF_HOME=str(scope/'hf'),TORCHINDUCTOR_CACHE_DIR=str(scope/'inductor'),TORCH_COMPILE_DEBUG_DIR=str(scope/'compile_debug'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        result['disabled_ipv6_probe']=helper.disable_local_ipv6_probe(site,v8)
        import torch
        from startup_gpu_reservation import reserve_before_assets
        from assigned_gpu_resources import AssignedGPUResources
        reservation,monitor,receipt=reserve_before_assets(torch);result['reservation']=receipt;torch.set_num_threads(2);check()
        if 'A100' not in torch.cuda.get_device_name(0) or torch.distributed.is_initialized(): raise RuntimeError('Single-process A100 scope required')
        result['runtime']={'python':sys.version.split()[0],'torch':torch.__version__,'CUDA':torch.version.cuda,'device':torch.cuda.get_device_name(0),'matmul_TF32':torch.backends.cuda.matmul.allow_tf32,'cudnn_TF32':torch.backends.cudnn.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision()}
        torch.cuda.reset_peak_memory_stats()
        src=hf/plan['source_root_HF_relative'];sys.path.insert(0,str(src))
        # No additional subprocess or network use after imports begin.
        checkpoint=(hf/assets['checkpoint']['HF_relative']).resolve()
        binary=Path(shutil.which('nvidia-smi') or '').resolve(); spec=plan['nvidia_smi']
        if spec['sha256']==PENDING or str(binary)!=spec['path'] or sha(binary)!=spec['sha256']: raise ValueError('nvidia-smi identity differs')
        sys.addaudithook(install_audit(scope,checkpoint,AssignedGPUResources.query.__code__,binary,os.environ['FFNDS_PHYSICAL_GPU']))
        from vbench.dynamic_degree import compute_dynamic_degree, DynamicDegree, RAFT
        if Path(sys.modules['vbench.dynamic_degree'].__file__).resolve() != (src/'vbench/dynamic_degree.py').resolve() or Path(sys.modules[RAFT.__module__].__file__).resolve() != (src/'vbench/third_party/RAFT/core/raft.py').resolve(): raise RuntimeError('Official module origin differs')
        snapshot=scope/'EXECUTED_SOURCES';snapshot.mkdir()
        for x in assets['sources']+[plan['vbench_init']]:
            target=snapshot/Path(x['HF_relative']).relative_to(plan['source_root_HF_relative']);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(hf/x['HF_relative'],target)
        for field in ('tool','test','helper','v8_protocol'):shutil.copyfile(ROOT/plan[field],snapshot/Path(plan[field]).name)
        for item in plan['repo_sources']:
            target=snapshot/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/item['path'],target)
        shutil.copyfile(args.protocol,snapshot/'protocol.json');check()
        original_load=torch.load
        def load_once(path,*a,**kw):
            if result['counts']['checkpoint_loads']>=1 or Path(path).resolve()!=checkpoint or sha(checkpoint)!=assets['checkpoint']['sha256']: raise RuntimeError('Checkpoint load denied')
            result['counts']['checkpoint_loads']+=1
            return original_load(path,*a,**kw)
        torch.load=load_once
        original_init=RAFT.__init__
        def init_once(obj,*a,**kw):
            if result['counts']['model_initializations']>=1: raise RuntimeError('RAFT initialization budget exceeded')
            result['counts']['model_initializations']+=1
            return original_init(obj,*a,**kw)
        RAFT.__init__=init_once;RAFT.forward=counted_forward(RAFT.forward,result['counts'])
        original_score=DynamicDegree.get_score;original_infer=DynamicDegree.infer;original_frames=DynamicDegree.get_frames
        context={'scores':None,'selected':None}
        def frames(obj,*a,**kw):
            values=original_frames(obj,*a,**kw);context['selected']=len(values)
            if len(values)!=41:raise ValueError('Unexpected selected frame count')
            return values
        def score(obj,*a,**kw):return observed_score(original_score,context['scores'])(obj,*a,**kw)
        def infer(obj,path,*a,**kw):
            index=charge_video(result['counts'],[v['path'] for v in assets['videos']],path)
            context['scores']=[];context['selected']=None
            value=original_infer(obj,path,*a,**kw)
            if len(context['scores'])!=40:raise ValueError('Unexpected scores per video')
            result['clips'].append({'source_index':assets['videos'][index]['source_index'],'seed':assets['videos'][index]['seed'],'label':assets['videos'][index]['label'],'video_sha256':assets['videos'][index]['sha256'],'selected_frames':context['selected'],'params':dict(obj.params),'pair_scores':context['scores'],'official_binary':bool(value)})
            check();write(scope/'RUN.json',result)
            return value
        DynamicDegree.get_frames=frames;DynamicDegree.get_score=score;DynamicDegree.infer=infer
        doc=[{'dimension':['dynamic_degree'],'video_list':[v['path'] for v in assets['videos']],'prompt_en':''}]
        input_path=scope/'OFFICIAL_INPUT.json';write(input_path,doc);result['official_input_sha256']=sha(input_path)
        aggregate,details=compute_dynamic_degree(str(input_path),torch.device('cuda'),{'model':str(checkpoint)})
        if result['counts']!=LIMITS or len(details)!=4: raise RuntimeError('Incomplete call ledger')
        if [v['video_path'] for v in details] != [v['path'] for v in assets['videos']] or [bool(v['video_results']) for v in details] != [v['official_binary'] for v in result['clips']]: raise RuntimeError('Original official outputs differ from observed clips')
        torch.cuda.synchronize()
        result['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated();result['peak_torch_reserved_bytes']=torch.cuda.max_memory_reserved()
        result.update(status='COMPLETE_OFFICIAL_DYNAMIC_DEGREE',aggregate=float(aggregate),official_results=[{'video_results':bool(v['video_results'])} for v in details],elapsed_seconds=time.monotonic()-started,checkpoint_compatibility='LOAD_AND_STATE_DICT_ACCEPTED')
    except BaseException as e:
        result.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0)
        if monitor:
            monitor.close();result['assigned_gpu_resources']=monitor.report()
            if result['assigned_gpu_resources']['violations']:result['status']='FAILED_GPU_MONITOR'
        write(scope/'RUN.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--hf-root',type=Path,required=True);run(parser.parse_args())
