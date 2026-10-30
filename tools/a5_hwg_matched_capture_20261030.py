"""Finite, reviewed HiF4 full160 matched acquisition, not a sparse method.

Every actual run is guarded by exact protocol/review/job/ACTIVE bindings.
Import and geometry fixtures do not execute a model or initialize CUDA.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import importlib.metadata
import importlib.machinery
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
SELF='tools/a5_hwg_matched_capture_20261030.py'
RES={'GPU':1,'CPU_threads':4,'GPU_reserved_max_bytes':76<<30,
     'host_available_min_bytes':128<<30,'peak_RSS_max_bytes':320<<30,
     'model_forecast_bytes':160<<30,'host_max_models':2,'session_max_GPUs':4,
     'new_HF_bytes':128<<30,'free_disk_min_bytes':132<<30,
     'maximum_single_file_bytes':512<<20,
     'asset_validation_read_max_bytes':128<<30,
     'wall_seconds':14400,'outer_seconds':14520,'automatic_retry':False}
VERIFY_RES={'GPU':0,'CPU_threads':1,'wall_seconds':7200,'outer_seconds':7320,
            'read_bytes_max':160<<30,'new_HF_bytes':16<<20,
            'metadata_read_bytes_max':256<<20,
            'peak_RSS_max_bytes':512<<20,'automatic_retry':False}
STATE_NAMES={'raw_noise_FP32','initial_noise',*(f'flow_step{i}'for i in range(4)),*(f'latent_step{i}'for i in range(4))}
CONTROL_READ=None
CURRENT_BOOTSTRAP=None
VIDEO_INSPECTION={'reader_contexts_per_video':1,'get_meta_data_calls_per_video':1,
                  'count_frames_calls_per_video':1,'deadline_seconds_per_video':60,
                  'internal_file_read_bytes':'UNKNOWN_OPAQUE_IMAGEIO_FFMPEG_IO',
                  'not_in_output_SHA_byte_count':True,'quality_or_speed_claim':False}


def need(value,message):
    if not value:raise RuntimeError(message)


def sha(path,*,counted=True):
    meter=CONTROL_READ if counted else None
    if meter:meter.before(Path(path).stat().st_size,str(path))
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):
            h.update(b)
            if meter:meter.after(len(b))
    if meter:meter.finish()
    return h.hexdigest()


def read(path):
    if CONTROL_READ:CONTROL_READ.before(Path(path).stat().st_size,str(path))
    b=Path(path).read_bytes()
    if CONTROL_READ:CONTROL_READ.after(len(b));CONTROL_READ.finish()
    return json.loads(b)
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def bind(path,root):
    p=Path(path);return {'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)}
def check_binding(root,ref):
    p=(root/ref['path']).resolve();need(p.is_relative_to(root.resolve())and p.is_file(),'binding confinement')
    need(p.stat().st_size==ref['bytes']and sha(p)==ref['sha256'],'bound file changed: '+ref['path']);return p


def matched_rows(grid,source_ordinal,seed):
    """One genuine global M64 tile inside each tau, same rows at every site."""
    need(grid==[21,29,52],'actual grid differs')
    width=grid[1]*grid[2];starts=[]
    for tau in range(grid[0]):
        first=(tau*width+63)//64
        stop=((tau+1)*width-64)//64
        key=f'A5_HWG_M64_V1|{source_ordinal}|{seed}|{tau}'
        offset=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(stop-first+1)
        starts.append((first+offset)*64)
    rows=[r for start in starts for r in range(start,start+64)]
    need(len(rows)==1344 and len(set(rows))==1344,'coordinate count')
    need(all(s%64==0 and s//width==(s+63)//width==t for t,s in enumerate(starts)),'tile not inside tau')
    return rows,starts


def expected_sample_shapes():
    d={k:[1344,5120]for k in ['P','C','R','U','Y','Delta_writeback_operand','X_next','Q1_input','V1']}
    d.update({k:[1344,13824]for k in ['Z','H','Q2_input','V2']});d.update(b2=[5120],gate_broadcast=[5120]);return d


REQUIRED_CHECKS={'sampled_native_P_plus_C_matches_R','cpu_rng_unchanged','cuda_rng_unchanged',
                 'sampled_gelu_tanh_matches_H','sampled_native_post_matches_derived_reference',
                 'actual_runtime_W2_matches_cache'}


def validate_packet(d,packet,p,hf,source,seen):
    step,block=packet['step'],packet['block'];expert='high_noise_model'if step<2 else'low_noise_model'
    need((d['step'],d['block'])==(step,block)and d['expert']==expert,'packet/meta site mismatch')
    need(d['source_id']==p['case']['source_id']and d['native_source_ordinal']==p['case']['native_source_ordinal']and d['seed']==42 and d['grid']==[21,29,52]and d['numeric_family']=='HIF4_FAKE_QDQ_BF16','packet source/seed/numeric identity')
    rows,starts=matched_rows([21,29,52],p['case']['native_source_ordinal'],42)
    need(d['selected_rows']==rows and d['tile_starts']==starts and d['row_count']==1344,'packet coordinates')
    shapes=expected_sample_shapes();need(set(d['tensor_schema'])==set(shapes),'fixed fifteen tensor fields')
    for key,shape in shapes.items():
        s=d['tensor_schema'][key];need(s['shape']==shape and s['dtype']=='bfloat16'and len(s['raw_tensor_sha256'])==64,'fixed sampled tensor shape/dtype/hash')
    need(REQUIRED_CHECKS<=set(d['checks'])and all(d['checks'][k]is True for k in REQUIRED_CHECKS),'required native checks')
    need(d['quantized_cache']==p['cache'],'actual cache identity')
    calls=d['QDQ_calls'];need(len(calls)==2,'two native QDQ calls')
    for call,suffix in zip(calls,['0','2']):
        slot='1'if suffix=='0'else'2'
        need(call['module']==f'blocks.{block}.ffn.{suffix}'and call['input']==f'Q{slot}_input'and call['output']==f'V{slot}'and call['force_fp32']is True and call['input_dtype']==call['output_dtype']=='torch.bfloat16'and call['native_group_size']==64 and call['dim']==-1,'native QDQ boundary identity')
    w=d['effective_W2'];need(w['runtime_shape']==[5120,13824]and w['dtype']=='torch.bfloat16'and w['runtime_cache_bytes_match']is True and w['cache_record_key']==[expert,f'blocks.{block}.ffn.2','weight_qdq']and len(w['runtime_operand_sha256'])==64,'actual W2 cache binding')
    need(d['full_V1']['shape']==[31668,5120]and d['full_V1']['dtype']=='bfloat16','fixed fullM V1 schema')
    need(d['gate_native_shape']in ([5120],[1,5120],[1,1,5120]),'native gate broadcast geometry')
    expected=[(packet['metadata'],f'step{step}_block{block:02d}.json'),(d['payload'],f'step{step}_block{block:02d}.safetensors'),(d['full_V1'],f'fullV1_step{step}_block{block:02d}.safetensors')]
    for ref,name in expected:
        path=(hf/ref['path']).resolve();need(path==source/'observed/ffn'/name and path.is_relative_to(source),'site path confinement')
        st=path.stat();key=(st.st_dev,st.st_ino);need(str(path)not in seen['paths']and key not in seen['inodes'],'duplicate site path or inode')
        seen['paths'].add(str(path));seen['inodes'].add(key)


def payload_budget():
    # Exactly the inherited observer's fields, all native BF16 in this recipe.
    d_fields=['P','C','R','U','Y','Delta_writeback_operand','X_next','Q1_input','V1']
    h_fields=['Z','H','Q2_input','V2']
    per={k:1344*5120*2*160 for k in d_fields}
    per.update({k:1344*13824*2*160 for k in h_fields})
    per.update(b2=5120*2*160,gate_broadcast=5120*2*160,fullM_V1=31668*5120*2*160)
    # Includes both phases' 20 state files, small sources, headers, metadata,
    # failure partials. Writes are synchronous, no duplicate tensor temp file.
    return {'tensor_payload_bytes_by_field':per,'total_tensor_payload_bytes':sum(per.values()),
            'metadata_states_video_and_failure_reserve':2<<30,
            'file_write_bytes_upper':sum(per.values())+(2<<30),
            'full_file_SHA_read_upper':sum(per.values())+(2<<30),
            'raw_tensor_SHA_memory_read_upper':sum(per.values())+(2<<30)+320*5120*13824*2,
            'observer_W2_SHA_memory_read_upper':320*5120*13824*2,
            'independent_verification_file_read_upper':sum(per.values())+(2<<30)}


class ReadBudget:
    def __init__(self,maximum,check=lambda:None,persist=lambda:None):
        self.maximum,self.check,self.persist=maximum,check,persist
        self.predebited=0;self.actual=0;self.inflight=None
    def before(self,n,label):
        self.check();need(self.inflight is None and type(n)is int and n>=0 and self.predebited+n<=self.maximum,'read predebit budget')
        self.predebited+=n;self.inflight={'label':label,'reserved_bytes':n,'actual_read_bytes':0,'completion':'INFLIGHT_UNKNOWN'};self.persist()
    def after(self,n):
        need(self.inflight is not None and self.inflight['actual_read_bytes']+n<=self.inflight['reserved_bytes'],'read exceeded predebit')
        self.actual+=n;self.inflight['actual_read_bytes']+=n;self.check()
    def finish(self):
        need(self.inflight is not None and self.inflight['actual_read_bytes']==self.inflight['reserved_bytes'],'short completed read')
        self.inflight=None;self.persist()
    def summary(self):return {'predebited_bytes':self.predebited,'actual_returned_read_bytes':self.actual,'inflight_unknown':self.inflight,'limit_bytes':self.maximum}


class StartupLedger:
    def __init__(self,a,kind,started):
        self.started=started;self.status='BOOTSTRAP_METADATA_ONLY';self.error=None
        # One bounded job peek locates the supervisor-owned fresh directory.
        # No protocol/source/model payload is read until durable predebit works.
        size=a.job_spec.stat().st_size;need(0<size<=65536,'restricted job peek bound')
        data=a.job_spec.read_bytes();need(len(data)==size,'restricted job peek changed');job=json.loads(data)
        hf=Path(job['hf_root']).resolve();out=Path(job['output_dir']).resolve()
        need(out.is_relative_to(hf)and out!=hf and out.is_dir()and Path(job['run_json']).resolve()==out/'results/RUN.json','bootstrap result confinement')
        need(Path(job['protocol_path']).resolve()==a.protocol.resolve(),'bootstrap protocol path')
        self.path=out/'STARTUP_READ_LEDGER.json';need(not self.path.exists(),'bootstrap already attempted')
        self.id=job['id'];self.spec_sha=hashlib.sha256(data).hexdigest();self.peek_bytes=size
        maximum=RES['asset_validation_read_max_bytes']if kind=='run'else VERIFY_RES['metadata_read_bytes_max']
        self.seconds=RES['wall_seconds']if kind=='run'else VERIFY_RES['wall_seconds']
        self.budget=ReadBudget(maximum,self.check,self.persist)
        self.budget.predebited=size;self.budget.actual=size;self.persist()
    def check(self):need(time.monotonic()-self.started<=self.seconds,'startup/actual total wall budget')
    def persist(self):
        d={'id':self.id,'spec_sha256':self.spec_sha,'status':self.status,'bounded_initial_job_peek_bytes':self.peek_bytes,'read_ledger':self.budget.summary(),'error':self.error,'wall_seconds':time.monotonic()-self.started,'automatic_retry':False}
        b=(json.dumps(d,indent=2,allow_nan=False)+'\n').encode();need(len(b)<=1<<20,'bootstrap output bound')
        tmp=self.path.with_suffix('.tmp')
        with tmp.open('xb')as f:need(f.write(b)==len(b),'short bootstrap write')
        tmp.replace(self.path)


def execution_bootstrap(kind):
    def decorate(fn):
        def wrapped(a):
            global CONTROL_READ,CURRENT_BOOTSTRAP
            started=time.monotonic();seconds=RES['wall_seconds']if kind=='run'else VERIFY_RES['wall_seconds']
            signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('whole execution deadline')));signal.alarm(seconds)
            boot=StartupLedger(a,kind,started);CURRENT_BOOTSTRAP=boot;CONTROL_READ=boot.budget
            try:
                value=fn(a);boot.status='COMPLETE_CHILD_RETURN_PENDING_SUPERVISOR_EXIT';return value
            except BaseException as e:
                boot.status='FAILED_PRESERVED_NO_RETRY';boot.error={'type':type(e).__name__,'message':str(e)[:1000]};raise
            finally:
                signal.alarm(0);CONTROL_READ=None;boot.persist();CURRENT_BOOTSTRAP=None
        return wrapped
    return decorate


class BoundTransformersFinder:
    def __init__(self,root,rows):self.root=root.resolve();self.rows={r['path']:r for r in rows};self.origins={}
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='transformers'and not fullname.startswith('transformers.'):return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        need(spec is not None and spec.origin is not None,'missing Transformers module')
        origin=Path(spec.origin).resolve();need(origin.is_relative_to(self.root),'Transformers import escaped verified copy')
        ref=self.rows.get(str(origin.relative_to(self.root)));need(ref is not None,'unbound Transformers import')
        check_binding(self.root,ref);self.origins[fullname]=ref['path'];return spec


def validate_import_origins(p):
    rows={b['path']:b for b in p['repo_sources']};seen={}
    prefixes=('HiFloat4','quant','quant_cy','hif4_gpu','hifxg_quant','lightx2v','lightx2v_platform')
    for name,module in list(sys.modules.items()):
        if not any(name==s or name.startswith(s+'.')for s in prefixes):continue
        origin=getattr(module,'__file__',None)
        need(origin is not None,'native namespace module lacks bound source')
        path=Path(origin).resolve();need(path.is_relative_to(ROOT),'native import outside frozen source roots')
        rel=str(path.relative_to(ROOT));need(rel in rows,'native import is absent from source closure: '+rel)
        check_binding(ROOT,rows[rel]);seen[name]=rel
    need(any(k.startswith('quant.hif4_backend')for k in seen),'actual HiF4 backend was not imported')
    return seen


@contextmanager
def patch(owner,name,value):
    old=getattr(owner,name);setattr(owner,name,value)
    try:yield old
    finally:setattr(owner,name,old)


def available_memory():
    return int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024


class HostReservation:
    """Serialize admission and account for already admitted models' future RSS."""
    def __init__(self):self.folder=Path('/tmp')/f'ffnds-a5-hwg-{os.getuid()}';self.path=None
    def __enter__(self):
        self.folder.mkdir(exist_ok=True)
        with (self.folder/'admission.lock').open('a+')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);others=[]
            for p in self.folder.glob('slot*.json'):
                d=read(p)
                try:
                    fields=Path(f'/proc/{d["pid"]}/stat').read_text();ticks=int(fields[fields.rfind(')')+2:].split()[19])
                    need(ticks==d['start_ticks'],'stale PID')
                    rss=int(Path(f'/proc/{d["pid"]}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                except (OSError,RuntimeError):p.unlink();continue
                others.append((p,max(0,d['forecast_bytes']-rss)))
            need(len(others)<RES['host_max_models'],'two model host slots already assigned')
            need(available_memory()-sum(x[1]for x in others)>=RES['host_available_min_bytes']+RES['model_forecast_bytes'],'joint host forecast fails')
            used={x[0].name for x in others};self.path=next(self.folder/f'slot{i}.json'for i in range(2)if f'slot{i}.json'not in used)
            fields=Path('/proc/self/stat').read_text();ticks=int(fields[fields.rfind(')')+2:].split()[19])
            with self.path.open('x')as f:json.dump({'pid':os.getpid(),'start_ticks':ticks,'forecast_bytes':RES['model_forecast_bytes']},f)
        return self
    def __exit__(self,*args):
        if self.path:self.path.unlink(missing_ok=True)


class Writer:
    def __init__(self,root,hf,check):
        self.root,self.hf,self.check=root,hf,check;self.written=0;self.raw_sha_bytes=0;self.file_sha_bytes=0;self.files={};self.holds={};self.inflight={};self.errors=[]
        self.journal_path=root/'OPERATIONS.jsonl';self.journal=self.journal_path.open('xb',buffering=0);self.holds['journal']=16<<20
    def note(self,event,**values):
        data=(json.dumps({'event':event,**values},separators=(',',':'))+'\n').encode()
        need(len(data)<=self.holds['journal'],'operation journal reserve');n=self.journal.write(data);self.journal.flush();need(n==len(data),'short journal write')
        self.holds['journal']-=n;self.written+=n
    def reserve(self,n):need(n>=0 and self.written+sum(self.holds.values())+n<=RES['new_HF_bytes'],'HF output budget');self.check()
    def hold_file(self,path,n):
        path=path.resolve();need(path.is_relative_to(self.root.resolve())and str(path)not in self.holds and not path.exists(),'output hold identity')
        self.reserve(n);self.holds[str(path)]=n;self.inflight[str(path)]='NOT_STARTED_RESERVED'
        self.note('FILE_RESERVED',path=str(path.relative_to(self.hf)),reserved_bytes=n)
    def begin_file(self,path,n=None):
        path=path.resolve()
        if str(path)not in self.holds:self.hold_file(path,n)
        self.inflight[str(path)]='INFLIGHT_ACTUAL_BYTES_UNKNOWN'
        self.note('FILE_WRITE_BEGIN_UNKNOWN',path=str(path.relative_to(self.hf)),reserved_bytes=self.holds[str(path)])
    def account_file(self,path,completed,known_sha=None):
        path=path.resolve();key=str(path);allowance=self.holds.get(key)
        need(allowance is not None,'file lacks persistent reservation')
        size=path.stat().st_size if path.exists()else 0
        old=self.files.get(key,{}).get('bytes',0);self.written+=size-old
        self.holds.pop(key);self.files[key]={'bytes':size,'sha256':known_sha,'complete':completed};self.inflight.pop(key,None)
        self.note('FILE_WRITE_ACCOUNTED',path=str(path.relative_to(self.hf)),actual_bytes=size,reserved_bytes=allowance,complete=completed)
        need(size<=allowance,'file exceeded reservation; partial accounted')
        self.reserve(0)
        if not path.exists():return None
        if known_sha is None:
            # Output hash reads have their own durable full-file predebit;
            # they do not consume the distinct input/identity read allowance.
            self.note('OUTPUT_FILE_HASH_READ_PREDEBIT_INFLIGHT_UNKNOWN',path=str(path.relative_to(self.hf)),reserved_read_bytes=size)
            h=hashlib.sha256();returned=0
            try:
                with path.open('rb')as f:
                    for b in iter(lambda:f.read(8<<20),b''):h.update(b);returned+=len(b);self.file_sha_bytes+=len(b)
                need(returned==size,'output file changed during hash');known_sha=h.hexdigest()
            finally:self.note('OUTPUT_FILE_HASH_READ_RETURNED',path=str(path.relative_to(self.hf)),actual_read_bytes=returned,complete=returned==size)
        self.files[key]['sha256']=known_sha
        return {'path':str(path.relative_to(self.hf)),'bytes':size,'sha256':known_sha}
    def tensor(self,path,tensors):
        from safetensors.torch import save_file
        n=sum(t.numel()*t.element_size()for t in tensors.values());self.begin_file(path,n+(1<<16));complete=False;ref=None
        try:save_file(tensors,str(path));complete=True
        finally:ref=self.account_file(path,complete)
        return ref
    def raw_sha(self,tensor):
        import torch
        need(tensor.device.type=='cpu'and tensor.is_contiguous(),'raw SHA requires owned CPU contiguous')
        self.raw_sha_bytes+=tensor.numel()*tensor.element_size()
        return hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest()
    def json(self,path,data):
        b=(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();old=path.stat().st_size if path.exists()else 0
        tmp=path.with_suffix(path.suffix+'.tmp');self.begin_file(tmp,len(b));complete=False
        try:
            with tmp.open('xb')as f:need(f.write(b)==len(b),'short JSON write')
            complete=True
        finally:self.account_file(tmp,complete,hashlib.sha256(b).hexdigest()if complete else None)
        tmp.replace(path);self.files.pop(str(tmp.resolve()),None);self.files[str(path.resolve())]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'complete':True};self.written-=old
    def external(self,path):
        return self.account_file(path,True)
    def finish_pending(self):
        for name in list(self.inflight):
            path=Path(name)
            if path.exists():
                try:self.account_file(path,False)
                except BaseException as e:self.errors.append(type(e).__name__+':'+str(e)[:200])
        self.note('FINAL_ACCOUNTING',inflight_unknown=dict(self.inflight),errors=self.errors)
    def summary(self):return {'tracked_live_bytes':self.written,'unconsumed_holds':dict(self.holds),'inflight_unknown':dict(self.inflight),'accounting_errors':self.errors,'journal':str(self.journal_path.relative_to(self.hf))}


@contextmanager
def bounded_files(maximum=None):
    maximum=RES['maximum_single_file_bytes']if maximum is None else maximum
    old=resource.getrlimit(resource.RLIMIT_FSIZE);handler=signal.getsignal(signal.SIGXFSZ)
    need(old[1]==resource.RLIM_INFINITY or old[1]>=maximum,'hard file limit too small')
    signal.signal(signal.SIGXFSZ,lambda *_:(_ for _ in ()).throw(OSError('bounded output file limit')))
    resource.setrlimit(resource.RLIMIT_FSIZE,(maximum,old[1]))
    try:yield
    finally:resource.setrlimit(resource.RLIMIT_FSIZE,old);signal.signal(signal.SIGXFSZ,handler)


@contextmanager
def shorter_deadline(seconds):
    prior=signal.getitimer(signal.ITIMER_REAL);started=time.monotonic()
    need(prior[0]>0,'whole-job deadline must already be active')
    signal.setitimer(signal.ITIMER_REAL,min(float(seconds),prior[0]))
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,max(.001,prior[0]-(time.monotonic()-started)),prior[1])


def observer_class():
    import torch
    from hif4_observer import HiF4Observer
    class MatchedObserver(HiF4Observer):
        series='A5_HWG_MATCHED_FULL160_HIF4_20261030'
        def __init__(self,*args,writer,ordinal,check,**kwargs):
            super().__init__(*args,rows=512,blocks=tuple(range(40)),**kwargs)
            self.writer,self.ordinal,self.check=writer,ordinal,check
            self.full_writes=0;self.d2h_bytes=0;self.observer_ops={'sample_index_select':0,'sample_GELU_tanh':0}
        def _sample(self,scope,tensor):
            self.observer_ops['sample_index_select']+=1;need(self.observer_ops['sample_index_select']<=24*160,'sample index budget')
            return super()._sample(scope,tensor)
        def _new_scope(self,tag,x):
            scope=super()._new_scope(tag,x)
            need(tag['expert']==('high_noise_model'if tag['step']<2 else'low_noise_model'),'expert mismatch')
            rows,starts=matched_rows(tag['grid'],self.ordinal,self.seed)
            scope.update(row_count=len(rows),selected_rows=rows,tile_starts=starts,
                         latent_t_h_w=[[r//1508,(r%1508)//52,r%52]for r in rows],
                         sampling='ONE_GLOBAL_ALIGNED_M64_RASTER_TILE_PER_TAU_NOT_WHOLE_SPATIAL_COVERAGE',
                         inclusion_probabilities=None,design_weights=None,
                         coordinate_provenance='NATIVE_GRID_AND_ORIGINAL_FLATTEN_INDICES_FIXED_ACROSS_ALL160',
                         preQ1_fullM='NOT_CAPTURED',fullM_boundary='ACTUAL_POSTQ1_NATIVE_V1')
            scope['_indices']=torch.tensor(rows,dtype=torch.long,device=x.device)
            # Reserve the entire declared packet before inherited sampled D2H.
            self.writer.reserve((9*5120+4*13824)*1344*2+2*5120*2+(1<<16))
            self.writer.note('SITE_OBSERVATION_BEGIN_D2H_UNKNOWN',step=tag['step'],block=tag['block'])
            return scope
        def __enter__(self):
            super().__enter__()
            from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
            import torch.nn.functional as F
            old_gelu=F.gelu;observer=self
            def counted_gelu(x,*args,**kwargs):
                if observer.pending is not None and list(x.shape)==[1344,13824]:
                    observer.observer_ops['sample_GELU_tanh']+=1;need(observer.observer_ops['sample_GELU_tanh']<=160,'sample GELU budget')
                return old_gelu(x,*args,**kwargs)
            self.methods.append((F,'gelu',old_gelu));F.gelu=counted_gelu
            old=adapter.native_hif4_activation_qdq;observer=self
            def observed(activation,*,force_fp32=True):
                result=old(activation,force_fp32=force_fp32)
                scope=observer.pending
                if scope is not None and scope.get('_current_linear','').endswith('.ffn.0'):
                    need('full_V1'not in scope,'duplicate fullM Q1')
                    flat=result.reshape(-1,result.shape[-1]);need(list(flat.shape)==[31668,5120]and flat.dtype==torch.bfloat16,'fullM V1 shape/dtype')
                    observer.writer.reserve(flat.numel()*flat.element_size()+(1<<16));observer.check()
                    # Synchronous .cpu() establishes the original production
                    # stream dependency before the runtime may reuse a buffer.
                    cpu=flat.detach().cpu().contiguous();observer.d2h_bytes+=cpu.numel()*cpu.element_size()
                    need(torch.equal(cpu.index_select(0,scope['_indices'].cpu()),scope['_samples']['V1']),'full/sample Q1 identity')
                    file=observer.root/f'fullV1_step{scope["step"]}_block{scope["block"]:02d}.safetensors'
                    scope['full_V1']={**observer.writer.tensor(file,{'V1':cpu}),'raw_tensor_sha256':observer.writer.raw_sha(cpu),'shape':list(cpu.shape),'dtype':'bfloat16','runtime_stride':list(flat.stride()),'runtime_data_ptr':flat.data_ptr(),'boundary':'ORIGINAL_FULL_NATIVE_Q1_RETURN_BEFORE_W1'}
                    observer.full_writes+=1
                return result
            self.methods.append((adapter,'native_hif4_activation_qdq',old));adapter.native_hif4_activation_qdq=observed
            return self
        def _save_scope(self,scope):
            need(len(scope.get('QDQ_calls',[]))==2 and scope['effective_W2'].get('runtime_cache_bytes_match'),'missing actual QDQ/W2')
            need(set(scope['_samples'])==self.required_tensors and all(scope['checks'].values()),'sample boundary checks')
            samples=scope['_samples'];scope['gate_native_shape']=list(samples['gate_broadcast'].shape)
            samples['gate_broadcast']=samples['gate_broadcast'].reshape(-1)
            need(all(t.dtype==torch.bfloat16 for t in samples.values()),'declared BF16 tensor budget changed')
            need({k:list(t.shape)for k,t in samples.items()}==expected_sample_shapes(),'declared sample shapes changed')
            need(all(bool(torch.isfinite(t).all())for t in samples.values()),'nonfinite observation')
            file=self.root/f'step{scope["step"]}_block{scope["block"]:02d}.safetensors'
            payload=self.writer.tensor(file,samples)
            public={k:v for k,v in scope.items()if not k.startswith('_')}
            public.update(payload=payload,tensor_schema={k:{'shape':list(t.shape),'dtype':'bfloat16','raw_tensor_sha256':self.writer.raw_sha(t)}for k,t in samples.items()},
                          Delta_status='DERIVED_NATIVE_DTYPE_Y_TIMES_GATE_CHECKED_AGAINST_ACTUAL_POSTWRITE',
                          source_id=self.source_id,native_source_ordinal=self.ordinal,seed=self.seed,numeric_family='HIF4_FAKE_QDQ_BF16')
            meta=self.root/f'step{scope["step"]}_block{scope["block"]:02d}.json';self.writer.json(meta,public)
            self.completed.append({'step':scope['step'],'block':scope['block'],'metadata':bind(meta,self.writer.hf)})
            self.seen.add((scope['step'],scope['block']));self.d2h_bytes+=sum(t.numel()*t.element_size()for t in samples.values());self._manifest();self.writer.note('SITE_OBSERVATION_COMPLETE',step=scope['step'],block=scope['block']);self.check()
        def _manifest(self,error=None):
            self.writer.json(self.root/'CAPTURE_MANIFEST.json',{'status':self.status,'series':self.series,'source_id':self.source_id,'seed':self.seed,'blocks':list(range(40)),'steps':[0,1,2,3],'expected_scopes':160,'completed_scopes':len(self.completed),'scopes':self.completed,'full_V1_files':self.full_writes,'logical_observation_D2H_bytes':self.d2h_bytes,'observer_actual_ops':self.observer_ops,'error':None if error is None else type(error).__name__,'postQ1_replay_not_fullQ1_reconstruction':True})
    return MatchedObserver


class StateTap:
    def __init__(self,torch,pipe,folder,writer,check):self.torch,self.pipe,self.folder,self.writer,self.check=torch,pipe,folder,writer,check;self.states={};self.stack=ExitStack();self.draws=0
    def save(self,value,name):
        torch=self.torch;need(name not in self.states and list(value.shape)==[16,21,58,104],'state identity')
        expected=torch.float32 if name=='raw_noise_FP32'or name.startswith('flow')else torch.bfloat16
        need(value.dtype==expected,'state dtype')
        self.writer.reserve(value.numel()*value.element_size()+(1<<16));cpu=value.detach().cpu().contiguous();need(bool(torch.isfinite(cpu).all()),'nonfinite state')
        self.states[name]={**self.writer.tensor(self.folder/(name+'.safetensors'),{'value':cpu}),'shape':list(cpu.shape),'dtype':str(cpu.dtype),'raw_tensor_sha256':self.writer.raw_sha(cpu)}
    def __enter__(self):
        torch=self.torch;s=self.pipe.runner.scheduler;oldnoise=s.prepare_latents;oldpre=s.step_pre;oldpost=s.step_post
        def noise(seed,shape,dtype=torch.float32):
            oldrand=torch.randn
            def randn(*a,**kw):
                need(self.draws==0 and list(a)==[16,21,58,104]and kw.get('dtype')==torch.float32 and kw.get('generator')is s.generator,'raw draw identity')
                result=oldrand(*a,**kw);self.draws+=1;self.save(result,'raw_noise_FP32');return result
            with patch(torch,'randn',randn):return oldnoise(seed,shape,dtype=dtype)
        def pre(*a,**kw):
            result=oldpre(*a,**kw)
            if int(s.step_index)==0:self.save(s.latents,'initial_noise')
            return result
        def post(*a,**kw):
            step=int(s.step_index);self.save(s.noise_pred,f'flow_step{step}');result=oldpost(*a,**kw);self.save(s.latents,f'latent_step{step}');self.check();return result
        for name,fn in [('prepare_latents',noise),('step_pre',pre),('step_post',post)]:self.stack.enter_context(patch(s,name,fn))
        return self
    def __exit__(self,*a):self.stack.close()
    def finish(self):need(self.draws==1 and set(self.states)==STATE_NAMES,'ten state coverage')


def authorize(p,protocol,job,kind):
    need(p['resources']==RES and p['payload_budget']==payload_budget(),'fixed resource/payload scope differs')
    need(p['design']['video_geometry_inspection']==VIDEO_INSPECTION,'opaque decoder scope differs')
    active=read(ROOT/'research/ACTIVE_RESEARCH.json')
    scope_id=p['id']+('_VERIFY'if kind=='verify'else'')
    scopes=[s for s in active['experiment_authority']['current_user_authorized_scopes']if s.get('id')==scope_id]
    need(len(scopes)==1,'exact ACTIVE scope missing');s=scopes[0]
    need(s['execution_status']in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')and s['protocol_sha256']==sha(protocol)and s['job_spec_sha256']==sha(job),'ACTIVE binding differs')
    need(s['session_max_GPUs']==4 and s['max_gpu_cards']==(0 if kind=='verify'else 1),'ACTIVE GPU cap')
    need(sum(v.get('max_gpu_cards',0)for v in active['experiment_authority']['current_user_authorized_scopes']if v.get('execution_status')=='RUNNING_FIXED_SCOPE'and v.get('native_group_id')=='A5_HWG_CAPTURE_20261030')<=4,'four GPU session cap')
    review=read(check_binding(ROOT,s['review']))
    need(review['state']=='REVIEWER_RECHECK_APPROVED'and review['recheck_disposition']=='APPROVE'and review['author_agent_id']!=review['reviewer_agent_id']and review['scope_id']==scope_id and review['approval_scope_fingerprint']==canonical(p),'independent exact review missing')
    need(review['job_spec_sha256']==sha(job),'reviewed job differs')
    for b in p['repo_sources']:check_binding(ROOT,b)
    for b in review['review_evidence']:check_binding(ROOT,b)
    from worker_identity import require_worker
    require_worker();j=read(job)
    need(j['id']==scope_id and j['protocol_sha256']==sha(protocol)and Path(j['protocol_path']).resolve()==protocol.resolve()and Path(j['cwd']).resolve()==ROOT,'job identity')
    need(Path(j['run_json']).resolve().is_relative_to(Path(j['hf_root']).resolve())and j['argv'][0]=='env','job confinement')
    need(s['reviewed_git_commit']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'frozen local commit differs')
    return s


def private_assets(p,path):
    need(sha(path)==p['private_assets_sha256'],'private asset locator changed')
    data=read(path);profile=Path(data['worker_profile']).resolve()
    need(Path(os.environ.get('FFNDS_WORKER_CONFIG','')).resolve()==profile and sha(profile)==p['private_worker_profile_sha256']==data['worker_profile_sha256'],'worker profile changed')
    return data


def runtime_assets(p,private):
    need(sha(ROOT/'.runtime/model_view.json')==p['model_view']['sha256'],'logical model view differs')
    view=read(ROOT/'.runtime/model_view.json');need(view['status']=='PASS','model view status')
    for row in view['files']:
        path=Path(row['view_path']);need(path.stat().st_size==row['bytes']and path.samefile(row['blob'])and sha(path)==row['sha256'],'actual original model SHA differs')
    cache=Path(private['cache_path']);need(cache.stat().st_size==p['cache']['bytes'],'cache size')
    need(sha(cache)==p['cache']['sha256'],'full actual cache SHA differs')
    check_binding(Path(private['hf_root']),p['case']['image'])
    return view,cache


@execution_bootstrap('run')
def run(a):
    global CONTROL_READ
    started=CURRENT_BOOTSTRAP.started;asset_reads=CURRENT_BOOTSTRAP.budget
    CONTROL_READ=asset_reads
    p=read(a.protocol);authorize(p,a.protocol,a.job_spec,'run');private=private_assets(p,a.private_assets)
    need(sorted(os.sched_getaffinity(0))==p['cpu_affinity'],'four CPU affinity differs')
    hf=Path(private['hf_root']).resolve();out=hf/p['output_HF_relative'];need(not out.exists(),'single attempt output already exists');out.mkdir()
    monitor=None;report={'id':p['id'],'protocol_sha256':sha(a.protocol),'status':'RUNNING','numeric_family':'HIF4_FAKE_QDQ_BF16','source':p['case'],'phases':[],'automatic_retry':False,'quality_or_speed_accepted':False}
    def check():
        need(time.monotonic()-started<=RES['wall_seconds'],'wall budget exceeded')
        need(available_memory()>=RES['host_available_min_bytes'],'host memory reserve exceeded')
        need(shutil.disk_usage(out).free>=4<<30,'free disk reserve exceeded')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RES['peak_RSS_max_bytes'],'RSS budget exceeded')
        if monitor:monitor.check()
        if 'torch'in sys.modules:need(sys.modules['torch'].cuda.max_memory_reserved()<=RES['GPU_reserved_max_bytes'],'GPU reserve exceeded')
    writer=Writer(out,hf,check);persist=lambda:writer.json(out/'RUN.json',report)
    persist()
    finder=None
    try:
        need(shutil.disk_usage(out).free>=RES['free_disk_min_bytes'],'initial disk budget')
        with HostReservation(),bounded_files():
            for arm in ['plain','observed']:writer.hold_file(out/arm/'output.mp4',RES['maximum_single_file_bytes'])
            os.environ.update(DTYPE='BF16',SENSITIVE_LAYER_DTYPE='BF16',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HIF4_MAIN_REPO=str(ROOT/'.runtime/code/HiF4_Sp'))
            sys.path[:0]=[str(ROOT/'.runtime/code/LightX2V'),str(ROOT/'.runtime/code/HiF4_Sp'),str(ROOT/'.runtime/code/Wan2.2-I2V-A14B-W4A4')]
            local=Path(private['local_package_root']);need(local.is_dir(),'local package missing');sys.path.insert(0,str(local))
            package=p['local_transformers'];inventory=read(check_binding(hf,package['inventory']))
            need(canonical(inventory['files'])==package['inventory_sha256'],'package inventory identity')
            for ref in inventory['files']:check_binding(local/'transformers',ref)
            finder=BoundTransformersFinder(local/'transformers',inventory['files']);sys.meta_path.insert(0,finder)
            import torch
            torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
            from startup_gpu_reservation import reserve_before_assets
            reservation,monitor,report['CUDA_reservation']=reserve_before_assets(torch)
            need(torch.cuda.device_count()==1 and'A100'in torch.cuda.get_device_name(),'one assigned A100 required')
            need({n:importlib.metadata.version(n)for n in p['runtime_versions']}==p['runtime_versions'],'software versions differ')
            view,cache=runtime_assets(p,private);check();report['asset_identity_validation_reads']=asset_reads.summary();persist()
            from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
            from quant import hif4_backend
            hif4_backend.get_hif4_api()
            base=Path(view['view']);pipe=adapter.build_lightx2v_distill_pipeline(base_model_dir=base,high_noise_ckpt=base/'high_noise_model/wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors',low_noise_ckpt=base/'low_noise_model/wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors')
            adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
            runtime=adapter.HIF4CacheRuntime(cache);adapter.activate_lightx2v_hooks(runtime=runtime)
            expected={f'blocks.{b}.{n}'for b in range(40)for n in ['self_attn.q','self_attn.k','self_attn.v','self_attn.o','cross_attn.q','cross_attn.k','cross_attn.v','cross_attn.o','ffn.0','ffn.2']}
            need(all(set(runtime.records[e])==expected for e in ['high_noise_model','low_noise_model']),'800 cached modules')
            report['actual_native_import_origins']=validate_import_origins(p)
            report['model_setup_seconds']=time.monotonic()-started;persist()
            for arm in ['plain','observed']:
                check();folder=out/arm;folder.mkdir();pipe.runner.scheduler.generator=None
                writer.begin_file(folder/'output.mp4')
                for model in pipe.runner.model.model:model.transformer_infer.offload_manager.need_init_first_buffer=True
                before=runtime.report();observer=None
                if arm=='observed':observer=observer_class()(folder/'ffn',runtime=runtime,cache_manifest=p['cache'],source_id=p['case']['source_id'],seed=42,writer=writer,ordinal=p['case']['native_source_ordinal'],check=check)
                tap=StateTap(torch,pipe,folder,writer,check);at=time.monotonic()
                with torch.inference_mode(),tap,ExitStack()as stack:
                    if observer is not None:stack.enter_context(observer)
                    pipe.generate(seed=42,image_path=str(hf/p['case']['image']['path']),prompt=p['case']['prompt'],negative_prompt='',save_result_path=str(folder/'output.mp4'))
                torch.cuda.synchronize();tap.finish();after=runtime.report()
                delta={k:after[k]-before[k]for k in ['linear_forward_calls','activation_qdq_calls','cache_misses']}
                need(delta=={'linear_forward_calls':1600,'activation_qdq_calls':1600,'cache_misses':0},'native QDQ/linear coverage')
                import imageio.v2 as imageio
                decoding={'arm':arm,**VIDEO_INSPECTION,'status':'INFLIGHT_UNKNOWN','known_input_video_bytes':(folder/'output.mp4').stat().st_size}
                report.setdefault('video_geometry_inspection',[]).append(decoding)
                writer.note('VIDEO_GEOMETRY_INSPECTION_BEGIN_OPAQUE_IO_UNKNOWN',**decoding);persist();decode_started=time.monotonic()
                try:
                    with shorter_deadline(VIDEO_INSPECTION['deadline_seconds_per_video']):
                        with imageio.get_reader(folder/'output.mp4')as reader:info=reader.get_meta_data();frames=reader.count_frames()
                    decoding['status']='COMPLETE_CALLS_INTERNAL_IO_BYTES_UNKNOWN'
                finally:
                    decoding['wall_seconds']=time.monotonic()-decode_started
                    writer.note('VIDEO_GEOMETRY_INSPECTION_RETURNED',**decoding);persist()
                need(frames==81 and abs(float(info['fps'])-16)<.001 and list(info['size'])==[832,464],'full video geometry')
                phase={'arm':arm,'states':tap.states,'video':writer.external(folder/'output.mp4'),'native_counter_delta':delta,'wall_seconds_including_capture':time.monotonic()-at,'video_geometry':[frames,832,464,16]}
                if observer is not None:
                    need(observer.status=='PASS_CAPTURE_STRUCTURE_ONLY'and observer.full_writes==160 and observer.seen=={(s,b)for s in range(4)for b in range(40)},'full160 observation missing')
                    need(observer.observer_ops['sample_GELU_tanh']==160,'all160 additional observer GELU controls required')
                    phase['capture_manifest']=bind(folder/'ffn/CAPTURE_MANIFEST.json',hf)
                report['phases'].append(phase);persist()
                if arm=='observed':
                    old=report['phases'][0]
                    need(all(phase['states'][n]['sha256']==old['states'][n]['sha256']for n in STATE_NAMES),'10-state neutral mismatch')
                    need(phase['video']['sha256']==old['video']['sha256'],'video neutral mismatch')
                    report['ten_states_and_video_neutral']=True
            report['actual_native_import_origins']=validate_import_origins(p)
            adapter.deactivate_lightx2v_hooks();report['status']='COMPLETE_MATCHED_FULL160_CAPTURE_EXIT_PENDING'
    except BaseException as e:
        report.update(status='FAILED_MATCHED_CAPTURE_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:1000]);raise
    finally:
        CONTROL_READ=None
        signal.alarm(0)
        writer.finish_pending()
        if monitor:
            monitor.close();report['assigned_GPU_monitor']=monitor.report()
            if report['assigned_GPU_monitor']['violations']:report['status']='FAILED_ASSIGNED_GPU_MONITOR_PRESERVED_NO_RETRY'
        report.update(wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,output_accounting_before_final_RUN=writer.summary(),output_written_bytes=writer.written,raw_tensor_SHA_memory_read_bytes=writer.raw_sha_bytes,file_SHA_read_bytes=writer.file_sha_bytes)
        report['asset_identity_validation_reads']=asset_reads.summary()
        if finder:
            report['actual_transformers_import_origins']=finder.origins
            if finder in sys.meta_path:sys.meta_path.remove(finder)
        if 'torch'in sys.modules:report['peak_CUDA_reserved_bytes']=sys.modules['torch'].cuda.max_memory_reserved()
        writer.check=lambda:None  # Bounded postmortem metadata only, no more model operations.
        persist()
        writer.journal.close()
        if monitor:monitor.check()


def verify_tensor(path,expected_schema,meter):
    """One sequential file pass checks file and each raw tensor SHA together."""
    before=path.stat();file_hash=hashlib.sha256();result={}
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    with path.open('rb')as f:
        prefix=f.read(8);need(len(prefix)==8,'short safetensors prefix')
        n=struct.unpack('<Q',prefix)[0];need(0<n<=1<<20,'safetensors header bound')
        raw_header=f.read(n);header=json.loads(raw_header);file_hash.update(prefix);file_hash.update(raw_header);meter(n+8)
        rows=sorted(((v['data_offsets'][0],k,v)for k,v in header.items()if k!='__metadata__'))
        need({k for _,k,_ in rows}==set(expected_schema),'tensor field set differs')
        cursor=0
        for start,key,meta in rows:
            end=meta['data_offsets'][1];expected=expected_schema[key]
            need(start==cursor and end>=start and meta['shape']==expected['shape'],'tensor offsets/shape differ')
            dtype='BF16'if expected['dtype']in ('bfloat16','torch.bfloat16')else'F32'
            need(meta['dtype']==dtype and end-start==math.prod(meta['shape'])*(2 if dtype=='BF16'else 4),'tensor dtype/length differs')
            raw_hash=hashlib.sha256();remaining=end-start
            while remaining:
                b=f.read(min(8<<20,remaining));need(bool(b),'short payload');remaining-=len(b);raw_hash.update(b);file_hash.update(b);meter(len(b))
            need(raw_hash.hexdigest()==expected['raw_tensor_sha256'],'raw tensor SHA differs')
            result[key]={'shape':meta['shape'],'dtype':meta['dtype'],'data_offset':8+n+start}
            cursor=end
        need(not f.read(1)and identity(path.stat())==identity(before)and identity(os.fstat(f.fileno()))==identity(before)and f.tell()==before.st_size,'tensor changed or trailing bytes')
    return file_hash.hexdigest(),result


@execution_bootstrap('verify')
def verify(a):
    global CONTROL_READ
    started=CURRENT_BOOTSTRAP.started
    def check():
        need(time.monotonic()-started<=VERIFY_RES['wall_seconds'],'verifier clock budget')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=VERIFY_RES['peak_RSS_max_bytes'],'verifier RSS budget')
    metadata=CURRENT_BOOTSTRAP.budget;metadata.check=check;CONTROL_READ=metadata
    p=read(a.protocol);authorize(p,a.protocol,a.job_spec,'verify');private=private_assets(p,a.private_assets)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==''and len(os.sched_getaffinity(0))==1,'CPU verification isolation')
    hf=Path(private['hf_root']).resolve();source=hf/p['output_HF_relative'];dest=source.parent.parent/'verification/worker/results';dest.mkdir()
    report={'id':p['id']+'_VERIFY','protocol_sha256':sha(a.protocol),'status':'VERIFYING','read_bytes':0,'files_verified':0,'automatic_retry':False};output=dest/'RUN.json'
    def persist():
        report['payload_read_ledger']=payload.summary();report['metadata_read_ledger']=metadata.summary()
        b=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode();need(len(b)<=1<<20,'verifier report output bound')
        tmp=output.with_suffix('.tmp');tmp.write_bytes(b);tmp.replace(output)
    payload=ReadBudget(VERIFY_RES['read_bytes_max'],check,persist);metadata.persist=persist
    def meter(n):payload.after(n);report['read_bytes']+=n
    persist()
    try:
        receipt=read(source.parent/'EXIT_RECEIPT.json');r=read(source/'RUN.json')
        parent=[s for s in read(ROOT/'research/ACTIVE_RESEARCH.json')['experiment_authority']['current_user_authorized_scopes']if s.get('id')==p['id']];need(len(parent)==1,'producer registration missing')
        need(receipt['process_exited']is True and receipt['child_returncode']==0 and receipt['timeout']is False and receipt['automatic_retry']is False,'producer did not EXIT0')
        need(receipt['spec_sha256']==parent[0]['job_spec_sha256']and receipt['protocol_sha256']==sha(a.protocol)and receipt['run_sha256']==sha(source/'RUN.json'),'producer final receipt binding')
        need(r['status']=='COMPLETE_MATCHED_FULL160_CAPTURE_EXIT_PENDING'and r['ten_states_and_video_neutral']is True and r['id']==p['id'],'producer completion identity')
        need(len(r['phases'])==2 and [v['arm']for v in r['phases']]==['plain','observed'],'phase identity')
        for phase in r['phases']:
            need(set(phase['states'])==STATE_NAMES,'state coverage')
            for name,state in phase['states'].items():
                path=hf/state['path'];need(path.stat().st_size==state['bytes'],'state size')
                dtype='torch.float32'if name=='raw_noise_FP32'or name.startswith('flow')else'torch.bfloat16'
                need(state['shape']==[16,21,58,104]and state['dtype']==dtype,'independent fixed state schema')
                need(path.resolve().is_relative_to(source/phase['arm']),'state confinement')
                payload.before(state['bytes'],state['path']);observed,_=verify_tensor(path,{'value':state},meter);payload.finish();need(observed==state['sha256'],'state file SHA');report['files_verified']+=1
            video=phase['video'];path=(hf/video['path']).resolve();need(path==source/phase['arm']/'output.mp4'and path.stat().st_size==video['bytes'],'video identity')
            payload.before(video['bytes'],video['path']);h=hashlib.sha256()
            with path.open('rb')as f:
                for b in iter(lambda:f.read(8<<20),b''):h.update(b);meter(len(b))
            payload.finish();need(h.hexdigest()==video['sha256'],'video file SHA');report['files_verified']+=1
        old,new=r['phases'];need(all(old['states'][n]['sha256']==new['states'][n]['sha256']for n in STATE_NAMES)and old['video']['sha256']==new['video']['sha256'],'rechecked neutral SHA')
        manifest=read(check_binding(hf,new['capture_manifest']));need(manifest['status']=='PASS_CAPTURE_STRUCTURE_ONLY'and manifest['full_V1_files']==160 and manifest['completed_scopes']==160,'capture manifest completeness')
        need({(s['step'],s['block'])for s in manifest['scopes']}=={(s,b)for s in range(4)for b in range(40)},'all160 sites')
        rows,starts=matched_rows([21,29,52],p['case']['native_source_ordinal'],42)
        seen={'paths':set(),'inodes':set()}
        for packet in manifest['scopes']:
            meta_ref=packet['metadata'];d=read(check_binding(hf,meta_ref))
            validate_packet(d,packet,p,hf,source,seen)
            for ref,schema in [(d['payload'],d['tensor_schema']),(d['full_V1'],{'V1':d['full_V1']})]:
                path=hf/ref['path'];need(path.stat().st_size==ref['bytes'],'observed tensor size')
                payload.before(ref['bytes'],ref['path']);observed,headers=verify_tensor(path,schema,meter);payload.finish();need(observed==ref['sha256'],'observed tensor file SHA');report['files_verified']+=1
            # Independently recover the actual fixed-row bytes from fullM V1.
            h=hashlib.sha256();offset=headers['V1']['data_offset']
            payload.before(21*64*5120*2,d['full_V1']['path']+':FIXED21_M64_SLICES')
            with (hf/d['full_V1']['path']).open('rb')as f:
                for start in starts:
                    f.seek(offset+start*5120*2);b=f.read(64*5120*2);need(len(b)==64*5120*2,'fullM sampled slice short');h.update(b);meter(len(b))
            payload.finish()
            need(h.hexdigest()==d['tensor_schema']['V1']['raw_tensor_sha256'],'fullM/sample actual Q1 bytes differ')
            report['matched_fullM_sample_sites']=report.get('matched_fullM_sample_sites',0)+1
            persist()
        report.update(status='PASS_INDEPENDENT_FILE_AND_RAW_SHA_FULL160_EXIT_PENDING',producer_RUN=bind(source/'RUN.json',hf),producer_exit=bind(source.parent/'EXIT_RECEIPT.json',hf),ten_states_and_video_neutral=True)
    except BaseException as e:report.update(status='FAILED_VERIFICATION_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:1000]);raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-started;persist();CONTROL_READ=None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['run','verify']);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--job-spec',type=Path,required=True);ap.add_argument('--private-assets',type=Path,required=True);a=ap.parse_args()
    if a.command=='run':
        with bounded_files():run(a)
    else:
        with bounded_files(4<<20):verify(a)


if __name__=='__main__':main()
