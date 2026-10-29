"""Offline-source-reviewed Llama3 byte bridge, single anonymous acquisition scope.

Draft and inspect do not fetch or convert. Run requires an independently
reviewed, exactly registered ACTIVE scope and a separately bound supervisor.
This program never imports torch, runs a model, or accesses a gated endpoint.
"""
import argparse,base64,hashlib,http.client,json,os,resource,signal,ssl,sys,time,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
HF=Path('/home/xiang/.cache/huggingface')
ID='E0_TEAL_LLAMA3_TOKENIZER_BYTE_BRIDGE_20261029'
SELF='tools/e0_teal_llama3_tokenizer_byte_bridge.py';TEST='tests/test_e0_teal_llama3_tokenizer_byte_bridge.py'
HELPERS=('tools/worker_identity.py','tools/durable_worker_job.py',
         'research/literature/20261029/P0_TEAL_TOKENIZER_BRIDGE_SOURCE_LEDGER_20261029.json',
         'research/literature/20261029/P0_TEAL_PUBLIC_REDISTRIBUTION_METADATA_20261029.json',
         'research/method_discussion/discoveries/E0/p0-teal-llama3-tokenizer-byte-bridge-design-20261029.md')
OUTPUT='ffnds-derived/e0_teal_llama3_tokenizer_byte_bridge_20261029/attempt1/worker/results'
REV='315b20096dc791d381d514deb5f8bd9c8d6d3061'
INPUT_URL='https://huggingface.co/NousResearch/Meta-Llama-3-8B/resolve/'+REV+'/tokenizer.json'
INPUT_BYTES=9085698;INPUT_BLOB='b197f72effb9d5ed16ee0f5663e11e4cfac2ba62'
TARGET_BYTES=2183982;TARGET_SHA='82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55'
HF_BINDINGS={
 'ffnds-research-sources/p0_teal_full_llm_20261029/tokenizer_bridge_sources/tiktoken_load.py':'43c3cfb29dc48a74e70e0c85bb01cfd96a3de069a98de8778ccab7f72a2836cc',
 'ffnds-research-sources/p0_teal_full_llm_20261029/tokenizer_bridge_sources/transformers_gpt2_tokenizer.py':'e7e13dec555c9315d894fbb31b084c6ef23c551a208cd6e856ec6413be8fe83d',
 'ffnds-research-sources/p0_teal_full_llm_20261029/tokenizer_bridge_sources/transformers_convert_slow_tokenizer.py':'9851ca3b96e111f4deb28f59c65529007f1fe02e956d06089a220a846176d564',
 'ffnds-research-sources/p0_teal_full_llm_20261029/redistribution_metadata/nous3_LICENSE':'475211637354ce4c14b9c3dacccbefbaba735fe1b0db97d6c4fd11cd1356819f',
 'ffnds-research-sources/p0_teal_full_llm_20261029/redistribution_metadata/nous3_README.md':'28cd72a6a5d4c29aa8af406147e941eacc51aab61591010b5c162e36ca7741cd',
 'ffnds-research-sources/p0_teal_full_llm_20261029/redistribution_metadata/nous3_api.json':'5855be8df9dd7cbbeb36d56ca7bd566b663b0b7e4bcc037d2850786b8cfa30e9',
 'ffnds-research-sources/p0_teal_full_llm_20261029/redistribution_metadata/official3_api.json':'7168d07b0290e8c2b0a0047ec442dbd4bdeb3d00979c903ed2b192505e1a18ad'}
DESIGN={'public_repo':'NousResearch/Meta-Llama-3-8B','revision':REV,'input_url':INPUT_URL,
 'input_bytes':INPUT_BYTES,'input_git_blob_sha1':INPUT_BLOB,'target_bytes':TARGET_BYTES,'target_sha256':TARGET_SHA,
 'license_url':'https://huggingface.co/NousResearch/Meta-Llama-3-8B/blob/'+REV+'/LICENSE',
 'notice_url':'https://huggingface.co/NousResearch/Meta-Llama-3-8B/blob/'+REV+'/README.md',
 'network':'ANONYMOUS_HTTPS_FIXED_PATH_BOUNDED_SAME_HOST_REDIRECTS_NO_PROXIES_NO_CREDENTIALS',
 'format':'RANK_SORTED_BASE64_SPACE_DECIMAL_LF','tokenizer_runtime':'NOT_RUN',
 'vocab_max_count':200000,'added_tokens_max_count':1024,'max_token_bytes':4096,'single_byte_coverage':256,
 'rank_policy':'CONTIGUOUS_ZERO_BASED','redirect_limit':2,'chunk_bytes':65536,'cpu_affinity':[0]}
RESOURCES={'GPU':0,'CPU_threads':1,'wall_seconds':120,'outer_seconds':180,'peak_address_space_bytes':512<<20,
 'peak_RSS_bytes':512<<20,'download_bytes':10<<20,'new_HF_bytes':16<<20,'automatic_retry':False}
LIMITS={'http_requests':3,'input_download':1,'tokenizer_conversion':1,'model_calls':0,'GPU_calls':0,'torch_imports':0}

def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def binding(p,root):
    p=Path(p).resolve();root=Path(root).resolve();need(p.is_relative_to(root) and p.is_file(),'Binding outside root')
    return {'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)}
def bound(root,b):
    p=(Path(root)/b['path']).resolve();need(p.is_relative_to(Path(root).resolve()) and p.is_file() and p.stat().st_size==b['bytes'] and sha(p)==b['sha256'],'Binding changed: '+b['path']);return p

def byte_unicode_inverse():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];extra=0
    for b in range(256):
        if b not in bs:bs.append(b);cs.append(256+extra);extra+=1
    inv={chr(c):b for b,c in zip(bs,cs)};need(len(inv)==256,'Byte map not bijective');return inv

def no_duplicate_pairs(pairs):
    d={}
    for k,v in pairs:
        need(k not in d,'Duplicate JSON key');d[k]=v
    return d

def parse_tokenizer(data):
    need(isinstance(data,bytes),'Expected JSON bytes')
    try:obj=json.loads(data.decode('utf-8'),object_pairs_hook=no_duplicate_pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    except (UnicodeDecodeError,json.JSONDecodeError) as e:raise ValueError('Invalid UTF-8/JSON') from e
    need(isinstance(obj,dict) and isinstance(obj.get('model'),dict) and obj['model'].get('type')=='BPE','Expected BPE model')
    vocab=obj['model'].get('vocab');need(isinstance(vocab,dict) and 256<=len(vocab)<=DESIGN['vocab_max_count'],'Vocabulary count invalid')
    inv=byte_unicode_inverse();rank_to_token={};tokens=set()
    for text,rank in vocab.items():
        need(isinstance(text,str) and text and type(rank) is int and 0<=rank<len(vocab),'Invalid token/rank')
        try:token=bytes(inv[c] for c in text)
        except KeyError as e:raise ValueError('Unknown ByteLevel codepoint') from e
        need(0<len(token)<=DESIGN['max_token_bytes'] and rank not in rank_to_token and token not in tokens,'Duplicate or invalid token/rank')
        rank_to_token[rank]=token;tokens.add(token)
    need(set(rank_to_token)==set(range(len(vocab))) and all(bytes([i]) in tokens for i in range(256)),'Ranks or byte coverage invalid')
    added=obj.get('added_tokens');need(isinstance(added,list) and 0<len(added)<=DESIGN['added_tokens_max_count'],'Special token list invalid')
    ids=set();contents=set()
    for a in added:
        need(isinstance(a,dict) and type(a.get('id')) is int and len(vocab)<=a['id']<len(vocab)+DESIGN['added_tokens_max_count'] and a.get('special') is True
             and isinstance(a.get('content'),str) and a['content'] and a['content'] not in vocab
             and a['id'] not in ids and a['content'] not in contents,'Special token conflict')
        for key in ('single_word','lstrip','rstrip','normalized'):
            need(key not in a or type(a[key]) is bool,'Invalid special token option')
        ids.add(a['id']);contents.add(a['content'])
    return rank_to_token,{'vocab_count':len(vocab),'added_count':len(added),'added_id_min':min(ids),'added_id_max':max(ids),'byte_coverage':256}

def write_canonical(rank_to_token,handle,limit,progress=None):
    n=0
    for rank in range(len(rank_to_token)):
        line=base64.b64encode(rank_to_token[rank])+b' '+str(rank).encode('ascii')+b'\n'
        n+=len(line);need(n<=limit,'Candidate output limit exceeded');need(handle.write(line)==len(line),'Short candidate write')
        if progress is not None:progress(n)
    return n

def verify_canonical(path,expected):
    count=0;seen=set()
    with Path(path).open('rb') as f:
        for line in f:
            need(line.endswith(b'\n') and line.count(b' ')==1,'Invalid dump line')
            encoded,rawrank=line[:-1].split(b' ');need(rawrank==str(count).encode('ascii'),'Noncanonical rank')
            try:token=base64.b64decode(encoded,validate=True)
            except Exception as e:raise ValueError('Invalid base64') from e
            need(base64.b64encode(token)==encoded and token==expected.get(count) and token not in seen,'Roundtrip mismatch')
            seen.add(token);count+=1
    need(count==len(expected),'Roundtrip count differs');return count

def git_blob_hash(path):
    path=Path(path);n=path.stat().st_size;h=hashlib.sha1(b'blob '+str(n).encode('ascii')+b'\x00')
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def worker_output_bytes(worker):
    return sum(p.stat().st_size for p in Path(worker).rglob('*') if p.is_file() and not p.is_symlink())

def allowed_url(url):
    u=urllib.parse.urlsplit(url);allowed={'/NousResearch/Meta-Llama-3-8B/resolve/'+REV+'/tokenizer.json',
      '/api/resolve-cache/models/NousResearch/Meta-Llama-3-8B/'+REV+'/tokenizer.json'}
    try:return u.scheme=='https' and u.hostname=='huggingface.co' and u.port in (None,443) and not u.username and not u.password and not u.fragment and u.path in allowed
    except ValueError:return False

def draft():
    sources=[binding(ROOT/p,ROOT) for p in (SELF,TEST,*HELPERS)]
    assets=[binding(HF/p,HF) for p in HF_BINDINGS];need(all(b['sha256']==HF_BINDINGS[b['path']] for b in assets),'Source/notice differs')
    exe=Path('/usr/bin/python3').resolve()
    return {'id':ID,'status':'DRAFT_REVIEW_REQUESTED','design':DESIGN,'resources':RESOURCES,'limits':LIMITS,
      'repo_sources':sources,'source_notice_assets':assets,'output_HF_relative':OUTPUT,
      'python':{'path':'/usr/bin/python3','sha256':sha(exe),'version':'3.10'},'runtime':runtime_snapshot(),
      'worker_config_sha256':None,'review':None,'automatic_retry':False}

def fingerprint(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='review'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate(p,frozen=False):
    need(p['id']==ID and p['design']==DESIGN and p['resources']==RESOURCES and p['limits']==LIMITS and p['output_HF_relative']==OUTPUT and p['automatic_retry'] is False,'Scope differs')
    need({b['path'] for b in p['repo_sources']}=={SELF,TEST,*HELPERS} and {b['path']:b['sha256'] for b in p['source_notice_assets']}==HF_BINDINGS,'Source closure differs')
    need(p['python']['path']=='/usr/bin/python3' and p['python']['version']=='3.10' and isinstance(p['runtime'],dict)
         and p['runtime'].get('python_sha256')==p['python']['sha256'],'Interpreter differs')
    if frozen:need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['review'] is None and isinstance(p['worker_config_sha256'],str) and len(p['worker_config_sha256'])==64,'Frozen identity missing')

def authorize(p,digest):
    validate(p,True);a=read(ROOT/'research/ACTIVE_RESEARCH.json');scopes=[s for s in a['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==ID]
    need(a.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True and len(scopes)==1,'Exact ACTIVE scope required')
    s=scopes[0];need(s.get('protocol_sha256')==digest and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('resources')==RESOURCES and s.get('limits')==LIMITS and s.get('max_gpu_cards')==0,'ACTIVE budget differs')
    review=read(bound(ROOT,s['review']));need(review.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and review.get('scope_id')==ID
      and review.get('state')=='REVIEWER_RECHECK_APPROVED' and review.get('reviewer_disposition') in ('REVISE','APPROVE') and review.get('recheck_disposition')=='APPROVE'
      and review.get('author_agent_id') and review.get('reviewer_agent_id') and review['author_agent_id']!=review['reviewer_agent_id']
      and review.get('responses_to_review') and review.get('adversarial_counterexample') and review.get('approval_scope_fingerprint')==fingerprint(p),'Independent review required')
    need(review.get('critical_source_hashes')=={b['path']:b['sha256'] for b in p['repo_sources']},'Review closure differs')
    for key in ('author_review','independent_review','reviewer_recheck'):bound(ROOT,review[key])
    need(isinstance(s.get('launch_spec_sha256'),str) and len(s['launch_spec_sha256'])==64,'Supervisor binding missing');return s

def runtime_snapshot():
    files={}
    for name,module in (('json',json),('base64',base64),('hashlib',hashlib),('http.client',http.client),('ssl',ssl),('urllib.parse',urllib.parse)):
        f=Path(module.__file__).resolve();files[name]={'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)}
    return {'python_path':'/usr/bin/python3','python_sha256':sha(Path('/usr/bin/python3').resolve()),
            'python_version':'.'.join(map(str,sys.version_info[:3])),'openssl':ssl.OPENSSL_VERSION,'stdlib_files':files}

def download(url,path,report,persist):
    need(allowed_url(url),'Origin not permitted');report['predebited']['input_download']=1;report['download']={'bytes_received':0,'bytes_written':0,'requests':[],
      'unrecorded_application_bytes_max':DESIGN['chunk_bytes'],
      'counter_semantics':'APPLICATION_LEVEL_READ_WRITE_PROGRESS_NOT_POWER_LOSS_DURABLE_OR_PHYSICAL_NETWORK_BYTES',
      'physical_network_bytes':'UNKNOWN','automatic_retry':False};persist()
    current=url;total=0;redirects=0
    while True:
        need(allowed_url(current) and report['predebited']['http_requests']<LIMITS['http_requests'],'Redirect/request budget exceeded')
        u=urllib.parse.urlsplit(current);target=urllib.parse.urlunsplit(('', '',u.path,u.query,''))
        report['predebited']['http_requests']+=1;report['inflight']='http_request';persist()
        conn=http.client.HTTPSConnection('huggingface.co',443,timeout=20,context=ssl.create_default_context())
        try:
            conn.request('GET',target,headers={'User-Agent':'FFNDS-byte-bridge/1','Accept-Encoding':'identity','Accept':'application/json'})
            response=conn.getresponse();status=response.status
            report['completed']['http_requests']+=1;report['inflight']=None;report['download']['requests'].append({'status':status});persist()
            if status in (301,302,303,307,308):
                location=response.getheader('Location');need(location is not None and redirects<DESIGN['redirect_limit'],'Invalid redirect')
                current=urllib.parse.urljoin(current,location);need(allowed_url(current),'Redirect target forbidden');redirects+=1;continue
            need(status==200,'Non-200 HTTP response')
            encoding=response.getheader('Content-Encoding');need(encoding in (None,'identity'),'Unexpected transfer encoding')
            length=response.getheader('Content-Length')
            if length is not None:need(length.isdigit() and int(length)==INPUT_BYTES,'HTTP length differs')
            report['inflight']='input_download';persist()
            blob=hashlib.sha1();blob.update(b'blob '+str(INPUT_BYTES).encode('ascii')+b'\x00');h=hashlib.sha256()
            fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'wb') as f:
                while True:
                    chunk=response.read(DESIGN['chunk_bytes'])
                    if not chunk:break
                    total+=len(chunk);report['download']['bytes_received']=total
                    need(total<=INPUT_BYTES and total<=RESOURCES['download_bytes'],'Download too large')
                    need(f.write(chunk)==len(chunk),'Short input write');h.update(chunk);blob.update(chunk);report['download']['bytes_written']=total;persist()
                f.flush();os.fsync(f.fileno())
            need(total==INPUT_BYTES and blob.hexdigest()==INPUT_BLOB,'Input size/blob differs')
            report['download'].update(bytes_received=total,bytes_written=total,sha256=h.hexdigest(),git_blob_sha1=blob.hexdigest(),
                                      unrecorded_application_bytes_max=0,redirects=redirects)
            report['completed']['input_download']=1;report['inflight']=None;persist();return
        finally:conn.close()

def run(args):
    need(not {'torch','triton','tiktoken','transformers'} & set(sys.modules),'Fresh stdlib-only process required')
    p=read(args.protocol);digest=sha(args.protocol);scope=authorize(p,digest)
    need(args.job_spec and sha(args.job_spec)==scope['launch_spec_sha256'],'Supervisor spec differs');job=read(args.job_spec)
    argv=job.get('argv');need(isinstance(argv,list) and len(argv)==16 and argv[1].startswith('FFNDS_WORKER_CONFIG='),'Job argv differs')
    profile=Path(argv[1].split('=',1)[1]);need(profile.is_absolute() and sha(profile)==p['worker_config_sha256'],'Worker profile differs')
    expected=['env',argv[1],'CUDA_VISIBLE_DEVICES=','taskset','-c','0','/usr/bin/python3','-I','-S','-B',str(ROOT/SELF),'run','--protocol',str(args.protocol.resolve()),'--job-spec',str(args.job_spec.resolve())]
    out=HF/OUTPUT
    need(argv==expected and job.get('id')==ID and job.get('protocol_sha256')==digest and Path(job['protocol_path']).resolve()==args.protocol.resolve()
         and Path(job['cwd']).resolve()==ROOT and job['hf_root']==str(HF) and Path(job['output_dir']).resolve()==out.parent
         and Path(job['run_json']).resolve()==out/'RUN.json' and job['external_timeout_seconds']==RESOURCES['outer_seconds'],'Supervisor chain differs')
    need(sys.flags.isolated==1 and sys.flags.no_site==1 and sys.version_info[:2]==(3,10)
         and Path(sys.executable).resolve()==Path('/usr/bin/python3').resolve() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
         and sorted(os.sched_getaffinity(0))==DESIGN['cpu_affinity'],'CPU/isolated environment differs')
    need(runtime_snapshot()==p['runtime'] and sha(Path('/usr/bin/python3').resolve())==p['python']['sha256'],'Runtime identity differs')
    for b in p['repo_sources']:bound(ROOT,b)
    for b in p['source_notice_assets']:bound(HF,b)
    import worker_identity
    worker_identity.require_worker();need(sha(worker_identity.private_config_path())==p['worker_config_sha256'],'Worker differs')
    need(not out.exists(),'Fresh output required');out.mkdir(parents=True,mode=0o700)
    started=time.monotonic();cpu0=time.process_time();report={'id':ID,'protocol_sha256':digest,'launch_spec_sha256':scope['launch_spec_sha256'],
       'status':'STARTED_NO_RETRY','design':DESIGN,'limits':LIMITS,'predebited':dict.fromkeys(LIMITS,0),'completed':dict.fromkeys(LIMITS,0),
       'inflight':None,'automatic_retry':False,'model_or_gpu_execution':False}
    def persist():write(out/'RUN.json',report)
    def tracked_read(name,path,fn):
        size=Path(path).stat().st_size;row={'operation':name,'predebited_logical_bytes':size,'completed_logical_bytes':0,'inflight':True}
        report.setdefault('file_reads',[]).append(row);persist()
        value=fn();row.update(completed_logical_bytes=size,inflight=False);persist();return value
    persist()
    try:
        resource.setrlimit(resource.RLIMIT_AS,(RESOURCES['peak_address_space_bytes'],RESOURCES['peak_address_space_bytes']))
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bridge wall budget exceeded')));signal.alarm(RESOURCES['wall_seconds'])
        input_path=out/'tokenizer.json';candidate=out/'tokenizer.model.candidate';target=out/'tokenizer.model'
        need(not os.path.lexists(input_path) and not os.path.lexists(candidate) and not os.path.lexists(target),'Fresh files required')
        download(INPUT_URL,input_path,report,persist)
        need(input_path.stat().st_size==INPUT_BYTES and tracked_read('input_sha256',input_path,lambda:sha(input_path))==report['download']['sha256'],'Downloaded file changed')
        report['predebited']['tokenizer_conversion']=1;report['inflight']='tokenizer_conversion';persist()
        data=tracked_read('input_parse',input_path,input_path.read_bytes);rank_map,metadata=parse_tokenizer(data);report['parsed']=metadata;persist()
        report['candidate_progress']={'logical_bytes_written':0,'unrecorded_application_bytes_max':262144+8192,
          'counter_semantics':'APPLICATION_WRITES_NOT_POWER_LOSS_DURABLE'};persist()
        next_mark=[262144]
        def progress(n):
            if n>=next_mark[0]:report['candidate_progress']['logical_bytes_written']=n;persist();next_mark[0]=n+262144
        fd=os.open(candidate,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'wb') as f:
            count=write_canonical(rank_map,f,RESOURCES['new_HF_bytes']-INPUT_BYTES-65536,progress);f.flush();os.fsync(f.fileno())
        report['candidate_progress']={'logical_bytes_written':count,'unrecorded_application_bytes_max':0,
                                      'counter_semantics':'APPLICATION_WRITES_NOT_POWER_LOSS_DURABLE'};persist()
        report['candidate']={'bytes':candidate.stat().st_size,'sha256':tracked_read('candidate_sha256',candidate,lambda:sha(candidate)),
                             'logical_bytes_written':count,'status':'PENDING_TARGET_MATCH'};persist()
        need(candidate.stat().st_size==count and tracked_read('candidate_roundtrip',candidate,lambda:verify_canonical(candidate,rank_map))==len(rank_map),'Candidate roundtrip failed')
        need(count==TARGET_BYTES and report['candidate']['sha256']==TARGET_SHA,'Official target size or SHA mismatch')
        os.link(candidate,target);candidate.unlink();actual_sha=tracked_read('published_sha256',target,lambda:sha(target))
        need(target.stat().st_size==TARGET_BYTES and actual_sha==TARGET_SHA,'Published bytes mismatch')
        report['candidate']['status']='MATCHED_PUBLISHED';report['output']={'path':OUTPUT+'/tokenizer.model','bytes':TARGET_BYTES,'sha256':actual_sha}
        report['completed']['tokenizer_conversion']=1;report['inflight']=None;report['status']='COMPLETE_BYTE_IDENTICAL_NOT_TOKENIZER_RUNTIME';persist()
    except BaseException as e:
        report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-started;report['cpu_seconds']=time.process_time()-cpu0
        report['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['output_bytes_before_final_RUN']=worker_output_bytes(out.parent)
        if report['peak_RSS_bytes']>RESOURCES['peak_RSS_bytes'] or report['output_bytes_before_final_RUN']>RESOURCES['new_HF_bytes']-65536:
            report['status']='FAILED_RESOURCE_LIMIT_PRESERVED';persist();raise ValueError('Resource limit exceeded')
        persist()

def verify_closed(args):
    p=read(args.protocol);validate(p,True);d=sha(args.protocol);out=HF/OUTPUT
    need(args.run.resolve()==(out/'RUN.json').resolve() and args.exit.resolve()==(out.parent/'EXIT_RECEIPT.json').resolve(),'Receipt path differs')
    r=read(args.run);e=read(args.exit);rb=binding(args.run,HF)
    need(e.get('id')==ID and e.get('protocol_sha256')==d and e.get('run_sha256')==rb['sha256'] and e.get('process_exited') is True
         and e.get('child_returncode')==0 and e.get('timeout') is False and e.get('automatic_retry') is False and e.get('spec_sha256')==r.get('launch_spec_sha256'),'Actual EXIT0 required')
    target=out/'tokenizer.model';input_file=out/'tokenizer.json';need(r.get('status')=='COMPLETE_BYTE_IDENTICAL_NOT_TOKENIZER_RUNTIME' and r.get('protocol_sha256')==d
      and r.get('design')==DESIGN and r.get('model_or_gpu_execution') is False and r.get('automatic_retry') is False
      and r['download']['git_blob_sha1']==INPUT_BLOB and r['download']['bytes_received']==INPUT_BYTES
      and input_file.is_file() and not input_file.is_symlink() and input_file.stat().st_size==INPUT_BYTES
      and git_blob_hash(input_file)==INPUT_BLOB and sha(input_file)==r['download']['sha256']
      and r['completed']['input_download']==r['completed']['tokenizer_conversion']==1
      and r['predebited']==r['completed'] and 1<=r['completed']['http_requests']<=LIMITS['http_requests']
      and r['inflight'] is None and r['output']=={'path':OUTPUT+'/tokenizer.model','bytes':TARGET_BYTES,'sha256':TARGET_SHA}
      and target.is_file() and not target.is_symlink() and target.stat().st_size==TARGET_BYTES and sha(target)==TARGET_SHA
      and 0<=r['wall_seconds']<=RESOURCES['wall_seconds'] and r['peak_RSS_bytes']<=RESOURCES['peak_RSS_bytes']
      and r['output_bytes_before_final_RUN']<=RESOURCES['new_HF_bytes']-65536
      and worker_output_bytes(out.parent)<=RESOURCES['new_HF_bytes'],'Bridge output/metrics differ')
    for b in p['repo_sources']:bound(ROOT,b)
    for b in p['source_notice_assets']:bound(HF,b)
    return {'id':ID,'status':'PASS_CLOSED_BYTE_IDENTICAL_NOT_TOKENIZER_RUNTIME','RUN':rb,'exit':binding(args.exit,HF),
            'tokenizer':binding(target,HF),'verification_GPU_calls':0,'verification_model_calls':0}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','inspect','run','verify-closed'));ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--exit',type=Path);ap.add_argument('--job-spec',type=Path);a=ap.parse_args()
    if a.action=='draft':
        need(a.out and not a.out.exists(),'Fresh output required');p=draft();p['runtime']=runtime_snapshot();a.out.parent.mkdir(parents=True,exist_ok=True);write(a.out,p)
    elif a.action=='inspect':
        p=read(a.protocol);validate(p);print(json.dumps({'id':ID,'fingerprint':fingerprint(p),'design':DESIGN,'resources':RESOURCES},indent=2))
    elif a.action=='run':run(a)
    else:print(json.dumps(verify_closed(a),indent=2))
if __name__=='__main__':main()
