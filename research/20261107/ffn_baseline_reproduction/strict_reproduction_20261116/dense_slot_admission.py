"""Read-only evidence audit of 11 fixed Stage-4 Dense candidates.

No model execution, remote writes or automatic admission. Old V2 tap video can
be quality-eligible, but without an actual final latent it is not a complete
main output slot. CLI writes a new manifest only when explicitly invoked.
"""
import argparse,hashlib,json,os,re,subprocess,threading,zipfile
from pathlib import Path

DEV=('vbi_1013','vbi_1040','vbi_0886','vbi_0947','vbi_1109','vbi_1084','vbi_0950','vbi_0176','vbi_0800','vbi_0392','vbi_0056')
NEW=('vbi_1013','vbi_0800')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))

def file_spec(path):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(str(p))
    return {'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size}

def verify_latent_storage(path, recorded):
    """Hash torch.save tensor storage without executing pickle instructions."""
    if recorded['shape']!=[16,21,58,104] or recorded['dtype']!='torch.bfloat16':
        raise ValueError('final latent recorded shape/dtype mismatch')
    expected_bytes=16*21*58*104*2
    with zipfile.ZipFile(path) as z:
        data=[x for x in z.infolist() if re.fullmatch(r'[^/]+/data/0',x.filename)]
        if len(data)!=1 or data[0].file_size!=expected_bytes:
            raise ValueError('final latent storage size or count mismatch')
        if sum(1 for x in z.infolist() if re.fullmatch(r'[^/]+/data/\d+',x.filename))!=1:
            raise ValueError('unexpected extra tensor storage')
        h=hashlib.sha256()
        with z.open(data[0]) as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=recorded['sha256']:
            raise ValueError('final latent raw tensor hash mismatch')
    return h.hexdigest()
def decode_video(exe,video):
    """Decode actual frame RGB bytes, require 81 frames/832x464/16fps/no audio."""
    proc=subprocess.Popen([str(exe),'-hide_banner','-i',str(video),'-f','rawvideo','-pix_fmt','rgb24','-vsync','0','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    digest=hashlib.sha256();frame_size=832*464*3;count=0;errors=[]
    def drain():
        kept=0
        while True:
            chunk=proc.stderr.read(65536)
            if not chunk:break
            if kept<262144:
                errors.append(chunk[:262144-kept]);kept+=len(chunk)
    reader=threading.Thread(target=drain,daemon=True);reader.start()
    try:
        while True:
            left=frame_size
            while left:
                chunk=proc.stdout.read(min(left,1024*1024))
                if not chunk:break
                digest.update(chunk);left-=len(chunk)
            if left==frame_size:break
            if left:raise ValueError('truncated RGB frame')
            count+=1
            if count>81:raise ValueError('too many frames')
        status=proc.wait(timeout=120)
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        proc.stdout.close();reader.join();proc.stderr.close()
    log=b''.join(errors).decode('utf8',errors='replace')
    input_log=log.split('Stream mapping:',1)[0]
    streams=[line for line in input_log.splitlines() if 'Stream #' in line and '->' not in line]
    video_streams=[line for line in streams if 'Video:' in line]
    audio_streams=[line for line in streams if 'Audio:' in line]
    if status!=0 or count!=81 or len(video_streams)!=1 or audio_streams:
        raise ValueError(f'frame/stream/decoder mismatch frames={count} video_streams={len(video_streams)} audio_streams={len(audio_streams)} exit={status}')
    if not re.search(r'\b832x464\b',video_streams[0]) or not re.search(r'\b16(?:\.0+)? fps\b',video_streams[0]):
        raise ValueError('video geometry/fps mismatch')
    return {'decoded_rgb24_sha256':digest.hexdigest(),'frames':81,'width':832,'height':464,'fps':16,'audio_streams':0}

def verify_common(spec):
    """Bind both producer scopes to the identical V2 source/weight/config."""
    v2=Path(spec['v2_stage']).resolve(); new=Path(spec['stage1_stage']).resolve()
    vscope=read_json(v2/'scope.json');nscope=read_json(new/'scope.json')
    if sha(v2/'scope.json')!=spec['v2_scope_sha256'] or sha(new/'scope.json')!=spec['stage1_scope_sha256']:
        raise ValueError('producer scope identity mismatch')
    if nscope['reference_sha256']['scope.json']!=spec['v2_scope_sha256']:
        raise ValueError('new Stage1 does not bind exact V2 producer scope')
    if nscope['stage']!='STAGE1_DENSE_FULL_U' or nscope['seed']!=42 or nscope['sources']!=list(NEW):
        raise ValueError('new producer source/seed mismatch')
    if nscope['checkpoints']!=vscope['checkpoints']:
        raise ValueError('new and V2 checkpoints differ')
    proof_path=Path(spec['checkpoint_preflight_path']).resolve()
    if sha(proof_path)!=spec['checkpoint_preflight_sha256']:raise ValueError('checkpoint preflight proof identity mismatch')
    q1_scope_path=Path(spec['checkpoint_preflight_scope']).resolve()
    if sha(q1_scope_path)!=spec['checkpoint_preflight_scope_sha256']:raise ValueError('preflight job scope identity mismatch')
    q1_scope=read_json(q1_scope_path);q1_run=read_json(q1_scope_path.parent/'receipt'/'RUN.json')
    if q1_run.get('scope_sha256')!=spec['checkpoint_preflight_scope_sha256'] or q1_scope['checkpoints']!=vscope['checkpoints']:
        raise ValueError('preflight job scope/checkpoint mismatch')
    if proof_path!=(Path(q1_scope['output'])/'CHECKPOINT_PREFLIGHT.json').resolve():
        raise ValueError('checkpoint preflight path differs from job scope')
    proof=read_json(proof_path);checkpoint_stats={}
    for expert,ck in vscope['checkpoints'].items():
        record=proof[expert];path=Path(ck['path']).resolve();st=path.stat()
        current=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
        # st_dev is host mount-specific (login versus compute); retain both
        # device IDs, compare inode/size/mtime/ctime across the shared mount.
        previous=record['stat_dev_ino_size_mtime_ns_ctime_ns']
        if record['path']!=str(path) or record['sha256']!=ck['sha256'] or previous[1:]!=current[1:]:
            raise ValueError('checkpoint inode/size/mtime/ctime differs from prior SHA preflight')
        checkpoint_stats[expert]={'sha256_from_prior_preflight':record['sha256'],'prior_stat':previous,'current_stat':current}
    vc=read_json(v2/'output'/'EFFECTIVE_CONFIG.json');nc=read_json(new/'output'/'EFFECTIVE_CONFIG.json')
    for key,val in vscope['required_runtime_config'].items():
        if vc.get(key)!=val or nc.get(key)!=val:raise ValueError('effective config mismatch '+key)
    if vc!=nc:raise ValueError('V2 and new effective configs differ')
    conditions_path=Path(nscope['v2_reference_dir'])/'v2_conditions.json'
    if sha(conditions_path)!=nscope['reference_sha256']['v2_conditions.json']:
        raise ValueError('V2 condition identity mismatch')
    conditions=read_json(conditions_path);byid={x['record_id']:x for x in conditions}
    if len(byid)!=22 or any(x not in byid for x in DEV):raise ValueError('V2 cohort identity')
    for source in DEV:
        c=byid[source]
        if c['seed']!=42 or c['cycle_role']!='DEV' or hashlib.sha256(c['prompt'].encode()).hexdigest()!=c['prompt_sha256']:
            raise ValueError('source/prompt/seed mismatch '+source)
        if sha(c['image_path'])!=c['image_sha256']:raise ValueError('condition image changed '+source)
    exe=Path(spec['ffmpeg_path'])
    if sha(exe)!=spec['ffmpeg_sha256']:raise ValueError('ffmpeg identity mismatch')
    return v2,new,vscope,nscope,byid,exe,checkpoint_stats


def audit(spec,*,decode=decode_video):
    v2,new,vscope,nscope,byid,exe,checkpoint_stats=verify_common(spec)
    errors={};bank=noise=nsummary=None
    try:
        vexit=read_json(v2/'EXIT.json');vrun=read_json(v2/'RUN.json');vshell=read_json(v2/'SHELL_EXIT.json');vsummary=read_json(v2/'output'/'SUMMARY.json')
        if (vexit.get('exit_code')!=0 or vshell.get('shell_exit_code')!=0 or
            vexit.get('scope_sha256')!=spec['v2_scope_sha256'] or vrun.get('scope_sha256')!=spec['v2_scope_sha256'] or
            str(vshell.get('job_id'))!=str(vexit.get('allocation',{}).get('job_id')) or
            str(vrun.get('allocation',{}).get('job_id'))!=str(vshell.get('job_id')) or
            vsummary.get('status')!='V2_CAPTURE_COMPLETE' or vsummary.get('neutral_summary')!='PASS'):
            raise ValueError('V2 global producer receipt incomplete')
        bank_path=v2/'output'/'BANK_MANIFEST.json';noise_path=v2/'output'/'INITIAL_NOISE_IDENTITIES.json'
        if sha(bank_path)!=vsummary['bank_manifest_sha256']:raise ValueError('V2 bank manifest mismatch')
        bank,noise=read_json(bank_path),read_json(noise_path)
    except (KeyError,ValueError,OSError) as exc:
        errors['legacy_v2_tap']=f'{type(exc).__name__}: {exc}'
    try:
        nexit=read_json(new/'receipt'/'EXIT.json');nrun=read_json(new/'receipt'/'RUN.json');nsummary=read_json(new/'output'/'SUMMARY.json')
        if nexit.get('exit_code')!=0 or nrun.get('scope_sha256')!=spec['stage1_scope_sha256'] or str(nexit.get('slurm_job_id'))!=str(nrun.get('job_id')) or nsummary.get('status')!='COMPLETE':
            raise ValueError('Stage1 global producer receipt incomplete')
    except (KeyError,ValueError,OSError) as exc:
        errors['stage1_full_u']=f'{type(exc).__name__}: {exc}'
    rows=[]
    for source in DEV:
        c=byid[source];is_new=source in NEW;stage=new if is_new else v2;out=stage/'output'
        row={'source':source,'arm':'Dense','seed':42,'origin':'stage1_full_u' if is_new else 'legacy_v2_tap',
             'quality_video_eligible':False,'main_slot_complete':False,'missing':[]}
        try:
            if row['origin'] in errors: raise ValueError(errors[row['origin']])
            if is_new:
                summary=nsummary['per_source'][source]
                if not summary['initial_noise_equal'] or not summary['decoded_equal'] or any(len(summary[k])!=4 or not all(summary[k]) for k in ('flows_equal','latents_equal','time_sigma_equal')):
                    raise ValueError('Stage1 plain-vs-tap neutral failed')
                receipt=read_json(out/f'{source}__full_u.json');video=out/f'{source}__full_u.mp4'
                if receipt['source']!=source or receipt['arm']!='full_u' or receipt['image_sha256']!=c['image_sha256'] or receipt['prompt_sha256']!=c['prompt_sha256']:
                    raise ValueError('Stage1 source condition mismatch')
                if receipt['dispatch']!=[{'step':i,'expert':0 if i<2 else 1} for i in range(4)]:raise ValueError('Stage1 dispatch mismatch')
                if len(receipt['steps'])!=4 or [x['step'] for x in receipt['steps']]!=list(range(4)):
                    raise ValueError('Stage1 step coverage mismatch')
                for i,item in enumerate(receipt['steps']):
                    if (item['timestep'],item['sigma'])!=(nscope['schedule'][i]['timestep'],nscope['schedule'][i]['sigma']):raise ValueError('Stage1 actual schedule mismatch')
                vr=receipt['video'];latent=receipt['steps'][-1].get('final_latent_artifact')
                if not latent:row['missing'].append('final_latent_artifact')
                else:
                    p=(out/latent['path']).resolve()
                    if not p.is_relative_to(out.resolve()) or sha(p)!=latent['sha256']:raise ValueError('final latent identity mismatch')
                    tensor_sha=verify_latent_storage(p,receipt['steps'][-1]['latent'])
                    row['final_latent']={'path':str(p),'sha256':latent['sha256'],'bytes':p.stat().st_size,
                                         'raw_tensor_sha256':tensor_sha}
            else:
                b=bank[source]
                for key in ('image_sha256','prompt_sha256','source_cluster_id'):
                    if b[key]!=c[key]:raise ValueError('V2 source identity mismatch: '+key)
                if b['sites']!=160:raise ValueError('V2 incomplete capture sites')
                receipt=read_json(out/f'CAPTURE_{source}.json');video=out/f'{source}__tap.mp4'
                if receipt['record_id']!=source or receipt['arm']!='tap' or receipt['initial_noise']!=noise[source] or receipt['initial_noise']['sha256']!=b['initial_noise_sha256']:
                    raise ValueError('V2 execution/noise mismatch')
                if receipt['expert_dispatch']!=[{'step':i,'expert':0 if i<2 else 1} for i in range(4)] or [x['step'] for x in receipt['latent_digests']]!=list(range(4)):
                    raise ValueError('V2 trajectory coverage mismatch')
                for step in range(4):
                    side_path=out/'bank'/source/f'{source}__seed42__s{step:02d}__b04.pt.json'
                    side=read_json(side_path)
                    if side['source']!=source or side['step']!=step or side['timestep']!=nscope['schedule'][step]['timestep'] or side['checkpoint_sha256']!=vscope['checkpoints']['high' if step<2 else 'low']['sha256']:
                        raise ValueError('V2 actual scheduler/site/weight mismatch')
                vr=receipt
                # Original V2 producer persisted latent digests, not final tensor.
                row['missing'].append('final_latent_artifact')
            if sha(video)!=vr['mp4_sha256']:raise ValueError('video SHA differs from producer receipt')
            decoded=decode(exe,video)
            if decoded['decoded_rgb24_sha256']!=vr['decoded_rgb24_sha256']:raise ValueError('decoded RGB differs from producer')
            row.update(video=file_spec(video),decoded=decoded,quality_video_eligible=True,
                       initial_noise=receipt['initial_noise'], scheduler=nscope['schedule'],
                       input_identity={'source':source,'seed':42,'image_sha256':c['image_sha256'],
                           'prompt_sha256':c['prompt_sha256'],'source_cluster_id':c['source_cluster_id'],
                           'checkpoint_sha256':{e:vscope['checkpoints'][e]['sha256'] for e in ('high','low')}},
                       producer_receipt=file_spec(out/(f'{source}__full_u.json' if is_new else f'CAPTURE_{source}.json')),
                       producer_scope_sha256=spec['stage1_scope_sha256'] if is_new else spec['v2_scope_sha256'])
            row['main_slot_complete']=not row['missing']
        except (KeyError,ValueError,OSError,subprocess.SubprocessError) as e:
            row['missing'].append(f'{type(e).__name__}: {e}')
        rows.append(row)
    return {'schema':'FFNDS_DENSE_ADMISSION_V1','expected_sources':list(DEV),'slots':rows,
            'complete_main_slots':sum(x['main_slot_complete'] for x in rows),
            'quality_eligible_videos':sum(x['quality_video_eligible'] for x in rows),
            'missing_main_sources':[x['source'] for x in rows if not x['main_slot_complete']],
            'checkpoint_stats':checkpoint_stats,
            'checkpoint_verification':'prior full SHA proof plus current inode/size/mtime/ctime; device ID may differ by host; new compute must rehash',
            'rule':'video and actual final latent are required for a complete Dense output slot; digest alone is insufficient'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('spec_sha256');p.add_argument('output');a=p.parse_args()
    if sha(a.spec)!=a.spec_sha256:raise ValueError('audit spec SHA mismatch')
    result=audit(read_json(a.spec))
    with Path(a.output).open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'output':str(Path(a.output).resolve()),'sha256':sha(a.output),
                      'complete_main_slots':result['complete_main_slots'],'quality_eligible_videos':result['quality_eligible_videos']}))









