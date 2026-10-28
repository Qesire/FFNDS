"""Streaming byte audit of two completed raw BF16 full-depth collector parts."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import time
import sys

ROOT=Path(__file__).resolve().parents[1]
ID='E0_RAW_BF16_FULL_DEPTH_ASSET_AUDIT_20261029'
SHAPES={'V1':[256,5120],'h':[256,13824],'Y':[256,5120],'r':[256,5120],'u':[256,5120],'gate':[5120],'shift':[5120],'scale':[5120],'rows':[256]}


def need(ok,msg):
    if not ok: raise ValueError(msg)


def digest(data): return hashlib.sha256(data).hexdigest()


def bound(root,b):
    p=Path(b['path']); need(not p.is_absolute() and '..' not in p.parts,'invalid bound path')
    f=(root/p).resolve(); need(f.is_relative_to(root.resolve()),'path outside root')
    data=f.read_bytes(); need(len(data)==b['bytes'] and digest(data)==b['sha256'],'bound identity mismatch')
    return json.loads(data)


def draw(ordinal,step,block):
    seed=42+(ordinal+1)*1000003+1009*step+37*block
    return sorted(random.Random(seed).sample(range(31668),512)[:256])


def audit_packet(path,binding,metadata,expected_rows,shapes=SHAPES,count=lambda n:None):
    """Read each payload byte once while updating its and the whole-file digest."""
    whole=hashlib.sha256(); observed={}; rows_raw=bytearray()
    with path.open('rb') as f:
        need(os.fstat(f.fileno()).st_size==binding['bytes'],'file size differs before read')
        def get(n):
            data=f.read(n);count(len(data));return data
        prefix=get(8);need(len(prefix)==8,'truncated prefix'); whole.update(prefix)
        n=struct.unpack('<Q',prefix)[0];need(0<n<=1048576,'header size')
        raw=get(n);need(len(raw)==n,'truncated header');whole.update(raw);hdr=json.loads(raw)
        need(set(hdr)-{'__metadata__'}==set(shapes),'tensor names')
        order=sorted(shapes,key=lambda k:hdr[k]['data_offsets'][0]);end=0
        for key in order:
            entry=hdr[key]; meta=metadata[key];a,b=entry['data_offsets'];dtype='I64' if key=='rows' else 'BF16'; size=1
            for d in shapes[key]:size*=d
            size*=8 if key=='rows' else 2
            need(entry['dtype']==dtype and entry['shape']==shapes[key]==meta['shape'] and meta['dtype']==('int64' if key=='rows' else 'bfloat16') and a==end and b-a==size,'tensor header/layout')
            t=hashlib.sha256();left=size
            while left:
                chunk=get(min(left,1048576));need(bool(chunk),'truncated tensor');whole.update(chunk);t.update(chunk);left-=len(chunk)
                if key=='rows':rows_raw.extend(chunk)
            value=t.hexdigest();need(value==meta['raw_tensor_sha256'],'tensor raw SHA');observed[key]=value;end=b
        need(os.fstat(f.fileno()).st_size==binding['bytes'],'packet size changed')
    need(path.stat().st_size==binding['bytes']==8+n+end and whole.hexdigest()==binding['sha256'],'packet SHA/bytes')
    need(list(struct.unpack('<'+'q'*(len(rows_raw)//8),rows_raw))==expected_rows,'row payload differs')
    return observed


def audit_state(path,binding,count=lambda n:None):
    total=hashlib.sha256(); tensor=hashlib.sha256()
    with path.open('rb') as f:
        need(os.fstat(f.fileno()).st_size==binding['bytes'],'file size differs before read')
        def get(n):
            data=f.read(n);count(len(data));return data
        prefix=get(8);need(len(prefix)==8,'state header prefix');total.update(prefix)
        n=struct.unpack('<Q',prefix)[0];need(0<n<=1048576,'state header size')
        raw=get(n);need(len(raw)==n,'state header truncated');total.update(raw); hdr=json.loads(raw)
        keys=set(hdr)-{'__metadata__'};need(keys in ({'flow'},{'latents'}),'state tensor key');t=hdr[next(iter(keys))]
        dtype={'float32':'F32','bfloat16':'BF16'}[binding['dtype']];size=16*21*58*104*(4 if dtype=='F32' else 2)
        need(t=={'dtype':dtype,'shape':[16,21,58,104],'data_offsets':[0,size]} and binding['shape']==[16,21,58,104],'state header contract')
        left=size
        while left:
            chunk=get(min(left,1048576));need(bool(chunk),'state truncated');left-=len(chunk);total.update(chunk);tensor.update(chunk)
        need(os.fstat(f.fileno()).st_size==binding['bytes'],'state size changed')
    need(binding['bytes']==8+n+size and path.stat().st_size==binding['bytes'] and total.hexdigest()==binding['sha256'] and tensor.hexdigest()==binding['raw_tensor_sha256'],'state SHA/bytes')


def journal_counts(hf,binding):
    path=(hf/binding['path']).resolve();need(path.is_relative_to(hf.resolve()),'journal path')
    data=path.read_bytes();need(len(data)==binding['bytes'] and digest(data)==binding['sha256'],'journal SHA/bytes')
    counts=defaultdict(Counter)
    for line in data.splitlines():
        event=json.loads(line);need(event['kind'] in ('reserved_before_call','observed_return'),'unexpected journal event')
        need(all(type(n) is int and n>0 for n in event['costs'].values()),'invalid journal costs')
        counts[(event['run_id'],event['kind'])].update(event['costs'])
    return counts


def validate_finalize(proof,verified,receipt,part,phase):
    expected={'protocol':('config/e0_raw_full_depth_video_finalize_ord288_20261029_v1.json',2660,'6f46ef91fcf1dcdf795ad9e0e4658de0d2da04042e37081aa4cdf845b4639c6c'),
      'RUN':('ffnds-derived/e0_raw_full_depth_video_finalize_ord288_20261029/attempt1/worker/results/RUN.json',1978,'f715ce32daf2e417dbfb69c17cb022f6d22bd17577b29f929753d651f1db7822'),
      'exit':('ffnds-derived/e0_raw_full_depth_video_finalize_ord288_20261029/attempt1/worker/EXIT_RECEIPT.json',528,'62c7c0d07c4e03027563cd1f864ba41a3c8a3addf8b8e6475189879a632b3cb5')}
    need(all((proof[k]['path'],proof[k]['bytes'],proof[k]['sha256'])==v for k,v in expected.items()),'finalize frozen binding')
    need(verified['parent']=={k:part[k] for k in ('protocol','RUN','exit')} and verified['original_native_scope_remains_failed'] is True and verified['automatic_retry'] is False,'finalize parent identity')
    need(verified['id']=='E0_RAW_FULL_DEPTH_VIDEO_FINALIZE_ORD288' and verified['status']=='COMPLETE_VIDEO_FULL_DECODE' and verified['native_source_ordinal']==288 and verified['protocol_sha256']==proof['protocol']['sha256'] and verified['decoded_frames']==81 and verified['decoded_fps']==16 and verified['decoded_size']==[832,464] and verified['model_calls']==verified['GPU_calls']==verified['new_video_generations']==0 and verified['predebited_existing_video_full_decodes']==verified['observed_existing_video_full_decodes']==1,'finalize decode contract')
    need(receipt['id']==verified['id'] and receipt['run_sha256']==proof['RUN']['sha256'] and receipt['protocol_sha256']==proof['protocol']['sha256'] and receipt['process_exited'] is True and type(receipt['child_returncode']) is int and receipt['child_returncode']==0 and receipt['timeout'] is False and receipt['automatic_retry'] is False,'finalize actual exit')
    video=verified['video'];first=phase['packets'][0]['metadata']['path'];expected_video=str(Path(first).parent/'output.mp4')
    need(video['root']=='hf' and video['path']==expected_video and video['sha256']=='2ee0edb617d562784fec019b32662fe3365cd7a22e08312e04a567ea02c7901d','finalize source video differs')


def check_generation(gen,core,phase,case,protocol_sha):
    need(gen['run_id']==phase['run_id']==f"raw_full_depth_ord{case['native_source_ordinal']}_seed42_dense" and gen['protocol_sha256']==protocol_sha and gen['source_id']==case['source_id'] and gen['native_source_ordinal']==case['native_source_ordinal'] and gen['source_index']==case['consumer20_index'] and gen['status']=='COMPLETE_RAW_BF16_TEACHER_STATE_FULL160' and gen['numeric_path']=='BF16_ORIGINAL' and gen['video']==phase['video'],'generation identity differs')
    need(core['identity']['run_sha256']==phase['generation_receipt']['sha256'] and core['identity']['source_id']==case['image_sha256'] and core['identity']['input_prompt_digest']==case['prompt_sha256'] and core['identity']['protocol_sha256']==protocol_sha and core['identity']['seed']==42 and core['candidate']['sha256']==gen['video']['sha256'] and core['metrics']['artifact']['verified_full_decode_frames']==81,'generation core identity differs')


def stream_video(path,b,count):
    h=hashlib.sha256(); size=0
    with path.open('rb') as f:
        need(os.fstat(f.fileno()).st_size==b['bytes'],'288 video size')
        while size<b['bytes']:
            chunk=f.read(min(1048576,b['bytes']-size))
            need(bool(chunk),'288 video truncated')
            count(len(chunk));size+=len(chunk);h.update(chunk)
        need(os.fstat(f.fileno()).st_size==b['bytes'],'288 video size changed')
    need(size==b['bytes'] and h.hexdigest()==b['sha256'],'288 video identity')


def audit(hf, parts, progress=lambda:None, read=lambda key,n,fn:fn(lambda n:None)):
    need(len(parts)==5,'five closed scopes required'); cases_out=[];seen=set(); conditions={};part_out=[]
    for part in parts:
        plan=bound(ROOT,part['protocol']);run=bound(hf,part['RUN']);ex=bound(hf,part['exit']); ps=part['protocol']['sha256']
        need(plan['part'] in (0,1) and plan['id'] not in seen,'duplicate scope');seen.add(plan['id'])
        need(plan['numeric_path']=='BF16_ORIGINAL' and run['protocol_sha256']==ex['protocol_sha256']==ps and run['id']==ex['id']==plan['id'] and ex['run_sha256']==part['RUN']['sha256'] and ex['process_exited'] is True and type(ex['child_returncode']) is int and ex['timeout'] is False and ex['automatic_retry'] is False,'actual exit mismatch')
        disposition=part['disposition'];need(disposition in ('FAILED_V2_TIMEOUT_PREFIX','FAILED_V3_MONITOR_PREFIX','COMPLETE_V3','COMPLETE_V4'),'scope disposition')
        failed=disposition in ('FAILED_V2_TIMEOUT_PREFIX','FAILED_V3_MONITOR_PREFIX')
        if disposition=='FAILED_V2_TIMEOUT_PREFIX':
            need(plan['id'].endswith('_V2') and ex['child_returncode']==1 and run['status']=='FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY' and run['error_type']=='TimeoutError' and ps=={0:'277e482c70cd494b7e48e2eec44dc40353a8e62713a3568d0c7e03c0082c9457',1:'ac718a803883e8d3443fa9ea962735fd3ddd6f7602f298c85b96cb7ea96de563'}[plan['part']],'v2 failure identity')
        elif disposition=='FAILED_V3_MONITOR_PREFIX':
            need(plan['part']==0 and plan['id'].endswith('_V3') and ps=='3cd30a95cfc7cfd252e94b431a31519dce33e0387b12da0698104476ad586e47' and part['RUN']['sha256']=='89f33384a570b02201489dd2b481d5adf1126b45ab75d2352cefed27140068ba' and ex['child_returncode']==1 and run['status']=='FAILED_FULL_DEPTH_PART_PRESERVED_NO_RETRY' and run['error_type']=='RuntimeError' and run['error']=='Assigned GPU monitor failed: TimeoutExpired','v3 monitor failure identity')
            monitor=run['assigned_GPU_resources'];need(monitor['sampled_isolation_pass'] is False and monitor['violations']==['Assigned GPU monitor failed: TimeoutExpired'] and bool(monitor['samples']) and all(x['foreign_process_count_on_assigned_GPU']==0 for x in monitor['samples']) and all(a['monotonic_seconds']<=b['monotonic_seconds'] for a,b in zip(monitor['samples'],monitor['samples'][1:])),'v3 monitor observations differ')
            need([(x['native_source_ordinal'],x['status'],len(x['packets'])) for x in run['phases']]==[(894,'COMPLETE_RAW_BF16_TEACHER_STATE_FULL160',160),(896,'COMPLETE_RAW_BF16_TEACHER_STATE_FULL160',160),(240,'FAILED_SOURCE_FULL_DEPTH_PRESERVED_NO_RETRY',18)],'v3 exact completed prefix differs')
        elif disposition=='COMPLETE_V4':
            need(ps=='0a95a5572564efca25c67db81c75b2de153d6ba4fc31d95d79504448974ea2a7' and plan['id']=='E0_ORIGINAL_BF16_FULL_DEPTH_ORD240_20261029_V4' and ex['child_returncode']==0 and run['status']=='COMPLETE_RAW_BF16_FULL_DEPTH_SOURCE240_V4_REUSED_DEV_NEUTRAL' and run['captured_native_sites']==160,'v4 actual complete differs')
            prev=plan['failed_predecessor'];v3=next(x for x in parts if x['disposition']=='FAILED_V3_MONITOR_PREFIX');need(prev['protocol']==v3['protocol'] and prev['RUN']==v3['RUN'] and prev['launch_exit']==v3['exit'],'v4 predecessor triple differs')
            prior=bound(hf,v3['RUN']);dev=next(x for x in prior['phases'] if x['native_source_ordinal']==894);reused=plan['reused_development_neutral'];need(reused['native_source_ordinal']==894 and reused['GENERATION']==dev['generation_receipt'] and reused['states']==dev['states'] and run['reused_development_neutral']==reused and run['local_development_neutral_passed'] is True and plan['limits']['neutral_state_byte_checks']==0,'v4 reused neutral differs')
        else:
            need(plan['id'].endswith('_V3') and ex['child_returncode']==0 and run['status']=='COMPLETE_RAW_BF16_FULL_DEPTH_V3_REMAINING_CAL_WITH_DEV_NEUTRAL','v3 complete status')
        led=run['operation_ledger'];need(not led['forbidden_attempts'],'forbidden operation')
        if disposition!='FAILED_V3_MONITOR_PREFIX':need(run.get('assigned_GPU_resources',{}).get('sampled_isolation_pass') is True and not run['assigned_GPU_resources'].get('violations'),'monitor failed')
        if not failed:need(all(led[x].get(k,0)==v for x in ('predebited','observed') for k,v in plan['limits'].items()) and run['captured_native_sites']==(160 if disposition=='COMPLETE_V4' else {0:480,1:320}[plan['part']]),'completed part ledger')
        need(run['local_development_neutral_passed'] is True and plan['limits']['neutral_state_byte_checks']==(0 if disposition=='COMPLETE_V4' else 10),'dev neutral absent')
        need(Path(part['journal']['path'])==Path(part['RUN']['path']).parent/'OPERATIONS.jsonl','journal scope directory differs');journal=journal_counts(hf,part['journal']); selected=part['selected_native_ordinals']; expected_selection={('FAILED_V2_TIMEOUT_PREFIX',0):[897],('FAILED_V2_TIMEOUT_PREFIX',1):[899,288],('FAILED_V3_MONITOR_PREFIX',0):[894,896],('COMPLETE_V3',1):[256,304],('COMPLETE_V4',0):[240]};need(selected==expected_selection[(disposition,plan['part'])],'selected source scope differs')
        for ordinal in selected:
            progress();case=next(c for c in plan['cases'] if c['native_source_ordinal']==ordinal);phase=next(p for p in run['phases'] if p['native_source_ordinal']==ordinal)
            need(phase['run_id']==f"raw_full_depth_ord{ordinal}_seed42_dense" and phase['run_id'] in plan['run_ids'] and phase['source_id']==case['source_id'] and phase['source_index']==case['consumer20_index'] and phase['native_source_ordinal']==case['native_source_ordinal'] and phase['role']==case['role'] and phase['history']==case['history'] and phase['source_state']=='teacher_input' and phase['seed']==42,'source mismatch')
            counts=journal[(phase['run_id'],'reserved_before_call')];returns=journal[(phase['run_id'],'observed_return')]
            required={'native_full_model_flows':4,'native_FFN_calls':160,'native_FFN_linears':320,'raw_postwrite_packets':160,'packet_safetensor_writes':160,'packet_metadata_writes':160,'packet_payload_validations':160,'scheduler_step_pre':4,'scheduler_step_post':4,'native_FP32_flow_files':4,'native_BF16_post_files':4,'raw_FP32_noise_files':1,'initial_BF16_noise_files':1}
            need(all(counts[k]==returns[k]==v for k,v in required.items()),'source journal incomplete')
            if ordinal==288:
                need(failed and phase['status']=='FAILED_SOURCE_FULL_DEPTH_PRESERVED_NO_RETRY' and phase['error_type']=='TimeoutError' and counts['video_geometry_checks']==1 and returns['video_geometry_checks']==0,'288 actual failure differs')
                proof=part['video_finalize'];final_protocol=bound(ROOT,proof['protocol']);verified=bound(hf,proof['RUN']);receipt=bound(hf,proof['exit'])
                need(final_protocol['id']=='E0_RAW_FULL_DEPTH_VIDEO_FINALIZE_ORD288','finalize protocol id')
                validate_finalize(proof,verified,receipt,part,phase)
                video=verified['video'];vp=Path(video['path']);need(not vp.is_absolute() and '..' not in vp.parts and (hf/vp).resolve().is_relative_to(hf.resolve()),'288 video path');vf=hf/vp;read('video_files',video['bytes'],lambda count:stream_video(vf,video,count))
            else:
                need(phase['status']=='COMPLETE_RAW_BF16_TEACHER_STATE_FULL160','source not complete')
                if case['role']=='calibration':
                    gen=bound(hf,phase['generation_receipt']);core=bound(hf,phase['core_metrics']);check_generation(gen,core,phase,case,ps)
            state_keys={'raw_noise_FP32','initial_noise',*[f'flow_step{i}' for i in range(4)],*[f'latent_step{i}' for i in range(4)]}
            need(set(phase['states'])==state_keys,'source state coverage')
            for name in sorted(state_keys):
                state=phase['states'][name]; expected_dtype='float32' if name=='raw_noise_FP32' or name.startswith('flow_') else 'bfloat16'
                need(state['dtype']==expected_dtype and state['root']=='hf','state dtype/root')
                path=Path(state['path']);need(not path.is_absolute() and '..' not in path.parts and (hf/path).resolve().is_relative_to(hf.resolve()),'state path')
                if case['role']=='development':
                    parent=case['controls'][name];need(all(state[k]==parent[k] for k in ('dtype','shape','raw_tensor_sha256')),'neutral parent state differs')
                read('state_files',state['bytes'],lambda count,path=path,state=state:audit_state(hf/path,state,count))
            need(len(phase['packets'])==160,'packet count'); positions=set(); entries=[]
            for item in phase['packets']:
                s,b=item['step'],item['block'];need((s,b) not in positions and s in range(4) and b in range(40),'duplicate position');positions.add((s,b))
                meta=bound(hf,item['metadata']); rows=draw(case['native_source_ordinal'],s,b)
                need(meta['schema']=='ffnds.raw_bf16.full_depth.operands.v1' and meta['protocol_sha256']==ps and meta['run_id']==phase['run_id'] and meta['source_id']==case['source_id'] and meta['image_sha256']==case['image_sha256'] and meta['prompt_sha256']==case['prompt_sha256'] and meta['consumer20_index']==case['consumer20_index'] and meta['native_source_ordinal']==case['native_source_ordinal'] and meta['role']==case['role'] and meta['history']==case['history'] and meta['seed']==42 and meta['source_state']=='teacher_input' and meta['numeric_path']=='BF16_ORIGINAL' and meta['step']==s and meta['block']==b and meta['expert']==('high' if s<2 else 'low') and meta['grid']==[21,29,52] and meta['population_rows']==31668 and meta['selected_rows']==case['rows'][str(s)][str(b)]==rows,'packet metadata identity')
                bind=meta['payload'];p=Path(bind['path']);need(bind['root']=='hf' and not p.is_absolute() and '..' not in p.parts and (hf/p).resolve().is_relative_to(hf.resolve()),'payload path')
                tensors=read('packet_files',bind['bytes'],lambda count:audit_packet(hf/p,bind,meta['tensor_metadata'],rows,count=count))
                for key in ('gate','shift','scale'):conditions.setdefault((s,b,key),set()).add(tensors[key])
                entries.append({'step':s,'block':b,'metadata':item['metadata'],'payload':bind,'tensor_metadata':meta['tensor_metadata']})
                progress()
            cases_out.append({'source_id':case['source_id'],'image_sha256':case['image_sha256'],'prompt_sha256':case['prompt_sha256'],'consumer20_index':case['consumer20_index'],'native_source_ordinal':case['native_source_ordinal'],'role':case['role'],'history':case['history'],'part':plan['part'],'run_id':phase['run_id'],'source_state':'teacher_input','states':phase['states'],'packets':entries})
        part_out.append({**part,'scope_status':run['status'],'actual_exit_code':ex['child_returncode'],'original_operation_ledger':led,'prefix_isolation':'COMPLETE_PREFIX_BEFORE_LATER_MONITOR_QUERY_TIMEOUT' if disposition=='FAILED_V3_MONITOR_PREFIX' else 'SCOPE_MONITOR_PASS'})
    need(len(seen)==5 and len(cases_out)==8 and sum(x['role']=='development' for x in cases_out)==2 and sum(x['role']=='calibration' for x in cases_out)==6 and len({x['image_sha256'] for x in cases_out})==len({x['source_id'] for x in cases_out})==len({x['native_source_ordinal'] for x in cases_out})==8,'cohort union')
    obs=[{'step':s,'block':b,'tensor':k,'distinct_raw_sha256_count':len(v),'all_eight_same':len(v)==1} for (s,b,k),v in sorted(conditions.items())]
    return {'schema':'ffnds.raw_bf16.full_depth.operands_index.v1','id':ID,'parts':part_out,'cases':cases_out,'conditioning_SHA_observations':obs,'totals':{'sources':8,'calibration':6,'development':2,'packets':1280,'native_positions':160},'scope':'teacher input only; historically exposed consumer20; SHA consistency is observational'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--hf-root',type=Path,required=True);args=ap.parse_args();start=time.monotonic(); raw=args.protocol.read_bytes();spec=json.loads(raw)
    resources={'GPU':0,'CPU_threads':2,'wall_seconds':900,'outer_seconds':960,'new_HF_bytes':16*1024*1024}
    limits={'packet_files':1280,'state_files':80,'video_files':1,'total_payload_bytes_max':26*1024**3,'video_bytes_max':4*1024**2}
    need(spec['id']==ID and spec['status']=='FROZEN_REGISTERED' and spec['resources']==resources and spec['limits']==limits,'fixed registered scope differs')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules and 'numpy' not in sys.modules,'CUDA-hidden stdlib worker required')
    for b in spec['repo_sources']:
        p=(ROOT/b['path']).resolve();need(p.is_relative_to(ROOT) and p.stat().st_size==b['bytes'] and digest(p.read_bytes())==b['sha256'],'audit source binding differs')
    active=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text());authority=active['experiment_authority'];standing=active['user_continuation_authorization_20261027'];scopes=[x for x in authority['current_user_authorized_scopes'] if x.get('id')==ID]
    need(standing.get('status')=='ACTIVE_CURRENT_USER_DIRECTED_SESSION' and standing.get('new_scope_registration_allowed') is True and standing.get('automatic_retry') is False and len(scopes)==1 and scopes[0].get('protocol_sha256')==digest(raw) and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and scopes[0].get('resources')==resources and scopes[0].get('limits')==limits,'ACTIVE scope mismatch')
    need(len(os.sched_getaffinity(0))==2,'exactly two CPU affinity required');hf=args.hf_root.resolve();out=(hf/spec['output_HF_relative']).resolve();need(out.is_relative_to(hf) and not out.exists(),'fresh confined output required');out.mkdir(parents=True)
    report={'id':ID,'status':'STARTED','protocol_sha256':digest(raw),'predebited':{'packet_files':0,'state_files':0,'video_files':0,'bytes':0},'observed':{'packet_files':0,'state_files':0,'video_files':0,'bytes':0},'actual_payload_bytes_read':0,'automatic_retry':False}
    def save():
        data=(json.dumps(report,indent=2)+'\n').encode();need(len(data)<=16*1024*1024,'output budget');tmp=out/'RUN.tmp';tmp.write_bytes(data);os.replace(tmp,out/'RUN.json')
    def check():need(time.monotonic()-start<=900,'wall budget')
    def read(key,n,fn):
        need(report['predebited'][key]<limits[key] and (key!='video_files' or n<=limits['video_bytes_max']) and report['predebited']['bytes']+n<=limits['total_payload_bytes_max'],'read budget');report['predebited'][key]+=1;report['predebited']['bytes']+=n;save();result=fn(lambda size:report.__setitem__('actual_payload_bytes_read',report['actual_payload_bytes_read']+size));report['observed'][key]+=1;report['observed']['bytes']+=n;return result
    save()
    try:
        manifest=audit(hf,spec['parts'],check,read);need(report['predebited']==report['observed'] and report['observed']['packet_files']==1280 and report['observed']['state_files']==80 and report['observed']['video_files']==1,'audit counts incomplete')
        data=(json.dumps(manifest,indent=2)+'\n').encode();need(len(data)+len(json.dumps(report).encode())<16*1024*1024,'combined output budget');(out/'MANIFEST.json').write_bytes(data)
        report.update(status='COMPLETE_ASSET_BYTES_AUDITED',manifest={'root':'hf','path':str((out/'MANIFEST.json').relative_to(hf)),'bytes':len(data),'sha256':digest(data)},wall_seconds=time.monotonic()-start);save()
    except BaseException as e:
        report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e),wall_seconds=time.monotonic()-start);save();raise

if __name__=='__main__':main()
