"""Fixed ten-source original-image extraction; registered CPU only, provisional."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import stat
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
HF=Path('/home/xiang/.cache/huggingface')
REV='577ba45c1653ede26e29996737f42841f2ec9eea'
ARCHIVE='hub/models--lxl-158--vbench-i2v/snapshots/'+REV+'/vbench2_beta_i2v/data.tar.gz'
METADATA='hub/models--lxl-158--vbench-i2v/snapshots/'+REV+'/vbench2_beta_i2v/vbench2_i2v_full_info.json'
ARCHIVE_SHA='aa6af89775242f35dbcd1f4c53421a6547d0700b1a99206e98b87a28ea393865'
META_SHA='8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8'
ID='E0_RAW_BF16_FRESH10_SOURCE_ARCHIVE_CPU_20261029'
OUTPUT='ffnds-derived/e0_raw_bf16_fresh10_source_archive_cpu_20261029/attempt1/worker/results'
ORDINALS=(909,976,921,960,964,592,440,600,608,448)
STRATA=('subject',)*5+('background',)*2+('camera',)*3
CAMERA=('Slowly pan the camera to the right.','Slowly dolly the camera forward.','Slowly pan the camera to the left.')
BANKS=('calibration12','calibration24','formal20','disjoint20','diverse_confirmation20','r17_fresh23','e0_fresh_official4_20261026')
RESOURCES={'GPU':0,'CPU_threads':2,'wall_seconds':300,'outer_seconds':600,'archive_stream_bytes_max':3<<30,'new_HF_bytes':64<<20,'single_image_bytes_max':4<<20,'automatic_retry':False}
LIMITS={'archive_streams':1,'selected_original_images':10,'model_calls':0,'GPU_calls':0,'new_crops':0}
SOURCES=('tools/raw_bf16_fresh_source_extract.py','tests/test_raw_bf16_fresh_source_extract.py','tools/worker_identity.py','research/20261029/RAW_BF16_FRESH_SOURCE_INVENTORY_20261029.json')


def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(data):return hashlib.sha256(data).hexdigest()
def bind(path,root):
    p=Path(path).resolve();r=Path(root).resolve();need(p.is_relative_to(r),'binding escaped');v=p.read_bytes();return {'path':str(p.relative_to(r)),'bytes':len(v),'sha256':sha(v)}
def bound(root,item):
    p=(root/item['path']).resolve();need(bind(p,root)==item,'binding mismatch');return p

def write(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def history(bank_paths):
    hashes={};names=set();indices=set();image_paths={}
    def walk(value,bank):
        if isinstance(value,dict):
            for key,val in value.items():
                if key in ('image_sha256','legacy_image_fingerprint') and isinstance(val,str) and len(val)==64:hashes.setdefault(val,set()).add(bank)
                walk(val,bank)
        elif isinstance(value,list):
            for x in value:walk(x,bank)
    for path in bank_paths:
        rows=json.loads(path.read_text());walk(rows,path.name)
        for row in rows:
            if not isinstance(row,dict):continue
            if row.get('image_name'):names.add(row['image_name'])
            if row.get('image_path'):image_paths[row['image_path']]=row.get('input_provenance',{}).get('image_sha256');names.add(Path(row['image_path']).name)
            for p in row.get('img_paths',[]):image_paths[p]=row.get('input_provenance',{}).get('image_sha256');names.add(Path(p).name)
            if row.get('source_group','').startswith(('i2v_subject:','i2v_background:')):indices.add(row['source_index'])
    return hashes,names,indices,image_paths


def make_draft(hf):
    metadata=(hf/METADATA).read_bytes();need(sha(metadata)==META_SHA,'metadata identity differs');rows=json.loads(metadata)
    paths=[ROOT/'.runtime/inputs'/b/'canonical.json' for b in BANKS]+sorted((ROOT/'.runtime/inputs').glob('task*.json'))
    hashes,names,indices,_=history(paths);source=[]
    for i,(ordinal,stratum) in enumerate(zip(ORDINALS,STRATA)):
        row=rows[ordinal];name=row['image_name'];original=row['prompt_en'];dim='i2v_subject' if stratum=='subject' else 'i2v_background'
        need(dim in row['dimension'] and ordinal not in indices and name not in names and '/' not in name and '\\' not in name,'fixed candidate conflicts with observed history')
        prompt=original if stratum!='camera' else original.rstrip()+' '+CAMERA[i-7]
        source.append({'source_id':f'lxl-158/vbench-i2v@{REV}:row{ordinal}','native_source_ordinal':ordinal,'cohort_index':i,'part':i%2,
            'role':'NEW_DEVELOPMENT','collector_role':'observation','dimension':dim,'scene_stratum':stratum,'image_type':row['image_type'],
            'original_prompt':original,'original_prompt_sha256':sha(original.encode()),'prompt':prompt,'prompt_sha256':sha(prompt.encode()),
            'archive_member':'data/crop/16-9/'+name,'history':'PROVISIONAL_UNTIL_HISTORICAL_AND_NEAR_DUPLICATE_REVIEW',
            'exposure_history':{'known_name_ordinal_overlap':False,'exact_image_sha_checked':False,'near_duplicate_review':'PENDING'}})
    archive=(hf/ARCHIVE).resolve();st=archive.stat();need(stat.S_ISREG(st.st_mode) and 0<st.st_size<=RESOURCES['archive_stream_bytes_max'],'archive stat differs')
    prior=ROOT/'.runtime/inputs/formal20/recovery_report.json'
    need(json.loads(prior.read_text())['archive_sha256']==ARCHIVE_SHA,'historic archive proof differs')
    return {'id':ID,'status':'DRAFT','assets_finalized':False,'automatic_retry':False,'resources':RESOURCES,'limits':LIMITS,
        'archive':{'path':ARCHIVE,'bytes':st.st_size,'sha256':ARCHIVE_SHA},'metadata':{'path':METADATA,'bytes':len(metadata),'sha256':META_SHA},
        'output_HF_relative':OUTPUT,'source_candidates':source,'history_sources':[bind(p,ROOT) for p in paths],
        'repo_sources':[bind(ROOT/p,ROOT) for p in SOURCES], 'known_history_hash_count':len(hashes),
        'archive_provenance':bind(prior,ROOT),
        'independent_review':None,'source_readiness':'PROVISIONAL_PENDING_ALL_HISTORY_AND_ROOT_VISUAL_REVIEW'}


class HashReader:
    def __init__(self,stream,report):self.stream=stream;self.report=report;self.hash=hashlib.sha256()
    def read(self,n=-1):
        x=self.stream.read(n);self.report['actual_archive_bytes']+=len(x);self.hash.update(x)
        need(self.report['actual_archive_bytes']<=RESOURCES['archive_stream_bytes_max'],'archive read budget exceeded');return x


def extract_members(fileobj, wanted, max_bytes):
    """Return only exact named regular files; tar stream is consumed once."""
    found={}
    with tarfile.open(fileobj=fileobj,mode='r|gz') as tar:
        for member in tar:
            name=member.name
            if name not in wanted:continue
            need(name not in found and member.isfile() and not name.startswith('/') and '..' not in Path(name).parts
                 and 0<member.size<=max_bytes,'duplicate, unsafe or oversized selected member')
            stream=tar.extractfile(member);need(stream is not None,'missing member reader');data=stream.read(member.size+1)
            need(len(data)==member.size,'truncated selected member');found[name]=data
    need(set(found)==set(wanted),'selected original member missing')
    return found


def image_info(data):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im.load();w,h=im.size;need(w>0 and h>0 and im.format in ('JPEG','PNG','WEBP'),'unsupported image')
        pixels=list(im.convert('L').resize((9,8)).getdata())
        code=sum((pixels[y*9+x]>pixels[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
        return {'width':w,'height':h,'format':im.format,'dhash64':f'{code:016x}'}


def run(args):
    started=time.monotonic();p=json.loads(args.protocol.read_text());d=sha(args.protocol.read_bytes())
    need(p['id']==ID and p['resources']==RESOURCES and p['limits']==LIMITS and p['status']=='FROZEN_REGISTERED' and p['assets_finalized'] is True and p.get('independent_review'),'protocol not frozen')
    active=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text());scopes=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==ID]
    need(active['user_continuation_authorization_20261027'].get('new_scope_registration_allowed') is True and len(scopes)==1 and scopes[0].get('protocol_sha256')==d and scopes[0].get('resources')==RESOURCES and scopes[0].get('limits')==LIMITS and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and scopes[0].get('max_gpu_cards')==0,'ACTIVE scope differs')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=2 and not {'torch','numpy'} & set(sys.modules),'CPU-only fresh worker required')
    from worker_identity import require_worker
    require_worker();hf=args.hf_root.resolve();fresh=make_draft(hf)
    need(all(p[k]==fresh[k] for k in ('archive','metadata','source_candidates','history_sources','repo_sources','archive_provenance','output_HF_relative')),'fixed source/archive/history closure differs')
    out=(hf/OUTPUT).resolve();need(out.is_relative_to(hf) and not out.exists(),'fresh output required');out.mkdir(parents=True)
    report={'id':ID,'protocol_sha256':d,'status':'STARTED_CPU_SOURCE_EXTRACTION_NO_RETRY','predebited_archive_bytes':0,'actual_archive_bytes':0,'sources':[],'model_calls':0,'GPU_calls':0,'automatic_retry':False}
    def timeout(*_):raise TimeoutError('source extraction wall exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(RESOURCES['wall_seconds']);write(out/'RUN.json',report)
    try:
        for b in p['repo_sources']+p['history_sources']+[p['archive_provenance']]:bound(ROOT,b)
        need(json.loads(bound(ROOT,p['archive_provenance']).read_text())['archive_sha256']==ARCHIVE_SHA,'historic archive SHA proof differs')
        metadata=bound(hf,p['metadata']).read_bytes();need(sha(metadata)==META_SHA,'metadata changed')
        banks=[bound(ROOT,b) for b in p['history_sources']];hashes,names,indices,history_paths=history(banks)
        need([x['native_source_ordinal'] for x in p['source_candidates']]==list(ORDINALS),'fixed ordinal differs')
        old=[];missing=[]
        for name,expected in history_paths.items():
            path=Path(name)
            if not path.is_file():missing.append(name);continue
            data=path.read_bytes();h=sha(data)
            if expected:need(h==expected,'historical image SHA changed')
            try:info=image_info(data)
            except (OSError,ValueError):missing.append(name);continue
            hashes.setdefault(h,set()).add('actual_history_image');old.append((h,info['dhash64']))
        report['historical_image_paths_checked']=len(old);report['historical_image_paths_unavailable']=len(missing)
        archive=(hf/p['archive']['path']).resolve();need(archive.is_relative_to(hf),'archive escaped');before=archive.stat();need(before.st_size==p['archive']['bytes'] and p['archive']['sha256']==ARCHIVE_SHA,'archive binding differs')
        report['predebited_archive_bytes']=before.st_size;write(out/'RUN.json',report)
        with archive.open('rb',buffering=0) as f:
            reader=HashReader(f,report);payload=extract_members(reader,{x['archive_member'] for x in p['source_candidates']},RESOURCES['single_image_bytes_max'])
            while reader.read(8<<20):pass
        after=archive.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and report['actual_archive_bytes']==before.st_size and reader.hash.hexdigest()==ARCHIVE_SHA,'archive full SHA/stat differs')
        for case in p['source_candidates']:
            data=payload[case['archive_member']];h=sha(data);info=image_info(data);need(h not in hashes and case['native_source_ordinal'] not in indices and Path(case['archive_member']).name not in names,'FAIL_UNUSABLE_COHORT_KNOWN_HISTORY_ALIAS')
            need(not any(x['image']['sha256']==h for x in report['sources']),'FAIL_UNUSABLE_COHORT_DUPLICATE_IMAGE')
            path=out/f"source_{case['cohort_index']:02}.jpg";need(sum(len(v) for v in payload.values())<RESOURCES['new_HF_bytes']-(1<<20),'output bound exceeded');path.write_bytes(data)
            near=sorted(({'history_sha256':oldsha,'hamming':(int(info['dhash64'],16)^int(oldhash,16)).bit_count()} for oldsha,oldhash in old),key=lambda x:(x['hamming'],x['history_sha256']))[:5]
            within=[{'source':x['native_source_ordinal'],'hamming':(int(info['dhash64'],16)^int(x['image_info']['dhash64'],16)).bit_count()} for x in report['sources']]
            report['sources'].append({**case,'image':bind(path,hf),'image_info':info,'nearest_checked_history_dhash':near,
                'earlier_candidate_dhash':within,
                'exposure_history':{'known_name_ordinal_overlap':False,'known_image_sha_overlap':False,'checked_history_banks':list(BANKS),'near_duplicate_review':'PENDING_ROOT_VISUAL_REVIEW','history_coverage':'SCANNED_BANKS_AND_TASKS_NOT_GLOBAL_HISTORY'}});write(out/'RUN.json',report)
        need(len(report['sources'])==10,'source count differs')
        report.update(status='COMPLETE_PROVISIONAL_SOURCE_BYTES_PENDING_HUMAN_AND_HISTORY_REVIEW',source_bank_status='PROVISIONAL_NOT_AUTHORIZED_FOR_MODEL')
    except BaseException as e:
        report.update(status='FAILED_SOURCE_EXTRACTION_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-started
        report['new_output_bytes_before_final_RUN']=sum(x.stat().st_size for x in out.iterdir() if x.is_file())
        write(out/'RUN.json',report)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','run'));ap.add_argument('--hf-root',type=Path,default=HF);ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    if a.action=='draft':need(a.out is not None and not a.out.exists(),'fresh draft required');write(a.out,make_draft(a.hf_root.resolve()))
    else:run(a)
if __name__=='__main__':main()
