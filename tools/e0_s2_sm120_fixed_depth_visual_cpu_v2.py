"""V2 CPU-only contact sheets; independent namespace and durable operation journal.

Derived from frozen V1 tools/e0_s2_sm120_fixed_depth_visual_cpu.py,
SHA256 13c36ae107f2018e5f1d5b55eec78f548c8e2865a3f98968043562a79ce6f56b.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'tools'))
def private(name, file):
    spec=importlib.util.spec_from_file_location(name, ROOT/file)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
old=private('_sm120_visual_old', 'tools/e0_s2_new12_dense_visual_cpu.py')
motion=private('_sm120_visual_parent', 'tools/e0_s2_sm120_fixed_depth_motion_metrics.py')
base=motion.base; need=base.require
SELF='tools/e0_s2_sm120_fixed_depth_visual_cpu_v2.py'
TEST='tests/test_e0_s2_sm120_fixed_depth_visual_cpu_v2.py'
RESOURCES={'GPU':0,'CPU_threads':2,'wall_seconds':900,'outer_seconds':1020,
           'new_HF_bytes':96<<20,'max_RSS_bytes':4<<30,'automatic_retry':False}
LIMITS={'videos':12,'successful_decoded_frames':972,'eof_checks':12,'JPEG_pages':24,
        'model_calls':0,'GPU_calls':0,'RAFT_calls':0,'video_generations':0,'video_encodes':0}
STATUS='COMPLETE_SM120_FOUR_ARM_ALL81_CONTACT_SHEETS_NOT_HUMAN_REVIEW'
INTERPRETATION='CONTACT_SHEETS_FOR_SEPARATE_AI_VISUAL_REVIEW_NOT_HUMAN_FULL_PLAYBACK'
LAYOUT={'columns':6,'rows':7,'tile_width':320,'tile_image_height':178,'label_height':24,
        'jpeg_quality':90,'frame_count':81,'fps':16,'source_size':[832,464],
        'page_frames':[old.page_positions(0),old.page_positions(1)]}


V1_PROTOCOL='config/e0_s2_sm120_fixed_depth_visual_part0_20261029_v1.json'
V1_WORKER='ffnds-derived/e0_s2_sm120_fixed_depth_all81_visual_cpu_20261029/part0/attempt1/worker'
V1_SHAS={'protocol':'160a6f6b1dc6770e17ec83ec09bf5a8967a7754028f82bc721133c234260bd19',
         'RUN':'f828975f2b0db8798ca99acc7f6fb1428c7257812b82ecd0e987fe3410548913'}

def failure_bindings(hf):
    result={'protocol':base.binding(ROOT/V1_PROTOCOL,ROOT),
            'RUN':base.binding(Path(hf)/V1_WORKER/'results/RUN.json',hf),
            'exit':base.binding(Path(hf)/V1_WORKER/'EXIT_RECEIPT.json',hf)}
    need(result==failure_expected(),'V1 frozen failure differs')
    return result

def failure_expected():
    return {'protocol':{'path':V1_PROTOCOL,'bytes':295631,'sha256':V1_SHAS['protocol']},
            'RUN':{'path':V1_WORKER+'/results/RUN.json','bytes':722349,'sha256':V1_SHAS['RUN']},
            'exit':{'path':V1_WORKER+'/EXIT_RECEIPT.json','bytes':544,'sha256':'fb0cc281aed519e59714b46eda404e10b7a7ed31478b29db4cb2b39a5fda629b'}}

def verify_failure(plan,hf):
    proof=plan['v1_failure'];p=base.read_bound(ROOT,proof['protocol']);r=base.read_bound(hf,proof['RUN']);e=base.read_bound(hf,proof['exit'])
    need(p['id']==r['id']==e['id']=='E0_S2_SM120_FIXED_DEPTH_ALL81_VISUAL_CPU_20261029_PART0'
         and r['protocol_sha256']==e['protocol_sha256']==proof['protocol']['sha256']
         and e['run_sha256']==proof['RUN']['sha256'] and e['child_returncode']==1
         and e['process_exited'] is True and e['timeout'] is False and e['automatic_retry'] is False
         and r['status']=='FAILED_SM120_VISUAL_PRESERVED_NO_RETRY' and r['error_type']=='ValueError'
         and r['error']=='Page exceeds 1 MiB' and r['completed']['JPEG_pages']==16
         and r['predebited']['JPEG_pages']==r['invoked']['JPEG_pages']==17
         and len(r['videos'])==9 and sum(v['status']=='COMPLETE' for v in r['videos'])==8
         and r['videos'][-1]['ordinal']==616, 'V1 failure evidence differs')

SIDES=('predebited','invoked','completed')
def replay_journal(path):
    counts={side:dict.fromkeys(LIMITS,0) for side in SIDES}
    with open(path,encoding='utf-8') as f:
        for line in f:
            event=json.loads(line);need(set(event)=={'side','key'} and event['side'] in SIDES and event['key'] in LIMITS,'Invalid journal entry')
            side,key=event['side'],event['key'];counts[side][key]+=1
            need(counts['predebited'][key]<=LIMITS[key] and counts['completed'][key]<=counts['invoked'][key]<=counts['predebited'][key]
                 and counts['predebited'][key]-counts['completed'][key]<=1,'Invalid journal order or budget')
    return counts

class JournalLedger:
    def __init__(self,report,path):
        self.report=report;self.stream=open(path,'x',encoding='utf-8')
        for side in SIDES:report[side]=dict.fromkeys(LIMITS,0)
    def mark(self,side,key):
        self.stream.write(json.dumps({'side':side,'key':key},separators=(',',':'))+'\n');self.stream.flush()
        self.report[side][key]+=1
    def call(self,key,fn):
        need(key in LIMITS and self.report['predebited'][key]<LIMITS[key],'Budget exceeded: '+key)
        self.mark('predebited',key);self.mark('invoked',key)
        value=fn();self.mark('completed',key);return value
    def finish(self):
        need(all(self.report[side]==LIMITS for side in SIDES),'Actual operation coverage incomplete')
    def close(self):self.stream.close()


def identity(part):
    need(type(part) is int and part in range(4),'Four fixed visual shards required')
    return f'E0_S2_SM120_FIXED_DEPTH_ALL81_VISUAL_CPU_V2_20261029_PART{part}'


def output(part):
    identity(part)
    return f'ffnds-derived/e0_s2_sm120_fixed_depth_all81_visual_cpu_v2_20261029/part{part}/attempt1/worker/results'


def parent_rows(proof,hf,part):
    from types import SimpleNamespace
    need(proof['protocol']['path']==motion.PARENTS.format(part),'Wrong native parent protocol')
    parent=base.read_bound(ROOT,proof['protocol']); run=base.read_bound(hf,proof['RUN'])
    base.exited(base.read_bound(hf,proof['exit']),proof['RUN'],proof['protocol']['sha256'],parent['id'])
    motion.video_parent.verify_closed(SimpleNamespace(protocol=ROOT/proof['protocol']['path'],hf_root=Path(hf),
        run=Path(hf)/proof['RUN']['path'],exit=Path(hf)/proof['exit']['path']))
    need(parent['part']==part and run['part']==part and run['id']==parent['id']
         and run['protocol_sha256']==proof['protocol']['sha256']
         and [c['native_source_ordinal'] for c in parent['cases']]==list(motion.SHARDS[part])
         and len(run['phases'])==14 and [p['run_id'] for p in run['phases']]==parent['run_ids'],
         'Closed parent source/phase coverage differs')
    rows=[]
    for case in parent['cases']:
        phases=[p for p in run['phases'][2:] if p['native_source_ordinal']==case['native_source_ordinal']]
        need([p['mode'] for p in phases]==list(motion.video_parent.order(case)), 'Parent arm order differs')
        for p in phases:
            gen=base.read_bound(hf,p['generation_receipt'])
            need(p['id']==parent['id'] and p['protocol_sha256']==proof['protocol']['sha256']
                 and p['numeric_path']=='BF16_ORIGINAL' and p['seed']==case['seed']==42
                 and p['source_id']==case['source_id'] and p['image_sha256']==case['image']['sha256']
                 and p['prompt_sha256']==case['prompt_sha256'] and gen['states']==p['states']
                 and gen['video']==p['video'] and gen['run_id']==p['run_id'] and gen['mode']==p['mode']
                 and p['video']['frames']==81 and p['video']['size']==[832,464] and p['video']['fps']==16,
                 'Parent generation/source identity differs')
            rows.append({'ordinal':case['native_source_ordinal'],'mode':p['mode'],'case':case,
                         'generation':p['generation_receipt'],'video':base.clip_binding(p['video'],hf),'states':p['states']})
    paths=[base.bound(hf,r['video']) for r in rows]
    need(len(rows)==len({(r['ordinal'],r['mode']) for r in rows})==12
         and len({r['case']['image']['sha256'] for r in rows})==3
         and len(set(paths))==len({(p.stat().st_dev,p.stat().st_ino) for p in paths})==12,
         'Distinct source and clip identity required')
    return rows


def draft(hf,part):
    parent=motion.parent_data(hf,part); rows=parent_rows(parent,hf,part)
    protocol=base.read_bound(ROOT,parent['protocol'])
    paths={b['path'] for b in protocol['repo_sources']} | {SELF,TEST,old.SELF,
        'tools/e0_s2_new12_dense_motion_metrics.py','tools/e0_s2_sm120_fixed_depth_visual_cpu.py',motion.SELF,motion.PARENTS.format(part)}
    return {'id':identity(part),'part':part,'status':'DRAFT','assets_finalized':False,
        'resources':copy.deepcopy(RESOURCES),'limits':copy.deepcopy(LIMITS),
        'output_HF_relative':output(part),'automatic_retry':False,'parent':parent,'v1_failure':failure_bindings(hf),'videos':rows,
        'repo_sources':[base.binding(ROOT/p,ROOT) for p in sorted(paths)],
        'page_layout':copy.deepcopy(LAYOUT),'interpretation':INTERPRETATION}


def validate(plan,frozen=False):
    part=plan['part']; proof=plan['parent']; parent=base.read_bound(ROOT,proof['protocol'])
    need(proof['protocol']['path']==motion.PARENTS.format(part) and plan['id']==identity(part)
         and plan['resources']==RESOURCES and plan['limits']==LIMITS and plan['output_HF_relative']==output(part)
         and plan['automatic_retry'] is False and plan['page_layout']==LAYOUT and plan['interpretation']==INTERPRETATION,
         'Fixed visual scope differs')
    need(plan.get('v1_failure')==failure_expected(), 'Fixed V1 failure binding differs')
    rows=plan['videos'];need(len(rows)==12 and [(x['ordinal'],x['mode']) for x in rows]==
        [(o,m) for o in motion.SHARDS[part] for m in motion.video_parent.order(next(v['case'] for v in rows if v['ordinal']==o))],
        'Fixed source/arm order differs')
    paths={b['path'] for b in parent['repo_sources']} | {SELF,TEST,old.SELF,
        'tools/e0_s2_new12_dense_motion_metrics.py','tools/e0_s2_sm120_fixed_depth_visual_cpu.py',motion.SELF,motion.PARENTS.format(part)}
    actual=[b['path'] for b in plan['repo_sources']]
    need(len(actual)==len(set(actual)) and set(actual)==paths,'Repository closure differs')
    if frozen:need(plan['status']=='FROZEN_REGISTERED' and plan['assets_finalized'] is True,'Frozen protocol required')


def authorize(plan,digest,active=None):
    validate(plan,True); active=base.read(ROOT/'research/ACTIVE_RESEARCH.json') if active is None else active
    scopes=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==plan['id']]
    need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True
         and len(scopes)==1 and scopes[0].get('protocol_sha256')==digest
         and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
         and scopes[0].get('resources')==RESOURCES and scopes[0].get('limits')==LIMITS
         and scopes[0].get('max_gpu_cards')==0,'Exact registered CPU scope required')
    return scopes[0]


def stat(path):
    s=Path(path).stat();return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}


def inputs(plan):
    rows=[('REPO',b) for b in plan['repo_sources']]
    rows += [('REPO',plan['parent']['protocol']),('REPO',plan['v1_failure']['protocol'])]+[('HF',plan['parent'][k]) for k in ('RUN','exit')]+[('HF',plan['v1_failure'][k]) for k in ('RUN','exit')]
    for v in plan['videos']:
        rows += [('HF',v['video']),('HF',v['generation']),('HF',v['case']['image'])]
        rows += [('HF',b) for b in v['states'].values()]
    items={}
    for root,b in rows:
        item={k:b[k] for k in ('path','bytes','sha256')};key=(root,item['path'])
        need(key not in items or items[key]==item,'Input binding collision');items[key]=item
    return [(r,b) for (r,_),b in sorted(items.items())]


def capture_inputs(plan,hf):
    result=[]
    for root,b in inputs(plan):
        p=(ROOT if root=='REPO' else hf)/b['path'];before=stat(p);actual=base.bound(ROOT if root=='REPO' else hf,b);after=stat(actual)
        need(before==after,'Input changed while hashing');result.append({'root':root,'binding':b,'stat':after})
    return result


def check_stats(records,hf):
    for r in records:
        p=(ROOT if r['root']=='REPO' else hf)/r['binding']['path']
        need(stat(p)==r['stat'],'Bound input changed after hashing')


def verify_parent(plan,hf):
    need(parent_rows(plan['parent'],hf,plan['part'])==plan['videos'], 'Closed native parent/video lineage differs')


class Sheets(old.Sheets):
    def add(self,frame,source,index):
        super().add(frame,source['ordinal'],index)
        cv2=self.cv2;slot=index%42;y=(slot//6)*202;x=(slot%6)*320
        cv2.rectangle(self.image,(x,y+178),(x+319,y+201),(32,32,32),-1)
        cv2.putText(self.image,'{} {} f{:02d} {:.3f}s'.format(source['ordinal'],source['mode'],index,index/16),
                    (x+3,y+195),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1,cv2.LINE_AA)


def extract(reader,renderer,source,ledger,write_page,checkpoint=lambda:None):
    for index in range(81):
        def read():
            ok,frame=reader.read();need(ok and frame is not None,f'Short video at frame {index}');return frame
        frame=ledger.call('successful_decoded_frames',read);renderer.add(frame,source,index)
        if index==41:
            ledger.call('JPEG_pages',lambda:write_page(renderer.finish(0),0,old.page_positions(0)));checkpoint()
    ok,frame=ledger.call('eof_checks',reader.read);need(not ok and frame is None,'Video has extra frames')
    ledger.call('JPEG_pages',lambda:write_page(renderer.finish(1),1,old.page_positions(1)));checkpoint()


def run(args):
    hf=args.hf_root.resolve();plan=base.read(args.protocol);digest=base.sha(args.protocol);authorize(plan,digest)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=2
         and not {'torch','cv2','numpy'} & set(sys.modules),'Fresh hidden CUDA/two CPU worker required')
    from worker_identity import require_worker
    require_worker();out=(hf/output(plan['part'])).resolve()
    need(out.is_relative_to(hf) and out!=hf and not out.exists(),'Fresh confined output required');out.mkdir(parents=True)
    start=time.monotonic();report={'id':plan['id'],'protocol_sha256':digest,'status':'STARTED_SM120_ALL81_VISUAL',
        'part':plan['part'],'parent':plan['parent'],'automatic_retry':False,'videos':[],
        'visual_review':'PENDING_SEPARATE_AI_INSPECTION','human_full_playback_review':'NOT_PERFORMED','excess_frames_detected':0}
    def persist():
        base.write(out/'RUN.json',report)
        need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())<=RESOURCES['new_HF_bytes'], 'HF output limit exceeded')
    ledger=JournalLedger(report,out/'OPERATIONS.jsonl');persist()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Visual CPU wall budget exceeded')));signal.alarm(RESOURCES['wall_seconds'])
    resource.setrlimit(resource.RLIMIT_FSIZE,(8<<20,8<<20)); no_torch=old.NoTorch();sys.meta_path.insert(0,no_torch)
    try:
        records=capture_inputs(plan,hf)
        verify_failure(plan,hf);verify_parent(plan,hf);check_stats(records,hf)
        base.write(out/'INPUT_PROOF.json',{'id':plan['id'],'protocol_sha256':digest,'records':records})
        report['input_proof']=base.binding(out/'INPUT_PROOF.json',hf);persist()
        import cv2
        import numpy as np
        need('torch' not in sys.modules,'Torch forbidden');cv2.setNumThreads(2)
        report['environment']={'python':sys.executable,'opencv_version':cv2.__version__,'numpy_version':np.__version__,
            'opencv_origin':str(Path(cv2.__file__).resolve()),'numpy_origin':str(Path(np.__file__).resolve()),
            'cuda_visible_devices':'','CPU_affinity':sorted(os.sched_getaffinity(0))};persist()
        for source in plan['videos']:
            ordinal=source['ordinal'];mode=source['mode'];entry={**copy.deepcopy(source),'pages':[],'decoded_frames':0,'status':'STARTED'}
            report['videos'].append(entry);persist();path=base.bound(hf,source['video']);before=stat(path)
            def open_video():
                cap=cv2.VideoCapture(str(path));need(cap.isOpened(),'Video open failed');return cap
            cap=ledger.call('videos',open_video)
            try:
                geom={'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      'fps':cap.get(cv2.CAP_PROP_FPS),'reported_frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
                need(geom=={'width':832,'height':464,'fps':16.0,'reported_frames':81},'Video geometry differs')
                entry['decoded_geometry']=geom;renderer=Sheets(cv2,np)
                class Reader:
                    def read(self):
                        ok,frame=cap.read()
                        if ok:
                            entry['decoded_frames']+=1
                            if entry['decoded_frames']>81:report['excess_frames_detected']+=1
                        return ok,frame
                def write_page(image,page,positions):
                    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,90]);need(ok,'JPEG encoding failed')
                    payload=encoded.tobytes();need(len(payload)<=3<<20,'Page exceeds 3 MiB')
                    current=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
                    need(current+len(payload)+(4<<20)<=RESOURCES['new_HF_bytes'],'Output byte limit exceeded')
                    dest=out/'pages'/f'source{ordinal}_{mode}_page{page}.jpg';dest.parent.mkdir(exist_ok=True)
                    with dest.open('xb') as f:f.write(payload)
                    labels=[None if n is None else {'source_ordinal':ordinal,'mode':mode,'frame':n,'timestamp_seconds':n/16} for n in positions]
                    entry['pages'].append({'page':page,'frames':positions,'labels':labels,'jpeg':base.binding(dest,hf),
                                           'width':1920,'height':1414,'quality':90})
                extract(Reader(),renderer,source,ledger,write_page,persist)
                need(entry['decoded_frames']==81 and len(entry['pages'])==2 and stat(path)==before,'Video or source changed')
                entry['status']='COMPLETE';persist()
            finally:cap.release()
        ledger.finish();check_stats(records,hf)
        need(len(report['videos'])==12 and report['excess_frames_detected']==0 and 'torch' not in sys.modules,'Incomplete or forbidden work')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;need(rss<=RESOURCES['max_RSS_bytes'],'RSS limit exceeded')
        manifest={'schema':'ffnds.sm120.fixed_depth.all81.contact_sheets.v2','id':plan['id'],'protocol_sha256':digest,
                  'parent':plan['parent'],'videos':report['videos'],'interpretation':INTERPRETATION}
        base.write(out/'MANIFEST.json',manifest)
        report.update(status=STATUS,manifest=base.binding(out/'MANIFEST.json',hf),max_RSS_bytes=rss)
    except BaseException as e:
        report.update(status='FAILED_SM120_VISUAL_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0);ledger.close();report['journal']=base.binding(out/'OPERATIONS.jsonl',hf)
        need(replay_journal(out/'OPERATIONS.jsonl')=={k:report[k] for k in ('predebited','invoked','completed')},'Journal ledger mismatch')
        report['wall_seconds']=time.monotonic()-start;persist()
        if no_torch in sys.meta_path:sys.meta_path.remove(no_torch)


def verify_closed(args):
    hf=args.hf_root.resolve();plan=base.read(args.protocol);validate(plan,True);digest=base.sha(args.protocol);out=(hf/output(plan['part'])).resolve()
    need(args.run.resolve()==out/'RUN.json' and args.exit.resolve()==out.parent/'EXIT_RECEIPT.json','Wrong run/exit path')
    binding=base.binding(args.run,hf);report=base.read(args.run);base.exited(base.read(args.exit),binding,digest,plan['id'])
    need(report.get('id')==plan['id'] and report.get('protocol_sha256')==digest and report.get('status')==STATUS
         and all(report.get(k)==LIMITS for k in ('predebited','invoked','completed'))
         and report.get('excess_frames_detected')==0 and 0<=report.get('wall_seconds',-1)<=RESOURCES['wall_seconds']
         and 0<=report.get('max_RSS_bytes',-1)<=RESOURCES['max_RSS_bytes'], 'Closed run differs')
    proof=base.read_bound(hf,report['input_proof']);records=proof['records']
    need(proof['id']==plan['id'] and proof['protocol_sha256']==digest and [(r['root'],r['binding']) for r in records]==inputs(plan),'Input audit differs')
    need(base.bound(hf,report['input_proof'])==out/'INPUT_PROOF.json' and base.bound(hf,report['journal'])==out/'OPERATIONS.jsonl','Proof paths differ')
    need(replay_journal(out/'OPERATIONS.jsonl')=={k:report[k] for k in ('predebited','invoked','completed')},'Journal ledger mismatch')
    check_stats(records,hf);verify_failure(plan,hf);verify_parent(plan,hf)
    manifest=base.read_bound(hf,report['manifest'])
    need(manifest.get('schema')=='ffnds.sm120.fixed_depth.all81.contact_sheets.v2'
         and manifest.get('id')==plan['id'] and manifest.get('protocol_sha256')==digest
         and manifest.get('parent')==plan['parent'] and manifest.get('interpretation')==INTERPRETATION
         and manifest.get('videos')==report['videos'] and len(manifest['videos'])==12,'Manifest differs')
    paths=[]
    for actual,expected in zip(manifest['videos'],plan['videos']):
        need(all(actual.get(k)==v for k,v in expected.items()) and actual['decoded_frames']==81
            and actual['decoded_geometry']=={'width':832,'height':464,'fps':16.0,'reported_frames':81}
            and actual['status']=='COMPLETE' and len(actual['pages'])==2,'Visual video identity differs')
        for i,page in enumerate(actual['pages']):
            positions=old.page_positions(i);labels=[None if n is None else {'source_ordinal':actual['ordinal'],'mode':actual['mode'],
                        'frame':n,'timestamp_seconds':n/16} for n in positions]
            need(page['page']==i and page['frames']==positions and page['labels']==labels and page['width']==1920
                 and page['height']==1414 and page['quality']==90 and page['jpeg']['bytes']<=3<<20,'Page metadata differs')
            p=base.bound(hf,page['jpeg']);need(p==(out/'pages'/f'source{actual["ordinal"]}_{actual["mode"]}_page{i}.jpg').resolve(),'Page path differs');paths.append(p)
    need(len(paths)==len(set(paths))==len({(p.stat().st_dev,p.stat().st_ino) for p in paths})==24,'Page aliases')
    return {'id':plan['id'],'status':'PASS_CLOSED_SM120_ALL81_VISUAL_CPU','RUN':binding,'exit':base.binding(args.exit,hf),
            'manifest':report['manifest'],'model_calls':0,'GPU_calls':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('draft','inspect','run','verify-closed'))
    p.add_argument('--part',type=int,choices=range(4));p.add_argument('--protocol',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));p.add_argument('--run',type=Path);p.add_argument('--exit',type=Path);a=p.parse_args()
    if a.action=='draft':
        need(a.part is not None and a.out is not None and not a.out.exists(),'Fresh draft required');a.out.parent.mkdir(parents=True,exist_ok=True);base.write(a.out,draft(a.hf_root,a.part))
    elif a.action=='inspect':validate(base.read(a.protocol));print('STATIC_ONLY_NOT_MODEL')
    elif a.action=='run':run(a)
    else:print(json.dumps(verify_closed(a),indent=2))

if __name__=='__main__':main()
