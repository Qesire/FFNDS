"""Generate two sequential contact sheets per existing New12 Dense video.

A registered CPU-only scope binds actual generation and exited RAFT parents.
Frames are decoded in order; no model, metric, GPU or video encoding is used.
"""
import argparse
import copy
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('_new12_visual_motion_parent',ROOT/'tools/e0_s2_new12_dense_motion_metrics.py')
motion=importlib.util.module_from_spec(spec);spec.loader.exec_module(motion)
base=motion.base
need=base.require
ID='E0_S2_NEW12_DENSE_ALL81_VISUAL_CPU_20261029'
REPORT='research/20261029/S2_NEW12_DENSE_CONTINUOUS_MOTION_RESULTS_20261029.json'
OUTPUT='ffnds-derived/e0_s2_new12_dense_all81_visual_cpu_20261029/attempt1/worker/results'
SELF='tools/e0_s2_new12_dense_visual_cpu.py'
TEST='tests/test_e0_s2_new12_dense_visual_cpu.py'
SOURCES=(911,912,978,979,1039,1040,922,923,616,624,456,472)
RESOURCES={'GPU':0,'CPU_threads':2,'wall_seconds':300,'outer_seconds':600,
           'new_HF_bytes':64<<20,'max_RSS_bytes':4<<30,'automatic_retry':False}
LIMITS={'videos':12,'successful_decoded_frames':972,'eof_checks':12,'JPEG_pages':24,
        'model_calls':0,'GPU_calls':0,'RAFT_calls':0,'video_generations':0,'video_encodes':0}
STATUS='COMPLETE_NEW12_DENSE_ALL81_CONTACT_SHEETS_NOT_HUMAN_REVIEW'


def page_positions(page):
    need(page in (0,1),'Only two pages')
    start=page*42
    return [n if n<81 else None for n in range(start,start+42)]


def source_view(row):
    return {k:copy.deepcopy(row[k]) for k in ('native_source_ordinal','source_id','image_id','prompt_sha256','prompt',
                                               'image_type','input_scene_stratum','generation','video','metric_protocol_id')}


def verify_parent(report,hf):
    need(report.get('status')=='COMPLETE_12_REAL_DENSE_VIDEOS_3_SHARDS_ACTUAL_EXIT0'
         and [x['native_source_ordinal'] for x in report['sources']]==list(SOURCES)
         and len(report['metric_parts'])==3 and report.get('seed')==42,'Exact parent source cohort required')
    paths=[]; observations={}; videos={}
    for i,parent in enumerate(report['metric_parts']):
        plan=base.read_bound(ROOT,parent['protocol']); run=base.read_bound(hf,parent['RUN'])
        base.exited(base.read_bound(hf,parent['exit']),parent['RUN'],parent['protocol']['sha256'],plan['id'])
        motion.validate(plan,True);need(plan['shard']==i and run['protocol_sha256']==parent['protocol']['sha256'], 'Wrong metric shard')
        motion.validate_result(plan,run,hf);motion.verify_lineage(plan,hf)
        for v,obs in zip(plan['videos'],run['RAFT_observations']):
            need(v['ordinal'] not in videos,'Duplicate metric source')
            videos[v['ordinal']]=v;observations[v['ordinal']]=obs
    for row in report['sources']:
        ordinal=row['native_source_ordinal']; v=videos[ordinal];case=v['case'];obs=observations[ordinal]
        need(row['video']==v['video'] and row['generation']==v['generation']
             and row['image_id']==case['image']['sha256'] and row['source_id']==case['source_id']
             and row['prompt']==case['prompt'] and row['prompt_sha256']==case['prompt_sha256']
             and hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
             and row['metric_protocol_id']==motion.identity(next(i for i,s in enumerate(motion.SHARDS) if ordinal in s))
             and row['summary']==obs['summary'] and row['scores_top5percent_flow_px_per_2frame_pair']==obs['scores'],
             'Source report/metric/generation mismatch')
        path=base.bound(hf,row['video']);paths.append(path)
    need(len(videos)==len(paths)==len({x['image_id'] for x in report['sources']})
         ==len({x['video']['sha256'] for x in report['sources']})
         ==len(set(paths))==len({(p.stat().st_dev,p.stat().st_ino) for p in paths})==12,
         'Distinct source/video SHA and inode required')


def draft(hf):
    report=base.read(ROOT/REPORT);verify_parent(report,hf)
    repos={SELF,TEST,REPORT,'tools/e0_s2_new12_dense_motion_metrics.py'}
    for part in report['metric_parts']:
        repos.add(part['protocol']['path'])
        old=base.read(ROOT/part['protocol']['path'])
        repos.update(b['path'] for b in old['repo_sources'])
    return {'id':ID,'status':'DRAFT','assets_finalized':False,'resources':copy.deepcopy(RESOURCES),
        'limits':copy.deepcopy(LIMITS),'output_HF_relative':OUTPUT,'automatic_retry':False,
        'parent_report':base.binding(ROOT/REPORT,ROOT),'metric_parts':copy.deepcopy(report['metric_parts']),
        'sources':[source_view(x) for x in report['sources']],
        'repo_sources':[base.binding(ROOT/p,ROOT) for p in sorted(repos)],
        'page_layout':{'columns':6,'rows':7,'tile_width':320,'tile_image_height':178,'label_height':24,
                       'jpeg_quality':90,'frame_count':81,'fps':16,'source_size':[832,464],
                       'page_frames':[page_positions(0),page_positions(1)]},
        'interpretation':'CONTACT_SHEETS_FOR_SEPARATE_AI_VISUAL_REVIEW_NOT_HUMAN_FULL_PLAYBACK'}


def validate(plan,frozen=False):
    need(plan['id']==ID and plan['resources']==RESOURCES and plan['limits']==LIMITS
         and plan['output_HF_relative']==OUTPUT and plan['automatic_retry'] is False
         and plan['parent_report']['path']==REPORT
         and plan['interpretation']=='CONTACT_SHEETS_FOR_SEPARATE_AI_VISUAL_REVIEW_NOT_HUMAN_FULL_PLAYBACK',
         'Fixed visual scope differs')
    report=base.read_bound(ROOT,plan['parent_report'])
    need(plan['metric_parts']==report['metric_parts'] and plan['sources']==[source_view(x) for x in report['sources']]
         and [x['native_source_ordinal'] for x in plan['sources']]==list(SOURCES),'Report cohort differs')
    expected={SELF,TEST,REPORT,'tools/e0_s2_new12_dense_motion_metrics.py'}
    for p in plan['metric_parts']:
        expected.add(p['protocol']['path']);old=base.read_bound(ROOT,p['protocol']);expected.update(b['path'] for b in old['repo_sources'])
    paths=[b['path'] for b in plan['repo_sources']]
    need(len(paths)==len(set(paths)) and set(paths)==expected,'Source closure differs')
    need(plan['page_layout']=={'columns':6,'rows':7,'tile_width':320,'tile_image_height':178,'label_height':24,
         'jpeg_quality':90,'frame_count':81,'fps':16,'source_size':[832,464],
         'page_frames':[page_positions(0),page_positions(1)]},'Page layout differs')
    if frozen:need(plan['status']=='FROZEN_REGISTERED' and plan['assets_finalized'] is True,'Not frozen')


def authorize(plan,digest,active=None):
    validate(plan,True)
    active=base.read(ROOT/'research/ACTIVE_RESEARCH.json') if active is None else active
    rows=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==ID]
    need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True
         and len(rows)==1 and rows[0].get('protocol_sha256')==digest
         and rows[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
         and rows[0].get('resources')==RESOURCES and rows[0].get('limits')==LIMITS
         and rows[0].get('max_gpu_cards')==0,'Exact registered CPU scope required')
    return rows[0]


class Ledger:
    def __init__(self,report,persist=lambda:None):
        self.report,self.persist=report,persist
        for side in ('predebited','invoked','completed'): report[side]=dict.fromkeys(LIMITS,0)
    def call(self,key,fn):
        need(key in LIMITS and self.report['predebited'][key]<LIMITS[key], 'Budget exceeded: '+key)
        self.report['predebited'][key]+=1;self.persist()
        self.report['invoked'][key]+=1;self.persist()
        value=fn();self.report['completed'][key]+=1;self.persist();return value
    def finish(self):
        need(all(self.report[side]==LIMITS for side in ('predebited','invoked','completed')),'Actual operation coverage incomplete')


def extract_one(reader,renderer,source,ledger,write_page):
    """Reader yields the actual decoded frames; renderer accepts source-labelled frames."""
    ordinal=source['native_source_ordinal']
    for index in range(81):
        def read_expected():
            success,frame=reader.read()
            need(success and frame is not None,'Short or corrupt video at frame {}'.format(index))
            return frame
        frame=ledger.call('successful_decoded_frames',read_expected)
        renderer.add(frame,ordinal,index)
        if index==41:
            ledger.call('JPEG_pages',lambda:write_page(renderer.finish(0),0,page_positions(0)))
    success,frame=ledger.call('eof_checks',reader.read)
    need(not success and frame is None,'Video has extra decoded frames')
    ledger.call('JPEG_pages',lambda:write_page(renderer.finish(1),1,page_positions(1)))


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='torch' or fullname.startswith('torch.'):
            raise ImportError('Torch forbidden in visual CPU scope')


class Sheets:
    def __init__(self,cv2,np):
        self.cv2,self.np=cv2,np;self.image=None;self.page=-1
    def _new(self,page):
        self.page=page;self.image=self.np.full((7*202,6*320,3),32,dtype=self.np.uint8)
    def add(self,frame,ordinal,index):
        cv2=self.cv2; page=index//42
        if self.page!=page:self._new(page)
        need(frame.shape==(464,832,3),'Decoded frame size differs')
        slot=index%42;y=(slot//6)*202;x=(slot%6)*320
        image=cv2.resize(frame,(320,178),interpolation=cv2.INTER_AREA)
        self.image[y:y+178,x:x+320]=image
        cv2.putText(self.image,'source {}  frame {:02d}  {:.3f}s'.format(ordinal,index,index/16),
                    (x+3,y+195),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    def finish(self,page):
        need(self.page==page,'Wrong image page')
        if page==1:
            for slot in range(39,42):
                y=(slot//6)*202;x=(slot%6)*320
                self.cv2.putText(self.image,'EMPTY',(x+8,y+100),self.cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,self.cv2.LINE_AA)
        result=self.image;self.image=None;self.page=-1;return result


def run(args):
    hf=args.hf_root.resolve();plan=base.read(args.protocol);digest=base.sha(args.protocol);authorize(plan,digest)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))<=2
         and not {'torch','cv2','numpy'} & set(sys.modules),'Fresh hidden-CUDA two-CPU worker required')
    from worker_identity import require_worker
    require_worker()
    out=(hf/OUTPUT).resolve();need(out.is_relative_to(hf) and not out.exists(),'Fresh confined output required');out.mkdir(parents=True)
    started=time.monotonic();report={'id':ID,'protocol_sha256':digest,'status':'STARTED_NEW12_ALL81_VISUAL',
        'automatic_retry':False,'sources':[],'visual_review':'PENDING_SEPARATE_AI_INSPECTION',
        'human_full_playback_review':'NOT_PERFORMED','excess_frames_detected':0}
    def persist():base.write(out/'RUN.json',report)
    ledger=Ledger(report,persist);persist()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Visual CPU wall budget exceeded')))
    signal.alarm(RESOURCES['wall_seconds'])
    resource.setrlimit(resource.RLIMIT_FSIZE,(3<<20,3<<20))
    no_torch=NoTorch();sys.meta_path.insert(0,no_torch)
    try:
        for item in plan['repo_sources']:base.bound(ROOT,item)
        parent=base.read_bound(ROOT,plan['parent_report']);verify_parent(parent,hf)
        import cv2
        import numpy as np
        need('torch' not in sys.modules,'Torch import forbidden')
        cv2.setNumThreads(2)
        report['environment']={'python':sys.executable,'opencv_version':cv2.__version__,
            'numpy_version':np.__version__,'opencv_origin':str(Path(cv2.__file__).resolve()),
            'numpy_origin':str(Path(np.__file__).resolve()),'cuda_visible_devices':'',
            'CPU_affinity':sorted(os.sched_getaffinity(0))}
        report['source_closure_sha256']=base.state_sha(plan['repo_sources']);persist()
        for source in plan['sources']:
            ordinal=source['native_source_ordinal'];entry={**copy.deepcopy(source),'pages':[],
                'decoded_frames':0,'status':'STARTED'};report['sources'].append(entry);persist()
            path=base.bound(hf,source['video'])
            def open_video():
                cap=cv2.VideoCapture(str(path))
                need(cap.isOpened(),'Video open failed')
                return cap
            cap=ledger.call('videos',open_video)
            try:
                geometry={'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                          'fps':cap.get(cv2.CAP_PROP_FPS),'reported_frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
                need(geometry=={'width':832,'height':464,'fps':16.0,'reported_frames':81},'Video header geometry differs')
                entry['decoded_geometry']=geometry;renderer=Sheets(cv2,np)
                class Reader:
                    def read(self):
                        success,frame=cap.read()
                        if success:
                            entry['decoded_frames']+=1
                            if entry['decoded_frames']>81:report['excess_frames_detected']+=1
                        return success,frame
                def write_page(image,page,positions):
                    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,90]);need(ok,'JPEG encoding failed')
                    payload=encoded.tobytes();need(len(payload)<=5<<19,'JPEG per-page byte limit exceeded')
                    current=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
                    need(current+len(payload)+(4<<20)<=RESOURCES['new_HF_bytes'],'Visual output budget exceeded')
                    dest=out/'pages'/('source{}_page{}.jpg'.format(ordinal,page));dest.parent.mkdir(exist_ok=True)
                    with dest.open('xb') as f:f.write(payload)
                    entry['pages'].append({'page':page,'frames':positions,
                        'labels':[None if n is None else {'source_ordinal':ordinal,'frame':n,'timestamp_seconds':n/16} for n in positions],
                        'jpeg':base.binding(dest,hf),'width':1920,'height':1414,'quality':90})
                extract_one(Reader(),renderer,source,ledger,write_page)
                need(entry['decoded_frames']==81 and len(entry['pages'])==2,'Incomplete 81-frame source')
                entry['status']='COMPLETE';persist()
            finally:cap.release()
        ledger.finish()
        need(len(report['sources'])==12 and report['excess_frames_detected']==0
             and 'torch' not in sys.modules,'Source coverage or forbidden import differs')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        need(rss<=RESOURCES['max_RSS_bytes'],'Visual RSS budget exceeded')
        manifest={'schema':'ffnds.new12.dense.all81.contact_sheets.v1','id':ID,'protocol_sha256':digest,
            'parent_report':plan['parent_report'],'metric_parts':plan['metric_parts'],
            'sources':report['sources'],'interpretation':plan['interpretation']}
        base.write(out/'MANIFEST.json',manifest)
        report.update(status=STATUS,manifest=base.binding(out/'MANIFEST.json',hf),max_RSS_bytes=rss)
    except BaseException as e:
        report.update(status='FAILED_NEW12_VISUAL_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-started;persist()
        if no_torch in sys.meta_path:sys.meta_path.remove(no_torch)


def verify_closed(args):
    hf=args.hf_root.resolve();plan=base.read(args.protocol);validate(plan,True);digest=base.sha(args.protocol)
    out=(hf/OUTPUT).resolve()
    for item in plan['repo_sources']:base.bound(ROOT,item)
    verify_parent(base.read_bound(ROOT,plan['parent_report']),hf)
    need(args.run.resolve()==out/'RUN.json' and args.exit.resolve()==out.parent/'EXIT_RECEIPT.json','Wrong actual run/exit path')
    run_binding=base.binding(args.run,hf);report=base.read(args.run)
    base.exited(base.read(args.exit),run_binding,digest,ID)
    need(report.get('id')==ID and report.get('protocol_sha256')==digest and report.get('status')==STATUS
         and all(report.get(k)==LIMITS for k in ('predebited','invoked','completed'))
         and report.get('excess_frames_detected')==0 and 0<=report.get('wall_seconds',-1)<=RESOURCES['wall_seconds']
         and 0<=report.get('max_RSS_bytes',-1)<=RESOURCES['max_RSS_bytes'],'Closed visual run differs')
    manifest=base.read_bound(hf,report['manifest'])
    need(manifest.get('schema')=='ffnds.new12.dense.all81.contact_sheets.v1' and manifest.get('id')==ID
         and manifest.get('interpretation')==plan['interpretation']
         and manifest.get('protocol_sha256')==digest and manifest.get('sources')==report['sources']
         and manifest.get('parent_report')==plan['parent_report'] and manifest.get('metric_parts')==plan['metric_parts']
         and len(manifest['sources'])==12,'Manifest coverage differs')
    page_paths=[]
    for actual,expected in zip(manifest['sources'],plan['sources']):
        need(all(actual.get(k)==v for k,v in expected.items()) and actual['decoded_frames']==81
             and actual['decoded_geometry']=={'width':832,'height':464,'fps':16.0,'reported_frames':81}
             and actual['status']=='COMPLETE' and len(actual['pages'])==2,'Visual source identity differs')
        for index,page in enumerate(actual['pages']):
            positions=page_positions(index)
            labels=[None if n is None else {'source_ordinal':actual['native_source_ordinal'],'frame':n,'timestamp_seconds':n/16} for n in positions]
            need(page['page']==index and page['frames']==positions and page['labels']==labels
                 and page['width']==1920 and page['height']==1414 and page['quality']==90,'Page label/geometry mismatch')
            path=base.bound(hf,page['jpeg'])
            need(path==(out/'pages'/('source{}_page{}.jpg'.format(actual['native_source_ordinal'],index))).resolve(), 'Unexpected page locator')
            page_paths.append(path)
    need(len(page_paths)==len(set(page_paths))==len({(p.stat().st_dev,p.stat().st_ino) for p in page_paths})==24, 'Page aliases detected')
    return {'id':ID,'status':'PASS_CLOSED_ALL81_VISUAL_CPU','RUN':run_binding,
        'exit':base.binding(args.exit,hf),'manifest':report['manifest'],'model_calls':0,'GPU_calls':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('draft','inspect','run','verify-closed'))
    p.add_argument('--protocol',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'))
    p.add_argument('--run',type=Path);p.add_argument('--exit',type=Path);a=p.parse_args()
    if a.action=='draft':
        need(a.out is not None and not a.out.exists(),'Fresh draft required');a.out.parent.mkdir(parents=True,exist_ok=True);base.write(a.out,draft(a.hf_root))
    elif a.action=='inspect':validate(base.read(a.protocol));print('STATIC_ONLY_NOT_MODEL')
    elif a.action=='run':run(a)
    else:print(json.dumps(verify_closed(a),indent=2))

if __name__=='__main__':main()
