"""Scoped three-arm contact sheets and synchronized video; no model evaluation."""
import argparse, hashlib, json, os, shutil, signal, time
import filecmp
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='UB1_DISJOINT20_VISUAL_MEDIA_20261027'
ARMS=('dense','neutral','candidate')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4<<20),b''):h.update(x)
 return h.hexdigest()

def validate_source(source):
 if type(source['index']) is not int or not 0<=source['index']<20 or source['seed']!=42 or not source['source_id'] or not source['prompt_sha256']:raise ValueError('Source identity mismatch')
 if set(source['arms'])!=set(ARMS):raise ValueError('Three arms required')
 for arm in ARMS:
  item=source['arms'][arm]
  if item['source_id']!=source['source_id'] or item['seed']!=source['seed'] or item['prompt_sha256']!=source['prompt_sha256']:raise ValueError('Arm/source mismatch')
  if item['frames']!=81 or item['fps']!=16 or item['size']!=[832,464]:raise ValueError('Original video geometry mismatch')
 if source['arms']['neutral']['initial_noise_status']!='NOT_SAVED_HISTORICALLY_NO_BYTE_PAIRING_CLAIM':raise ValueError('Historical noise provenance mismatch')
 return True

def validate_generation(source, arm, receipt, hf):
 item=source['arms'][arm]
 if (receipt.get('status')!='PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'
     or receipt.get('source_index')!=source['index'] or receipt.get('source_id')!=source['source_id']
     or receipt.get('image_sha256')!=source['source_id'] or receipt.get('prompt_sha256')!=source['prompt_sha256']
     or receipt.get('seed')!=source['seed'] or receipt.get('mode')!=arm):raise ValueError('Generation/source identity differs')
 if receipt.get('initial_noise')!=item['initial_noise'] or receipt['video']['sha256']!=item['sha256'] or Path(receipt['video']['path']).resolve()!=(hf/item['path']).resolve():raise ValueError('Generation video/noise identity differs')
 timings=receipt.get('FFN_boundary_timings',[])
 if len(timings)!=160 or {(x['step'],x['block']) for x in timings}!={(s,b) for s in range(4) for b in range(40)} or receipt.get('native_counter_delta',{}).get('linear_forward_calls')!=1600 or receipt['native_counter_delta'].get('activation_qdq_calls')!=1600:raise ValueError('Generation full160 evidence differs')
 return True

def compose(frames,width,cv2,np):
 if len(frames)!=3 or any(x is None or x.shape!=(464,832,3) for x in frames):raise ValueError('Frame shape mismatch')
 height=round(464*width/832);panels=[]
 for label,frame in zip(('DENSE','HISTORICAL UB1','CONSUMER'),frames):
  x=cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA);cv2.putText(x,label,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA);panels.append(x)
 return np.concatenate(panels,axis=1)

def run(args):
 import cv2,numpy as np,imageio,imageio_ffmpeg
 cv2.setNumThreads(2)
 plan=json.loads(args.protocol.read_text());digest=sha(args.protocol);hf=args.hf_root.resolve()
 if plan['id']!=ID or plan['assets_finalized'] is not True or sha(__file__)!=plan['tool_sha256'] or sha(ROOT/plan['test'])!=plan['test_sha256']:raise ValueError('Frozen tool/assets required')
 active=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text());scopes=active['experiment_authority']['current_user_authorized_scopes']
 if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in scopes):raise ValueError('Exact authorization required')
 if plan['limits']!={'cpu_threads':2,'memory_bytes':2147483648,'wall_seconds':1200,'outer_wall_seconds':1500,'output_bytes':536870912,'attempts':1,'gpu':0,'model_calls':0}:raise ValueError('Budget mismatch')
 codec=plan['encoder']; ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
 if imageio.__version__!=codec['imageio_version'] or imageio_ffmpeg.__version__!=codec['imageio_ffmpeg_version'] or ffmpeg!=(ROOT/codec['ffmpeg_repo_relative']).resolve() or sha(ffmpeg)!=codec['ffmpeg_sha256']:raise ValueError('Frozen ffmpeg identity differs')
 if len(plan['sources'])!=20 or sorted(x['index'] for x in plan['sources'])!=list(range(20)):raise ValueError('Twenty fixed sources required')
 for x in plan['sources']:validate_source(x)
 view=(hf/plan['historical_view']['path']).resolve()
 if not view.is_relative_to(hf) or sha(view)!=plan['historical_view']['sha256']:raise ValueError('Historical view identity mismatch')
 old=json.loads(view.read_text())
 if old['initial_noise']!='NOT_SAVED_HISTORICALLY_NO_BYTE_PAIRING_CLAIM':raise ValueError('Historical noise status differs')
 historical={v['index']:v for v in old['arms']['neutral']['sources']}
 for source in plan['sources']:
  prior=historical[source['index']];neutral=source['arms']['neutral']
  if any(prior[k]!=source[v] for k,v in (('image_sha256','source_id'),('source_group','source_group'),('seed','seed'),('prompt_sha256','prompt_sha256'))) or prior['video']['sha256']!=neutral['sha256'] or prior['video']['path']!=neutral['path']:raise ValueError('Historical neutral identity differs')
  pair=[]
  for arm in ('dense','candidate'):
   receipt_binding=source['arms'][arm]['generation']; receipt_path=(hf/receipt_binding['path']).resolve()
   if not receipt_path.is_relative_to(hf) or not receipt_path.is_file() or receipt_path.stat().st_size!=receipt_binding['bytes'] or sha(receipt_path)!=receipt_binding['sha256']:raise ValueError('Generation receipt binding differs')
   validate_generation(source,arm,json.loads(receipt_path.read_text()),hf)
   noise=source['arms'][arm]['initial_noise']; path=(hf/noise['path']).resolve()
   if not path.is_relative_to(hf) or not path.is_file() or sha(path)!=noise['sha256']:raise ValueError('Fresh noise binding differs')
   pair.append(path)
  if pair[0].samefile(pair[1]) or not filecmp.cmp(*pair,shallow=False):raise ValueError('Fresh paired noise bytes differ or alias')
 for item in plan['bindings']:
  path=(hf/item['path']).resolve()
  if not path.is_relative_to(hf) or not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Input binding mismatch')
 out=(hf/plan['output_HF_relative']).resolve()
 if not out.is_relative_to(hf) or out==hf:raise ValueError('Output outside scope')
 out.mkdir(parents=True,exist_ok=False)
 with (out/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest},f)
 started=time.monotonic();report={'id':ID,'status':'STARTED','protocol_sha256':digest,'sources':[],'scope':'Existing clips; visual comparisons only, no human acceptance; historic neutral initial noise not saved.'}
 def budget():
  if time.monotonic()-started>1200 or sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())>plan['limits']['output_bytes']:raise RuntimeError('Resource budget exceeded')
 def alarm(*args):raise TimeoutError('CPU media wall limit')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(1200)
 try:
  for source in sorted(plan['sources'],key=lambda x:x['index']):
   budget();paths=[(hf/source['arms'][arm]['path']).resolve() for arm in ARMS]
   for arm,path in zip(ARMS,paths):
    item=source['arms'][arm]
    if not path.is_relative_to(hf) or not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Video identity mismatch')
   if len({(p.stat().st_dev,p.stat().st_ino) for p in paths})!=3:raise ValueError('Distinct arm video files required')
   # Equal arm bytes can be scientifically meaningful; record rather than claim independent samples.
   equal_bytes=[(ARMS[i],ARMS[j]) for i in range(3) for j in range(i) if sha(paths[i])==sha(paths[j])]
   captures=[cv2.VideoCapture(str(p)) for p in paths]
   if any(not c.isOpened() for c in captures):raise RuntimeError('Video open failed')
   if any(abs(c.get(cv2.CAP_PROP_FPS)-16)>1e-3 or round(c.get(cv2.CAP_PROP_FRAME_COUNT))!=81 or round(c.get(cv2.CAP_PROP_FRAME_WIDTH))!=832 or round(c.get(cv2.CAP_PROP_FRAME_HEIGHT))!=464 for c in captures):raise ValueError('Encoded video metadata mismatch')
   directory=out/f'source{source["index"]:02d}';directory.mkdir();movie=directory/'comparison.mp4';height=round(464*512/832)
   writer=imageio.v2.get_writer(str(movie),format='FFMPEG',mode='I',fps=16,codec='libx264',pixelformat='yuv420p',macro_block_size=1,output_params=['-threads','2'])
   tiles=[];pages=[]
   try:
    for i in range(81):
     decoded=[c.read() for c in captures]
     if not all(ok for ok,_ in decoded):raise ValueError('Incomplete source decode')
     frames=[v for _,v in decoded];tile=compose(frames,256,cv2,np);cv2.putText(tile,f'frame {i:02d}',(4,tile.shape[0]-5),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA);tiles.append(tile);writer.append_data(cv2.cvtColor(compose(frames,512,cv2,np),cv2.COLOR_BGR2RGB))
     if len(tiles)==9:
      page=directory/f'frames_{i-8:02d}_{i:02d}.jpg';sheet=np.concatenate(tiles,axis=0)
      if not cv2.imwrite(str(page),sheet,[cv2.IMWRITE_JPEG_QUALITY,88]):raise RuntimeError('Contact sheet write failed')
      pages.append({'path':str(page.relative_to(out)),'sha256':sha(page),'frames':list(range(i-8,i+1))});tiles=[];budget()
    if any(c.read()[0] for c in captures):raise ValueError('Extra decoded source frame')
   finally:
    writer.close()
    for c in captures:c.release()
   check=cv2.VideoCapture(str(movie));decoded=0
   if not check.isOpened():raise RuntimeError('Comparison video reopen failed')
   while check.read()[0]:decoded+=1
   movie_fps=check.get(cv2.CAP_PROP_FPS);check.release()
   if decoded!=81 or abs(movie_fps-16)>1e-3 or len(pages)!=9:raise ValueError('Output coverage mismatch')
   budget();report['sources'].append({'index':source['index'],'source_id':source['source_id'],'source_group':source['source_group'],'seed':42,'input_sha256':{a:source['arms'][a]['sha256'] for a in ARMS},'equal_arm_bytes':equal_bytes,'pages':pages,'comparison_video':{'path':str(movie.relative_to(out)),'sha256':sha(movie),'bytes':movie.stat().st_size,'frames':decoded,'fps':movie_fps},'human_fullplay':'NOT_RUN'})
   (out/'RUN.json').write_text(json.dumps(report,indent=2)+'\n')
  report.update(status='COMPLETE_MEDIA_NOT_HUMAN_ACCEPTANCE',elapsed_seconds=time.monotonic()-started)
 except BaseException as e:report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
 finally:
  signal.alarm(0);(out/'RUN.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--hf-root',type=Path,required=True);run(p.parse_args())
