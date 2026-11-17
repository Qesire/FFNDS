"""Create fixed-frame DEV3 contact sheets from verified actual videos.

Pure visual summaries; missing arms stay blank and no quality inference is made.
The separately reviewed guard supplies generation proof and decode validation.
"""
import hashlib,json,os,subprocess
from pathlib import Path
from quality_pair_report import ARMS

DEV3=('vbi_1013','vbi_0886','vbi_0800')
FRAMES=(0,20,40,60,80)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def create(scope,*,authorize,verify_generation,validate_video):
    authorize(scope)
    if tuple(scope['sources'])!=DEV3 or tuple(scope['arms'])!=ARMS or tuple(scope['frame_indices'])!=FRAMES:
        raise ValueError('DEV3 / method / frame order differs from frozen design')
    exe=Path(scope['ffmpeg_path']).resolve()
    if sha(exe)!=scope['ffmpeg_sha256']:raise ValueError('ffmpeg identity mismatch')
    records=scope['records'];index={}
    for r in records:
        key=(r['source'],r['arm'])
        if key in index or key[0] not in DEV3 or key[1] not in ARMS:raise ValueError('duplicate or unknown contact-sheet video')
        if sha(r['video_path'])!=r['video_sha256']:raise ValueError('video identity mismatch')
        proof=verify_generation(r,scope)
        if not isinstance(proof,dict) or proof.get('status')!='PASS' or proof.get('proof_sha256')!=r.get('generation_proof_sha256') or not proof.get('proof_sha256'):
            raise ValueError('unverified generation provenance')
        validate_video(r['video_path'],scope)
        index[key]=(r,proof['proof_sha256'])
    from PIL import Image,ImageDraw
    out=Path(scope['output']);out.mkdir(parents=True,exist_ok=False)
    results=[];w,h=208,116;left,top=125,35
    for source in DEV3:
        canvas=Image.new('RGB',(left+5*w,top+5*(h+24)),(248,248,248));draw=ImageDraw.Draw(canvas)
        for j,frame in enumerate(FRAMES):draw.text((left+j*w+4,8),f'frame {frame}  {frame/16:.2f}s',fill=(10,10,10))
        inputs=[]
        for i,arm in enumerate(ARMS):
            y=top+i*(h+24);draw.text((5,y+5),arm,fill=(10,10,10))
            item=index.get((source,arm))
            if item is None:
                draw.text((left+4,y+5),'MISSING',fill=(130,30,30));continue
            r,proof_sha=item
            expression='+'.join(f'eq(n\\,{n})' for n in FRAMES)
            cmd=[str(exe),'-hide_banner','-loglevel','error','-i',str(r['video_path']),'-vf','select='+expression,
                 '-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
            proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=180)
            frame_bytes=832*464*3
            if len(proc.stdout)!=len(FRAMES)*frame_bytes:raise RuntimeError('contact frame count or geometry mismatch')
            for j in range(len(FRAMES)):
                frame=Image.frombytes('RGB',(832,464),proc.stdout[j*frame_bytes:(j+1)*frame_bytes])
                canvas.paste(frame.resize((w,h),Image.Resampling.LANCZOS),(left+j*w,y))
            inputs.append({'arm':arm,'video_sha256':r['video_sha256'],'generation_proof_sha256':proof_sha})
        path=out/f'{source}__contact.png'
        with path.open('xb') as f:canvas.save(f,format='PNG');f.flush();os.fsync(f.fileno())
        results.append({'source':source,'file':path.name,'sha256':sha(path),'inputs':inputs,
                        'missing_arms':[a for a in ARMS if (source,a) not in index]})
    metadata={'sources':DEV3,'arms':ARMS,'frame_indices':FRAMES,'fps':16,'frame_times_seconds':[n/16 for n in FRAMES],
              'meaning':'fixed-frame visual context; no qualitative ratings inferred','sheets':results}
    with (out/'CONTACT_MANIFEST.json').open('x',encoding='utf8') as f:
        json.dump(metadata,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return metadata

