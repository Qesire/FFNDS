"""Decode every paired video frame and prepare full-duration review artifacts.

CPU/media only. This checks completeness and produces reviewable media; it
never automatically calls the candidate visually safe or marks smoke PASS.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from rollout_contract import ROOT,atomic,read,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    from worker_identity import require_worker
    require_worker()
    import imageio_ffmpeg
    from PIL import Image,ImageDraw
    import numpy as np
    root=Path(os.environ['HF_HOME'])/'derived/ffnds/R15V0_paired_rollout'
    report=read(root/'RUN.json')
    if report['status']!='COMPLETE_PAIRED_FULL_VIDEO_AWAITING_DECODE_AND_REVIEW':
        raise RuntimeError('Both complete free-running videos required before review')
    out=root/'review'
    out.mkdir(exist_ok=False)
    readers=[];metadata=[]
    for request in report['requests']:
        path=Path(os.environ['HF_HOME'])/request['video_HF_relative']
        if sha(path)!=request['video_sha256']:raise RuntimeError('Generated video changed')
        reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
        readers.append(reader);metadata.append(next(reader))
    if metadata[0]['size']!=metadata[1]['size'] or metadata[0]['fps']!=metadata[1]['fps']:
        raise RuntimeError('Paired dimensions/FPS mismatch')
    width,height=metadata[0]['size'];fps=metadata[0]['fps']
    frame=0;pages=[];metrics=[];canvas=None
    for pair in __import__('itertools').zip_longest(*readers):
        if any(value is None for value in pair):raise RuntimeError('Paired frame counts differ')
        images=[];luma=[]
        for value in pair:
            array=np.frombuffer(value,dtype=np.uint8).reshape(height,width,3)
            images.append(Image.fromarray(array))
            luma.append(float((array[...,0]*.299+array[...,1]*.587+array[...,2]*.114).mean()))
        if frame%12==0:
            if canvas is not None:
                page=out/f'all_frames_{len(pages):02d}.jpg';canvas.save(page,quality=92)
                pages.append({'HF_relative':str(page.relative_to(Path(os.environ['HF_HOME']))),'sha256':sha(page)})
            thumb_w=416;thumb_h=round(height*thumb_w/width)
            canvas=Image.new('RGB',(2*thumb_w,12*(thumb_h+24)),(255,255,255))
        y=(frame%12)*(thumb_h+24);draw=ImageDraw.Draw(canvas)
        draw.text((6,y+3),f'frame {frame:02d}  t={frame/fps:.3f}s | left dense D-Q | right candidate',fill=(0,0,0))
        for col,img in enumerate(images):canvas.paste(img.resize((thumb_w,thumb_h)),(col*thumb_w,y+24))
        metrics.append({'frame':frame,'time_seconds':frame/fps,'mean_luma_dense':luma[0],'mean_luma_candidate':luma[1]})
        frame+=1
    if canvas is not None:
        page=out/f'all_frames_{len(pages):02d}.jpg';canvas.save(page,quality=92)
        pages.append({'HF_relative':str(page.relative_to(Path(os.environ['HF_HOME']))),'sha256':sha(page)})
    if frame!=81:raise RuntimeError('Original native81-frame output required')
    first,second=[Path(os.environ['HF_HOME'])/r['video_HF_relative'] for r in report['requests']]
    side=out/'paired_left_dense_right_candidate.mp4'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-i',str(first),'-i',str(second),
             '-filter_complex','hstack=inputs=2','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(side)]
    subprocess.run(command,check=True)
    result={'id':'R15V0_COMPLETE_VIDEO_DECODE_AND_REVIEW_MEDIA','status':'BOTH_COMPLETE_AWAITING_FULL_VISUAL_REVIEW',
            'raw_run_sha256':sha(root/'RUN.json'),'paired_frames':frame,'fps':fps,'size':list(metadata[0]['size']),
            'duration_seconds':frame/fps,'decoded_every_frame':True,'all_frame_contact_pages':pages,
            'paired_video':{'HF_relative':str(side.relative_to(Path(os.environ['HF_HOME']))),'sha256':sha(side)},
            'luma_time_series':metrics,'automatic_visual_PASS':False,'new_model_generations':0,
            'limits':'Full-frame decode and global mean luminance are completeness/descriptive checks. They do not certify stable spatial lighting, motion, small objects or video quality. Inspect the entire paired video.'}
    if sum(f.stat().st_size for f in root.rglob('*') if f.is_file())>1<<28:
        raise RuntimeError('Combined generation/review output cap exceeded')
    atomic(root/'VIDEO_INSPECTION.json',result)
    print(json.dumps({'status':result['status'],'frames':frame,'contact_pages':len(pages),'duration_seconds':frame/fps}),flush=True)


if __name__=='__main__':main()
