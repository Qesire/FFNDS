"""Export fixed-source formal video frames after input/cache/geometry checks."""
import argparse
import json
from pathlib import Path

from decord import VideoReader,cpu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ar2_checkpoint import atomic_json,digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',type=Path,nargs='+',required=True)
    ap.add_argument('--labels',nargs='+',required=True)
    ap.add_argument('--frames',type=int,nargs='+',default=[0,20,40,60,80])
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if len(args.cases)!=len(args.labels) or len(args.cases)<2:ap.error('Need matching labels for at least two cases')
    if any(i not in range(81) for i in args.frames):ap.error('Frames must be0..80')
    cases=[json.loads(p.read_text()) for p in args.cases]
    first=cases[0]
    for case in cases:
        if (case['status']!='PASS_FORMAL_INPUT_GENERATION_ONLY' or case['input']!=first['input']
                or case['model_assets']!=first['model_assets'] or case['video']['frames']!=81
                or case['video']['fps']!=16 or case['video']['size']!=first['video']['size']
                or digest(case['video']['path'])!=case['video']['sha256']):
            raise RuntimeError('Formal source/cache/video identity or geometry differs')
    fig,axes=plt.subplots(len(cases),len(args.frames),figsize=(15,2.35*len(cases)),squeeze=False,layout='constrained')
    for i,(case,label) in enumerate(zip(cases,args.labels)):
        reader=VideoReader(case['video']['path'],ctx=cpu(0),num_threads=2)
        if len(reader)!=81:raise RuntimeError('Decoded frame count differs')
        for j,index in enumerate(args.frames):
            ax=axes[i,j];ax.imshow(reader[index].asnumpy());ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(f'Frame {index} / {index/16:.2f}s')
            if j==0:ax.set_ylabel(label,fontsize=12)
    fig.suptitle(f'Fixed formal input{first["index"]:02d}, seed42: D-Q and sparse controls\nVisual diagnostic; each official quality metric is evaluated separately',fontsize=13)
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=160)
    atomic_json(args.output.with_suffix('.json'),{'input':first['input'],'frames':args.frames,
        'cases':[{'report':str(p.resolve()),'sha256':digest(p),'video':c['video'],'label':label}
            for p,c,label in zip(args.cases,cases,args.labels)],
        'output_sha256':digest(args.output),'source_sha256':digest(Path(__file__)),
        'quality_or_speed_acceptance':False})


if __name__=='__main__':main()
