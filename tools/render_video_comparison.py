"""Export a scientific contact sheet from identity-matched generated videos."""
import argparse
import hashlib
import json
from pathlib import Path

from decord import VideoReader, cpu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metas',type=Path,nargs='+',required=True)
    ap.add_argument('--labels',nargs='+',required=True)
    ap.add_argument('--frames',type=int,nargs='+',default=[0,20,40,60,80])
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if len(args.metas)!=len(args.labels):
        ap.error('A label is required for each video')
    reports=[json.loads(p.read_text()) for p in args.metas]
    identity_keys=['seed','source_id','prompt_sha256','image_sha256','checkpoint_sha256','resolved_contract']
    identity={k:reports[0][k] for k in identity_keys}
    for report in reports:
        if (report['status']!='PASS' or {k:report[k] for k in identity_keys}!=identity
                or report['video']['frames']!=81 or report['video']['fps']!=16
                or sha(report['video']['path'])!=report['video']['sha256']):
            raise RuntimeError('Video/input/model contract mismatch')
    fig,axes=plt.subplots(len(reports),len(args.frames),figsize=(15,3.2*len(reports)),squeeze=False,layout='constrained')
    for i,(report,label) in enumerate(zip(reports,args.labels)):
        reader=VideoReader(report['video']['path'],ctx=cpu(0),num_threads=2)
        for j,index in enumerate(args.frames):
            axes[i,j].imshow(reader[index].asnumpy())
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(f'Frame {index} ({index/16:.2f}s)')
            if j==0:axes[i,j].set_ylabel(label,fontsize=12)
    fig.suptitle('Same source, seed and D-Q cache: full closed-loop video comparison\nVisual diagnostic; no formal20 quality or FFN speed acceptance',fontsize=15)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=150)
    manifest={'identity':identity,'frames':args.frames,'source_sha256':sha(Path(__file__)),
        'inputs':[{'meta':str(p.resolve()),'sha256':sha(p),'video':r['video'],'label':label}
                  for p,r,label in zip(args.metas,reports,args.labels)],
        'output_sha256':sha(args.output),'quality_or_speed_acceptance':False}
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
