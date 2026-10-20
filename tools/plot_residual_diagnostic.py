"""Render paired local risks; this figure does not report video quality or speed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--centered',type=Path,required=True)
    ap.add_argument('--residual',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    centered=list(csv.DictReader(args.centered.open()))
    residual=list(csv.DictReader(args.residual.open()))
    residual.sort(key=lambda r:(int(r['task']),int(r['seed']),int(r['step'])))
    labels=[];values=[[],[],[]]
    for row in residual:
        scope=row['scope']
        parent={r['mode']:float(r['native_risk']) for r in centered if r['scope']==scope and r['policy']=='C'}
        if not np.isclose(parent['centered_gain_bias'],float(row['base_native_SSE']),rtol=1e-7):
            raise RuntimeError('The residual and centered-base measurements differ')
        values[0].append(parent['uncentered_gain'])
        values[1].append(parent['centered_gain_bias'])
        values[2].append(float(row['native_SSE']))
        labels.append(f"task{row['task']} / seed{row['seed'][-2:]} / step{row['step']}")
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    y=np.arange(len(labels));width=.24
    for i,(name,color) in enumerate(zip(['C static + uncentered gain','C static + centered gain/bias','C centered + RR1 rank256'],['#8b9aa7','#3682b6','#14937a'])):
        ax.barh(y+(i-1)*width,values[i],width,label=name,color=color)
    ax.set_yticks(y,labels);ax.invert_yaxis()
    ax.axhline(3.5,color='#cccccc',linewidth=1)
    ax.set_xlabel('Local normalized SSE against captured D-Q writeback (lower is better)')
    ax.set_title('Calibration-only parameters on two independent research sources\nBlock20, steps0/2; native CPU BF16; no video-quality or speed acceptance')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower right',frameon=False)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'),dpi=180)
    fig.savefig(args.output.with_suffix('.pdf'))
    identity={'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.centered,args.residual]},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'matplotlib':matplotlib.__version__,'scope_count':len(labels),'quality_or_speed_acceptance':False}
    args.output.with_suffix('.json').write_text(json.dumps(identity,indent=2)+'\n')


if __name__=='__main__':
    main()
