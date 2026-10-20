"""Plot paired formal trajectory differences without treating them as quality scores."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ar2_checkpoint import atomic_json,digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--csv',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();report=json.loads(args.report.read_text())
    if (report['status']!='COMPLETED_PAIRED_FORMAL20_LATENT_DRIFT_DESCRIPTION'
            or digest(args.csv)!=report['per_source_csv_sha256']):raise RuntimeError('Latent audit identity differs')
    rows=list(csv.DictReader(args.csv.open()));fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    steps=np.arange(4);summary=report['summary']
    for index in range(20):
        case=sorted([r for r in rows if int(r['index'])==index],key=lambda r:int(r['step']))
        if len(case)!=4:raise RuntimeError('Incomplete source trajectory')
        axes[0].plot(steps,[100*float(r['relative_RMSE']) for r in case],color='#bac4d2',alpha=.5,lw=.7)
    axes[0].plot(steps,[100*r['pooled_relative_RMSE'] for r in summary],'-o',color='#203e65',lw=2,label='Pooled relative RMSE')
    axes[0].set_ylabel('Relative latent RMSE (%)');axes[0].legend(frameon=False)
    axes[1].plot(steps,[r['centered_variance_ratio_mean'] for r in summary],'-o',color='#a64b2a',label='Centered energy ratio')
    axes[1].plot(steps,[r['norm_ratio_mean'] for r in summary],'-s',color='#287969',label='Norm ratio')
    axes[1].axhline(1,color='#555555',lw=.8,ls='--');axes[1].set_ylabel('Candidate / dense (mean over20 sources)')
    axes[1].legend(frameon=False);axes[1].set_ylim(.75,1.05)
    for ax in axes:
        ax.set_xticks(steps,labels=['0 / 1000','1 / 750','2 / 500','3 / 250'])
        ax.set_xlabel('Native step / denoising timestep');ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('RR1 vs D-Q: paired formal20 latent trajectories',fontsize=12)
    fig.text(.5,-.03,'Descriptive trajectory differences; no video-quality or causal-mechanism acceptance.',ha='center',fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    paths=[]
    for suffix in ['.png','.pdf']:
        path=args.output.with_suffix(suffix);fig.savefig(path,dpi=180,bbox_inches='tight');paths.append(path)
    atomic_json(args.output.with_suffix('.json'),{'report_sha256':digest(args.report),'csv_sha256':digest(args.csv),
        'source_sha256':digest(Path(__file__)),'files':[{'path':str(p.resolve()),'sha256':digest(p)} for p in paths],
        'quality_or_speed_acceptance':False})


if __name__=='__main__':main()
