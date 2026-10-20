"""Export all three formal20 relative drops with the exact2% acceptance limit."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparison',type=Path,required=True)
    ap.add_argument('--candidate-label',default='RR1 calibration12')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();data=json.loads(args.comparison.read_text())
    if data['status']!='COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON' or data['unique_source_count']!=20:
        raise RuntimeError('Complete paired formal20 comparison required')
    names=['aesthetic_quality','imaging_quality','motion_smoothness']
    values=[100*data['metrics'][k]['relative_drop'] for k in names]
    colors=['#218a68' if data['metrics'][k]['pass'] else '#be4b48' for k in names]
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    ax.barh(range(3),values,color=colors,height=.55)
    ax.set_yticks(range(3),['Aesthetic quality','Imaging quality','Motion smoothness']);ax.invert_yaxis()
    ax.axvline(2,color='#4b667a',linestyle='--',label='Maximum allowed drop:2%')
    ax.set_xlim(min(0,min(values)*1.2),max(3,max(values)*1.2))
    for i,(key,value) in enumerate(zip(names,values)):
        row=data['metrics'][key]
        ax.text(max(value,0)+.12,i,f'{value:.3f}%   ({row["D_Q"]:.5f} → {row["candidate"]:.5f})',va='center',fontsize=10)
    ax.set_xlabel('Relative score drop from frozen D-Q baseline (%)')
    ax.set_title(f'{args.candidate_label}: all three formal20 metrics\n20 independent sources, seed42; each metric must pass separately')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.legend(loc='lower right',frameon=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'),dpi=180);fig.savefig(args.output.with_suffix('.pdf'))
    args.output.with_suffix('.json').write_text(json.dumps({'comparison_sha256':sha(args.comparison),
        'source_sha256':sha(Path(__file__)),'candidate_policy_sha256':data['candidate_policy_sha256'],
        'formal20_quality_pass':data['formal20_quality_pass'],'matplotlib':matplotlib.__version__},indent=2)+'\n')


if __name__=='__main__':main()
