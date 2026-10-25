"""Export the shared-column/decision-time mechanism comparison."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--closure',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    r=json.loads(a.input.read_text());c=json.loads(a.closure.read_text())
    if a.out.with_suffix('.png').exists():raise FileExistsError('Preserve first figure')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    blocks=[0,10,20,30,39];x=np.arange(5)
    fig,axes=plt.subplots(2,2,figsize=(13.5,8.4),layout='constrained')
    for key,label,color in [('pointwise_posthoc_diagonal','Pointwise posthoc','#999999'),
                            ('posthoc_shared_diagonal','Shared posthoc','#4477aa'),
                            ('preW1_micro16_gelu_value','Pre-W1 micro16 GELU/value','#228833'),
                            ('preW1_mean_preact','Pre-W1 mean preactivation','#cc6677'),
                            ('source_fixed_LOSO','Other-source fixed','#aa3377')]:
        y=[100*r['by_block'][str(b)]['results']['0.25'][key]['relative_innovation_SSE'] for b in blocks]
        axes[0,0].plot(x,y,'o-',label=label,color=color)
    axes[0,0].set_title('25% retained columns; original M128 token tiles')
    axes[0,0].set_ylabel('Error / signed innovation energy (%)');axes[0,0].legend(fontsize=8)
    y=[100*r['by_block'][str(b)]['innovation_token_mean_energy_fraction'] for b in blocks]
    bars=axes[0,1].bar(x,y,color='#4477aa');axes[0,1].bar_label(bars,fmt='%.1f',padding=3)
    axes[0,1].set_ylim(0,100);axes[0,1].set_title('Common token component depends strongly on block')
    axes[0,1].set_ylabel('Tile-mean / full signed innovation energy (%)')
    names=['posthoc_shared_diagonal','preW1_micro16_gelu_value','preW1_mean_preact','group64_closed_shared_diagonal']
    labels=['Shared\nposthoc','Pre-W1\nmicro16','Pre-W1\npreactivation','Group64\nclosed']
    for j,g in enumerate([4,8,64]):
        y=[100*statistics.mean(v for d in c['cases'] for v in d['results']['0.25'][name]['group_closure'][str(g)]) for name in names]
        axes[1,0].bar(np.arange(4)+(j-1)*.24,y,.23,label=f'Original group{g}')
    axes[1,0].set_xticks(np.arange(4),labels,fontsize=8);axes[1,0].set_ylim(0,108)
    axes[1,0].set_title('25% columns touch many original quantizer groups')
    axes[1,0].set_ylabel('Expanded operand fraction (%)');axes[1,0].legend(fontsize=8)
    for j,key in enumerate(['posthoc_shared_diagonal','preW1_micro16_gelu_value','group64_closed_shared_diagonal']):
        y=[100*statistics.mean(d['results']['0.25'][key]['relative_innovation_SSE'] for d in c['cases'] if d['block']==b) for b in blocks]
        axes[1,1].plot(x,y,'o-',label={'posthoc_shared_diagonal':'Arbitrary shared posthoc','preW1_micro16_gelu_value':'Pre-W1 micro16 GELU/value','group64_closed_shared_diagonal':'Original64-group closed posthoc'}[key])
    axes[1,1].set_title('Group closure has a functional price at equal channels')
    axes[1,1].set_ylabel('Innovation relative SSE (%)');axes[1,1].legend(fontsize=8)
    for ax in [axes[0,0],axes[0,1],axes[1,1]]:ax.set_xticks(x,blocks);ax.set_xlabel('Block')
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Wan2.2 four-step FFN · 2 exposed sources, seed42 · saved diagnostics, no sparse speed/quality claim',fontsize=12)
    fig.savefig(a.out.with_suffix('.png'),dpi=170);fig.savefig(a.out.with_suffix('.pdf'))


if __name__=='__main__':main()
