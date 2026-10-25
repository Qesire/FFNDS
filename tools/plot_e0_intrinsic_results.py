"""Scientific figures for full-hidden concentration and native full-M support."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True);args=p.parse_args()
    r=json.loads(args.input.read_text());args.outdir.mkdir(parents=True,exist_ok=True)
    output=args.outdir/'intrinsic_full_hidden'
    if output.with_suffix('.png').exists() or output.with_suffix('.pdf').exists():
        raise FileExistsError('Preserve first figure')
    fig,axes=plt.subplots(2,2,figsize=(13.5,8.4),layout='constrained');blocks=[0,10,20,30,39]
    positions=np.arange(5)
    for mode,label,color in [('absolute','Absolute coefficients','#4477aa'),
                             ('same_expert_innovation','Same-expert innovation','#228833')]:
        v=[100*r['by_block_mode'][str(b)][mode]['neuron']['0.25']['value_diagonal']['relative_SSE'] for b in blocks]
        axes[0,0].plot(positions,v,'o-',label=label,color=color)
    axes[0,0].set_title('Innovation is more concentrated at equal 25% retention')
    axes[0,0].set_ylabel('SSE / corresponding signed-update energy (%)')
    axes[0,0].set_xticks(positions,blocks);axes[0,0].set_xlabel('Block');axes[0,0].legend(fontsize=8)
    for unit,name,label,color in [('neuron','value_diagonal','Individual channels: diagonal rank','#228833'),
                                  ('neuron','source_fixed','Individual channels: source-fixed','#aa3377'),
                                  ('group','actual_group_energy','Whole64: posthoc group-energy rank','#ee7733')]:
        v=[100*r['by_block_mode'][str(b)]['same_expert_innovation'][unit]['0.25'][name]['relative_SSE'] for b in blocks]
        axes[0,1].plot(positions,v,'o-',label=label,color=color)
    axes[0,1].set_title('Whole64 grouping loses much of the local advantage')
    axes[0,1].set_ylabel('Innovation relative SSE (%)');axes[0,1].set_xlabel('Block')
    axes[0,1].set_xticks(positions,blocks);axes[0,1].legend(fontsize=8)
    fractions=['0.125','0.25','0.5'];x=np.arange(3)
    for shift,name,label,color in [(-.2,'original_post_Q2','Absolute post-Q2','#4477aa'),
                                  (.2,'same_expert_post_Q2_innovation','Same-expert innovation','#228833')]:
        v=[100*r['full_M_support'][name]['fractions'][f]['row_tiles']['128']['retained_dense_tile_work_fraction'] for f in fractions]
        bars=axes[1,0].bar(x+shift,v,.36,label=label,color=color)
        axes[1,0].bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
    axes[1,0].set_xticks(x,['12.5%','25%','50%']);axes[1,0].set_ylim(0,112)
    axes[1,0].set_xlabel('Retained elements per token');axes[1,0].set_ylabel('Occupied M128 × K64 work bound (%)')
    axes[1,0].set_title('Full native token grid: sparse elements fill tiles');axes[1,0].legend(fontsize=8)
    for tile,label,color in [('1','M1 × K64','#929292'),('16','M16 × K64','#4477aa'),('128','M128 × K64','#228833')]:
        v=[100*r['full_M_support_by_block'][str(b)]['same_expert_post_Q2_innovation']['fractions']['0.125']['row_tiles'][tile]['retained_dense_tile_work_fraction'] for b in blocks]
        axes[1,1].plot(positions,v,'o-',label=label,color=color)
    axes[1,1].set_ylim(0,106);axes[1,1].set_xticks(positions,blocks);axes[1,1].set_xlabel('Block')
    axes[1,1].set_ylabel('Retained tile-work bound (%)');axes[1,1].set_title('12.5% innovation support in original layout')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.suptitle('Wan2.2 four-step FFN · 6 saved sources / 2 unchanged native videos · no sparse speed or quality claim',fontsize=12)
    fig.savefig(output.with_suffix('.png'),dpi=170);fig.savefig(output.with_suffix('.pdf'))


if __name__=='__main__':main()
