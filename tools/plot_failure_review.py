"""Standalone scientific figures for saved-source errors and synthetic timings."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ar2_checkpoint import atomic_json, digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/20261021'


def main():
    paths=[OUT/n for n in ['failure_data_audit_R7.json','weight_2of4_shape_preflight.json',
                           'weight_2of4_fused_layout_preflight.json']]
    a,old,new=[json.loads(p.read_text()) for p in paths]
    fig,axes=plt.subplots(3,1,figsize=(12,11),layout='constrained')
    arms=a['quality']['cohorts']['disjoint20']['arms']
    sources=sorted(arms['UB1']['imaging_quality']['per_source'])
    matrix=np.array([[100*(1-v['candidate']/v['D_Q']) for s in sources
                      for v in [arms[m]['imaging_quality']['per_source'][s]]] for m in ['UB1','LR2','UB2']])
    im=axes[0].imshow(matrix,aspect='auto',cmap='coolwarm',vmin=-10,vmax=30)
    axes[0].set_yticks(range(3),['UB1','LR2','UB2'])
    axes[0].set_xticks(range(20),[s.replace('i2v_','') for s in sources],rotation=55,ha='right',fontsize=8)
    axes[0].set_title('Supplement20: imaging loss by original source (development-exposed)')
    fig.colorbar(im,ax=axes[0],label='Individual relative drop (%)',shrink=.8)
    for y in range(3):
        for x in range(20):axes[0].text(x,y,f'{matrix[y,x]:.1f}',ha='center',va='center',fontsize=7)
    flow=a['flow']['pooled'];x=np.arange(4)
    axes[1].bar(x-.16,[100*(1-r['per_source_hindsight_scalar_remaining_SSE_fraction']) for r in flow],
                width=.3,label='Error removable by per-source hindsight scalar')
    axes[1].bar(x+.16,[100*r['per_channel_constant_bias_error_energy_fraction'] for r in flow],
                width=.3,label='Error energy in per-channel constant subspace')
    axes[1].set_xticks(x,[f'step{s}' for s in x]);axes[1].set_ylabel('Fraction of original flow SSE (%)')
    axes[1].set_title('16 same-DQ-state flow pairs: description only, no learned correction')
    axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
    labels=[];values=[];samples=[]
    for run,name in [(old,'default layout'),(new,'fused output layout')]:
        for key,label in [('dense_full','dense'),('sparse_2of4','2:4'),('sparse_2of4_plus_R256_workload','2:4 + R256 work')]:
            labels.append(name+'\n'+label);values.append(run['median_milliseconds'][key]);samples.append(run['milliseconds_samples'][key])
    axes[2].bar(range(6),values,color=['#999999','#bb5555','#cc8888','#999999','#447799','#77aabb'])
    for i,s in enumerate(samples):axes[2].scatter([i]*len(s),s,s=9,color='black',zorder=3)
    axes[2].set_xticks(range(6),labels,fontsize=8);axes[2].set_ylabel('CUDA event time (ms)')
    axes[2].set_title('Synthetic M31668 / D5120 / H13824; medians and all5 samples, not Wan speed')
    axes[2].grid(axis='y',alpha=.2)
    for i,v in enumerate(values):axes[2].text(i,v+1,f'{v:.2f}',ha='center',fontsize=8)
    for suffix in ['png','pdf']:
        dest=OUT/f'failure_review_R7.{suffix}'
        if dest.exists():raise RuntimeError('Preserve completed figure')
        fig.savefig(dest,dpi=180)
    atomic_json(OUT/'failure_review_R7.figure.json',{'sources':{str(p.relative_to(ROOT)):digest(p) for p in paths},
        'plot_source_sha256':digest(Path(__file__)),'quality_or_speed_acceptance':False,
        'scope':'Replots saved exposed-source scores and synthetic non-model timings; distinct runs retain their own dense control.'})


if __name__=='__main__':main()
