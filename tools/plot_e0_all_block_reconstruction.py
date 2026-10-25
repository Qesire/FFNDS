"""Descriptive all-depth matched-calibration plots; no model operations."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();d=json.loads(a.result.read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
names=['calibration_input_ridge','affine_plus_input_ridge','whole_g94_ridge','affine_residual_g54_ridge']
labels=['Input ridge','Affine + input ridge','94 groups + ridge','Affine +54 groups + ridge']
colors=['#999999','#d6a047','#bd604d','#147d92']
for name,label,color in zip(names,labels,colors):
 vals=[100*d['by_block'][str(i)]['methods'][name]['uniform_disjoint']['relative_target_SSE']['mean'] for i in range(40)]
 axes[0].plot(range(40),vals,label=label,color=color,lw=1.8)
axes[0].set(xlabel='Block',ylabel='Gated output SSE / current target energy (%)',title='All40 blocks, same four-source calibration')
axes[0].legend(fontsize=8)
vals=[100*d['overall_all_controls']['methods'][n]['uniform_disjoint']['relative_target_SSE']['mean'] for n in names]
axes[1].barh(labels[::-1],vals[::-1],color=colors[::-1])
for i,v in enumerate(vals[::-1]):axes[1].text(v+.4,i,f'{v:.2f}%',va='center')
axes[1].set(xlim=(0,34),xlabel='Equal-case mean relative SSE (%)',title='320 held cases; earlier values reproduced')
fig.supxlabel('Calibration4/8/16/20, 256 rows per expert/block. Held0/12, actual seeds42/54. Local outputs, not video quality.',fontsize=9)
a.out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.out.with_suffix('.png'),dpi=180);fig.savefig(a.out.with_suffix('.pdf'));plt.close(fig)
