"""Plot the eight fixed-D-input OD1 scopes without pooling them as eight images."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ar2_checkpoint import atomic_json,digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=json.loads(args.report.read_text())
    if report['status']!='COMPLETED_OD1_EIGHT_NATIVE_CONTROLS_AND_LOCAL_FUNCTION_COMPARISONS':
        raise RuntimeError('Completed native OD1 evidence required')
    rows=report['comparisons']
    if len(rows)!=8 or len({r['task'] for r in rows})!=2 or not all(all(c['checks'].values()) for c in report['controls']):
        raise RuntimeError('Eight bitwise controls and two source images required')
    labels=[f"s{r['step']} / img{r['task']} / seed{str(r['seed'])[-2:]}" for r in rows]
    x=np.arange(len(rows));fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for i,(key,label,color) in enumerate([('D_vs_O','D-BF16 vs O-BF16','#22848b'),('DQ_vs_D','D-Q vs D-BF16','#c1772c')]):
        axes[0].bar(x+(i-.5)*.36,[100*r['comparison'][key]['same_D_gate']['relative_RMSE'] for r in rows],.36,label=label,color=color)
    axes[0].set(ylabel='Relative RMSE (%)',title='Local FFN output, same D input and gate')
    axes[0].legend(frameon=False)
    keys=[('O_BF16','O-BF16','#4066a5'),('D_BF16','D-BF16','#22848b'),('D_Q_on_DBF16_U','D-Q','#c1772c')]
    for key,label,color in keys:
        axes[1].plot(x,[100*r['C_concentration'][key]['mean_row_oracle_top108_retained_C'] for r in rows],marker='o',label=label,color=color,alpha=.8)
    axes[1].set(ylabel='Mean oracle top108 C retained (%)',title='Original-coordinate group contribution diagnostic')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x,labels,rotation=45,ha='right');ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('OD1: 2 images x 2 seeds x 2 steps, fixed block20; no O trajectory or online selector',fontsize=12)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=170)
    fig.savefig(args.output.with_suffix('.pdf'))
    atomic_json(args.output.with_suffix('.json'),{'status':'PLOTTED_COMPLETE_FIXED_INPUT_OD1','input_sha256':digest(args.report),
        'source_sha256':digest(Path(__file__)),'png_sha256':digest(args.output),'pdf_sha256':digest(args.output.with_suffix('.pdf')),
        'independent_images':2,'scopes':8,'inference':'descriptive only; no independent eight-image inference'})


if __name__=='__main__': main()
