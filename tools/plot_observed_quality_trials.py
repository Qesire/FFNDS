"""Compare fixed policies on separate observed cohorts, using official point gates."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ar2_checkpoint import atomic_json,digest

DIMS=['aesthetic_quality','imaging_quality','motion_smoothness']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial',action='append',nargs=3,metavar=('LABEL','FORMAL_JSON','SUPPLEMENT_JSON'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    evidence=[];references={}
    for label,*paths in args.trial:
        rows=[json.loads(Path(p).read_text()) for p in paths]
        if (any(r['status']!='COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON' or r['unique_source_count']!=20 for r in rows)
                or rows[0]['candidate_policy_sha256']!=rows[1]['candidate_policy_sha256']):
            raise RuntimeError('Each plotted policy requires two complete same-policy20-source comparisons')
        for cohort,row in zip(['original20','supplement20'],rows):
            dense={dim:row['metrics'][dim]['D_Q'] for dim in DIMS}
            if cohort in references and references[cohort]!=dense:
                raise RuntimeError('Policy comparisons must use the identical per-cohort D-Q reference')
            references[cohort]=dense
            if any(row['metrics'][dim]['max_relative_drop']!=.02 for dim in DIMS):
                raise RuntimeError('The frozen2% per-metric point gate is required')
        evidence.append({'label':label,'policy_sha256':rows[0]['candidate_policy_sha256'],
            'comparisons':[{'path':str(Path(p).resolve()),'sha256':digest(p)} for p in paths],
            'relative_drop_percent':[[100*r['metrics'][dim]['relative_drop'] for dim in DIMS] for r in rows]})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharey=True)
    x=np.arange(3);width=.76/len(evidence)
    colors=['#3e7298','#c18b39','#328869','#965c99']
    for cohort,ax in enumerate(axes):
        for i,item in enumerate(evidence):
            values=item['relative_drop_percent'][cohort]
            positions=x+(i-(len(evidence)-1)/2)*width
            bars=ax.bar(positions,values,width,label=item['label'],color=colors[i%len(colors)])
            ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,fontsize=9)
        ax.axhline(2,color='#b64141',linestyle='--',linewidth=1.2,label='2% per-metric gate')
        ax.axhline(0,color='grey',linewidth=.6)
        ax.set_xticks(x,['Aesthetic','Imaging','Motion'])
        ax.set_title(['Original20 (observed)','Supplement20 (observed)'][cohort])
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Relative quality drop from paired D-Q (%)')
    axes[1].legend(frameon=False)
    fig.suptitle('Same frozen policy per trial; each cohort has20 independent sources,5 filename copies/source',fontsize=11)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180);fig.savefig(args.output.with_suffix('.pdf'))
    atomic_json(args.output.with_suffix('.json'),{'status':'PLOTTED_SEPARATE_OBSERVED_COHORT_POINT_GATES',
        'source_sha256':digest(Path(__file__)),'trials':evidence,'reference_scores':references,
        'png_sha256':digest(args.output),'pdf_sha256':digest(args.output.with_suffix('.pdf')),
        'scope':'both cohorts observed during development, no untouched-test claim; official point estimates, no uncertainty gate substitution'})


if __name__=='__main__':main()
