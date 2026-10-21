"""Plot measured TF2 errors at identical D-Q states, without causal attribution."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    record = json.loads(args.report.read_text())
    if record['status'] != 'COMPLETED_TF2_SAME_DENSE_STATE_DIAGNOSTIC' or record['quality_or_speed_acceptance']:
        raise RuntimeError('Complete paired TF2 diagnostic required')
    for arm in record['completed_arms']:
        if sha(arm['path']) != arm['sha256']:
            raise RuntimeError('Diagnostic arm changed')
        value = json.loads(Path(arm['path']).read_text())
        if [c['index'] for c in value['completed']] != [0,5,10,15]:
            raise RuntimeError('All four prespecified sources required')
        for case in value['completed']:
            if sha(case['path']) != case['sha256']:
                raise RuntimeError('Diagnostic case changed')
            actual = json.loads(Path(case['path']).read_text())
            if (arm['arm'] == 'UB1' and not all(s['input_bitwise_equals_DQ'] for s in actual['steps'])):
                raise RuntimeError('All candidate inputs must equal D-Q')
    rows = record['per_step']
    if [r['step'] for r in rows] != list(range(4)) or any(r['source_count'] != 4 for r in rows):
        raise RuntimeError('Four steps pooled over the fixed four sources required')
    series = [([100*r['flow_error']['pooled_relative_RMSE'] for r in rows], 'Flow relative RMSE (%)'),
              ([100*r['one_step_post_error']['pooled_relative_RMSE'] for r in rows], 'One-step post relative RMSE (%)'),
              ([r['one_step_post_error']['squared_error'] for r in rows], 'One-step post squared error (sum)')]
    fig, axes = plt.subplots(1,3,figsize=(10.7,4.1),layout='constrained')
    for ax, (values, title) in zip(axes,series):
        bars = ax.bar(range(4),values,color=['#637f9d','#637f9d','#5d947f','#b77b49'])
        ax.set_xticks(range(4),['step0','step1','step2','step3'])
        ax.set_title(title,fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.17)
        ax.set_axisbelow(True)
        ax.set_ylim(0,max(values)*1.23)
        for bar, value in zip(bars,values):
            ax.annotate(f'{value:.3g}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
    fig.suptitle('Frozen UB1 on identical verified D-Q states: 4 fixed supplementary sources\n'
                 'Actual pre-restoration errors; not additive terminal attribution or quality acceptance',fontsize=10)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['.png','.pdf']:
        fig.savefig(args.output.with_suffix(ext),dpi=180)
    args.output.with_suffix('.json').write_text(json.dumps({'source_sha256':sha(Path(__file__)),
        'report_path':str(args.report.resolve()),'report_sha256':sha(args.report),
        'matplotlib':matplotlib.__version__,'quality_or_speed_acceptance':False},indent=2)+'\n')


if __name__ == '__main__':
    main()
