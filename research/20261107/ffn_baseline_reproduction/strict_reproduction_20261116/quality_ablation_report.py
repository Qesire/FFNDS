"""Paired DEV3 E4/E6 metric changes from independently verified metric cells."""
import json,math
from pathlib import Path
from quality_pair_report import DIMS,finite_number,load_verified_runs

DEV3=('vbi_1013','vbi_0886','vbi_0800')
COMPARISONS={'E4':('ICC','ICC 2R E4'),'E6':('Chipmunk Author Safe','Chipmunk Author Safe E6')}


def aggregate_ablation(contract,cells,comparison,*,e6_variable=None):
    if comparison not in COMPARISONS:raise ValueError('unknown ablation')
    if comparison=='E6' and e6_variable not in ('tail','precision'):raise ValueError('E6 variable must be frozen')
    parent,child=COMPARISONS[comparison];groups=contract['source_clusters'];floor=contract['motion_dense_floor']
    if not finite_number(floor) or floor<0 or any(s not in groups for s in DEV3):raise ValueError('invalid preregistered source/floor')
    index={}
    for row in cells:
        if row['source'] not in DEV3 or row['dimension'] not in DIMS or row['arm'] not in (parent,child):continue
        k=(row['source'],row['arm'],row['dimension'])
        if k in index:raise ValueError('duplicate ablation cell')
        if row['status']!='COMPLETE' or not finite_number(row['value']) or row.get('source_cluster_id')!=groups[row['source']]:
            raise ValueError('unverified or invalid ablation cell')
        if row['arm']==child and row.get('comparison')!=comparison:raise ValueError('child arm comparison identity mismatch')
        if comparison=='E6' and row['arm']==child and row.get('e6_variable')!=e6_variable:
            raise ValueError('E6 child is not the frozen variable')
        index[k]=row
    pairs=[];summaries=[]
    for dimension in DIMS:
        valid=[]
        for source in DEV3:
            a=index.get((source,parent,dimension));b=index.get((source,child,dimension))
            applicable=dimension!='camera_motion' or source in contract.get('explicit_camera_instruction_by_source',{})
            row={'source':source,'cluster':groups[source],'dimension':dimension,
                 'parent':parent,'child':child,'status':'NOT_APPLICABLE' if not applicable else 'MISSING','delta':None}
            if a and b:
                if not applicable:raise ValueError('camera metric provided for nonapplicable source')
                if not a.get('input_identity') or a['input_identity']!=b.get('input_identity'):
                    raise ValueError('ablation source condition/noise differs')
                if not a.get('metric_batch_identity') or a['metric_batch_identity']!=b.get('metric_batch_identity'):
                    raise ValueError('ablation pair not scored in the same fixed metric batch')
                if not a.get('metric_protocol_sha256') or a['metric_protocol_sha256']!=b.get('metric_protocol_sha256'):
                    raise ValueError('ablation metric protocol differs')
                row.update(status='COMPLETE',parent_value=a['value'],child_value=b['value'],delta=b['value']-a['value'])
                if dimension=='dynamic_degree':
                    x,y=a.get('continuous_motion'),b.get('continuous_motion')
                    if not finite_number(x) or not finite_number(y) or min(x,y)<0:raise ValueError('invalid paired RAFT motion')
                    row.update(parent_motion=x,child_motion=y,motion_delta=y-x,motion_ratio=y/x if x>floor else None)
                valid.append(row)
            pairs.append(row)
        clusters={}
        for x in valid:clusters.setdefault(x['cluster'],[]).append(x['delta'])
        means={k:math.fsum(v)/len(v) for k,v in clusters.items()};values=list(means.values())
        summaries.append({'dimension':dimension,'valid_sources':len(valid),'valid_clusters':len(values),
                          'cluster_equal_mean_delta':math.fsum(values)/len(values) if values else None,
                          'source_delta_range':[min(x['delta'] for x in valid),max(x['delta'] for x in valid)] if valid else None,
                          'leave_one_cluster_out':{k:math.fsum(v for n,v in means.items() if n!=k)/(len(values)-1) for k in means} if len(values)>1 else {}})
    return {'comparison':comparison,'e6_variable':e6_variable if comparison=='E6' else None,
            'parent':parent,'child':child,'pairs':pairs,'summaries':summaries,
            'interpretation':'paired DEV3 development diagnostic; no noninferiority claim'}

if __name__=='__main__':
    import argparse,hashlib
    p=argparse.ArgumentParser();p.add_argument('contract');p.add_argument('contract_sha256');p.add_argument('runs');p.add_argument('comparison',choices=('E4','E6'));p.add_argument('output');p.add_argument('--e6-variable',choices=('tail','precision'));a=p.parse_args()
    data=Path(a.contract).read_bytes()
    if hashlib.sha256(data).hexdigest()!=a.contract_sha256:raise ValueError('frozen contract identity mismatch')
    result=aggregate_ablation(json.loads(data),load_verified_runs(json.loads(Path(a.runs).read_text(encoding='utf8'))),a.comparison,e6_variable=a.e6_variable)
    with Path(a.output).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')

