"""Descriptive analysis of frozen source-level historical reports; no model or fitting."""
import argparse, csv, json, hashlib, math, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/worker_20261020'
QUALITY = {'formal': {'UB1':'formal20_UB1_paired_sources.csv','UB2':'UB2_formal20_COMPARISON.csv','C24':'formal20_C24_paired_sources.csv'}, 'disjoint': {'UB1':'disjoint20_UB1_paired_sources.csv','UB2':'UB2_disjoint20_COMPARISON.csv'}}
TRAJ = {(batch,method): f'{method}_{batch}20_trajectory_per_source_step.csv' for batch in ('formal','disjoint') for method in ('UB1','UB2')}

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name):
    with (BASE/name).open(newline='') as f: return list(csv.DictReader(f))
def quantile(xs,q):
    x=sorted(xs); p=(len(x)-1)*q; i=int(p); return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(p-i)
def describe(xs):
    return {'n':len(xs),'mean':statistics.mean(xs),'q10':quantile(xs,.1),'median':quantile(xs,.5),'q90':quantile(xs,.9),'min':min(xs),'max':max(xs)}
def checked_relative_drop(row):
    x=float(row['D_Q']); y=float(row['candidate']); z=float(row['relative_drop'])
    if not all(math.isfinite(t) for t in (x,y,z)) or x==0 or not math.isclose(z,(x-y)/x,rel_tol=1e-9,abs_tol=1e-9): raise ValueError('Invalid stored metric')
    return z
def paired_identity(a,b):
    return a['source_group']==b['source_group'] and a['dimension']==b['dimension'] and a['prompt']==b['prompt'] and float(a['D_Q'])==float(b['D_Q'])
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.protocol.read_text())
    if digest(Path(__file__))!=plan['tool_sha256']: raise ValueError('Tool identity mismatch')
    for name, expected in plan['inputs'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or digest(path)!=expected: raise ValueError('Input identity mismatch: '+name)
    expected_names=set(QUALITY['formal'].values())|set(QUALITY['disjoint'].values())|set(TRAJ.values())|{f'{m}_{b}20_trajectory_RUN.json' for b,m in TRAJ}|{'OP3_summary.json'}
    if set(plan['inputs'])!={str((BASE/n).relative_to(ROOT)) for n in expected_names}: raise ValueError('Unexpected input set')
    output_dir=args.output.resolve()
    if output_dir.exists(): raise ValueError('Output scope exists')
    output_dir.mkdir(parents=True)
    report={'quality':{},'matched_methods':{},'trajectory':{},'op3':{},'input_sha256':{},'quantile_method':'Linear interpolation of sorted values at (n-1)*p', 'notes':['Exploratory descriptive analysis of exposed sources; no fitting, test of significance or selection of future confirmation sources.','relative_drop=(dense-candidate)/dense; positive is worse for these higher-is-better metrics.','Trajectory rows are not independent sources; compare methods at identical source and step.']}
    all_data={}
    for batch, methods in QUALITY.items():
        report['quality'][batch]={}
        for method,name in methods.items():
            path=BASE/name; report['input_sha256'][str(path.relative_to(ROOT))]=digest(path); data=rows(name); table={}
            for r in data:
                key=(r['source_group'],r['dimension'])
                if key in table: raise ValueError('Duplicate source/metric')
                checked_relative_drop(r)
                table[key]=r
            all_data[(batch,method)]=table
            dims=sorted({k[1] for k in table}); metrics={}
            for dim in dims:
                subset=[r for (source,metric),r in table.items() if metric==dim]
                groups={}
                for group in sorted({r['source_group'].split(':')[0] for r in subset}):
                    part=[r for r in subset if r['source_group'].split(':')[0]==group]
                    groups[group]={'baseline':describe([float(r['D_Q']) for r in part]),'relative_drop':describe([float(r['relative_drop']) for r in part]),'sources_drop_above_2_percent':sum(float(r['relative_drop'])>.02 for r in part),'worst_sources':[{k:r[k] for k in ('source_group','D_Q','candidate','relative_drop')} for r in sorted(part,key=lambda r:float(r['relative_drop']),reverse=True)[:5]]}
                dense=sum(float(r['D_Q']) for r in subset); candidate=sum(float(r['candidate']) for r in subset)
                metrics[dim]={'baseline':describe([float(r['D_Q']) for r in subset]),'candidate':describe([float(r['candidate']) for r in subset]),'relative_drop':describe([float(r['relative_drop']) for r in subset]),'aggregate_drop':(dense-candidate)/dense,'sources_drop_above_2_percent':sum(float(r['relative_drop'])>.02 for r in subset),'category':groups,'worst_sources':[{k:r[k] for k in ('source_group','D_Q','candidate','relative_drop')} for r in sorted(subset,key=lambda r:float(r['relative_drop']),reverse=True)[:5]]}
            report['quality'][batch][method]=metrics
        for lhs in methods:
            for rhs in methods:
                if lhs>=rhs: continue
                a=all_data[(batch,lhs)];b=all_data[(batch,rhs)];common=sorted(set(a)&set(b)); identity=[k for k in common if not paired_identity(a[k],b[k])]
                result={'matched_source_metric_rows':len(common),'lhs_only_rows':len(set(a)-set(b)),
                        'rhs_only_rows':len(set(b)-set(a)), 'baseline_or_prompt_mismatch':len(identity),'metrics':{}}
                for dim in sorted({x[1] for x in common}):
                    valid=[k for k in common if k[1]==dim and k not in identity];diff=[float(a[k]['relative_drop'])-float(b[k]['relative_drop']) for k in valid]
                    if diff:result['metrics'][dim]={'paired_drop_difference_'+lhs+'_minus_'+rhs:describe(diff),'lhs_more_loss_count':sum(x>0 for x in diff),'rhs_more_loss_count':sum(x<0 for x in diff),'ties':sum(x==0 for x in diff)}
                report['matched_methods'][batch+':'+lhs+':'+rhs]=result
    report['cohort_source_overlap'] = sorted({x[0] for x in all_data['formal','UB1']} & {x[0] for x in all_data['disjoint','UB1']})
    trajectory_tables={}; trajectory_receipts={}
    for (batch,method),name in TRAJ.items():
        p=BASE/name;run=BASE/f'{method}_{batch}20_trajectory_RUN.json';d=json.loads(run.read_text()); report['input_sha256'][str(p.relative_to(ROOT))]=digest(p);report['input_sha256'][str(run.relative_to(ROOT))]=digest(run)
        if d['per_source_csv_sha256']!=digest(p): raise ValueError('Trajectory receipt mismatch')
        data=rows(name);table={}
        for r in data:
            key=(r['source_group'],int(r['step']))
            if key in table or key[1] not in range(4): raise ValueError('Duplicate or invalid trajectory source/step')
            table[key]=r
        sources={k[0] for k in table}
        if set(table)!={(source,step) for source in sources for step in range(4)}: raise ValueError('Incomplete trajectory')
        trajectory_tables[batch,method]=table;trajectory_receipts[batch,method]=d
        report['trajectory'][batch+':'+method]={str(step):describe([float(r['relative_RMSE']) for r in data if int(r['step'])==step]) for step in range(4)}
    report['matched_trajectory']={}
    for batch in ('formal','disjoint'):
        a=trajectory_tables[batch,'UB1'];b=trajectory_tables[batch,'UB2']; common=set(a)&set(b)
        same_dense=trajectory_receipts[batch,'UB1']['dense_suite_sha256']==trajectory_receipts[batch,'UB2']['dense_suite_sha256']
        bystep={}
        for step in range(4):
            keys=sorted(k for k in common if k[1]==step and a[k]['dimension']==b[k]['dimension'] and a[k]['dense_squared_norm']==b[k]['dense_squared_norm']) if same_dense else []
            differences=[{'source_group':k[0],'UB1_relative_RMSE':float(a[k]['relative_RMSE']),'UB2_relative_RMSE':float(b[k]['relative_RMSE']),'UB2_minus_UB1':float(b[k]['relative_RMSE'])-float(a[k]['relative_RMSE'])} for k in keys]
            bystep[str(step)]={'n':len(keys),'source_differences':differences,'relative_RMSE_equal_count':sum(a[k]['relative_RMSE']==b[k]['relative_RMSE'] for k in keys),'all_recorded_numeric_values_equal_count':sum(all(float(a[k][field])==float(b[k][field]) for field in ('elements','dense_squared_norm','candidate_squared_norm','squared_error','relative_RMSE','norm_ratio','cosine','centered_variance_ratio')) for k in keys)}
        report['matched_trajectory'][batch]={'dense_suite_same':same_dense,'only_UB1_rows':len(set(a)-set(b)),'only_UB2_rows':len(set(b)-set(a)),'steps':bystep}
    op=BASE/'OP3_summary.json';data=json.loads(op.read_text());report['input_sha256'][str(op.relative_to(ROOT))]=digest(op);report['op3']={'n_fold_step_comparisons':len(data['CV']),'new_over_same_n_old':describe([x['candidate_over_matched_old_state'] for x in data['CV']]),'new_over_frozen_UB1':describe([x['candidate_over_frozen'] for x in data['CV']]),'native_gate_pass':data['native_gate_pass'],'videos':data['videos']}
    out=output_dir/'UB1_SOURCE_GENERALIZATION_AUDIT_20261027.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    md=out.with_suffix('.md');md.write_text('# UB1 source generalization: descriptive audit\n\nFrozen exploratory source-level analysis; complete numeric tables, sources, sample counts and tails are in [JSON](UB1_SOURCE_GENERALIZATION_AUDIT_20261027.json). These previously exposed cohorts do not provide an independent confirmation set.\n')
if __name__=='__main__': main()
