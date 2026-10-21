"""Recount saved quality evidence and describe native TF2 flow errors on CPU.

No model execution, parameter fitting, new candidate or metric rerun. Oracle
projections below are descriptive hindsight bounds, never deployed corrections.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from ar2_checkpoint import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs/worker_restart_20261020'
COHORTS = {
    'formal20': ['RR1', 'C24', 'VM1', 'UB1', 'LR2', 'UB2'],
    'disjoint20': ['UB1', 'LR2', 'UB2'],
}


def quality_analysis():
    cohorts, bindings = {}, {}
    for cohort, methods in COHORTS.items():
        arms, baseline = {}, None
        for method in methods:
            folder = RUNS / f'{cohort}_{method}_comparison'
            rp, cp = folder / 'COMPARISON.json', folder / 'COMPARISON.csv'
            report = json.loads(rp.read_text())
            if report['unique_source_count'] != 20:
                raise RuntimeError('Quality unit must remain20 unique sources')
            rows = list(csv.DictReader(cp.open()))
            if len(rows) != 60 or digest(cp) != report['paired_csv_sha256']:
                raise RuntimeError('Paired quality CSV identity/coverage differs')
            by_metric, results = defaultdict(list), {}
            for row in rows:
                by_metric[row['dimension']].append(row)
            for metric, subset in by_metric.items():
                ids = [r['source_group'] for r in subset]
                if len(set(ids)) != 20:
                    raise RuntimeError('Duplicate quality source')
                dq, candidate = (np.array([float(r[k]) for r in subset]) for k in ['D_Q', 'candidate'])
                aggregate = 1 - candidate.mean() / dq.mean()
                if not math.isclose(aggregate, report['metrics'][metric]['relative_drop'], abs_tol=1e-12):
                    raise RuntimeError('Official aggregate differs from per-source reconstruction')
                groups = {}
                for group in sorted({s.split(':')[0] for s in ids}):
                    take = np.array([s.split(':')[0] == group for s in ids])
                    groups[group] = {'sources': int(take.sum()), 'drop_percent': 100 * (1-candidate[take].sum()/dq[take].sum())}
                deficit = dq - candidate
                order = np.argsort(-deficit, kind='stable')
                rng = np.random.default_rng(20261021)
                draws = rng.integers(0, 20, size=(10000, 20))
                boot = 100 * (1-candidate[draws].sum(axis=1)/dq[draws].sum(axis=1))
                needed = max(0., float(deficit.sum() - .02*dq.sum()))
                results[metric] = {
                    'D_Q_mean': float(dq.mean()), 'candidate_mean': float(candidate.mean()),
                    'drop_percent': float(100*aggregate), 'original_gate_pass': report['metrics'][metric]['pass'],
                    'source_bootstrap_percentile95_drop_percent': np.quantile(boot, [.025, .975]).tolist(),
                    'groups': groups, 'sources_worse': int((deficit > 0).sum()),
                    'sources_over2percent_individual_drop': int((candidate < .98*dq).sum()),
                    'sum_score_deficit': float(deficit.sum()), 'sum_score_recovery_needed_for_original_gate': needed,
                    'fraction_of_current_net_deficit_requiring_recovery': needed/float(deficit.sum()) if needed else 0.,
                    'largest3_deficits': [{'source_group': ids[i], 'score_deficit': float(deficit[i]),
                                            'individual_drop_percent': 100*float(deficit[i]/dq[i])} for i in order[:3]],
                    'largest3_share_of_net_deficit': float(deficit[order[:3]].sum()/deficit.sum()) if deficit.sum()>0 else None,
                    'per_source': {r['source_group']: {'D_Q': float(r['D_Q']), 'candidate': float(r['candidate'])} for r in subset},
                }
            base = {metric: {s: v['D_Q'] for s, v in val['per_source'].items()} for metric, val in results.items()}
            if baseline is not None and base != baseline:
                raise RuntimeError('Candidates do not share the exact original baseline')
            baseline, arms[method] = base, results
            bindings.update({str(p.relative_to(ROOT)): digest(p) for p in [rp, cp]})
        comparison = {}
        if 'LR2' in methods:
            metric = 'imaging_quality'
            ids = sorted(baseline[metric])
            losses = {m: np.array([arms[m][metric]['per_source'][s]['D_Q']-arms[m][metric]['per_source'][s]['candidate'] for s in ids]) for m in ['UB1', 'LR2', 'UB2']}
            d = np.array([baseline[metric][s] for s in ids])
            values = np.array([[arms[m][metric]['per_source'][s]['candidate'] for s in ids] for m in ['UB1','LR2','UB2']])
            best = values.max(axis=0)
            comparison = {'UB1_LR2_deficit_Pearson': float(np.corrcoef(losses['UB1'], losses['LR2'])[0,1]),
                          'LR2_recovers_fraction_of_UB1_net_deficit': float(1-losses['LR2'].sum()/losses['UB1'].sum()),
                          'LR2_sources_improved_over_UB1': int((values[1]>values[0]).sum()),
                          'three_saved_arm_imaging_only_hindsight_oracle_drop_percent': float(100*(1-best.sum()/d.sum())),
                          'oracle_scope': 'Only these three frozen whole-video outputs. Hindsight upper bound for imaging; no router, mixture, within-video policy or quality acceptance.'}
        cohorts[cohort] = {'arms': arms, 'cross_method_description': comparison}
    return {'cohorts': cohorts, 'source_bindings': bindings,
            'bootstrap_scope': '10000 fixed-seed paired source resamples of already-exposed sources, not seed uncertainty, independent final evaluation or a replacement for the original hard gate.'}


def describe_flow_error(teacher, student):
    # [C,T,H,W], real stored FP32 values, all reductions FP64.
    d, s = np.asarray(teacher, dtype=np.float64), np.asarray(student, dtype=np.float64)
    e = s-d
    sse, dn, sn = (float(np.square(v).sum()) for v in [e, d, s])
    dot = float((s*d).sum())
    along_teacher = float((e*d).sum())
    gain = dot/sn
    # Teacher amplitude regression, not a proposed gain on the FFN itself.
    oracle_sse = float(np.square(gain*s-d).sum())
    channel_bias = e.mean(axis=(1,2,3), keepdims=True)
    bias_energy = float(np.square(channel_bias).sum()*np.prod(e.shape[1:]))
    axis_details = {}
    for name, axis in [('time',1),('height',2),('width',3)]:
        de, dd = np.diff(e, axis=axis), np.diff(d, axis=axis)
        axis_details[name] = {'error_difference_energy': float(np.square(de).sum()),
                              'teacher_difference_energy': float(np.square(dd).sum()),
                              'relative_difference_RMSE': float(np.sqrt(np.square(de).sum()/np.square(dd).sum()))}
    return {'squared_error':sse, 'teacher_squared_norm':dn, 'student_squared_norm':sn,
            'student_teacher_dot':dot, 'error_teacher_dot':along_teacher,
            'relative_RMSE':math.sqrt(sse/dn), 'cosine':dot/math.sqrt(dn*sn),
            'teacher_parallel_error_energy_fraction': along_teacher**2/(dn*sse),
            'per_channel_constant_bias_error_energy_fraction':bias_energy/sse,
            'hindsight_optimal_scalar':gain, 'hindsight_optimal_scalar_remaining_SSE_fraction':oracle_sse/sse,
            'axis_differences':axis_details}


def flow_analysis():
    import torch
    from safetensors import safe_open
    from run_ar2_restart import tensor_sha
    from worker_identity import require_worker
    require_worker()
    torch.set_num_threads(8)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU diagnostic only')
    rows, bindings = [], {}
    for index in [0,5,10,15]:
        reports = {}
        for arm in ['DQ','UB1']:
            path = ROOT/f'reports/worker_20261020/TF2_{arm}_input{index:02d}.json'
            reports[arm] = json.loads(path.read_text())
            bindings[str(path.relative_to(ROOT))] = digest(path)
        if (reports['DQ']['status']!='PASS_TF2_NATIVE_DENSE_CONTROL'
                or reports['UB1']['status']!='PASS_TF2_SAME_DENSE_INPUT_STATES'
                or reports['DQ']['input'] != reports['UB1']['input']):
            raise RuntimeError('Require matched original TF2 input records')
        for step in range(4):
            tensors, records = {}, {}
            for arm in ['DQ','UB1']:
                record = reports[arm]['steps'][step]
                if record['step']!=step or digest(record['path'])!=record['sha256']:
                    raise RuntimeError('Original TF2 tensor identity changed')
                with safe_open(record['path'],framework='pt',device='cpu') as h:
                    if set(h.keys())!={'input','flow','post'}:
                        raise RuntimeError('Unexpected TF2 tensor fields')
                    tensors[arm]={k:h.get_tensor(k) for k in h.keys()}
                for key,t in tensors[arm].items():
                    if (t.shape!=(16,21,58,104) or t.dtype!=torch.float32
                            or not bool(torch.isfinite(t).all()) or tensor_sha(t)!=record[f'{key}_sha256']):
                        raise RuntimeError('Original TF2 decoded tensor shape/type/values/SHA differ')
                records[arm]=record
            if (not records['UB1']['input_bitwise_equals_DQ']
                    or not torch.equal(tensors['DQ']['input'],tensors['UB1']['input'])):
                raise RuntimeError('Same-state input control differs')
            description=describe_flow_error(tensors['DQ']['flow'].numpy(),tensors['UB1']['flow'].numpy())
            expected=records['UB1']['flow_error']['squared_error']
            post=float((tensors['DQ']['post'].double()-tensors['UB1']['post'].double()).square().sum())
            if (not math.isclose(description['squared_error'],expected,rel_tol=1e-12)
                    or not math.isclose(post,records['UB1']['one_step_post_error']['squared_error'],rel_tol=1e-12)):
                raise RuntimeError('Saved original TF2 errors did not reproduce')
            rows.append({'input_index':index,'step':step,**description,
                         'native_post_SSE':post,'original_error_reproduced':True,
                         'payload_sha256':{arm:records[arm]['sha256'] for arm in records}})
    if torch.cuda.is_initialized():
        raise RuntimeError('Unexpected CUDA initialization')
    pooled=[]
    for step in range(4):
        part=[r for r in rows if r['step']==step]
        sse=sum(r['squared_error'] for r in part)
        dot=sum(r['student_teacher_dot'] for r in part)
        sn=sum(r['student_squared_norm'] for r in part)
        dn=sum(r['teacher_squared_norm'] for r in part)
        pooled.append({'step':step,'source_count':4,'squared_error':sse,
                       'hindsight_pooled_optimal_scalar':dot/sn,
                       'hindsight_pooled_scalar_remaining_SSE_fraction':(dn-dot*dot/sn)/sse,
                       'per_source_hindsight_scalar_remaining_SSE_fraction':sum(r['squared_error']*r['hindsight_optimal_scalar_remaining_SSE_fraction'] for r in part)/sse,
                       'per_channel_constant_bias_error_energy_fraction':sum(r['squared_error']*r['per_channel_constant_bias_error_energy_fraction'] for r in part)/sse,
                       'teacher_parallel_error_energy_fraction':sum(r['squared_error']*r['teacher_parallel_error_energy_fraction'] for r in part)/sse})
    return {'rows':rows,'pooled':pooled,'source_bindings':bindings,'CUDA_initialized':False,
            'scope':'16 saved same-DQ-state predictions from4 already-exposed sources. No deployment rollout, fitted correction, post-processed video or metric change. Scalar oracles cannot bound general spatial/per-layer corrections.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--with-flow-tensors',action='store_true')
    args=parser.parse_args()
    if args.report.exists():
        raise RuntimeError('Preserve first completed analysis')
    result={'status':'COMPLETE_SAVED_EVIDENCE_FAILURE_ANALYSIS_CPU_ONLY',
            'evidence_base_commit':'16e14265b6a8908bc9e6f3fa7fc4fb860c350fb8',
            'quality':quality_analysis(),'source_sha256':digest(Path(__file__)),
            'models_loaded':0,'fitted_parameters':0,'GPU_jobs_started':0,'quality_or_speed_acceptance':False}
    if args.with_flow_tensors:
        result['flow']=flow_analysis()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.report,result)
    print(result['status'],flush=True)


if __name__=='__main__':
    main()
