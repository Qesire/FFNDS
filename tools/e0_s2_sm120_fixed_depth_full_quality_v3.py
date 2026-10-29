"""V3 immutable snapshot I/O repair for the eight-axis SM120 producer.

A new scope repeats all eight axes; V2's three completed axes are retained as
partial historical evidence, never reused as V3 calls. No automatic retry.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
q2=module('_sm120_quality_v2_for_v3','tools/e0_s2_sm120_fixed_depth_full_quality_v2.py')
budget_module=module('_sm120_raft_v2_budget_for_quality','tools/e0_s2_sm120_fixed_depth_motion_metrics_v2.py')
q=q2.q;base=q2.base;need=base.require
OLD_V2_DRAFT,OLD_V2_VALIDATE,OLD_V2_BINDINGS,OLD_V2_CPU=q2.draft,q2.validate,q2.all_bindings,q2.cpu_expected
SELF='tools/e0_s2_sm120_fixed_depth_full_quality_v3.py'
TEST='tests/test_e0_s2_sm120_fixed_depth_full_quality_v3.py'
V2_CONFIG='config/e0_s2_sm120_fixed_depth_full_quality_shard0_20261029_v2.json'
V2_CONFIG_SHA='8f9cd2c8c283ed6c245ae56af6e82f605a4ec11ad1ccd3b1e967ada117e20e93'
V2_FOLDER='ffnds-derived/e0_s2_sm120_fixed_depth_full_quality_v2_20261029/shard0/attempt1/worker'
V2_EXIT_SHA='54b3759acd5fcca1ba0e0d77da9fd65872099074ee82e4d7a9a2ca235eb37a2f'
V2_RUN_SHA='009ae6d3a6f3b6e6ba1ed2be3434cf0b0e434510f7518cac3ea21e21ab88c07e'
EXTRA={SELF,TEST,V2_CONFIG,'tools/e0_s2_sm120_fixed_depth_motion_metrics_v2.py',
       'tests/test_e0_s2_sm120_fixed_depth_motion_metrics_v2.py'}
REPAIR={'reason':'V2_LEDGER_REPEATED_IMMUTABLE_SNAPSHOT_WALKS_COORDINATOR_INTERRUPT',
        'prior_completed_axes':['imaging_quality','aesthetic_quality','motion_smoothness'],
        'V3_calls':'ALL_EIGHT_AXES_REPEATED_AND_FULLY_CHARGED_NO_RESULT_REUSE',
        'method':'ONCE_HASH_IMMUTABLE_SNAPSHOT_EACH_PERSIST_WALK_MUTABLE_ONLY_FINAL_FULL_AUDIT',
        'algorithm_changed':False,'automatic_retry':False}


def identity(shard):
    need(type(shard) is int and shard in range(4),'Invalid shard')
    return 'E0_S2_SM120_FIXED_DEPTH_EIGHT_AXIS_20261029_SHARD{}_V3'.format(shard)


def output(shard):
    return 'ffnds-derived/e0_s2_sm120_fixed_depth_full_quality_v3_20261029/shard{}/attempt1/worker/results'.format(shard)


def evidence(hf):
    return {'protocol':base.binding(ROOT/V2_CONFIG,ROOT),
            'RUN':base.binding(Path(hf)/V2_FOLDER/'results/RUN.json',hf),
            'exit':base.binding(Path(hf)/V2_FOLDER/'EXIT_RECEIPT.json',hf),
            'raft_software_diagnostic':base.binding(Path(hf)/budget_module.DIAGNOSTIC,hf)}


def verify_evidence(plan,hf):
    p=plan['v2_interrupted_evidence']
    need(p['protocol']['path']==V2_CONFIG and p['protocol']['sha256']==V2_CONFIG_SHA
         and p['RUN']['path']==V2_FOLDER+'/results/RUN.json' and p['RUN']['sha256']==V2_RUN_SHA
         and p['exit']['path']==V2_FOLDER+'/EXIT_RECEIPT.json' and p['exit']['sha256']==V2_EXIT_SHA
         and p['raft_software_diagnostic']['path']==budget_module.DIAGNOSTIC
         and p['raft_software_diagnostic']['sha256']==budget_module.DIAGNOSTIC_SHA,'V2/diagnostic evidence binding differs')
    protocol=base.read_bound(ROOT,p['protocol']);run=base.read_bound(hf,p['RUN']);ex=base.read_bound(hf,p['exit'])
    diag=base.read_bound(hf,p['raft_software_diagnostic'])
    need(protocol['id']=='E0_S2_SM120_FIXED_DEPTH_EIGHT_AXIS_20261029_SHARD0_V2'
         and protocol['status']=='FROZEN_REGISTERED' and ex['id']==protocol['id']
         and ex['protocol_sha256']==V2_CONFIG_SHA and ex['run_sha256']==V2_RUN_SHA
         and type(ex['child_returncode']) is int and ex['child_returncode']==1
         and ex['process_exited'] is True and ex['timeout'] is False and ex['automatic_retry'] is False
         and run['id']==protocol['id'] and run['protocol_sha256']==V2_CONFIG_SHA
         and run['status']=='FAILED_SM120_EIGHT_AXIS_PRESERVED_NO_RETRY'
         and run['error_type']=='KeyboardInterrupt' and run['active_dimension']=='dynamic_degree'
         and [d['dimension'] for d in run['completed_dimensions']]==REPAIR['prior_completed_axes']
         and run['predebited']['official_dimension_calls']==4 and run['completed']['official_dimension_calls']==3
         and run['predebited']['video_dimension_results']==48 and run['completed']['video_dimension_results']==36
         and run['predebited']['RAFT_forwards']==16 and run['completed']['RAFT_forwards']==15
         and run['predebited']['RAFT_iterations']==run['completed']['RAFT_iterations']==320
         and diag['status']=='SOFTWARE_DIAGNOSTIC_COORDINATOR_STOP'
         and diag['observed_snapshot_and_output_files']==1007 and diag['observed_output_bytes']==23669555
         and diag['one_full_output_walk_wall_seconds']==8.650316627928987,'Actual interrupted scope/diagnostic differs')
    paths=[str((Path(hf)/v['video']['path']).resolve()) for v in protocol['videos']]
    need(len(paths)==len(set(paths))==12,'Prior video coverage differs')
    for entry in run['completed_dimensions']:
        base.check_result(entry['result'],paths,entry['dimension'])
    return p


def draft(hf,shard):
    plan=OLD_V2_DRAFT(hf,shard);plan['v2_interrupted_evidence']=evidence(hf);verify_evidence(plan,hf)
    paths={b['path'] for b in plan['repo_sources']}|EXTRA
    plan.update(id=identity(shard),output_HF_relative=output(shard),snapshot_repair=copy.deepcopy(REPAIR),
                repo_sources=[base.binding(ROOT/p,ROOT) for p in sorted(paths)])
    return plan


def validate(plan,frozen=False):
    shard=plan['shard'];need(plan['id']==identity(shard) and plan['output_HF_relative']==output(shard)
         and plan['snapshot_repair']==REPAIR,'V3 identity/output/repair differs')
    paths=[b['path'] for b in plan['repo_sources']]
    need(len(paths)==len(set(paths)) and EXTRA<=set(paths),'V3 source closure incomplete')
    normalized=copy.deepcopy(plan);normalized['repo_sources']=[b for b in normalized['repo_sources'] if b['path'] not in EXTRA]
    OLD_V2_VALIDATE(normalized,frozen)
    p=plan['v2_interrupted_evidence']
    need(p['protocol']['path']==V2_CONFIG and p['protocol']['sha256']==V2_CONFIG_SHA
         and p['RUN']['path']==V2_FOLDER+'/results/RUN.json' and p['RUN']['sha256']==V2_RUN_SHA
         and p['exit']['path']==V2_FOLDER+'/EXIT_RECEIPT.json' and p['exit']['sha256']==V2_EXIT_SHA
         and p['raft_software_diagnostic']['path']==budget_module.DIAGNOSTIC
         and p['raft_software_diagnostic']['sha256']==budget_module.DIAGNOSTIC_SHA,'V3 predecessor identity differs')


def all_bindings(plan):
    rows=OLD_V2_BINDINGS(plan);proof=plan['v2_interrupted_evidence']
    rows.extend(('HF',proof[k]) for k in ('RUN','exit','raft_software_diagnostic'))
    unique={}
    for root,b in rows:
        key=(root,b['path']);item={k:b[k] for k in ('path','bytes','sha256')}
        need(key not in unique or unique[key]==item,'Conflicting asset binding');unique[key]=item
    return [(root,b) for (root,_),b in sorted(unique.items())]


def cpu_expected(plan,digest):
    return {**OLD_V2_CPU(plan,digest),'v2_interrupted_evidence':plan['v2_interrupted_evidence'],'snapshot_repair':REPAIR}


def preflight(args):
    plan=base.read(args.protocol);verify_evidence(plan,args.hf_root)
    return q2.preflight(args)


def run(args):
    hf=args.hf_root.resolve();plan=base.read(args.protocol);verify_evidence(plan,hf)
    out=(hf/plan['output_HF_relative']).resolve()
    budget=budget_module.SnapshotBudget(out,plan,args.protocol,q.RESOURCES['new_HF_bytes'])
    original=Path.rglob
    def fast_rglob(path,pattern):
        if path.resolve()!=out or pattern!='*':return original(path,pattern)
        if budget.initial is None:
            if not (budget.snapshot/'protocol.json').is_file():return original(path,pattern)
            budget.initialize()
        return iter([*budget.mutable_files(),budget_module._SnapshotBytes(budget.initial['bytes'])])
    Path.rglob=fast_rglob;completed=False
    try:
        q2.run(args);completed=True
        audit=budget.final();path=out/'RUN.json';report=base.read(path)
        report.update(snapshot_audit=budget.initial,output_budget_audit=audit,
                      snapshot_repair=REPAIR,v2_interrupted_evidence=plan['v2_interrupted_evidence'])
        base.write(path,report);budget.final()
    except BaseException as error:
        path=out/'RUN.json'
        if path.is_file():
            report=base.read(path)
            if completed:report['status']='FAILED_V3_OUTPUT_AUDIT_PRESERVED_NO_RETRY'
            report.update(v3_error_type=type(error).__name__,v3_error=str(error));base.write(path,report)
        raise
    finally:Path.rglob=original


def verify_closed(args):
    hf=args.hf_root.resolve();plan=base.read(args.protocol);verify_evidence(plan,hf);report=base.read(args.run)
    need(report.get('snapshot_repair')==REPAIR and report.get('v2_interrupted_evidence')==plan['v2_interrupted_evidence'], 'V3 lineage missing')
    budget=budget_module.SnapshotBudget(hf/plan['output_HF_relative'],plan,args.protocol,q.RESOURCES['new_HF_bytes'])
    need(budget.initialize()==report['snapshot_audit'],'Snapshot audit differs');now=budget.final();prior=report['output_budget_audit']
    need(prior['snapshot_bytes']==now['snapshot_bytes'] and prior['limit_bytes']==now['limit_bytes']
         and prior['total_bytes']<=prior['limit_bytes'] and now['total_bytes']<=now['limit_bytes'],'Output budget closure differs')
    return q2.verify_closed(args)


q2.identity,q2.output=identity,output
q.identity,q.validate,q.all_bindings,q.cpu_expected=identity,validate,all_bindings,cpu_expected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('draft','inspect','preflight','run','verify-closed'))
    p.add_argument('--shard',type=int,choices=range(4));p.add_argument('--protocol',type=Path)
    p.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'));p.add_argument('--out',type=Path)
    p.add_argument('--run',type=Path);p.add_argument('--exit',type=Path);a=p.parse_args()
    if a.action=='draft':
        need(a.shard is not None and a.out is not None and not a.out.exists(),'Fresh shard draft required')
        a.out.parent.mkdir(parents=True,exist_ok=True);base.write(a.out,draft(a.hf_root,a.shard))
    elif a.action=='inspect':
        plan=base.read(a.protocol);validate(plan);print(json.dumps({'id':plan['id'],'cpu_affinity':plan['cpu_affinity'],
            'limits':q.LIMITS,'resources':q.RESOURCES},indent=2))
    elif a.action=='preflight':preflight(a)
    elif a.action=='run':run(a)
    else:print(json.dumps(verify_closed(a),indent=2))

if __name__=='__main__':main()
