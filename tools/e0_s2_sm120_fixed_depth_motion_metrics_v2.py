"""V2 RAFT: exact immutable source snapshot accounting without repeated walks.

V1's supervisor-terminated runs are evidence, never retried or overwritten.
This isolated adapter retains the original numerical producer and budgets.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('_sm120_raft_v1_for_v2', ROOT / 'tools/e0_s2_sm120_fixed_depth_motion_metrics.py')
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
base = q.base; need = base.require
OLD_DRAFT, OLD_VALIDATE, OLD_BINDINGS, OLD_CPU_EXPECTED = q.draft, q.validate, q.all_bindings, q.cpu_expected
SELF = 'tools/e0_s2_sm120_fixed_depth_motion_metrics_v2.py'
TEST = 'tests/test_e0_s2_sm120_fixed_depth_motion_metrics_v2.py'
V1_CONFIGS = tuple('config/e0_s2_sm120_fixed_depth_motion_shard{}_20261029_v1.json'.format(i) for i in range(3))
V1_CONFIG_SHAS = ('fdfbb17502dcfc1d65eb7ae30ba7aedf8ddb2e858f14e429d1e5aade1a2b22e7',
                  '2c9ec23d3282fe498c0840626f33fea483ec805544427241a8f81a6fa13acd9f',
                  '12a032ca9dc7d94f12bb80678d89d8fac4deebaba92eea921995e52d3853fd63')
V1_WORKER = 'ffnds-derived/e0_s2_sm120_fixed_depth_motion_metrics_20261029/shard{}/attempt1/worker'
DIAGNOSTIC = 'ffnds-derived/e0_s2_sm120_raft_metadata_scan_diagnostic_cpu_20261029/DIAGNOSTIC.json'
DIAGNOSTIC_SHA = 'a5a1462fc697e165a577fc3ef0c5733d0ade4e6b8262f1a9fcc5f05e999e20bf'
EXTRA = {SELF, TEST, *V1_CONFIGS}
REPAIR = {'reason':'V1_LEDGER_RECURSIVE_IMMUTABLE_SNAPSHOT_WALK_COORDINATOR_STOP',
          'shards_0_1_2':'ACTUAL_SUPERVISOR_EXIT_MINUS15', 'shard_3':'NO_V1_GPU_EXECUTION',
          'method':'ONCE_HASH_IMMUTABLE_SNAPSHOT_EACH_PERSIST_WALK_MUTABLE_ONLY_FINAL_FULL_AUDIT',
          'algorithm_changed':False,'automatic_retry':False}


def identity(shard):
    need(type(shard) is int and shard in range(4), 'Invalid shard')
    return 'E0_S2_SM120_FIXED_DEPTH_CONTINUOUS_RAFT_20261029_SHARD{}_V2'.format(shard)


def output(shard):
    return 'ffnds-derived/e0_s2_sm120_fixed_depth_motion_metrics_v2_20261029/shard{}/attempt1/worker/results'.format(shard)


def evidence(hf):
    proofs = []
    for part in range(3):
        worker = Path(hf) / V1_WORKER.format(part); run = worker / 'results/RUN.json'
        proofs.append({'protocol':base.binding(ROOT / V1_CONFIGS[part],ROOT),
                       'exit':base.binding(worker / 'EXIT_RECEIPT.json',hf),
                       'RUN':base.binding(run,hf) if run.is_file() else None})
    return {'diagnostic':base.binding(Path(hf)/DIAGNOSTIC,hf), 'terminated_V1':proofs}


def verify_evidence(plan,hf):
    e = plan['repair_evidence']; d = e['diagnostic']
    need(d['path']==DIAGNOSTIC and d['sha256']==DIAGNOSTIC_SHA, 'Wrong coordinator diagnostic binding')
    diag = base.read_bound(hf,d)
    need(diag['status']=='SOFTWARE_DIAGNOSTIC_COORDINATOR_STOP' and diag['observed_snapshot_and_output_files']==1007
         and diag['observed_output_bytes']==23669555 and diag['one_full_output_walk_wall_seconds']==8.650316627928987,
         'Coordinator software diagnostic differs')
    need(len(e['terminated_V1'])==3, 'Three actual stopped scopes required')
    for i,proof in enumerate(e['terminated_V1']):
        need(proof['protocol']['path']==V1_CONFIGS[i] and proof['protocol']['sha256']==V1_CONFIG_SHAS[i]
             and proof['exit']['path']==V1_WORKER.format(i)+'/EXIT_RECEIPT.json', 'Wrong prior identity')
        old = base.read_bound(ROOT,proof['protocol']); ex = base.read_bound(hf,proof['exit'])
        need(old['id']=='E0_S2_SM120_FIXED_DEPTH_CONTINUOUS_RAFT_20261029_SHARD{}'.format(i)
             and ex['id']==old['id'] and ex['protocol_sha256']==V1_CONFIG_SHAS[i]
             and ex['process_exited'] is True and type(ex['child_returncode']) is int and ex['child_returncode']==-15
             and ex['timeout'] is False and ex['automatic_retry'] is False, 'Prior actual SIGTERM exit differs')
        if proof['RUN'] is None:
            need(ex['run_sha256'] is None, 'Missing prior RUN binding')
        else:
            need(proof['RUN']['path']==V1_WORKER.format(i)+'/results/RUN.json'
                 and ex['run_sha256']==proof['RUN']['sha256'], 'Prior actual RUN identity differs')
            r = base.read_bound(hf,proof['RUN'])
            need(r['id']==old['id'] and r['protocol_sha256']==V1_CONFIG_SHAS[i], 'Prior partial RUN identity differs')
    return e


def draft(hf,shard):
    plan = OLD_DRAFT(hf,shard)
    plan['repair_evidence']=evidence(hf);verify_evidence(plan,hf)
    paths={b['path'] for b in plan['repo_sources']} | EXTRA
    plan.update(id=identity(shard), output_HF_relative=output(shard), software_repair=copy.deepcopy(REPAIR),
                repo_sources=[base.binding(ROOT/p,ROOT) for p in sorted(paths)])
    return plan


def validate(plan,frozen=False):
    shard=plan['shard'];need(plan['id']==identity(shard) and plan['output_HF_relative']==output(shard)
        and plan['software_repair']==REPAIR, 'V2 identity/output/repair differs')
    paths=[b['path'] for b in plan['repo_sources']]
    need(len(paths)==len(set(paths)) and EXTRA <= set(paths), 'V2 source closure incomplete')
    normalized=copy.deepcopy(plan)
    normalized['output_HF_relative']='ffnds-derived/e0_s2_sm120_fixed_depth_motion_metrics_20261029/shard{}/attempt1/worker/results'.format(shard)
    normalized['repo_sources']=[b for b in normalized['repo_sources'] if b['path'] not in EXTRA]
    OLD_VALIDATE(normalized,frozen)
    proof=plan['repair_evidence']
    need(proof['diagnostic']['path']==DIAGNOSTIC and proof['diagnostic']['sha256']==DIAGNOSTIC_SHA
         and len(proof['terminated_V1'])==3, 'V2 proof closure differs')
    for i,p in enumerate(proof['terminated_V1']):
        need(p['protocol']['path']==V1_CONFIGS[i] and p['protocol']['sha256']==V1_CONFIG_SHAS[i]
             and p['exit']['path']==V1_WORKER.format(i)+'/EXIT_RECEIPT.json', 'Prior scope identity differs')


def all_bindings(plan):
    rows=OLD_BINDINGS(plan);e=plan['repair_evidence'];rows.append(('HF',e['diagnostic']))
    for p in e['terminated_V1']:
        rows.append(('HF',p['exit']))
        if p['RUN'] is not None:rows.append(('HF',p['RUN']))
    unique={}
    for root,b in rows:
        key=(root,b['path']);item={k:b[k] for k in ('path','bytes','sha256')}
        need(key not in unique or unique[key]==item,'Conflicting bound asset')
        unique[key]=item
    return [(r,b) for (r,_),b in sorted(unique.items())]


def cpu_expected(plan,digest):
    return {**OLD_CPU_EXPECTED(plan,digest),'repair_evidence':plan['repair_evidence'],'software_repair':REPAIR}


class SnapshotBudget:
    """Snapshot verified once, hot accounting excludes precisely that directory."""
    def __init__(self,out,plan,protocol,limit):
        self.out=Path(out).resolve();self.snapshot=self.out/'EXECUTED_SOURCES';self.limit=limit
        self.expected={b['path']:{k:b[k] for k in ('bytes','sha256')} for b in plan['repo_sources']}
        self.expected['protocol.json']={'bytes':Path(protocol).stat().st_size,'sha256':base.sha(protocol)}
        self.initial=None
    def snapshot_audit(self):
        need(self.snapshot.is_dir() and not self.snapshot.is_symlink(), 'Snapshot missing or symlink')
        files={}
        for directory,dirs,names in os.walk(self.snapshot,topdown=True,followlinks=False):
            root=Path(directory)
            need(all(not (root/d).is_symlink() for d in dirs), 'Snapshot directory symlink')
            for name in names:
                path=root/name;need(path.is_file() and not path.is_symlink(), 'Snapshot nonregular file')
                rel=path.relative_to(self.snapshot).as_posix()
                files[rel]={'bytes':path.stat().st_size,'sha256':base.sha(path)}
        need(files==self.expected, 'Snapshot file inventory or bytes/SHA changed')
        return {'file_count':len(files),'bytes':sum(x['bytes'] for x in files.values()),
                'inventory_sha256':base.state_sha(files)}
    def initialize(self):
        need(self.initial is None, 'Snapshot already initialized')
        self.initial=self.snapshot_audit();return self.initial
    def mutable_files(self):
        result=[]
        for directory,dirs,names in os.walk(self.out,topdown=True,followlinks=False):
            root=Path(directory)
            if root==self.out:
                dirs[:]=[x for x in dirs if x!='EXECUTED_SOURCES']
            for name in names:
                path=root/name
                if path.is_file() and not path.is_symlink():result.append(path)
        return result
    def total(self):
        need(self.initial is not None, 'Snapshot not initialized')
        mutable=sum(p.stat().st_size for p in self.mutable_files())
        total=mutable+self.initial['bytes'];need(total<=self.limit,'Output byte budget exceeded')
        return {'snapshot_bytes':self.initial['bytes'],'mutable_bytes':mutable,'total_bytes':total,'limit_bytes':self.limit}
    def final(self):
        need(self.snapshot_audit()==self.initial, 'Immutable snapshot changed')
        return self.total()


class _SnapshotBytes:
    def __init__(self,size):self.size=size
    def is_file(self):return True
    def is_symlink(self):return False
    def stat(self):return SimpleNamespace(st_size=self.size)


def preflight(args):
    plan=base.read(args.protocol);verify_evidence(plan,args.hf_root)
    return q.preflight(args)


def run(args):
    hf=args.hf_root.resolve();plan=base.read(args.protocol);verify_evidence(plan,hf)
    out=(hf/plan['output_HF_relative']).resolve();budget=SnapshotBudget(out,plan,args.protocol,q.RESOURCES['new_HF_bytes'])
    original=Path.rglob
    def fast_rglob(path,pattern):
        if path.resolve()!=out or pattern!='*':return original(path,pattern)
        if budget.initial is None:
            if not (budget.snapshot/'protocol.json').is_file():return original(path,pattern)
            budget.initialize()
        return iter([*budget.mutable_files(),_SnapshotBytes(budget.initial['bytes'])])
    Path.rglob=fast_rglob
    completed=False
    try:
        q.run(args)
        completed=True
        final=budget.final();path=out/'RUN.json';report=base.read(path)
        report.update(snapshot_audit=budget.initial,output_budget_audit=final,
                      software_repair=REPAIR,repair_evidence=plan['repair_evidence'])
        base.write(path,report)
        # Include the updated final report itself in the budget.
        checked=budget.final();need(checked['total_bytes']<=q.RESOURCES['new_HF_bytes'], 'Final output budget exceeded')
    except BaseException as error:
        path=out/'RUN.json'
        if path.is_file():
            report=base.read(path)
            if completed: report['status']='FAILED_V2_OUTPUT_AUDIT_PRESERVED_NO_RETRY'
            report.update(v2_error_type=type(error).__name__,v2_error=str(error));base.write(path,report)
        raise
    finally:Path.rglob=original


def verify_closed(args):
    hf=args.hf_root.resolve();plan=base.read(args.protocol);verify_evidence(plan,hf)
    report=base.read(args.run)
    need(report.get('software_repair')==REPAIR and report.get('repair_evidence')==plan['repair_evidence'], 'V2 evidence missing')
    budget=SnapshotBudget(hf/plan['output_HF_relative'],plan,args.protocol,q.RESOURCES['new_HF_bytes'])
    need(budget.initialize()==report['snapshot_audit'], 'Snapshot audit differs')
    now=budget.final();previous=report['output_budget_audit']
    need(previous['snapshot_bytes']==now['snapshot_bytes'] and previous['limit_bytes']==now['limit_bytes']
         and previous['total_bytes']<=previous['limit_bytes'] and now['total_bytes']<=now['limit_bytes'], 'Output budget closure differs')
    return q.verify_closed(args)


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
        plan=base.read(a.protocol);validate(plan);print(json.dumps({'id':plan['id'],'limits':q.LIMITS,'resources':q.RESOURCES},indent=2))
    elif a.action=='preflight':preflight(a)
    elif a.action=='run':run(a)
    else:print(json.dumps(verify_closed(a),indent=2))

if __name__=='__main__':main()
