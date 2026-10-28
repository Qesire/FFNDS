"""v4: initialize pinned stateless activation defaults under existing import guards."""
import importlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = 'E0_COMPACT6016_CONSUMER00_12_QUALITY_IMPORT_ORDER_V4_20261028'
V3_ID = 'E0_COMPACT6016_CONSUMER00_12_QUALITY_DIRFD_V3_20261028'
V3_RUN_SHA = 'c1cedd1d41ebb6e15ac49ac88c24eb6187da635315140da095058ac861be65e9'
POLICY = {'module':'transformers.activations','ordered_defaults':8,'before':0,'after':8,
          'timing':'imports-only first metric module, after original guards installed',
          'native_scoring_order_unchanged':True}

spec = importlib.util.spec_from_file_location('_ffnds_quality_v3_for_v4',ROOT/'tools/e0_compact6016_consumer_quality_v3.py')
v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)
base = v3.base
v3.ID = base.ID = ID
base.REQUIRED_SOURCES = base.REQUIRED_SOURCES | {'tools/e0_compact6016_consumer_quality_v4.py','tests/test_e0_compact6016_consumer_quality_v4.py'}
_original_metric_module = base.original_metric_module
_current = None


def bootstrap(result, site, activation, importer=importlib.import_module):
    if 'activation_bootstrap' in result:
        return
    events=result.get('stateless_activation_wrappers', [])
    base.require(len(events)==0 and result.get('predebited_official_module_imports')==1
                 and result.get('completed_official_module_imports')==0, 'Unexpected bootstrap order')
    expected=(Path(site)/activation['site_relative']).resolve()
    base.require(base.sha(expected)==activation['source_sha256'], 'Activation source SHA differs')
    module=importer('transformers.activations')
    base.require(Path(module.__file__).resolve()==expected and base.sha(expected)==activation['source_sha256'], 'Activation origin differs')
    expected_events=[dict(item,module='transformers.activations',source_sha256=activation['source_sha256']) for item in activation['instances']]
    base.require(len(expected_events)==8 and events==expected_events, 'Ordered activation defaults differ')
    result['activation_bootstrap']={'module':'transformers.activations','origin':str(expected),'sha256':activation['source_sha256'],'before':0,'after':8}


class EnvironmentProxy:
    def __init__(self,inner,v8,site): self.inner,self.v8,self.site=inner,v8,site
    def __getattr__(self,name): return getattr(self.inner,name)
    def import_audit_guard(self,directory,result):
        global _current
        value=self.inner.import_audit_guard(directory,result)
        if 'predebited_official_module_imports' in result:
            _current=(result,self.site,self.v8['stateless_activation_allowlist'])
        return value


def verify_environment(plan,hf):
    helper,v8=v3.verify_environment(plan,hf)
    return EnvironmentProxy(helper,v8,Path(hf)/plan['environment']['site']),v8


def metric_module(dimension,plan,hf):
    global _current
    if _current is not None:
        result,site,activation=_current
        bootstrap(result,site,activation)
        _current=None
    return _original_metric_module(dimension,plan,hf)


def verify_failed_v3(plan,hf):
    prev=plan['failed_import_predecessor_v3']; run=base.read_bound(hf,prev['RUN']); exit=base.read_bound(hf,prev['exit'])
    base.require(prev['RUN']['sha256']==V3_RUN_SHA and run['id']==exit['id']==V3_ID
      and run['status']=='FAILED_IMPORTS_ONLY_PRESERVED_NO_RETRY' and exit['child_returncode']==1
      and exit['process_exited'] is True and exit['timeout'] is False and exit['run_sha256']==V3_RUN_SHA
      and run['protocol_sha256']==exit['protocol_sha256']==prev['protocol']['sha256']
      and run['denied_attempts']==['POOL_OUTSIDE_EXACT_GLOBAL'] and run['stateless_activation_wrappers']==[]
      and run['predebited_official_module_imports']==1 and run['completed_official_module_imports']==0
      and run['functions']==[] and run['CUDA_initialized'] is False
      and all(run.get(k)==0 for k in ('GPU_calls','checkpoint_loads','model_constructions','parameter_creations','buffer_creations','model_forwards','official_scores')),
      'V3 actual failed import identity differs')
    protocol=base.read_bound(ROOT,prev['protocol']); base.require(protocol['id']==V3_ID,'V3 protocol differs')
    return prev


def validate_plan(plan,*,frozen=False):
    v3.validate_plan(plan,frozen=frozen)
    base.require(plan.get('activation_bootstrap_policy')==POLICY and plan.get('failed_import_predecessor_v3',{}).get('RUN',{}).get('sha256')==V3_RUN_SHA,'V4 policy differs')


def verify_assets(plan,hf,*,repo=ROOT):
    answer=v3.verify_assets(plan,hf,repo=repo); verify_failed_v3(plan,hf); return answer


def extension(plan):
    return {'activation_bootstrap_policy_sha256':base.state_sha(POLICY),'failed_import_v3_RUN_sha256':V3_RUN_SHA}


def cpu_report(plan,digest): return {**v3.cpu_report(plan,digest),**extension(plan)}
def import_report(plan,digest): return {**v3.import_report(plan,digest),**extension(plan)}


def require_import_receipt(scope,plan,digest,hf):
    proof=v3.require_import_receipt(scope,plan,digest,hf); report=base.read_bound(hf,proof['RUN'])
    _,v8=v3.verify_environment(plan,hf)
    activation=v8['stateless_activation_allowlist']
    expected=(Path(hf)/plan['environment']['site']/activation['site_relative']).resolve()
    record=report.get('activation_bootstrap',{})
    base.require(record.get('module')=='transformers.activations' and record.get('before')==0 and record.get('after')==8
      and record.get('origin')==str(expected) and record.get('sha256')==activation['source_sha256'] and len(report.get('stateless_activation_wrappers',[]))==8
      and all(x.get('source_sha256')==record['sha256'] for x in report['stateless_activation_wrappers']), 'Actual activation bootstrap proof absent')
    return proof


def make_draft(hf,*,repo=ROOT):
    plan=v3.make_draft(hf,repo=repo); worker=Path(hf)/'ffnds-derived/e0_compact6016_quality_isolated_cpu_20261028/v3/imports/worker'
    plan.update(id=ID,status='DRAFT_IMPORT_ORDER_V4',activation_bootstrap_policy=POLICY,
      failed_import_predecessor_v3={'protocol':base.binding(Path(repo)/'config/e0_compact6016_consumer_quality_20261028_v3.json',repo),
      'RUN':base.binding(worker/'imports/RUN.json',hf),'exit':base.binding(worker/'EXIT_RECEIPT.json',hf)},
      output_HF_relative='ffnds-derived/e0_compact6016_consumer00_12_quality_import_order_v4_20261028/attempt1/results')
    plan['repo_sources']=[base.binding(Path(repo)/x,repo) for x in sorted(base.REQUIRED_SOURCES)]
    plan['pending_freeze']='Independent review and new registered CPU byte and import proofs; preserve v1/v2/v3 failures.'
    return plan

base.validate_plan=validate_plan
base.verify_assets=verify_assets
base.verify_environment=verify_environment
base.cpu_report=cpu_report
base.import_report=import_report
base.require_import_receipt=require_import_receipt
base.original_metric_module=metric_module
base.make_draft=make_draft

if __name__=='__main__': base.main()
