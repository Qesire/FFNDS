"""Versioned LR2 runtime: replace exactly10 residuals, preserving frozen UB1 main."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch

from ar2_checkpoint import canonical_json, digest
from evaluate_residual_rr1 import open_verified
from run_ar2_restart import tensor_sha


def expected_rank(step, block, identity_control=False):
    if step not in range(4) or block not in range(40):
        raise ValueError('Native position outside the160-position contract')
    return 512 if not identity_control and step == 3 and block >= 30 else 256


def validate_policy(policy):
    if (policy['kind'] != 'LATE_BLOCK_RANK_RESIDUAL_REFERENCE'
            or type(policy['identity_control']) is not bool
            or policy['main_groups_by_step'] != [54,108,108,162]
            or policy['changed_positions'] != [[3,b] for b in range(30,40)]
            or policy['default_rank'] != 256 or policy['changed_rank'] != 512):
        raise RuntimeError('Only the frozen LR2 last10 schedule or its identity control is supported')


def create_runtime(cache_path, policy_path):
    from step_budget_runtime import create_runtime as parent_runtime
    policy = json.loads(Path(policy_path).read_text())
    validate_policy(policy)
    for name,sha in policy['runtime_sources'].items():
        if digest(name) != sha:
            raise RuntimeError('Frozen LR2 numerical runtime source changed')
    parent = policy['parent_policy']
    if digest(parent['path']) != parent['sha256']:
        raise RuntimeError('Frozen UB1 parent policy changed')
    runtime = parent_runtime(cache_path,parent['path'])
    if (runtime.policy['identity_control'] or runtime.policy['groups_by_step'] != policy['main_groups_by_step']
            or runtime.policy['cache_manifest'] != policy['cache_manifest']):
        raise RuntimeError('LR2 must preserve the exact active UB1 main branch')
    binding = policy['fit_report']
    if digest(binding['path']) != binding['sha256']:
        raise RuntimeError('LR2 full fit changed')
    fit,report,identity = open_verified(Path(binding['path']),'COMPLETED_LR2_TEN_LAST_STEP_RESIDUAL_FIT')
    cv = policy['CV_report']
    protocol = policy['protocol']
    if (fit.fingerprint != binding['fingerprint'] or len(report['positions']) != 10
            or identity['UB1_policy_sha256'] != parent['sha256']
            or identity['CV_report_sha256'] != cv['sha256'] or digest(cv['path']) != cv['sha256']
            or identity['protocol_sha256'] != protocol['sha256'] or digest(protocol['path']) != protocol['sha256']
            or identity['blocks'] != list(range(30,40)) or identity['rank'] != 512 or identity['step'] != 3):
        raise RuntimeError('Exact source-bound LR2 fit and passed CV required')
    cvreport = json.loads(Path(cv['path']).read_text())
    if (cvreport['status'] != 'COMPLETED_LR2_NATIVE_RISK_GATE_PASSED'
            or len(cvreport['CV']) != 4 or not all(v['gate_pass'] for v in cvreport['CV'])):
        raise RuntimeError('All4 LR2 native-risk checks must have passed')
    for name,sha in identity['sources'].items():
        if digest(name) != sha:
            raise RuntimeError('Frozen full-fit numerical producer changed')
    changed = []
    for block in range(30,40):
        part = fit.load('positions',f'step3_block{block:02d}')
        if part is None or part['metadata']['step'] != 3 or part['metadata']['block'] != block:
            raise RuntimeError('Missing or incorrect LR2 changed position')
        parameters = {k:torch.from_numpy(v).bfloat16() for k,v in part['arrays'].items()}
        shapes = {'first':(512,5120),'second':(5120,512),'bias_first':(512,),'bias_second':(5120,)}
        if (set(parameters) != set(shapes) or any(tuple(parameters[k].shape) != shape for k,shape in shapes.items())
                or any(not bool(torch.isfinite(v).all()) for v in parameters.values())
                or {k:tensor_sha(v) for k,v in parameters.items()} != part['metadata']['parameter_sha256']):
            raise RuntimeError('LR2 native BF16 parameter shapes/bits changed')
        if not policy['identity_control']:
            runtime.residual_parameters[3,block] = parameters
        changed.append({'step':3,'block':block,'parameter_sha256':part['metadata']['parameter_sha256']})
    old_apply,old_report = runtime.apply,runtime.report
    runtime.policy,runtime.policy_sha = policy,digest(policy_path)
    runtime.residual_fingerprint = hashlib.sha256(canonical_json({'parent':runtime.residual_fingerprint,
        'LR2_fit':fit.fingerprint,'policy':runtime.policy_sha}).encode()).hexdigest()

    def apply(self,mm_weight,input_tensor,*,expert,step_index,block_index,module_name):
        result = old_apply(mm_weight,input_tensor,expert=expert,step_index=step_index,
                           block_index=block_index,module_name=module_name)
        if module_name.endswith('.ffn.0'):
            actual = self.residual_parameters[step_index,block_index]['first'].shape[0]
            rank = expected_rank(step_index,block_index,policy['identity_control'])
            if actual != rank:
                raise RuntimeError('Actual residual operand differs from the frozen LR2 schedule')
            row = self.residual_coverage[f'step{step_index}_block{block_index}']
            row['auxiliary_GEMM_shapes'] = [[actual,5120],[5120,actual]]
            row['auxiliary_MACs_over_original_dense_FFN'] = actual/13824
        return result

    def report_runtime():
        result = old_report()
        packed = result['packed_reference']
        packed.update(source_sha256=digest(Path(__file__)),parent_runtime_source_sha256=digest(Path(__file__).with_name('step_budget_runtime.py')))
        packed['late_block_rank'] = {'identity_control':policy['identity_control'],'fit_fingerprint':fit.fingerprint,
            'changed_positions':changed,'default_rank':256,'changed_rank':512,
            'mean_rank':256 if policy['identity_control'] else 272,
            'scope':'static last10 final-step residual capacity; no dynamic support or inherited acceptance'}
        coverage = packed['residual_correction']['coverage']
        if len(coverage) == 160:
            rows = sum(v['rows'] for v in coverage.values())
            packed['late_block_rank']['actual_row_weighted_auxiliary_MAC_density'] = sum(
                v['rows']*v['auxiliary_GEMM_shapes'][0][0] for v in coverage.values())/(rows*13824)
        return result

    runtime.apply = MethodType(apply,runtime)
    runtime.report = report_runtime
    return runtime
