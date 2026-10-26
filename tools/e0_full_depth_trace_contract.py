"""Streaming accounting for a future native teacher/student full-depth trace.

No model runner, operator replacement, CUDA initialization or experiment launcher.
The caller must supply actual same-site tensors and immutable source provenance.
Synthetic software fixtures remain labeled synthetic in all exported records.
"""
import math


def _validate_chunk_rows(chunk_rows):
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError('Positive integer reduction chunk required')


def _require_finite_record(value):
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_record(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_record(item)
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError('Nonfinite observation metadata')


def _double(value):
    if hasattr(value,'detach'):
        import torch
        return value.detach().to(dtype=torch.float64)
    import numpy as np
    return np.asarray(value,dtype=np.float64)


def _sum(value):
    return float(value.sum())


def pair_geometry(reference,candidate,chunk_rows=256):
    _validate_chunk_rows(chunk_rows)
    if reference.shape!=candidate.shape or len(reference.shape)!=2:
        raise ValueError('Paired full-M matrices required')
    norm=delta=dot=0.
    for start in range(0,len(reference),chunk_rows):
        a,b=[_double(v[start:start+chunk_rows]) for v in [reference,candidate]]
        d=b-a;norm+=_sum(a*a);delta+=_sum(d*d);dot+=_sum(a*d)
    if not all(math.isfinite(v) for v in [norm,delta,dot]):raise ValueError('Nonfinite observed state')
    return {'rows':len(reference),'columns':reference.shape[1],'reference_energy':norm,
            'difference_SSE':delta,'reference_difference_dot':dot,
            'relative_SSE':delta/max(norm,1e-30)}


def four_arm_geometry(dense_teacher,approx_teacher,dense_student,approx_student,chunk_rows=256):
    _validate_chunk_rows(chunk_rows)
    arrays=[dense_teacher,approx_teacher,dense_student,approx_student]
    if len(dense_teacher.shape)!=2 or any(v.shape!=dense_teacher.shape for v in arrays):
        raise ValueError('All four observed outputs must share the same full-M shape')
    totals=dict.fromkeys(['reference_energy','teacher_operator_SSE','student_operator_SSE',
                         'prefix_response_SSE','total_SSE','operator_prefix_dot','recomposition_SSE'],0.)
    for start in range(0,len(dense_teacher),chunk_rows):
        a,b,c,d=[_double(v[start:start+chunk_rows]) for v in arrays]
        teacher=b-a;op=d-c;shift=c-a;total=d-a;closure=total-op-shift
        for name,value in [('reference_energy',a*a),('teacher_operator_SSE',teacher*teacher),
                           ('student_operator_SSE',op*op),('prefix_response_SSE',shift*shift),
                           ('total_SSE',total*total),('operator_prefix_dot',op*shift),
                           ('recomposition_SSE',closure*closure)]:totals[name]+=_sum(value)
    if not all(math.isfinite(v) for v in totals.values()):raise ValueError('Nonfinite observed outputs')
    scale=max(totals['total_SSE'],totals['student_operator_SSE'],totals['prefix_response_SSE'],1e-30)
    gap=totals['total_SSE']-(totals['student_operator_SSE']+totals['prefix_response_SSE']+2*totals['operator_prefix_dot'])
    if abs(gap)>1e-9*scale or totals['recomposition_SSE']>1e-18*scale:
        raise ValueError('Signed decomposition fails numerical closure')
    denominator=math.sqrt(totals['student_operator_SSE']*totals['prefix_response_SSE'])
    return {**totals,'rows':len(dense_teacher),'columns':dense_teacher.shape[1],
            'signed_energy_identity_gap':gap,
            'operator_prefix_cosine':totals['operator_prefix_dot']/denominator if denominator else None,
            'operator_relative_to_dense_teacher':totals['student_operator_SSE']/max(totals['reference_energy'],1e-30),
            'total_relative_to_dense_teacher':totals['total_SSE']/max(totals['reference_energy'],1e-30)}


def step_pre_geometry(teacher_before,student_before,teacher_after,student_after,chunk_rows=256):
    """Four actual states immediately before/after each native step_pre."""
    _validate_chunk_rows(chunk_rows)
    arrays=[teacher_before,student_before,teacher_after,student_after]
    if len(teacher_before.shape)<1 or any(v.shape!=teacher_before.shape for v in arrays):
        raise ValueError('Same-shaped states on both sides of step_pre required')
    totals=dict.fromkeys(['delta_before_SSE','delta_after_SSE','teacher_cast_SSE',
                         'student_cast_SSE','cast_difference_SSE','delta_cast_dot'],0.)
    for start in range(0,len(teacher_before),chunk_rows):
        a,b,c,d=[_double(v[start:start+chunk_rows]) for v in arrays]
        delta=b-a;teacher_cast=c-a;student_cast=d-b;change=student_cast-teacher_cast
        for key,value in [('delta_before_SSE',delta*delta),('delta_after_SSE',(d-c)**2),
                          ('teacher_cast_SSE',teacher_cast**2),('student_cast_SSE',student_cast**2),
                          ('cast_difference_SSE',change**2),('delta_cast_dot',delta*change)]:
            totals[key]+=_sum(value)
    _require_finite_record(totals)
    return {**totals,'shape':list(teacher_before.shape),'boundary':'before_step_pre_to_after_step_pre',
            'dtypes':{name:str(v.dtype) for name,v in zip(
                ['teacher_before','student_before','teacher_after','student_after'],arrays)}}


def scheduler_geometry(teacher_before,student_before,teacher_flow,student_flow,
                       teacher_after,student_after,*,sigma,sigma_next,chunk_rows=256):
    """before is AFTER step_pre; after is AFTER step_post. Entry cast is separate."""
    _validate_chunk_rows(chunk_rows)
    h=sigma-sigma_next
    arrays=[teacher_before,student_before,teacher_flow,student_flow,teacher_after,student_after]
    if (not all(math.isfinite(v) for v in [sigma,sigma_next,h]) or sigma_next<0 or h<=0
            or len(teacher_before.shape)<1 or any(v.shape!=teacher_before.shape for v in arrays)):
        raise ValueError('Same-shaped state/flow tensors and positive actual sigma gap required')
    totals=dict.fromkeys(['delta_before_SSE','flow_difference_SSE','delta_flow_dot',
                         'delta_after_SSE','ideal_update_defect_SSE','ideal_after_SSE'],0.)
    for start in range(0,len(teacher_before),chunk_rows):
        a,b,c,d,e,f=[_double(v[start:start+chunk_rows]) for v in arrays]
        delta=b-a;flow=d-c;after=f-e;ideal=delta-h*flow;defect=after-ideal
        for key,value in [('delta_before_SSE',delta*delta),('flow_difference_SSE',flow*flow),
                          ('delta_flow_dot',delta*flow),('delta_after_SSE',after*after),
                          ('ideal_update_defect_SSE',defect*defect),('ideal_after_SSE',ideal*ideal)]:
            totals[key]+=_sum(value)
    if not all(math.isfinite(v) for v in totals.values()):raise ValueError('Nonfinite scheduler state')
    return {**totals,'h':h,'sigma':sigma,'sigma_next':sigma_next,'shape':list(teacher_before.shape),
        'boundary':'after_step_pre_to_after_step_post','dtypes':{name:str(v.dtype) for name,v in zip(
        ['teacher_before','student_before','teacher_flow','student_flow','teacher_after','student_after'],arrays)},
        'ideal_energy_change':h*h*totals['flow_difference_SSE']-2*h*totals['delta_flow_dot'],
        'actual_energy_change':totals['delta_after_SSE']-totals['delta_before_SSE'],
        'interpretation':'Defect includes actual step_post ordering and output cast. step_pre cast is a separate record; the ideal equation is not native bitwise identity.'}


class FullDepthTrace:
    def __init__(self,shape,source_index,seed,provenance,*,state_shape,synthetic=False):
        if len(shape)!=2 or min(shape)<=0:raise ValueError('Declare full-M shape')
        if (not state_shape or min(state_shape)<=0 or
                (not synthetic and len(state_shape)!=4)):
            raise ValueError('Declare complete native C,F,H,W latent shape, not sampled state panels')
        required=['protocol_sha256','native_source_sha256','model_digest','bank_RUN_sha256']
        if not all(provenance.get(k) for k in required):raise ValueError('Missing execution provenance')
        self.shape=tuple(shape);self.source=source_index;self.seed=seed;self.provenance=provenance
        self.synthetic=synthetic;self.sites={};self.steps={}
        self.state_shape=tuple(state_shape)

    def add_site(self,step,block,expert,*,teacher_q1,student_q1,ffn_outputs,
                 native_writebacks,teacher_block,student_block,numerical_check):
        key=(step,block)
        if key in self.sites or step not in range(4) or block not in range(40):
            raise ValueError('Duplicate or unexpected full-depth site')
        if expert!=('high_noise_model' if step<2 else 'low_noise_model'):
            raise ValueError('Wrong expert for step')
        arms=('dense_teacher','approx_teacher','dense_student','approx_student')
        values=[teacher_q1,student_q1,teacher_block,student_block]
        if set(ffn_outputs)!=set(arms) or set(native_writebacks)!=set(arms):
            raise ValueError('Four actual native outputs AND four actual writebacks required')
        values+=list(ffn_outputs.values())+list(native_writebacks.values())
        if any(tuple(v.shape)!=self.shape for v in values):
            raise ValueError('Sample panels cannot certify full-M scalar coverage')
        if not numerical_check or numerical_check.get('status')!='MEASURED_SAME_INPUT':
            raise ValueError('Native-versus-continuous same-input check missing')
        if (not numerical_check.get('coefficient_provenance_sha256') or
                not numerical_check.get('row_ids') or
                not all(k in numerical_check for k in ['teacher_native_vs_continuous','student_native_vs_continuous'])):
            raise ValueError('Numerical comparison must bind its coefficients and explicit rows')
        row_ids=numerical_check['row_ids']
        _require_finite_record(numerical_check)
        if len(set(row_ids))!=len(row_ids) or any(i not in range(self.shape[0]) for i in row_ids):
            raise ValueError('Invalid numerical comparison row coordinates')
        self.sites[key]={'step':step,'block':block,'expert':expert,'scalar_coverage':'FULL_M',
            'input_boundary':'actual_post_Q1','shape':list(self.shape),
            'input_geometry':pair_geometry(teacher_q1,student_q1),
            'FFN':four_arm_geometry(*(ffn_outputs[k] for k in arms)),
            'native_writeback':four_arm_geometry(*(native_writebacks[k] for k in arms)),
            'block_output':pair_geometry(teacher_block,student_block),
            'numerical_check':numerical_check,
            'warning':'Four actual W(rD,yDD,gD), W(rD,ySD,gD), W(rS,yDS,gS), W(rS,ySS,gS) writebacks. The prefix term includes residual-base and gate changes; it is not pure FFN-input shift.'}

    def add_step(self,step,geometry,*,entry_cast,boundary_hashes,pre_dtype_teacher,post_dtype_teacher,pre_dtype_student,post_dtype_student):
        _require_finite_record(geometry)
        _require_finite_record(entry_cast)
        if step not in range(4) or step in self.steps:raise ValueError('Repeated or invalid denoising step')
        if not all([pre_dtype_teacher,post_dtype_teacher,pre_dtype_student,post_dtype_student]):
            raise ValueError('Actual step_pre and step_post dtypes required')
        if not {'delta_before_SSE','delta_after_SSE','delta_flow_dot','ideal_update_defect_SSE'}<=geometry.keys():
            raise ValueError('Missing signed full-state scheduler geometry')
        if (geometry.get('boundary')!='after_step_pre_to_after_step_post' or
                entry_cast.get('boundary')!='before_step_pre_to_after_step_pre' or
                entry_cast.get('shape')!=geometry.get('shape') or
                tuple(geometry.get('shape',[]))!=self.state_shape):
            raise ValueError('Distinct step_pre cast and step_post update boundaries required')
        required_hashes={f'{branch}_{boundary}' for branch in ['teacher','student']
                         for boundary in ['before_step_pre','after_step_pre','after_step_post']}
        if set(boundary_hashes)!=required_hashes or not all(boundary_hashes.values()):
            raise ValueError('All six actual scheduler boundary hashes required')
        if not self.synthetic and any(len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
                                      for v in boundary_hashes.values()):
            raise ValueError('Native boundary hashes must be SHA256 digests')
        if (entry_cast['dtypes']['teacher_after']!=pre_dtype_teacher or
                entry_cast['dtypes']['student_after']!=pre_dtype_student or
                geometry['dtypes']['teacher_before']!=pre_dtype_teacher or
                geometry['dtypes']['student_before']!=pre_dtype_student or
                geometry['dtypes']['teacher_after']!=post_dtype_teacher or
                geometry['dtypes']['student_after']!=post_dtype_student):
            raise ValueError('Inconsistent native scheduler dtype boundaries')
        if not math.isclose(entry_cast['delta_after_SSE'],geometry['delta_before_SSE'],rel_tol=1e-9,abs_tol=1e-20):
            raise ValueError('Entry-cast and update states are not the same observed boundary')
        self.steps[step]={'step':step,**geometry,'entry_cast':entry_cast,'boundary_hashes':boundary_hashes,'observed_dtypes':{
            'teacher_after_step_pre':pre_dtype_teacher,'teacher_after_step_post':post_dtype_teacher,
            'student_after_step_pre':pre_dtype_student,'student_after_step_post':post_dtype_student}}

    def finish(self):
        if set(self.sites)!={(s,b) for s in range(4) for b in range(40)} or set(self.steps)!=set(range(4)):
            raise ValueError('Missing all40x4 native paired sites or four signed scheduler records')
        for step in range(1,4):
            for branch in ['teacher','student']:
                if (self.steps[step-1]['boundary_hashes'][branch+'_after_step_post']!=
                        self.steps[step]['boundary_hashes'][branch+'_before_step_pre']):
                    raise ValueError('Latent continuity broken between steps')
        return {'schema':'ffnds.full_depth.paired_trace.v1',
                'status':'SYNTHETIC_SOFTWARE_VALIDATION_ONLY' if self.synthetic else 'NATIVE_VECTOR_ACCOUNTING_COMPLETE_NOT_QUALITY_ACCEPTANCE',
                'source_index':self.source,'seed':self.seed,'provenance':self.provenance,
                'full_state_shape':list(self.state_shape),
                'sites':[self.sites[k] for k in sorted(self.sites)],
                'steps':[self.steps[k] for k in sorted(self.steps)],'quality_accepted':False,
                'causal_attribution':'NOT_IDENTIFIED_BY_DUAL_TRAJECTORIES; hybrid interventions separately required'}
