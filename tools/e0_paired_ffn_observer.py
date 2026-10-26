"""Four native FFN arms at actual caller inputs; no model launcher.

All torch imports are lazy. The authorized driver owns model/source bindings,
exclusive hook access, stream lifetime, immutable bank identity and call budgets.
"""
from contextlib import contextmanager
import math

from e0_full_depth_trace_contract import four_arm_geometry, pair_geometry


def uniform_rows(rows, count):
    if rows < 1 or count < 1:
        raise ValueError('Positive row/sample counts required')
    return sorted({int(i * (rows - 1) / max(min(count, rows) - 1, 1))
                   for i in range(min(count, rows))})


def raster_strata(grid):
    """Nine disjoint frame-third x vertical-third bins; no semantic ROI claim."""
    frames, height, width = map(int, grid)
    if min(frames, height, width) < 3:
        raise ValueError('This frozen stratum design needs at least three frames and rows')
    bins = []
    for t in range(3):
        for y in range(3):
            ids = [((f * height + h) * width + w)
                   for f in range(t * frames // 3, (t + 1) * frames // 3)
                   for h in range(y * height // 3, (y + 1) * height // 3)
                   for w in range(width)]
            bins.append({'frame_third': t, 'vertical_third': y, 'rows': ids})
    return bins


class PairedFFNObserver:
    def __init__(self, runtime, packs, adapter, trace, coefficient_sha256, *,
                 numerical_rows=32, vector_rows=8, save_vectors=None, timing_events=False):
        self.runtime, self.packs, self.adapter, self.trace = runtime, packs, adapter, trace
        self.coefficients = coefficient_sha256
        self.numerical_rows, self.vector_rows, self.save_vectors = numerical_rows, vector_rows, save_vectors
        self.active = self.pending = None
        self.site = {}
        self.timing_events, self.events = timing_events, []
        self.counts = dict.fromkeys(['dense_main_GEMM', 'dense_shadow_GEMM', 'candidate_main_GEMM',
                                    'candidate_shadow_GEMM', 'native_QDQ', 'continuous_GEMM',
                                    'continuous_coeff_cast_bytes', 'completed_sites'], 0)

    def __enter__(self):
        import torch
        import torch.nn.functional as functional
        self.torch, self.functional = torch, functional
        self.base_apply = self.runtime.apply
        self.runtime.apply = self.apply
        return self

    def __exit__(self, *error):
        self.runtime.apply = self.base_apply
        if error[0] is None and (self.active is not None or self.pending is not None or self.site):
            raise RuntimeError('Incomplete paired FFN observer lifetime')

    def _quantize(self, value, scale):
        x = (value.float() / scale.to(device=value.device, dtype=self.torch.float32)).to(value.dtype)
        self.counts['native_QDQ'] += 1
        return self.adapter.native_hif4_activation_qdq(x, force_fp32=True).to(value.dtype)

    def _record(self, expert, block, suffix):
        record = self.runtime._record(expert, f'blocks.{block}.ffn.{suffix}')
        if (record is None or not record.get('activation_qdq', True) or
                not record.get('force_fp32', True) or self.runtime.activation_qdq_override is False):
            raise ValueError('This producer requires the frozen dense HiF4 activation path')
        if not self.torch.is_tensor(record.get('smooth_scale')):
            raise ValueError('Explicit native Smooth scale required')
        return record

    @staticmethod
    def _bias(mm):
        value = mm._get_actual_bias() if hasattr(mm, '_get_actual_bias') else getattr(mm, 'bias', None)
        return value

    def _project(self, value, mm, kind, suffix, *, q1=None, shadow=False):
        branch, primary_kind, expert, step, block = self.active
        torch, F = self.torch, self.functional
        if kind == 'dense':
            record = self._record(expert, block, suffix)
            scale = record['smooth_scale'].reshape(1, -1)
            q = self._quantize(value, scale)
            weight = self.runtime._weight(expert=expert, block_index=block,
                                          module_name=f'blocks.{block}.ffn.{suffix}', record=record,
                                          device=value.device, dtype=value.dtype)
            bias = self._bias(mm)
            if bias is not None:
                bias = bias.to(device=value.device, dtype=value.dtype)
            self.runtime.counters['linear_forward_calls'] += 1
            self.runtime.counters['activation_qdq_calls'] += 1
            y = F.linear(q, weight, bias)
            gemms = 1
        else:
            pack = self.packs[expert, block]
            q = self._quantize(value, pack['s1' if suffix == 0 else 's2'])
            y = F.linear(q, pack['w1' if suffix == 0 else 'v'], pack['b1' if suffix == 0 else 'bias'])
            gemms = 1
            if suffix == 2 and 'matrix' in pack:
                y = y + F.linear(q1, pack['matrix'])
                gemms += 1
        role = kind + ('_shadow_GEMM' if shadow else '_main_GEMM')
        self.counts[role] += gemms
        return q, y

    def apply(self, mm, value, *, expert, step_index, block_index, module_name):
        if not module_name.endswith(('.ffn.0', '.ffn.2')):
            return self.base_apply(mm, value, expert=expert, step_index=step_index,
                                   block_index=block_index, module_name=module_name)
        if self.active is None or (expert, step_index, block_index) != self.active[2:]:
            raise RuntimeError('FFN target context is missing or belongs to another trajectory')
        kind = self.active[1]
        if module_name.endswith('.ffn.0'):
            if self.pending is not None or value.ndim != 2 or value.dtype != self.torch.bfloat16:
                raise ValueError('Unexpected actual FFN caller boundary')
            q1, y = self._project(value, mm, kind, 0)
            self.pending = (q1, value)
            return y
        if self.pending is None:
            raise RuntimeError('FFN down projection without its up projection')
        q1, u = self.pending
        _, y = self._project(value, mm, kind, 2, q1=q1)
        self.pending = None
        self.completed = {'q1': q1, 'u': u, 'output': y}
        return y

    @contextmanager
    def branch(self, infer, *, branch, kind, expert, step, block):
        if self.active is not None or self.pending is not None or branch in self.site:
            raise RuntimeError('Paired FFN branches must be exclusive and complete')
        if branch not in ['teacher', 'student'] or kind not in ['dense', 'candidate']:
            raise ValueError('Unknown trajectory/operator')
        self.active = (branch, kind, expert, step, block)
        old_ffn, old_post = infer.infer_ffn, infer.post_process
        state = {}

        def ffn(phase, x, attention, shift, scale, gate=None):
            if state:
                raise RuntimeError('Exactly one FFN caller per block/trajectory required')
            if self.timing_events:
                begin, end = [self.torch.cuda.Event(enable_timing=True) for _ in range(2)]
                begin.record()
            y = old_ffn(phase, x, attention, shift, scale, gate)
            if self.pending is not None:
                raise RuntimeError('Native caller did not finish its FFN pair')
            primary = self.completed
            del self.completed
            other = 'candidate' if kind == 'dense' else 'dense'
            q1, up = self._project(primary['u'], phase.ffn_0, other, 0, shadow=True)
            _, shadow = self._project(self.functional.gelu(up, approximate='tanh'), phase.ffn_2,
                                      other, 2, q1=q1, shadow=True)
            if not self.torch.equal(primary['q1'], q1):
                raise RuntimeError('Dense/candidate post-Q1 values differ on the same actual U')
            if self.timing_events:
                end.record()
                self.events.append((step, block, branch, begin, end))
            state.update(q1=primary['q1'], outputs={kind: y, other: shadow}, residual=x, actual_kind=kind)
            if (step, block) == (0, 0):
                state['initial_residual'] = x.clone()
                state['initial_gate'] = gate.clone()
            return y

        def post(x, y, gate, pre=None):
            if 'outputs' not in state or 'writebacks' in state or x is not state['residual']:
                raise RuntimeError('Actual residual and native FFN output boundary mismatch')
            other = 'candidate' if kind == 'dense' else 'dense'
            shadow = old_post(x.clone(), state['outputs'][other], gate, pre)
            actual = old_post(x, y, gate, pre)
            state['writebacks'] = {kind: actual, other: shadow}
            del state['residual']
            return actual

        infer.infer_ffn, infer.post_process = ffn, post
        try:
            yield
            if not {'q1', 'outputs', 'writebacks', 'actual_kind'} <= set(state):
                raise RuntimeError('Missing genuine native FFN or postgate arm')
            self.site[branch] = state
        finally:
            infer.infer_ffn, infer.post_process = old_ffn, old_post
            self.active = None
            self.pending = None

    def _continuous(self, q1, pack):
        F = self.functional
        self.counts['continuous_GEMM'] += 3
        self.counts['continuous_coeff_cast_bytes'] += sum(pack[k].numel() * 4 for k in ['w1', 'b1', 'v', 'bias', 'matrix'])
        hidden = F.gelu(F.linear(q1.float(), pack['w1'].float(), pack['b1'].float()), approximate='tanh')
        return F.linear(hidden / pack['s2'].float(), pack['v'].float(), pack['bias'].float()) + F.linear(q1.float(), pack['matrix'].float())

    def finish_site(self, step, block, expert, teacher_block, student_block, grid):
        if set(self.site) != {'teacher', 'student'} or math.prod(grid) != teacher_block.shape[0]:
            raise RuntimeError('Two complete native prefixes and actual raster grid required')
        a, b = self.site['teacher'], self.site['student']
        if (step, block) == (0, 0) and not all(self.torch.equal(a[key], b[key])
                for key in ['q1', 'initial_residual', 'initial_gate']):
            raise RuntimeError('First native prefix differs before any approximation')
        arms = {'dense_teacher': a['outputs']['dense'], 'approx_teacher': a['outputs']['candidate'],
                'dense_student': b['outputs']['dense'], 'approx_student': b['outputs']['candidate']}
        writes = {'dense_teacher': a['writebacks']['dense'], 'approx_teacher': a['writebacks']['candidate'],
                  'dense_student': b['writebacks']['dense'], 'approx_student': b['writebacks']['candidate']}
        rows = uniform_rows(len(teacher_block), self.numerical_rows)
        check = {'status': 'MEASURED_SAME_INPUT', 'coefficient_provenance_sha256': self.coefficients[expert, block],
                 'row_ids': rows, 'definition': 'Actual full-M native candidate rows versus FP32 GELU/linear with same deployed BF16 coefficients promoted; no Q2/intermediate BF16 casts. Not original pre-BF16 fitted coefficients.'}
        for name, state in [('teacher', a), ('student', b)]:
            continuous = self._continuous(state['q1'][rows], self.packs[expert, block])
            check[name + '_native_vs_continuous'] = pair_geometry(continuous, state['outputs']['candidate'][rows])
        self.trace.add_site(step, block, expert, teacher_q1=a['q1'], student_q1=b['q1'],
                            ffn_outputs=arms, native_writebacks=writes,
                            teacher_block=teacher_block, student_block=student_block, numerical_check=check)
        record = self.trace.sites[step, block]
        record['actual_trajectory_operators'] = {'teacher': a['actual_kind'], 'student': b['actual_kind']}
        record['raster_grid'] = list(grid)
        record['spatiotemporal_strata'] = []
        for stratum in raster_strata(grid):
            ids = stratum.pop('rows')
            record['spatiotemporal_strata'].append({**stratum, 'row_count': len(ids),
                'FFN': four_arm_geometry(*(v[ids] for v in arms.values())),
                'native_writeback': four_arm_geometry(*(v[ids] for v in writes.values()))})
        if self.save_vectors is not None:
            ids = uniform_rows(len(teacher_block), self.vector_rows)
            values = {'teacher_q1': a['q1'][ids], 'student_q1': b['q1'][ids],
                      **{'FFN_' + k: v[ids] for k, v in arms.items()},
                      **{'writeback_' + k: v[ids] for k, v in writes.items()}}
            record['sampled_vectors'] = self.save_vectors(step, block, ids, values)
        self.site.clear()
        self.counts['completed_sites'] += 1

    def timing_report(self):
        return [{'step': s, 'block': b, 'branch': role, 'primary_plus_shadow_ms': start.elapsed_time(end)}
                for s, b, role, start, end in self.events]
