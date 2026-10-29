"""Per-block fixed W2 policy for the original Wan double-buffer offload path.

The caller changes policy only between complete phases, with no live NativeTap.
Prefetch routing is based on its own block index, independently of compute.
Torch is deliberately not imported here. This module is not an authorization or
an executed-model proof.
"""
from threading import Lock
import types

ORIGINAL_W2_BYTES = 141568000
EARLY = frozenset(range(10))
LATE = frozenset(range(30, 40))
_ALLOWED = (frozenset(), EARLY, LATE)


def need(value, message):
    if not value:
        raise ValueError(message)


class FixedLayerW2Runtime:
    """Phase-scoped W2 dispatch with per-buffer, single-use readiness markers.

    Same constructor and `set_mode` as FactorW2Runtime. In dense mode all
    blocks use original W2. In factor mode `dense_blocks` selects the original
    W2 blocks; the remaining blocks use the caller's registered factor bank.
    The mask is the same at all four steps, including prefetch across steps.
    `is_factor(block)` exposes this same immutable policy for transfer geometry.
    """
    def __init__(self, pipe, guard, bank, *, factor_forward, meter, synchronize,
                 record_stream, emit=lambda event: None):
        self.pipe, self.guard, self.bank = pipe, guard, bank
        self.factor_forward, self.meter = factor_forward, meter
        self.synchronize, self.record_stream, self.emit = synchronize, record_stream, emit
        self.mode = 'dense'
        self.dense_blocks = frozenset()
        self.markers = {}
        self.lock = Lock()
        self.saved = []
        self.objects = {}
        models = pipe.runner.model.model
        need(len(models) == 2, 'Two original experts required')
        for expert, model in enumerate(models):
            buffers = model.transformer_infer.offload_manager.cuda_buffers
            need(len(buffers) == 2, 'Two original CUDA buffers per expert required')
            for buffer in buffers:
                mm = buffer.compute_phases[2].ffn_2
                need(id(mm) not in self.objects and type(mm).__name__ == 'MMWeight'
                     and not mm.has_lora_branch and not mm.has_diff
                     and not hasattr(mm, 'weight_diff') and not hasattr(mm, 'bias_diff'),
                     'Unsupported W2 object/adapters')
                self.objects[id(mm)] = (mm, expert)
        need(len(self.objects) == 4, 'Four distinct W2 buffers required')

    def is_factor(self, block):
        need(type(block) is int and 0 <= block < 40, 'Invalid block')
        with self.lock:
            return self.mode == 'factor' and block not in self.dense_blocks

    def _set_policy(self, *, mode=None, dense_blocks=None):
        need(self.guard.active is None, 'Policy may change only at a phase boundary')
        self.synchronize()
        with self.lock:
            if mode is not None:
                self.mode = mode
            if dense_blocks is not None:
                self.dense_blocks = dense_blocks
            self.markers.clear()
        self.emit({'event': 'fixed_w2_policy', 'mode': self.mode,
                   'dense_blocks': sorted(self.dense_blocks),
                   'original_cuda_buffer_allocation_retained': True})

    def set_mode(self, mode):
        need(mode in ('dense', 'factor'), 'Unsupported runtime mode')
        self._set_policy(mode=mode)

    def set_dense_blocks(self, blocks):
        try:
            values = list(blocks)
        except TypeError as error:
            raise ValueError('Dense block policy must be iterable') from error
        need(all(type(b) is int and 0 <= b < 40 for b in values)
             and len(values) == len(set(values)), 'Invalid or duplicate dense blocks')
        policy = frozenset(values)
        need(policy in _ALLOWED, 'Dense blocks must be empty, 0..9, or 30..39')
        self._set_policy(dense_blocks=policy)

    def _install(self, obj, name, fn):
        present = name in obj.__dict__
        existing = obj.__dict__.get(name)
        self.saved.append((obj, name, present, existing))
        setattr(obj, name, types.MethodType(fn, obj))

    def __enter__(self):
        need(not self.saved and self.guard.active is None, 'Fresh phase-boundary context required')
        self.synchronize()
        try:
            for mm, expert in self.objects.values():
                old_apply, old_load, runtime = mm.apply, mm.load_state_dict, self

                def load(obj, destination, block_index, adapter_block_index=None,
                         *, _load=old_load, _expert=expert):
                    with runtime.lock:
                        runtime.markers.pop(id(obj), None)
                    need(type(block_index) is int and 0 <= block_index < 40
                         and adapter_block_index is None, 'Unsupported W2 prefetch block')
                    prefix = f'blocks.{block_index}.ffn.2.'
                    w, b = destination.get(prefix+'weight'), destination.get(prefix+'bias')
                    need(w is not None and b is not None and tuple(w.shape) == (13824, 5120)
                         and tuple(b.shape) == (5120,) and w.element_size() == b.element_size() == 2,
                         'Original W2 source shape/size differs')
                    with runtime.lock:
                        factor = runtime.mode == 'factor' and block_index not in runtime.dense_blocks
                    kind = 'factor' if factor else 'dense'
                    if factor:
                        runtime.emit({'event': 'original_W2_H2D_skipped', 'expert': _expert,
                                      'block': block_index, 'bytes': ORIGINAL_W2_BYTES})
                        result = None
                    else:
                        result = _load(destination, block_index, adapter_block_index)
                    with runtime.lock:
                        runtime.markers[id(obj)] = (_expert, block_index, kind)
                    return result

                def apply(obj, value, *, _apply=old_apply, _expert=expert):
                    active = runtime.guard.active
                    need(active is not None and active.block is not None, 'W2 outside native compute block')
                    expert_now, block = active.block
                    need(type(expert_now) is int and expert_now == _expert
                         and type(block) is int and 0 <= block < 40, 'Wrong expert/block')
                    step = active.step(expert_now)
                    need(type(step) is int and 0 <= step < 4
                         and expert_now == (0 if step < 2 else 1), 'Wrong expert/step')
                    with runtime.lock:
                        factor = runtime.mode == 'factor' and block not in runtime.dense_blocks
                        kind = 'factor' if factor else 'dense'
                        need(runtime.markers.get(id(obj)) == (_expert, block, kind),
                             'Current compute buffer expert/block/kind was not loaded')
                        del runtime.markers[id(obj)]
                    if not factor:
                        # old_apply is already wrapped by OriginalGuard; do not double meter it.
                        result = _apply(value)
                        runtime.emit({'event': 'fixed_W2_dispatch_returned', 'kind': 'dense',
                                      'expert': _expert, 'step': step, 'block': block})
                        return result
                    factors = runtime.bank[_expert, step, block]
                    left, right, bias = (factors[k] for k in ('left.weight', 'right.weight', 'right.bias'))
                    need(len(value.shape) == 2 and value.shape[1] == 13824
                         and len(left.shape) == 2 and left.shape[1] == 13824
                         and len(right.shape) == 2 and right.shape[1] == left.shape[0]
                         and right.shape[0] == 5120 and tuple(bias.shape) == (5120,),
                         'Factor dimensions differ')
                    for tensor in (left, right, bias):
                        runtime.record_stream(tensor)
                    meta = {'input': list(value.shape), 'call_label': 'factor_W2_two_GEMMs',
                            'factor_left': list(left.shape), 'factor_right': list(right.shape),
                            'bias': list(bias.shape), 'original_full_W2_loaded': False}
                    def factor_callback(_mm, input_value):
                        return runtime.factor_forward(input_value, factors, runtime.meter)
                    result = active.mm_call(obj, value, meta, factor_callback)
                    runtime.emit({'event': 'fixed_W2_dispatch_returned', 'kind': 'factor',
                                  'expert': _expert, 'step': step, 'block': block})
                    return result

                self._install(mm, 'load_state_dict', load)
                self._install(mm, 'apply', apply)
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        for obj, name, present, value in reversed(self.saved):
            if present:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        self.saved.clear()
        with self.lock:
            self.markers.clear()

    def __exit__(self, *_):
        try:
            self.synchronize()
        finally:
            self._restore()
            with self.lock:
                self.mode = 'dense'
