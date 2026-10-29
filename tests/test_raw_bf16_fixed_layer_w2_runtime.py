import unittest
from types import SimpleNamespace as NS
from tools.raw_bf16_fixed_layer_w2_runtime import FixedLayerW2Runtime, ORIGINAL_W2_BYTES, EARLY, LATE


class Tensor:
    def __init__(self, shape): self.shape = shape
    def element_size(self): return 2


class MMWeight:
    has_lora_branch = False
    has_diff = False
    def __init__(self): self.loads, self.calls, self.fail = [], [], False
    def apply(self, value): self.calls.append(value); return 'dense'
    def load_state_dict(self, destination, block, adapter_block_index=None):
        self.loads.append(block)
        if self.fail: raise RuntimeError('load failed')
        return 'loaded'


class Active:
    def __init__(self, expert, step, block): self.block = expert, block; self.step_number = step; self.calls = []
    def step(self, expert): return self.step_number
    def mm_call(self, mm, value, meta, original):
        self.calls.append(meta)
        return original(mm, value)


def destination(block):
    return {f'blocks.{block}.ffn.2.weight': Tensor((13824, 5120)),
            f'blocks.{block}.ffn.2.bias': Tensor((5120,))}


class FixedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.mm = [MMWeight() for _ in range(4)]
        models = []
        for expert in range(2):
            buffers = [NS(compute_phases=[None, None, NS(ffn_2=self.mm[expert*2+j])]) for j in range(2)]
            models.append(NS(transformer_infer=NS(offload_manager=NS(cuda_buffers=buffers))))
        pipe = NS(runner=NS(model=NS(model=models)))
        self.guard = NS(active=None)
        self.factors = {'left.weight': Tensor((4, 13824)), 'right.weight': Tensor((5120, 4)), 'right.bias': Tensor((5120,))}
        self.bank = {(expert, step, block): self.factors for expert, steps in ((0, (0, 1)), (1, (2, 3)))
                     for step in steps for block in range(40)}
        self.sync, self.records, self.events, self.forward = [], [], [], []
        def forward(value, factors, meter):
            self.forward.append((value, factors, meter)); return 'factor'
        self.runtime = FixedLayerW2Runtime(pipe, self.guard, self.bank, factor_forward=forward, meter='meter',
            synchronize=lambda: self.sync.append(True), record_stream=self.records.append, emit=self.events.append)
        self.value = Tensor((1, 13824))

    def active(self, expert, step, block): self.guard.active = Active(expert, step, block)
    def load(self, obj, block): return obj.load_state_dict(destination(block), block)

    def test_early_dense_prefetch_of_factor_is_independent_of_active_block(self):
        with self.runtime:
            self.runtime.set_dense_blocks(EARLY); self.runtime.set_mode('factor')
            self.active(0, 0, 9)
            self.assertEqual(self.load(self.mm[0], 9), 'loaded')
            self.assertIsNone(self.load(self.mm[1], 10))
            self.assertEqual(self.mm[1].loads, [])
            self.assertEqual(self.mm[0].apply(self.value), 'dense')
            self.guard.active.block = (0, 10)
            self.assertEqual(self.mm[1].apply(self.value), 'factor')
            self.assertEqual(len(self.records), 3)
            self.assertEqual(len(self.guard.active.calls), 1)
            self.assertEqual(self.guard.active.calls[0]['call_label'], 'factor_W2_two_GEMMs')
            self.assertEqual([x['bytes'] for x in self.events if x['event'] == 'original_W2_H2D_skipped'], [ORIGINAL_W2_BYTES])

    def test_late_boundary_wraparound_and_expert_step(self):
        with self.runtime:
            self.runtime.set_dense_blocks(LATE); self.runtime.set_mode('factor')
            self.active(1, 2, 29)
            self.load(self.mm[2], 29); self.load(self.mm[3], 30)
            self.assertEqual(self.mm[2].apply(self.value), 'factor')
            self.guard.active.block = (1, 30)
            self.assertEqual(self.mm[3].apply(self.value), 'dense')
            self.guard.active.block = (1, 39)
            self.load(self.mm[2], 39); self.load(self.mm[3], 0)
            self.assertEqual(self.mm[2].apply(self.value), 'dense')
            self.guard.active.block = (1, 0); self.guard.active.step_number = 3
            self.assertEqual(self.mm[3].apply(self.value), 'factor')

    def test_stale_markers_wrong_block_expert_step_and_reuse_rejected(self):
        with self.runtime:
            self.runtime.set_dense_blocks(EARLY); self.runtime.set_mode('factor')
            self.active(0, 0, 1)
            with self.assertRaises(ValueError): self.mm[0].apply(self.value)
            self.load(self.mm[0], 0)
            with self.assertRaises(ValueError): self.mm[0].apply(self.value)
            self.guard.active.block = (1, 0)
            with self.assertRaises(ValueError): self.mm[0].apply(self.value)
            self.guard.active.block = (0, 0); self.guard.active.step_number = 2
            with self.assertRaises(ValueError): self.mm[0].apply(self.value)
            self.guard.active.step_number = 0
            self.assertEqual(self.mm[0].apply(self.value), 'dense')
            with self.assertRaises(ValueError): self.mm[0].apply(self.value)

    def test_failure_invalidates_old_buffer_and_policy_change_syncs(self):
        with self.runtime:
            self.runtime.set_mode('factor'); self.active(0, 0, 11)
            self.load(self.mm[0], 11)
            self.mm[0].fail = True
            self.guard.active = None; before = len(self.sync)
            self.runtime.set_mode('dense'); self.assertEqual(len(self.sync), before+1)
            self.active(0, 0, 11)
            with self.assertRaises(RuntimeError): self.load(self.mm[0], 11)
            with self.assertRaises(ValueError): self.mm[0].apply(self.value)
            with self.assertRaises(ValueError): self.runtime.set_dense_blocks(EARLY)

    def test_dense_factor_policy_geometry_and_invalid_policies(self):
        with self.runtime:
            self.assertFalse(self.runtime.is_factor(39))
            self.runtime.set_dense_blocks(EARLY); self.runtime.set_mode('factor')
            self.assertFalse(self.runtime.is_factor(0)); self.assertFalse(self.runtime.is_factor(9))
            self.assertTrue(self.runtime.is_factor(10)); self.assertTrue(self.runtime.is_factor(39))
            self.runtime.set_dense_blocks(LATE)
            self.assertTrue(self.runtime.is_factor(29)); self.assertFalse(self.runtime.is_factor(30))
            self.runtime.set_dense_blocks([])
            self.assertTrue(all(self.runtime.is_factor(i) for i in range(40)))
            for invalid in ([0], list(range(10))+[0], [True], range(40)):
                with self.assertRaises(ValueError): self.runtime.set_dense_blocks(invalid)
            self.runtime.set_mode('dense'); self.assertFalse(any(self.runtime.is_factor(i) for i in range(40)))

    def test_bad_prefetch_cannot_leave_previous_marker_usable(self):
        with self.runtime:
            self.runtime.set_mode('factor'); self.active(0, 0, 11)
            self.load(self.mm[0], 11)
            with self.assertRaises(ValueError): self.mm[0].load_state_dict({}, 12)
            with self.assertRaises(ValueError): self.mm[0].apply(self.value)

    def test_missing_bank_and_wrong_kind_fail_closed(self):
        with self.runtime:
            self.runtime.set_mode('factor'); self.active(0, 0, 11)
            self.load(self.mm[0], 11)
            self.runtime.markers[id(self.mm[0])] = (0, 11, 'dense')
            with self.assertRaises(ValueError): self.mm[0].apply(self.value)
            self.load(self.mm[0], 11); del self.bank[(0, 0, 11)]
            with self.assertRaises(KeyError): self.mm[0].apply(self.value)
            with self.assertRaises(ValueError): self.mm[0].apply(self.value)

    def test_original_methods_restored_after_exception(self):
        original_apply = self.mm[0].apply.__func__; original_load = self.mm[0].load_state_dict.__func__
        with self.assertRaisesRegex(RuntimeError, 'boom'):
            with self.runtime:
                raise RuntimeError('boom')
        self.assertIs(self.mm[0].apply.__func__, original_apply)
        self.assertIs(self.mm[0].load_state_dict.__func__, original_load)
        self.assertEqual(self.runtime.mode, 'dense'); self.assertEqual(self.runtime.markers, {})


if __name__ == '__main__': unittest.main()
