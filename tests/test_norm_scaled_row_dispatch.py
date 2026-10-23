"""Synthetic hook chronology and restoration; no Torch, Triton or model."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from norm_scaled_row_dispatch import NormScaledEnergy


class Vec:
    def __init__(self, values): self.values = list(values)
    def numel(self): return len(self.values)
    def reshape(self, *shape): return self
    def square(self): return Vec([x * x for x in self.values])
    def __mul__(self, other): return Vec([a * b for a, b in zip(self.values, other.values)])


class ScaledEnergyHooks(unittest.TestCase):
    def fixture(self, mode, fail=False):
        modules = {}
        name = 'lightx2v.models.networks.wan.infer.transformer_infer'
        for i in range(1, len(name.split('.')) + 1):
            k = '.'.join(name.split('.')[:i]); modules[k] = types.ModuleType(k); modules[k].__path__ = []
        kernel = modules['row_dispatch_kernels'] = types.ModuleType('row_dispatch_kernels')
        kernel.gated_energy = lambda residual, gate: Vec([2., 5.])
        dispatch = types.SimpleNamespace(arm='dynamic_residual' if mode == 'scaled' else 'UB1', events=[])
        norm = types.SimpleNamespace(eps=1e-6, apply=lambda x: x,
            _get_actual_weight=lambda: None, _get_actual_bias=lambda: None)
        phase = types.SimpleNamespace(norm2=norm)
        infer = types.SimpleNamespace(scheduler=types.SimpleNamespace(step_index=1), block_idx=4,
                                     infer_dtype='BF16', sensitive_layer_dtype='BF16')
        def original(infer, phase, x, attention, shift, scale, gate):
            phase.norm2.apply(x)
            if fail: raise ValueError('synthetic failure')
            score = kernel.gated_energy(types.SimpleNamespace(shape=(2, 5120)), gate) if mode == 'scaled' else x
            dispatch.events.append({'step': 1, 'block': 4, 'pre_W1_signal': 'fixed_or_old'})
            return score
        cls = type('WanTransformerInfer', (), {'infer_ffn': original})
        modules[name].WanTransformerInfer = cls
        helper = lambda value, **kw: (value, None, Vec([.5, 2.]))
        return modules, kernel, dispatch, phase, infer, cls, helper

    def test_actual_new_information_changes_the_old_score(self):
        modules, kernel, dispatch, phase, infer, cls, helper = self.fixture('scaled')
        previous_ffn, previous_norm, previous_energy = cls.infer_ffn, phase.norm2.apply, kernel.gated_energy
        with patch.dict(sys.modules, modules), patch('norm_scaled_row_dispatch.layer_norm_with_statistics', helper):
            with NormScaledEnergy(dispatch) as observer:
                score = cls.infer_ffn(infer, phase, Vec([0., 0.]), None, None, None, None)
                self.assertEqual(score.values, [.5, 20.])
                self.assertEqual(observer.events[0]['scaled_score_calls'], 1)
                self.assertEqual(observer.events[0]['CPU_tensor_transfers'], 0)
                self.assertEqual(dispatch.events[0]['pre_W1_signal'], 'gated_R256_energy_times_current_norm2_rstd_squared')
        self.assertIs(cls.infer_ffn, previous_ffn); self.assertIs(phase.norm2.apply, previous_norm)
        self.assertIs(kernel.gated_energy, previous_energy)

    def test_observation_leaves_baseline_value_and_score_function_unchanged(self):
        modules, kernel, dispatch, phase, infer, cls, helper = self.fixture('observe_UB1')
        original_energy = kernel.gated_energy; value = Vec([3., 7.])
        with patch.dict(sys.modules, modules), patch('norm_scaled_row_dispatch.layer_norm_with_statistics', helper):
            with NormScaledEnergy(dispatch, 'observe_UB1') as observer:
                self.assertIs(cls.infer_ffn(infer, phase, value, None, None, None, None), value)
                self.assertIs(kernel.gated_energy, original_energy)
                self.assertEqual(observer.events[0]['scaled_score_calls'], 0)

    def test_exception_restores_hooks_and_active_state(self):
        modules, kernel, dispatch, phase, infer, cls, helper = self.fixture('scaled', fail=True)
        old_ffn, old_norm, old_energy = cls.infer_ffn, phase.norm2.apply, kernel.gated_energy
        with patch.dict(sys.modules, modules), patch('norm_scaled_row_dispatch.layer_norm_with_statistics', helper):
            with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                with NormScaledEnergy(dispatch) as observer:
                    cls.infer_ffn(infer, phase, Vec([1., 2.]), None, None, None, None)
        self.assertIsNone(observer.active); self.assertIs(cls.infer_ffn, old_ffn)
        self.assertIs(phase.norm2.apply, old_norm); self.assertIs(kernel.gated_energy, old_energy)

    def test_mismatched_arm_is_rejected(self):
        with self.assertRaises(ValueError): NormScaledEnergy(types.SimpleNamespace(arm='UB1'))


if __name__ == '__main__': unittest.main()
