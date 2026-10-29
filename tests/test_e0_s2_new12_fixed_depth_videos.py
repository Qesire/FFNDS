import importlib.util
from pathlib import Path
import unittest

R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('s2_fixed_depth',R/'tools/e0_s2_new12_fixed_depth_videos.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class FixedDepthPlan(unittest.TestCase):
    def test_equal_compute_different_prefetch_cost(self):
        early=m.expected(m.MODES[1]);late=m.expected(m.MODES[2])
        self.assertEqual(early['dense_compute_sites'],late['dense_compute_sites'])
        self.assertEqual(early['dense_compute_sites'],40)
        self.assertEqual(early['factor_GEMMs'],240)
        self.assertEqual((early['original_W2_prefetches'],late['original_W2_prefetches']),(42,40))
        self.assertEqual((early['original_W2_prefetches']-late['original_W2_prefetches'])*m.vm.ORIGINAL_W2_BYTES,283136000)

    def test_counterbalanced_orders_and_exact_native_totals(self):
        orders=[m.order({'index':i}) for i in range(0,12,2)]
        self.assertEqual([o[0] for o in orders],list(m.MODES)*2)
        records=[m.expected(m.MODES[0])]+[m.expected(mode) for _ in range(6) for mode in m.MODES]
        limits=m.limits(0)
        self.assertEqual(sum(r['factor_GEMMs'] for r in records),limits['BF16_factor_GEMM'])
        self.assertEqual(sum(r['original_W2_skips'] for r in records),limits['original_W2_H2D_skips'])
        self.assertEqual(sum(r['dense_compute_sites'] for r in records),480)
        self.assertEqual(sum(r['original_W2_prefetches']+r['original_W2_skips'] for r in records),limits['prefetch_transfer_observations'])
        self.assertEqual(limits['native_full_model_flows'],76)
        self.assertEqual(limits['native_FFN_calls'],3040)
        self.assertEqual(limits['state_output_writes'],190)

    def test_transfer_observer_uses_call_local_policy(self):
        from types import SimpleNamespace
        original=m._original_geometry;calls=[]
        class Runtime:
            mode='factor';objects={42:('mm',0)}
            def is_factor(self,block):return block>=10
        rt=Runtime()
        try:
            def observe(buf,dest,block,view):
                calls.append((block,view.mode,view.objects));return view.mode
            m._original_geometry=observe
            self.assertEqual(m.transfer_geometry(None,None,9,rt),'dense')
            self.assertEqual(m.transfer_geometry(None,None,10,rt),'factor')
            self.assertEqual(rt.mode,'factor')
            self.assertIs(calls[0][2],rt.objects)
        finally:m._original_geometry=original

    def test_actual_parent_registration_is_independent_of_runtime_globals(self):
        hf=Path('/home/xiang/.cache/huggingface')
        master=hf/'ffnds-derived/e0_s2_causal_additional_idle_a100_env_master_cpu_v5_20261029/attempt1/worker/EXIT_RECEIPT.json'
        if not master.exists():self.skipTest('Closed parent assets unavailable')
        previous=(m.vm.MODES,m.vm.phase_order,m.vm.run_id)
        try:
            m.vm.MODES=('wrong_legacy_mode',)
            m.vm.phase_order=lambda case:('wrong_legacy_mode',)
            m.vm.run_id=lambda case,mode:'colliding_legacy_id'
            for part in (0,1):
                plan=m.draft(part,hf);refs=m.references(plan,hf)
                manifest=m.registration(plan,hf,refs)
                self.assertEqual(len(manifest['sources']),6)
                self.assertEqual(len(manifest['samples']),24)
                self.assertEqual(len({s['run_id'] for s in manifest['samples']}),24)
                for source in manifest['sources']:
                    jobs=[s for s in manifest['samples'] if s['source_id']==source['source_id']]
                    self.assertEqual({s['variant'] for s in jobs},{'dense',*m.MODES})
                    self.assertEqual(len({s['initial_noise_id'] for s in jobs}),1)
        finally:m.vm.MODES,m.vm.phase_order,m.vm.run_id=previous


if __name__=='__main__':unittest.main()
