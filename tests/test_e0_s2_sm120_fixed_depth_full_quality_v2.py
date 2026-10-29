import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full_quality_v2_test', ROOT / 'tools/e0_s2_sm120_fixed_depth_full_quality_v2.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class V2Tests(unittest.TestCase):
    def test_per_shard_identity_and_affinity(self):
        self.assertEqual([m.cpus(i) for i in range(4)], [[16,17],[18,19],[20,21],[22,23]])
        self.assertEqual(len({m.identity(i) for i in range(4)}), 4)
        self.assertTrue(all(m.identity(i).endswith('_V2') for i in range(4)))
        with patch.object(m.os, 'sched_getaffinity', return_value={16,17}):
            self.assertEqual(m.check_affinity({'shard':0}), [16,17])
            with self.assertRaises(ValueError): m.check_affinity({'shard':1})

    def test_failure_receipt_actual_bound_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'repo'; hf = Path(tmp)/'hf'; root.mkdir(); hf.mkdir()
            config = root/'v1.json'; receipt = hf/'exit.json'; stderr = hf/'stderr.log'
            config.write_text(json.dumps({'id':'E0_S2_SM120_FIXED_DEPTH_EIGHT_AXIS_20261029_SHARD0','status':'FROZEN_REGISTERED'}))
            config_sha = m.base.sha(config)
            receipt.write_text(json.dumps({'id':'E0_S2_SM120_FIXED_DEPTH_EIGHT_AXIS_20261029_SHARD0',
                'protocol_sha256':config_sha,'process_exited':True,'child_returncode':1,'run_sha256':None,
                'timeout':False,'automatic_retry':False,'outer_seconds':0.03}))
            stderr.write_bytes(b'taskset: affinity failed')
            proof = {'protocol':m.base.binding(config, root),'exit':m.base.binding(receipt,hf),'stderr':m.base.binding(stderr,hf)}
            patches = [patch.object(m,'ROOT',root), patch.object(m,'V1_CONFIG','v1.json'), patch.object(m,'V1_CONFIG_SHA',config_sha),
                       patch.object(m,'V1_EXIT','exit.json'), patch.object(m,'V1_STDERR','stderr.log'),
                       patch.object(m,'EXIT_SHA',m.base.sha(receipt)),patch.object(m,'STDERR_SHA',m.base.sha(stderr))]
            with patches[0],patches[1],patches[2],patches[3],patches[4],patches[5],patches[6]:
                self.assertEqual(m.verify_failure({'v1_launch_failure':proof},hf),proof)
                stderr.write_bytes(b'taskset: tampered')
                with self.assertRaises(ValueError):m.verify_failure({'v1_launch_failure':proof},hf)

    def test_validate_normalizes_only_old_output_and_sources(self):
        def b(path): return {'path':path,'bytes':1,'sha256':'a'*64}
        plan = {'shard':0,'id':m.identity(0),'cpu_affinity':[16,17],'output_HF_relative':m.output(0),
                'launch_repair':copy.deepcopy(m.REPAIR), 'repo_sources':[b('base')]+[b(p) for p in sorted(m.EXTRA_PATHS)],
                'v1_launch_failure':{'protocol':{'path':m.V1_CONFIG,'sha256':m.V1_CONFIG_SHA},
                                     'exit':{'path':m.V1_EXIT,'sha256':m.EXIT_SHA},
                                     'stderr':{'path':m.V1_STDERR,'sha256':m.STDERR_SHA}}}
        with patch.object(m,'OLD_VALIDATE') as old:
            m.validate(plan,True)
            normalized, frozen = old.call_args.args
            self.assertTrue(frozen)
            self.assertEqual([b['path'] for b in normalized['repo_sources']], ['base'])
            self.assertIn('/e0_s2_sm120_fixed_depth_full_quality_20261029/', normalized['output_HF_relative'])
            self.assertEqual(plan['output_HF_relative'],m.output(0))
            broken = copy.deepcopy(plan); broken['cpu_affinity']=[24,25]
            with self.assertRaises(ValueError):m.validate(broken,True)

    def test_v2_assets_in_cpu_proof(self):
        def b(path): return {'path':path,'bytes':1,'sha256':'a'*64}
        plan = {'v1_launch_failure':{'exit':b('exit'),'stderr':b('stderr')}}
        with patch.object(m,'OLD_BINDINGS',return_value=[('REPO',b('v1_config'))]):
            rows = m.all_bindings(plan)
        self.assertEqual({(r,b['path']) for r,b in rows}, {('REPO','v1_config'),('HF','exit'),('HF','stderr')})

    def test_run_receipt_contains_actual_affinity_and_failure_binding(self):
        from types import SimpleNamespace
        plan = {'id':m.identity(0), 'shard':0, 'v1_launch_failure':{'identity':'proof'}}
        seen = []
        def underlying(args):
            m.base.write(Path('/tmp/RUN.json'), {'id':plan['id'], 'status':'STARTED'})
        original = m.base.write
        with patch.object(m.base, 'read', return_value=plan), patch.object(m, 'verify_failure'), \
             patch.object(m.os, 'sched_getaffinity', return_value={16,17}), \
             patch.object(m.base, 'write', side_effect=lambda path, value: seen.append(copy.deepcopy(value))), \
             patch.object(m.q, 'run', side_effect=underlying):
            m.run(SimpleNamespace(protocol=Path('p'),hf_root=Path('/tmp')))
        self.assertEqual(seen[0]['actual_CPU_affinity'],[16,17])
        self.assertEqual(seen[0]['v1_launch_failure'],plan['v1_launch_failure'])
        self.assertIs(m.base.write,original)

    def test_preflight_and_run_fail_before_execution_on_affinity_mismatch(self):
        from types import SimpleNamespace
        args = SimpleNamespace(protocol=Path('unused'),hf_root=Path('/tmp'))
        with patch.object(m.base,'read',return_value={'shard':0}),patch.object(m,'verify_failure'), \
             patch.object(m.os,'sched_getaffinity',return_value={24,25}), patch.object(m.q,'preflight') as pre, patch.object(m.q,'run') as run:
            with self.assertRaises(ValueError):m.preflight(args)
            with self.assertRaises(ValueError):m.run(args)
            pre.assert_not_called();run.assert_not_called()

if __name__ == '__main__': unittest.main()
