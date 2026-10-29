import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('quality_v3_test',ROOT/'tools/e0_s2_sm120_fixed_depth_full_quality_v3.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class V3Tests(unittest.TestCase):
    def test_scopes_keep_full_v1_budget_and_v2_affinity(self):
        self.assertEqual(m.q.LIMITS['official_dimension_calls'],8)
        self.assertEqual(m.q.LIMITS['video_dimension_results'],96)
        self.assertEqual(m.q.LIMITS['RAFT_forwards'],480)
        self.assertEqual(m.q.LIMITS['RAFT_iterations'],9600)
        self.assertEqual([m.q2.cpus(i) for i in range(4)],[[16,17],[18,19],[20,21],[22,23]])
        self.assertEqual(len({m.identity(i) for i in range(4)}),4)
        self.assertTrue(all(m.identity(i).endswith('_V3') for i in range(4)))
        self.assertIn('FULLY_CHARGED',m.REPAIR['V3_calls'])

    def test_v2_repair_evidence_rejects_changed_sha(self):
        proof={'protocol':{'path':m.V2_CONFIG,'sha256':m.V2_CONFIG_SHA},
          'RUN':{'path':m.V2_FOLDER+'/results/RUN.json','sha256':m.V2_RUN_SHA},
          'exit':{'path':m.V2_FOLDER+'/EXIT_RECEIPT.json','sha256':m.V2_EXIT_SHA},
          'raft_software_diagnostic':{'path':m.budget_module.DIAGNOSTIC,'sha256':m.budget_module.DIAGNOSTIC_SHA}}
        changed=copy.deepcopy(proof);changed['RUN']['sha256']='a'*64
        with self.assertRaises(ValueError):m.verify_evidence({'v2_interrupted_evidence':changed},Path('/tmp'))

    def test_validate_passes_only_v2_closure_to_original_validator(self):
        def b(p):return {'path':p,'bytes':1,'sha256':'a'*64}
        proof={'protocol':{'path':m.V2_CONFIG,'sha256':m.V2_CONFIG_SHA},
          'RUN':{'path':m.V2_FOLDER+'/results/RUN.json','sha256':m.V2_RUN_SHA},
          'exit':{'path':m.V2_FOLDER+'/EXIT_RECEIPT.json','sha256':m.V2_EXIT_SHA},
          'raft_software_diagnostic':{'path':m.budget_module.DIAGNOSTIC,'sha256':m.budget_module.DIAGNOSTIC_SHA}}
        plan={'shard':0,'id':m.identity(0),'output_HF_relative':m.output(0),'snapshot_repair':m.REPAIR,
              'repo_sources':[b('v2-source')]+[b(p) for p in sorted(m.EXTRA)],'v2_interrupted_evidence':proof}
        with patch.object(m,'OLD_V2_VALIDATE') as old:
            m.validate(plan,True)
            self.assertEqual([v['path'] for v in old.call_args.args[0]['repo_sources']],['v2-source'])
            altered=copy.deepcopy(plan);altered['repo_sources'].pop()
            with self.assertRaises(ValueError):m.validate(altered,True)

    def test_hot_walk_skips_snapshot_and_final_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';snap=out/'EXECUTED_SOURCES';snap.mkdir(parents=True)
            src=snap/'source.py';src.write_bytes(b'code');protocol=root/'protocol.json';protocol.write_bytes(b'protocol')
            (snap/'protocol.json').write_bytes(b'protocol');(out/'RUN.json').write_bytes(b'run')
            plan={'repo_sources':[{'path':'source.py','bytes':4,'sha256':m.base.sha(src)}]}
            budget=m.budget_module.SnapshotBudget(out,plan,protocol,16);budget.initialize()
            with patch.object(budget,'snapshot_audit',side_effect=AssertionError('snapshot hot scan')):
                self.assertEqual(budget.total()['total_bytes'],15)
            (out/'extra').write_bytes(b'12')
            with self.assertRaises(ValueError):budget.total()
            (out/'extra').unlink();src.write_bytes(b'evil')
            with self.assertRaises(ValueError):budget.final()

    def test_run_installs_and_restores_fast_walk(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf=Path(tmp);protocol=hf/'protocol.json';protocol.write_bytes(b'protocol');source=hf/'source.py';source.write_bytes(b'code')
            plan={'id':m.identity(0),'shard':0,'output_HF_relative':'output',
                  'repo_sources':[{'path':'source.py','bytes':4,'sha256':m.base.sha(source)}],
                  'v2_interrupted_evidence':{'binding':'test'}}
            def fake_run(_):
                out=hf/'output';snap=out/'EXECUTED_SOURCES';snap.mkdir(parents=True)
                (snap/'source.py').write_bytes(source.read_bytes());(snap/'protocol.json').write_bytes(protocol.read_bytes())
                (out/'RUN.json').write_text(json.dumps({'id':plan['id'],'status':'COMPLETE'}))
                for _ in range(3):self.assertGreaterEqual(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink()),12)
            original=Path.rglob
            with patch.object(m.base,'read',side_effect=lambda p:plan if Path(p)==protocol else json.loads(Path(p).read_text())), \
                 patch.object(m,'verify_evidence'),patch.object(m.q2,'run',side_effect=fake_run):
                m.run(SimpleNamespace(hf_root=hf,protocol=protocol))
            self.assertIs(Path.rglob,original)
            report=json.loads((hf/'output/RUN.json').read_text())
            self.assertEqual(report['snapshot_audit']['file_count'],2)
            self.assertEqual(report['output_budget_audit']['snapshot_bytes'],12)
            self.assertEqual(report['snapshot_repair'],m.REPAIR)

if __name__=='__main__':unittest.main()
