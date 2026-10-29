import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('raft_v2_test',ROOT/'tools/e0_s2_sm120_fixed_depth_motion_metrics_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class BudgetTests(unittest.TestCase):
    def fixture(self,tmp,limit=10000):
        root=Path(tmp);out=root/'out';snap=out/'EXECUTED_SOURCES';(snap/'tools').mkdir(parents=True)
        (snap/'tools/a.py').write_bytes(b'code');protocol=root/'protocol.json';protocol.write_bytes(b'protocol')
        (snap/'protocol.json').write_bytes(b'protocol');(out/'RUN.json').write_bytes(b'run')
        plan={'repo_sources':[{'path':'tools/a.py','bytes':4,'sha256':m.base.sha(snap/'tools/a.py')}]}
        return m.SnapshotBudget(out,plan,protocol,limit),snap,out

    def test_hot_path_skips_snapshot_and_final_detects_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget,snap,out=self.fixture(tmp);audit=budget.initialize()
            self.assertEqual(audit['file_count'],2);self.assertEqual(audit['bytes'],12)
            original=os.walk;visited=[]
            def observed(path,*args,**kwargs):
                for directory,dirs,files in original(path,*args,**kwargs):
                    visited.append(Path(directory));yield directory,dirs,files
            with patch.object(m.os,'walk',side_effect=observed):
                for _ in range(4):self.assertEqual(budget.total()['total_bytes'],15)
            self.assertNotIn(snap,visited);self.assertNotIn(snap/'tools',visited)
            (snap/'tools/a.py').write_bytes(b'evil')
            with self.assertRaises(ValueError):budget.final()

    def test_new_snapshot_file_and_mutable_budget_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget,snap,out=self.fixture(tmp,limit=16);budget.initialize()
            (out/'extra').write_bytes(b'12')
            with self.assertRaises(ValueError):budget.total()
            (out/'extra').unlink();(snap/'extra').write_bytes(b'x')
            with self.assertRaises(ValueError):budget.final()

    def test_snapshot_refuses_symlinks_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget,snap,out=self.fixture(tmp);(snap/'tools/a.py').unlink()
            with self.assertRaises(ValueError):budget.initialize()
        with tempfile.TemporaryDirectory() as tmp:
            budget,snap,out=self.fixture(tmp);(snap/'link').symlink_to(out)
            with self.assertRaises(ValueError):budget.initialize()

    def test_run_interposes_hot_walk_and_preserves_final_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf=Path(tmp); protocol=hf/'protocol.json';protocol.write_bytes(b'protocol')
            src=hf/'source.py';src.write_bytes(b'code')
            plan={'id':m.identity(0),'shard':0,'output_HF_relative':'output',
                  'repo_sources':[{'path':'source.py','bytes':4,'sha256':m.base.sha(src)}],
                  'repair_evidence':{'proof':'bound'}}
            args=SimpleNamespace(hf_root=hf,protocol=protocol)
            def fake_run(_):
                out=hf/'output';snap=out/'EXECUTED_SOURCES';snap.mkdir(parents=True)
                (snap/'source.py').write_bytes(src.read_bytes());(snap/'protocol.json').write_bytes(protocol.read_bytes())
                (out/'RUN.json').write_text(json.dumps({'id':plan['id'],'status':'COMPLETE'}))
                for _ in range(3):
                    total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
                    self.assertGreaterEqual(total,12)
            original=Path.rglob
            with patch.object(m.base,'read',side_effect=lambda p:plan if Path(p)==protocol else json.loads(Path(p).read_text())), \
                 patch.object(m,'verify_evidence'),patch.object(m.q,'run',side_effect=fake_run):
                m.run(args)
            self.assertIs(Path.rglob,original)
            report=json.loads((hf/'output/RUN.json').read_text())
            self.assertEqual(report['snapshot_audit']['file_count'],2)
            self.assertEqual(report['output_budget_audit']['snapshot_bytes'],12)
            self.assertEqual(report['software_repair'],m.REPAIR)

    def test_identity_and_original_budgets(self):
        self.assertEqual(len({m.identity(i) for i in range(4)}),4)
        self.assertTrue(all(m.identity(i).endswith('_V2') for i in range(4)))
        self.assertEqual(m.q.LIMITS['RAFT_forwards'],480)
        self.assertEqual(m.q.LIMITS['RAFT_iterations'],9600)
        self.assertEqual(m.q.LIMITS['decoded_frames'],972)
        self.assertEqual(m.q.RESOURCES['GPU'],1)

    def test_normalized_old_validation_cannot_drop_v1_or_v2_sources(self):
        def b(p):return {'path':p,'bytes':1,'sha256':'a'*64}
        proof={'diagnostic':{'path':m.DIAGNOSTIC,'sha256':m.DIAGNOSTIC_SHA},
            'terminated_V1':[{'protocol':{'path':m.V1_CONFIGS[i],'sha256':m.V1_CONFIG_SHAS[i]},
                              'exit':{'path':m.V1_WORKER.format(i)+'/EXIT_RECEIPT.json'}} for i in range(3)]}
        plan={'shard':0,'id':m.identity(0),'output_HF_relative':m.output(0),'software_repair':m.REPAIR,
              'repo_sources':[b('v1-source')]+[b(p) for p in sorted(m.EXTRA)],'repair_evidence':proof}
        with patch.object(m,'OLD_VALIDATE') as old:
            m.validate(plan,True);self.assertEqual([x['path'] for x in old.call_args.args[0]['repo_sources']],['v1-source'])
            plan['repo_sources'].pop()
            with self.assertRaises(ValueError):m.validate(plan,True)

if __name__=='__main__':unittest.main()
