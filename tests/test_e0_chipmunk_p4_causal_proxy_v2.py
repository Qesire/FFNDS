import pathlib, types, unittest
from unittest.mock import patch
from tools import e0_chipmunk_p4_causal_proxy_v2 as m

class EntryRegression(unittest.TestCase):
    def test_runtime_metadata_call_occurs_without_local_import_shadow(self):
        plan={'runtime_versions':{'torch':'expected'}}
        job={'protocol_sha256':'d','id':m.ID,'driver_argv':['driver'],'hf_root':'/tmp','cwd':str(m.ROOT),'cpu_affinity':[0,1,2,3]}
        scope={'launch_spec_sha256':'d'}
        arg=types.SimpleNamespace(protocol=pathlib.Path('/fake/protocol'),job_spec=pathlib.Path('/fake/job'),hf_root=pathlib.Path('/tmp'),preflight=False)
        class ReachedMetadata(Exception):pass
        def fake_load(text):return plan if text=='plan' else job
        with patch.object(pathlib.Path,'read_text',side_effect=['plan','job']),patch.object(m.json,'loads',side_effect=fake_load),patch.object(m,'sha',return_value='d'),patch.object(m,'authorize',return_value=scope),patch.object(m.sys,'argv',['driver']),patch.object(m.sys,'flags',types.SimpleNamespace(isolated=1)),patch.object(m.os,'sched_getaffinity',return_value={0,1,2,3}),patch.object(m.importlib.metadata,'version',side_effect=ReachedMetadata):
            with self.assertRaises(ReachedMetadata):m.run(arg)
    def test_draft_rejected(self):
        with self.assertRaises(ValueError):m.authorize({'id':m.ID,'assets_finalized':False},'a')
if __name__=='__main__':unittest.main()
