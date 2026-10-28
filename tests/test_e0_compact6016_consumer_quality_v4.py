import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from tools import e0_compact6016_consumer_quality_v4 as v4


class BootstrapTest(unittest.TestCase):
    def test_guarded_order_once(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'activations.py'; p.write_text('pinned'); sha=hashlib.sha256(p.read_bytes()).hexdigest()
            specs=[{'type_name':str(i),'line':i,'global_name':str(i)} for i in range(8)]
            activation={'site_relative':'activations.py','source_sha256':sha,'instances':specs}
            result={'predebited_official_module_imports':1,'completed_official_module_imports':0,'stateless_activation_wrappers':[]}
            calls=[]
            def importer(name):
                self.assertEqual(result['predebited_official_module_imports'],1)
                calls.append(name); result['stateless_activation_wrappers'].extend(dict(x,module=name,source_sha256=sha) for x in specs)
                return SimpleNamespace(__file__=str(p))
            v4.bootstrap(result,d,activation,importer); v4.bootstrap(result,d,activation,importer)
            self.assertEqual(calls,['transformers.activations']); self.assertEqual(result['activation_bootstrap']['after'],8)
            bad={'predebited_official_module_imports':0,'completed_official_module_imports':0}
            with self.assertRaises(Exception): v4.bootstrap(bad,d,activation,importer)
            bad={'predebited_official_module_imports':1,'completed_official_module_imports':0}
            with self.assertRaises(Exception): v4.bootstrap(bad,d,{**activation,'source_sha256':'bad'},importer)
            bad={'predebited_official_module_imports':1,'completed_official_module_imports':0,'stateless_activation_wrappers':[]}
            with self.assertRaises(Exception): v4.bootstrap(bad,d,activation,lambda name: SimpleNamespace(__file__=str(p)))

    def test_proxy_only_captures_imports_result(self):
        class Guard:
            def import_audit_guard(self,d,r): return (1,2,3)
        proxy=v4.EnvironmentProxy(Guard(),{'stateless_activation_allowlist':{}},'/tmp')
        v4._current=None; proxy.import_audit_guard('/tmp',{}); self.assertIsNone(v4._current)
        obj={'predebited_official_module_imports':0}; proxy.import_audit_guard('/tmp',obj); self.assertIs(v4._current[0],obj)
        v4._current=None

    def test_v3_run_binding_fixed(self):
        self.assertEqual(v4.V3_RUN_SHA,'c1cedd1d41ebb6e15ac49ac88c24eb6187da635315140da095058ac861be65e9')

if __name__=='__main__': unittest.main()
