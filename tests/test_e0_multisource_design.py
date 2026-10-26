"""Static source-contract regression tests; stdlib only."""
import importlib.util, json, copy, unittest
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/"tools/check_e0_multisource_design.py"
spec=importlib.util.spec_from_file_location("designcheck",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
plan=json.loads((root/"config/e0_multisource_directional_evaluation_20261026.json").read_text())
class Coverage(unittest.TestCase):
    def test_valid(self): self.assertEqual([],m.check(plan))
    def test_old_two_only(self):
        d=copy.deepcopy(plan);d["phase1"]["add_source_indices"]=[]
        self.assertTrue(m.check(d))
    def test_leak(self):
        d=copy.deepcopy(plan);d["phase1"]["source_records"][1]["role"]="INDEPENDENT_HOLDOUT"
        self.assertTrue(m.check(d))
    def test_sse_only(self):
        d=copy.deepcopy(plan);d["evaluation"]["global_SSE"]="PRIMARY"
        self.assertTrue(m.check(d))
if __name__=="__main__": unittest.main()
