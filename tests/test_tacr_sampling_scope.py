import sys
import unittest
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from check_tacr_sampling_scope import read,replay,verify,run
class SamplingTest(unittest.TestCase):
    def test_actual_r17_exposure(self):
        errors,a=run()
        self.assertEqual(errors,[])
        self.assertEqual((a["completed_generation_count"],a["quality_clip_count"],len(a["quality_complete_groups"])),(57,27,5))
        self.assertEqual([x["source_index"] for x in a["quality_complete_groups"]],[0,1,2,3,4])
        self.assertEqual([x["source_index"] for x in a["quality_partial_groups"]],[5])
        self.assertEqual(a["official_metric_calls"],0)
    def test_only_subject_has_complete_pairs(self):
        self.assertEqual({x["stratum"] for x in replay()["quality_complete_groups"]},{"i2v_subject"})
    def test_latent_time_cannot_use_81_decoded_frames(self):
        p=deepcopy(read("research/20261023/TACR_sampling_scope_v2.json"))
        p["sampling"]["remaining_time_bins"]=list(range(1,81))
        self.assertTrue(any("latent" in x for x in verify(p)))
    def test_hif4_group_support(self):
        p=deepcopy(read("research/20261023/TACR_sampling_scope_v2.json"))
        p["model"]["UB1_step_base_groups"]=[108]*4
        self.assertTrue(any("UB1" in x for x in verify(p)))
    def test_depth_strata(self):
        p=deepcopy(read("research/20261023/TACR_sampling_scope_v2.json"))
        p["sampling"]["depth"]["late"]=[30,41]
        self.assertTrue(any("depth" in x for x in verify(p)))
    def test_cannot_authorize_gpu_in_design(self):
        p=deepcopy(read("research/20261023/TACR_sampling_scope_v2.json"))
        p["no_new_GPU_authority"]=False
        self.assertTrue(any("eligibility" in x for x in verify(p)))
if __name__=="__main__":
    unittest.main()
