import importlib.util
from pathlib import Path
import unittest
import numpy as np
p=Path(__file__).resolve().parents[1]/'tools/ub1_consumer_latent_geometry.py';spec=importlib.util.spec_from_file_location('geometry',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class GeometryTests(unittest.TestCase):
    def test_signed_opposite_correction(self):
        out=m.stats(np.array([0.,0.]),np.array([3.,4.]),np.array([1.5,2.]))
        self.assertEqual(out['ub1_error_energy'],25.);self.assertEqual(out['candidate_error_energy'],6.25)
        self.assertEqual(out['correction_energy'],6.25);self.assertEqual(out['twice_cross'],-25.);self.assertEqual(out['cos_ub1_correction'],-1.)
    def test_invalid_shape_and_nonfinite(self):
        with self.assertRaises(ValueError):m.stats([0.],[1.,2.],[0.])
        with self.assertRaises(ValueError):m.stats([0.],[float('nan')],[1.])
    def test_identity_and_step_failure(self):
        case={'index':0,'seed':42,'source_id':'source','prompt_sha256':'prompt'}
        trajectory=[{'step':i} for i in range(4)];phase={'mode':'dense','source_id':'source','seed':42,'prompt_sha256':'prompt','trajectory':trajectory,'video':{'sha256':'v','frames':81}}
        run={'status':'PASS_FIXED_FULL_DEPTH_CONSUMER_EXECUTION_CORE_METRICS_QUALITY_PENDING','case':case,'phases':[phase],'protocol_sha256':'protocol'}
        core={'identity':{'source_id':'source','seed':42,'input_prompt_digest':'prompt','expected_frames':81,'protocol_sha256':'protocol'},'metrics':{'artifact':{'verified_full_decode_frames':81,'frame_shape':[464,832,3],'candidate_sha256':'v'}}}
        self.assertEqual(m.check_identity(case,'dense',run,phase,phase,core),trajectory)
        with self.assertRaises(ValueError):m.check_identity(dict(case,seed=43),'dense',run,phase,phase,core)
        bad=dict(phase,trajectory=[{'step':0}]*4);run2=dict(run,phases=[bad])
        with self.assertRaises(ValueError):m.check_identity(case,'dense',run2,bad,bad,core)
if __name__=='__main__':unittest.main()
