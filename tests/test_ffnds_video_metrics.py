"""Contract tests; synthetic videos are used only for the video-decoder smoke test."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from ffnds_video_metrics import SCHEMA, summary, validate, evaluate


class MetricContractTest(unittest.TestCase):
    def fixture(self, paired=True):
        def stat(n):
            return {"n":n,"mean":0.2,"p50":0.1,"p95":0.3,"max":0.4}
        n=4
        r={"schema":SCHEMA,
           "identity":{"source_id":"fixture","seed":42,"expected_frames":4,"input_prompt_digest":"inputsha","model_digest":"hash",
                       "protocol_sha256":"protocol","run_sha256":"run"},
           "cost":{"generation_wall_seconds":3.1,"ffn_boundary_seconds":.5,
                   "setup_seconds":4.2,"peak_gpu_memory_mib":500},
           "reference":{"sha256":"dense"} if paired else None,
           "reference_identity":{"source_id":"fixture","seed":42,"input_prompt_digest":"inputsha","run_sha256":"dense-run"} if paired else None,
           "metrics":{"artifact":{"decoded_frames":n,"verified_full_decode_frames":n,
                                  "fps":16,"frame_shape":[32,32,3]},
                      "runtime":{"generation_wall_seconds":3.1,"ffn_boundary_seconds":.5,"setup_seconds":4.2,"peak_gpu_memory_mib":500},
                      "temporal_motion":{"adjacent_RGB_MAE_0to1":stat(n-1),"downsampled_Farneback_mean_flow_px":stat(n-1)},
                      "spatial_detail":{"laplacian_abs_energy_candidate":stat(n)},
                      "luma_color":{"mean_encoded_BT709_luma":stat(n),"adjacent_encoded_luma_abs":stat(n-1)}} ,
           "external_quality":{k:{"status":"NOT_RUN","reason":"Missing actual VBench/manual evaluation"}
                               for k in ("vbench_i2v","vbench2","human_review")},
           "quality_accepted":False}
        r["per_frame"]=[dict(frame=i,laplacian_abs_edge_energy=.2,mean_encoded_luma=.3,
                            **({"candidate_adjacent_RGB_MAE_0to1":.02,"candidate_downsampled_Farneback_flow_mean_px":1.0} if i>0 else {}),
                            **({"paired_RGB_RMSE_0to1":0.1,
                      **({"paired_temporal_delta_RGB_RMSE_0to1":0.2} if i>0 else {})} if paired else {})) for i in range(n)]
        if paired:
            r["metrics"].update(paired_spatial={"frames":n,"RGB_RMSE_0to1":stat(n),
                                                "RGB_MAE_0to1":stat(n),
                                                "encoded_luma_RMSE_0to1":stat(n),
                                                "signed_encoded_luma_drift_0to1":stat(n),
                                                "candidate_to_reference_laplacian_energy_ratio":stat(n)},
                                paired_temporal={"adjacent_delta_RGB_RMSE_0to1":stat(n-1)},
                                paired_motion={"downsampled_Farneback_flow_L2_px":stat(n-1),
                                               "candidate_to_reference_flow_magnitude_ratio":stat(n-1)},
                                static_region={"reference_flow_threshold_downsampled_px":.25,
                                               "uncompensated_low_reference_motion_luma_delta_abs":stat(n-1),
                                               "warning":"Low-motion mask is not a semantic background mask"})
        return r

    def test_percentiles_are_auditable(self):
        x=summary([1,2,3,4])
        self.assertEqual(x["n"],4)
        self.assertAlmostEqual(x["p95"],3.85)

    def test_complete_core_does_not_claim_accepted_quality(self):
        report=self.fixture()
        self.assertEqual(validate(report),[])
        self.assertIn("quality acceptance not documented",validate(report,True))

    def test_missing_video_metric_or_temporal_family_fails(self):
        report=self.fixture()
        del report["metrics"]["paired_temporal"]
        self.assertTrue(any("paired_temporal" in e for e in validate(report)))

    def test_one_frame_or_unverified_decode_fails(self):
        report=self.fixture()
        report["metrics"]["artifact"]["verified_full_decode_frames"]=2
        self.assertIn("not all frames verified",validate(report))

    def test_cannot_claim_acceptance_without_official_and_human(self):
        report=self.fixture()
        report["quality_accepted"]=True
        self.assertIn("unsupported accepted quality",validate(report))

    def test_wrong_expected_frame_count_is_rejected(self):
        report=self.fixture()
        report["identity"]["expected_frames"]=81
        self.assertIn("not all expected frames decoded",validate(report))

    def test_unpaired_reference_has_own_temporal_metrics(self):
        self.assertEqual(validate(self.fixture(False)),[])

    def test_synthetic_video_decode_and_metrics(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/numpy optional")
        with tempfile.TemporaryDirectory() as td:
            a,b=[Path(td)/x for x in ("candidate.mp4","dense.mp4")]
            for path,modifier in ((a,20),(b,0)):
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),16,(64,64))
                if not writer.isOpened():
                    self.skipTest("No MP4 encoder")
                for t in range(4):
                    f=np.full((64,64,3),60+modifier,dtype=np.uint8)
                    x=8+5*t
                    cv2.rectangle(f,(x,15),(x+10,28),(220,220,220),-1)
                    writer.write(f)
                writer.release()
            metadata={"identity":{"source_id":"synthetic", "seed":42,"expected_frames":4,
                                  "input_prompt_digest":"inputsha","model_digest":"hash","protocol_sha256":"hash","run_sha256":"hash"},
                      "reference_identity":{"source_id":"synthetic","seed":42,"input_prompt_digest":"inputsha","run_sha256":"dense-run"},
                      "cost":{"generation_wall_seconds":1.0,"ffn_boundary_seconds":.2,
                              "setup_seconds":.5,"peak_gpu_memory_mib":100}}
            result=evaluate(a,b,metadata)
            self.assertEqual(validate(result),[])
            self.assertEqual(result["metrics"]["artifact"]["decoded_frames"],4)
            self.assertEqual(result["metrics"]["paired_temporal"]["adjacent_delta_RGB_RMSE_0to1"]["n"],3)
            self.assertFalse(result["quality_accepted"])
            self.assertEqual(len(result["per_frame"]),4)


if __name__=="__main__":
    unittest.main()
