import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from ffnds_video_metric_gate import inspect
from ffnds_video_metrics import SCHEMA, digest


class VideoGateTest(unittest.TestCase):
    def write_fixture(self,root):
        root=Path(root)
        video=root/"generated.mp4"
        video.write_bytes(b"synthetic video bytes for hash-only gate test")
        run=root/"RUN.json"
        run.write_text('{"status":"PASS_FIXED_GENERATION"}')
        run_sha=digest(run)
        n=4
        report={"schema":SCHEMA,
                "identity":{"source_id":"image-id","seed":42,"expected_frames":4,"input_prompt_digest":"inputsha","model_digest":"modelsha",
                            "protocol_sha256":"protocolsha","run_sha256":run_sha},
                "cost":{"generation_wall_seconds":10,"ffn_boundary_seconds":2,
                        "setup_seconds":3,"peak_gpu_memory_mib":24000},
                "candidate":{"sha256":digest(video)},
                "reference":None,
                "metrics":{"artifact":{"decoded_frames":n,"verified_full_decode_frames":n,"fps":16},
                           "runtime":{"generation_wall_seconds":10,"ffn_boundary_seconds":2,"setup_seconds":3,"peak_gpu_memory_mib":24000},
                           "temporal_motion":{"adjacent_RGB_MAE_0to1":{"n":3},"downsampled_Farneback_mean_flow_px":{"n":3}},
                           "spatial_detail":{"laplacian_abs_energy_candidate":{"n":4}},
                           "luma_color":{"mean_encoded_BT709_luma":{"n":4},"adjacent_encoded_luma_abs":{"n":3}}},
                "external_quality":{k:{"status":"NOT_RUN","reason":"Not executed"} for k in
                                    ("vbench_i2v","vbench2","human_review")},
                "quality_accepted":False,
                "per_frame":[{"frame":i,"laplacian_abs_edge_energy":.2,"mean_encoded_luma":.3,
                              **({"candidate_adjacent_RGB_MAE_0to1":.02,
                                  "candidate_downsampled_Farneback_flow_mean_px":1.0} if i>0 else {})}
                             for i in range(n)]}
        report_path=root/"metrics.json"
        report_path.write_text(json.dumps(report))
        bundle={"schema":"ffnds.video.bundle.v1","expected_video_count":1,
                "videos":[{"source_id":"image-id","seed":42,"variant":"dense",
                           "video_path":video.name,"video_sha256":digest(video),
                           "run_receipt_path":run.name,"run_sha256":run_sha,
                           "metric_report_path":report_path.name,
                           "metric_report_sha256":digest(report_path)}]}
        bundle_path=root/"bundle.json"
        bundle_path.write_text(json.dumps(bundle))
        return bundle_path,bundle,report_path

    def test_complete_video_record_passes_evidence_gate_not_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,_,_=self.write_fixture(temp)
            self.assertEqual(inspect(bundle),[])
            self.assertTrue(any("quality acceptance" in x for x in inspect(bundle,True)))

    def test_unreported_video_must_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,b,_=self.write_fixture(temp)
            b["videos"][0]["metric_report_path"]="missing_metrics.json"
            bundle.write_text(json.dumps(b))
            self.assertTrue(any("missing video, metric report or actual run receipt" in x for x in inspect(bundle)))

    def test_record_sha_must_be_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,b,_=self.write_fixture(temp)
            b["videos"][0]["video_sha256"]="forged"
            bundle.write_text(json.dumps(b))
            self.assertTrue(any("video digest mismatch" in x for x in inspect(bundle)))


if __name__=="__main__":
    unittest.main()
