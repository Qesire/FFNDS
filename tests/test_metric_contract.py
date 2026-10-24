"""Pure-stdlib regression checks for literature-aligned FFNDS measurement/claim grains."""
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from check_metric_contract import catalog,validate_catalog,validate_record
class MetricContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat=catalog()
        cls.map={m["id"]:m for m in cls.cat["metrics"]}
    def obs(self,id):
        m=self.map[id]
        return {"metric_id":id,"level":m["level"],"unit":m["unit"],"value":1.0,
          "sample_unit":"64 stripes","source":"SYNTHETIC_TEST_ONLY","spatial_grain":"group of four real-token positions",
          "temporal_grain":"step2 block20","population":"exposed development","aggregation":"sum per group",
          "comparison_reference":"same-state D-Q","execution_mode":"saved_native_same_state",
          "source_exposure":"development","implementation_sha":"a"*64,
          "evidence_locator":"SYNTHETIC_TEST_ONLY"}
    def test_catalog_is_well_formed(self):self.assertEqual(validate_catalog(self.cat),[])
    def test_r14n0_cannot_be_renamed_video_loss(self):
        r={"record_id":"fixture","evaluation_status":"MEASURED","claim":"video_noncollapse_smoke",
            "observations":[self.obs("post_writeback_sse")]}
        errs=validate_record(r,self.cat)
        self.assertIn("claim exceeds actually measured evidence levels",errs)
        self.assertIn("claim missing required metrics",errs)
    def test_legacy_sse_is_legitimate_local_diagnostic(self):
        r={"record_id":"fixture","evaluation_status":"MEASURED","claim":"local_quality_diagnostic",
            "observations":[self.obs("post_writeback_sse")]}
        self.assertEqual(validate_record(r,self.cat),[])
    def test_synthetic_backend_never_native_speed(self):
        r={"record_id":"fixture","evaluation_status":"MEASURED","claim":"native_speedup",
            "observations":[self.obs("synthetic_backend_speedup")]}
        self.assertIn("claim missing required metrics",validate_record(r,self.cat))
    def test_video_claim_needs_real_full_student(self):
        r={"record_id":"fixture","evaluation_status":"MEASURED","claim":"video_noncollapse_smoke",
            "observations":[self.obs("free_rollout_completed"),self.obs("visual_collapse_review")]}
        self.assertIn("video claim lacks true full free-running paired student evidence",validate_record(r,self.cat))
    def test_missing_granularity_is_detected(self):
        item=self.obs("post_writeback_sse");item.pop("temporal_grain")
        r={"record_id":"fixture","evaluation_status":"MEASURED","claim":"local_quality_diagnostic","observations":[item]}
        self.assertIn("post_writeback_sse missing temporal_grain",validate_record(r,self.cat))
    def test_unmeasured_remains_null(self):
        v=json.loads((ROOT/"research/metric_system/metric_record_template.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_record(v,self.cat),[])
    def test_historical_criteria_unchanged(self):
        d=json.loads((ROOT/"research/OBJECTIVE_CONTRACT.json").read_text(encoding="utf-8"))
        old=d["existing_joint_acceptance"]
        self.assertEqual((old["main_gemm_MAC_fraction_max"],old["complete_original_FFN_speedup_min"],old["per_axis_relative_quality_drop_max"]),(0.5,1.5,0.02))
        self.assertEqual(d["research_evaluation_policy"]["old_global_threshold_role"],"HISTORICAL_REFERENCE_ONLY_NOT_A_UNIVERSAL_HARD_GATE")
    def test_metric_catalog_reachable_without_literal_old_prompt_words(self):
        d=json.loads((ROOT/"research/OBJECTIVE_CONTRACT.json").read_text(encoding="utf-8"))
        active=json.loads((ROOT/"research/ACTIVE_RESEARCH.json").read_text(encoding="utf-8"))
        self.assertEqual(d["metric_evaluation_framework"]["catalog"],"research/metric_system/catalog.json")
        self.assertTrue((ROOT/d["metric_evaluation_framework"]["catalog"]).is_file())
        self.assertTrue((ROOT/"docs/76_基于学术文献的FFNDS分层指标体系_20261022.md").is_file())
        self.assertTrue((ROOT/"docs/77_FFNDS学术指标族与评测协议_20261023.md").is_file())
        self.assertIn("actual", active["questions"][2]["focus"].lower())
        self.assertEqual(active["experiment_authority"]["eligible_model_method_ids"], [])
if __name__=="__main__":unittest.main()
