"""R15 catalog integration and noninvented evidence; no model or GPU."""
import json
import pathlib
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]

def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

class R15CatalogIntegrityTests(unittest.TestCase):
    def test_ten_families_ninetythree_distinct_definitions(self):
        c=load("research/metric_system/academic_metric_families.json")
        self.assertEqual((c["family_count"],c["entry_count"]),(10,93))
        self.assertEqual((len(c["families"]),len(c["metrics"])),(10,93))
        ids=[m["id"] for m in c["metrics"]]
        self.assertEqual(len(ids),len(set(ids)))
        for f in c["families"]:
            self.assertEqual(set(f["metric_ids"]),{m["id"] for m in c["metrics"] if m["family"]==f["id"]})
        for m in c["metrics"]:
            self.assertEqual(m["measurement_status"],"DEFINITION_ONLY")
            self.assertTrue(m["source_ids"])
            self.assertTrue(all(src in c["sources"] for src in m["source_ids"]))
    def test_official_i2v_suite_and_no_claim_of_ninetythree_measurements(self):
        c=load("research/metric_system/academic_metric_families.json")
        self.assertEqual(c["suite_pin"]["repository"],"Vchitect/VBench")
        self.assertEqual(len(c["suite_pin"]["commit"]),40)
        ids={m["id"] for m in c["metrics"]}
        self.assertTrue({"i2v_subject","i2v_background","i2v_camera_motion","vbench_motion_smoothness","vbench_dynamic_degree"}.issubset(ids))
        d=load("research/CURRENT_DECISION.json")
        self.assertEqual(d["latest_metric_review"]["catalog_entries"],93)
        self.assertEqual(d["latest_metric_review"]["new_model_runs"],0)
        self.assertEqual(d["eligible_model_method_ids"],[])
        self.assertTrue((ROOT/d["latest_metric_review"]["handoff"]).is_file())
        self.assertIn("R14N0",d["review_id"])
    def test_historical_raw_results_and_metric_layers_remain_separate(self):
        c=load("research/metric_system/catalog.json")
        self.assertEqual(len(c["metrics"]),34)
        self.assertEqual(c["academic_families"],"research/metric_system/academic_metric_families.json")
        v=load("research/metric_system/records/R15_R14_backend_mapping.json")
        self.assertEqual(v["claim"],"synthetic_backend_diagnostic")
        self.assertEqual(v["observations"][0]["metric_id"],"synthetic_backend_speedup")
        n=load("research/metric_system/records/R15_R14N0_local_mapping.json")
        self.assertEqual(n["claim"],"local_quality_diagnostic")
        self.assertTrue(all(o["level"]=="L1" for o in n["observations"]))
        p=load("research/OBJECTIVE_CONTRACT.json")
        gates=p["existing_joint_acceptance"]
        self.assertEqual((gates["main_gemm_MAC_fraction_max"],gates["complete_original_FFN_speedup_min"],gates["per_axis_relative_quality_drop_max"]),(0.5,1.5,0.02))

if __name__ == "__main__":unittest.main()
