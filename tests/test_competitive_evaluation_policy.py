"""User clarified historical numeric acceptance thresholds are not global research gates."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

def load(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))

class CompetitiveEvaluationPolicy(unittest.TestCase):
    def test_new_policy_supersedes_global_gate_only(self):
        c=load("research/OBJECTIVE_CONTRACT.json")
        p=c["research_evaluation_policy"]
        self.assertEqual(p["old_global_threshold_role"], "HISTORICAL_REFERENCE_ONLY_NOT_A_UNIVERSAL_HARD_GATE")
        self.assertIn("Pareto", p["goal"])
        self.assertIn("not", p["authority"].lower())
        self.assertTrue(any("same hardware" in x.lower() for x in p["fair_comparison_requirements"]))
        old=c["existing_joint_acceptance"]
        self.assertEqual((old["main_gemm_MAC_fraction_max"],old["complete_original_FFN_speedup_min"],old["per_axis_relative_quality_drop_max"]),(0.5,1.5,0.02))
        self.assertIn("historical registered",old["scope"].lower())

    def test_decision_and_closed_results_not_reinterpreted(self):
        d=load("research/CURRENT_DECISION.json")
        if d["eligible_model_method_ids"]:
            self.assertEqual(d["eligible_model_method_ids"],["R14N0_COMPLETE_STRIPE_NATIVE_VALUE"])
            p=load(d['registered_diagnostic']['protocol'])
            self.assertEqual(load(p['authorization']['path'])['authorized_method_id'],d['eligible_model_method_ids'][0])
        self.assertEqual(d["research_goal_update_20261022"]["eligibility_effect"],"NONE")
        reviews=[d['latest_research_review'],*d.get('previous_research_reviews',[])]
        old=next(r for r in reviews if r['id']=='R12_D0_EQUAL_BUDGET_NATIVE_VALUE_AND_DEBIT')
        self.assertEqual(old['disposition'],"CLOSED_FIXED_R12_D0_RAW_VALUE_GATE_REJECTED")
        self.assertEqual(d['closed_method_statuses']['R12E1_D0_EQUAL_BUDGET_NATIVE_VALUE'],old['disposition'])

    def test_guides_and_skills_explicitly_nonbinding(self):
        for path in ("AGENTS.md",".agents/skills/ffnds-paper-discovery/SKILL.md","README.md",
                     "docs/72_与同前提方法竞争的质量速度评价准则_20261022.md"):
            text=(ROOT/path).read_text(encoding="utf-8").lower()
            self.assertIn("pareto",text,path)
            self.assertTrue("1.5" in text and "50%" in text,path)
        self.assertIn("new-study", (ROOT/"research/NEW_BRANCH_TEMPLATE.md").read_text(encoding="utf-8"))

    def test_historical_r13_and_new_eval_distinct(self):
        h=(ROOT/"research/20261022/HANDOFF_R13.md").read_text(encoding="utf-8")
        guide=(ROOT/"docs/72_与同前提方法竞争的质量速度评价准则_20261022.md").read_text(encoding="utf-8")
        self.assertIn("1.355",h)
        self.assertIn("不追溯",guide)
        self.assertIn("同前提",guide)

if __name__=="__main__":
    unittest.main()
