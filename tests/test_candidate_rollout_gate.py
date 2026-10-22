"""Guard that no FFNDS method can claim video-safe promotion from a local proxy."""
import copy
import unittest
from pathlib import Path
from tools.check_candidate_rollout_gate import validate_record, ROOT

class CandidateRolloutGateTests(unittest.TestCase):
    def sample(self):
        return {
            "candidate_id": "SYNTHETIC_FIXTURE_NOT_ACTUAL_VIDEO",
            "status": "ROLLOUT_PASSED_SMOKE",
            "selected_for_backup": True,
            "rollout": {
                "outcome": "PASS", "candidate_video_count": 1,
                "completed_student_steps": 4, "candidate_executed_at_intended_positions": True,
                "student_freely_evolved_across_steps": True,
                "intermediate_latent_reset_to_DQ": False,
                "video_decoded": True, "paired_dense_DQ_video_exists": True,
                "matched": {k: True for k in (
                    "same_model_and_checkpoint", "same_source_image", "same_prompt",
                    "same_seed", "same_initial_noise", "same_scheduler_and_timesteps",
                    "same_frames_and_resolution", "same_precision_regime"
                )},
                "candidate_video": {"sha256": "a"*64, "locator": "SYNTHETIC_ONLY"},
                "baseline_video": {"sha256": "b"*64, "locator": "SYNTHETIC_ONLY"},
                "visual_review": {
                    "full_duration_reviewed": True,
                    "subject_identity_and_scene_integrity": True,
                    "motion_coherence": True,
                    "static_background_and_small_objects_stable": True,
                    "stable_lighting_without_unexpected_flicker": True,
                    "no_catastrophic_artifacts": True,
                    "verdict": "NO_CATASTROPHIC_COLLAPSE",
                    "notes": "SYNTHETIC FIXTURE only: full-video review requirements are tested, no video exists."
                },
                "authorization_reference": "SYNTHETIC_ONLY",
                "review_report_path": "research/candidate_rollout/README.md"
            }
        }

    def test_valid_metadata_shape_only_not_video_proof(self):
        self.assertEqual(validate_record(self.sample(), ROOT), [])

    def test_pending_method_must_require_full_generation(self):
        x={"candidate_id": "backup", "status": "LOCALLY_SUPPORTED"}
        self.assertIn("selected candidate must list full free-running video smoke as next required milestone",
                      validate_record(x))
        x["next_required"]="COMPLETE_PAIRED_FREE_RUNNING_VIDEO_SMOKE"
        self.assertEqual(validate_record(x), [])

    def test_promoted_no_video_is_blocked(self):
        x=self.sample()
        x["rollout"]=None
        self.assertTrue(validate_record(x))

    def test_reset_to_dense_does_not_count(self):
        x=self.sample()
        x["rollout"]["intermediate_latent_reset_to_DQ"]=True
        self.assertTrue(any("restoration" in v for v in validate_record(x)))

    def test_single_frame_or_partial_path_not_full_generation(self):
        x=self.sample()
        x["rollout"]["completed_student_steps"]=1
        x["rollout"]["candidate_executed_at_intended_positions"]=False
        self.assertGreaterEqual(len(validate_record(x)),2)

    def test_lighting_or_background_instability_blocks_promotion(self):
        for key in ("stable_lighting_without_unexpected_flicker","static_background_and_small_objects_stable",
                    "motion_coherence", "subject_identity_and_scene_integrity"):
            x=self.sample()
            x["rollout"]["visual_review"][key]=False
            self.assertTrue(validate_record(x),key)

    def test_no_visual_review_or_artifact_provenance_is_blocked(self):
        x=self.sample()
        x["rollout"]["visual_review"]["notes"]=""
        x["rollout"]["candidate_video"]["sha256"]="missing"
        self.assertGreaterEqual(len(validate_record(x)),2)

    def test_benchmark_promotion_needs_next_validation(self):
        x=self.sample()
        x["status"]="READY_FOR_PARETO_BENCHMARK"
        self.assertTrue(validate_record(x))
        x["next_validation_plan"]="Pair new-source videos and measure full FFN and full generation latency"
        self.assertEqual(validate_record(x),[])

    def test_guide_and_contract_are_bound(self):
        import json
        contract=json.loads((ROOT/"research/OBJECTIVE_CONTRACT.json").read_text(encoding="utf-8"))
        policy=contract["candidate_full_rollout_policy"]
        self.assertIn("full",policy["mandatory_minimum"].lower())
        self.assertIn("authorization",policy["execution_authorization"].lower())
        self.assertIn("75_",policy["source_document"])
        for p in ("AGENTS.md",".agents/skills/ffnds-paper-discovery/SKILL.md",
                  ".github/copilot-instructions.md"):
            text=(ROOT/p).read_text(encoding="utf-8")
            self.assertIn("docs/75_",text,p)

if __name__=="__main__":
    unittest.main()
