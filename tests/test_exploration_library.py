"""Evidence/paper library integrity; these tests do not establish research novelty."""
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/"research/exploration/20261022"

class ExplorationLibrary(unittest.TestCase):
    def test_reference_documents_and_library_index(self):
        for p in ("history_full_mechanism_atlas_V1.md",
                  "prior_art_and_consistency_review.md",
                  "conditional_value_derivation_V3.md",
                  "full_trajectory_dynamic_ffn_seeds_V4.md",
                  "EXPLORATION_INDEX.md","REVIEW_DECISIONS.md"):
            s=(DIR/p).read_text(encoding="utf-8")
            self.assertGreater(len(s), 1000, p)
        self.assertTrue((ROOT/"docs/66_R12等预算动作的操作数与条件粒度审计.md").is_file())
        self.assertTrue((ROOT/"docs/67_后训练动态FFN开放探索与质量速度目标_20261022.md").is_file())

    def test_agents_not_locked_to_latest_hypotheses(self):
        agents=(ROOT/"AGENTS.md").read_text(encoding="utf-8")
        skill=(ROOT/".agents/skills/ffnds-paper-discovery/SKILL.md").read_text(encoding="utf-8")
        index=(DIR/"EXPLORATION_INDEX.md").read_text(encoding="utf-8")
        self.assertIn("not a mandatory experiment sequence", index)
        self.assertIn("not individually obligatory", agents)
        self.assertIn("no mandate to begin with block30/31", index)
        self.assertIn("none of those names are privileged", skill)

    def test_status_labels_and_scoped_facts(self):
        index=(DIR/"EXPLORATION_INDEX.md").read_text(encoding="utf-8")
        self.assertIn("neither a feasible pre-W1 policy", index)
        self.assertIn("not equal-budget R12 actions", index)
        self.assertIn("Model method eligibility remains **empty**", index)

if __name__=="__main__":
    unittest.main()
