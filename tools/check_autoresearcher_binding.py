#!/usr/bin/env python3
"""Repository-integrity check, NOT a scientific-quality/creativity validator."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "research/agent_protocol/SOURCE_MANIFEST.json").read_text())
    assert manifest["source_commit"] == "d24f18091e0ee1f75e5f7ce7fddb7c2f16d6b818"
    expected = manifest["copied_paths"]
    assert len(expected) == len(set(expected)) and len(expected) >= 10
    for path in expected:
        p = (ROOT / path).resolve()
        assert p.is_relative_to(ROOT.resolve()) and p.is_file() and p.stat().st_size > 100, path
    agents = (ROOT/"AGENTS.md").read_text()
    for phrase in ("Mandatory AutoResearcher v2.3", "Execute, not merely describe",
                   "POST_TRAIN_DYNAMIC_FFN_PAPER_PLAN.md", "best", "pre-W1"):
        assert phrase in agents, phrase
    for path in ("research/20261022/POST_TRAIN_DYNAMIC_FFN_PAPER_PLAN.md",
                 "research/20261022/AGENT_EXECUTION_HANDOFF.md",
                 ".agents/skills/ffnds-paper-discovery/SKILL.md",
                 "tools/analyze_counterfactual_actions.py"):
        assert (ROOT/path).is_file(), path
    contract = json.loads((ROOT/"research/OBJECTIVE_CONTRACT.json").read_text())
    assert "dynamic FFN" in contract["primary_scientific_goal"]
    decision = json.loads((ROOT/"research/CURRENT_DECISION.json").read_text())
    agenda = decision["exploratory_paper_agenda_20261022"]
    assert agenda["status"] == "EXPLORATION_ONLY_NOT_REGISTERED"
    assert agenda["model_execution_eligibility_change"] is False
    assert decision["closed_method_statuses"]["R4E1_FIXED_H_16GROUP_Q2"] == "CLOSED_REJECTED_REGISTERED_NEIGHBORHOOD"
    print("AutoResearcher v2.3 files, FFNDS agent instructions and objective: OK")
    print("This confirms file integrity, not agent obedience or science novelty.")


if __name__ == "__main__":
    main()
