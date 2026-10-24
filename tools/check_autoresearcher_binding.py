#!/usr/bin/env python3
"""Static verification of the ONE active AutoResearcher prompt + task route.

No imports from Wan, CUDA, model code or experimental analyses. No jobs or
model forward. This tests the source/config wiring, not research creativity.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
read = lambda p: (ROOT / p).read_text(encoding="utf-8")
def record(p):
    return json.loads(read(p))

def main():
    active = record("research/ACTIVE_RESEARCH.json")
    current = record("research/CURRENT_DECISION.json")
    triage = record("research/20261021/method_triage.json")
    pause = record("research/20261024/EXPERIMENT_PAUSE.json")
    agents = read("AGENTS.md")
    copilot = read(".github/copilot-instructions.md")
    readme = read("README.md")
    skill = read(".agents/skills/ffnds-paper-discovery/SKILL.md")
    workflow = read(".github/workflows/research-agent-validation.yml")
    assert active["autoResearcher"]["default"] is True
    assert active["autoResearcher"]["automatic_mode_selection"] is True
    sha = active["autoResearcher"]["source_revision"]
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
    assert sha[:8] in agents
    assert len(agents) < 9500 and len(readme) < 8500 and len(skill) < 6000
    assert all(x in agents for x in ("AutoResearcher is the default", "Information", "Decision", "Compute"))
    assert all(x in copilot for x in ("Default to AutoResearcher", "ACTIVE_RESEARCH.json"))
    assert all(x in skill for x in ("information", "net decision value", "actual"))
    assert current["current_research_source"] == triage["current_research_source"] == "research/ACTIVE_RESEARCH.json"
    assert current["next_active_research_task"]["id"] == triage["next_active_research_task"]["id"] == active["id"]
    assert current["eligible_model_method_ids"] == pause["eligible_model_method_ids"] == []
    assert current["experiment_pause"]["all_experiments_paused"] is True
    assert pause["status"] == "PAUSED_BY_EXPLICIT_USER_REQUEST"
    assert pause["resume_requires_explicit_user_instruction"] is True
    assert active["experiment_authority"]["automatic_followups"] is False
    assert current["next_active_research_task"]["automatic_followup"] is False
    assert "workflow_dispatch:" in workflow
    assert not any(line.strip() in ("push:", "pull_request:") for line in workflow.splitlines())
    assert not (ROOT / "docs/94_AutoResearcher_算子优先重新探索_20261024.md").exists()
    assert not (ROOT / "research/20261024/operator_first_autoresearcher_reexploration.json").exists()
    for linked in ("core/AGENT_HARNESS.md", "core/PROMPT_EVALUATION.md", "examples/RESEARCH_CONTEXT_PACKET.md", "research/agent_protocol/DEFAULT_BEHAVIOR_CASES.json"):
        assert (ROOT / linked).is_file(), linked
    behavior = record("research/agent_protocol/DEFAULT_BEHAVIOR_CASES.json")
    assert behavior["source_revision"] == sha and len(behavior["cases"]) >= 5
    assert any(x["id"] == "DEFAULT_RESEARCH_WITHOUT_TRIGGER" for x in behavior["cases"])
    assert (ROOT / active["active_document"]).is_file()
    assert (ROOT / active["context_packet"]).is_file()
    print("PASS: one active AutoResearcher route, matching Q1/Q2/Q3 task and experiment pause.")
    print("Scope: static repository wiring, NOT a scientific-performance evaluation.")

if __name__ == "__main__":
    main()
