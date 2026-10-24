#!/usr/bin/env python3
"""Pure-static FFNDS research routing and pinned AutoResearcher content audit.

This validates instructions/source/task wiring only. It never imports the model,
runs a CPU numerical experiment, launches a GPU job or evaluates agent quality.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def record(path):
    return json.loads(text(path))


def ensure(condition, message):
    if not condition:
        raise SystemExit("FAIL: " + message)


def git_blob_sha1(data):
    # Identical to git hash-object on a regular blob.
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def check_relative_markdown_links(path):
    for link in re.findall(r"\]\(([^)]+)\)", text(path)):
        if link.startswith(("https://", "http://", "#", "mailto:")):
            continue
        target = link.split("#", 1)[0].split("?", 1)[0]
        if not target:
            continue
        resolved = (ROOT / path).parent.joinpath(target).resolve()
        ensure(resolved.is_relative_to(ROOT.resolve()) and resolved.exists(),
               "Broken local Markdown link " + path + ": " + link)


def main():
    active = record("research/ACTIVE_RESEARCH.json")
    contract = record("research/OBJECTIVE_CONTRACT.json")
    current = record("research/CURRENT_DECISION.json")
    triage = record("research/20261021/method_triage.json")
    pause = record("research/20261024/EXPERIMENT_PAUSE.json")
    source = record("research/agent_protocol/ACTIVE_SOURCE_MANIFEST.json")
    agent = text("AGENTS.md")
    skill = text(".agents/skills/ffnds-paper-discovery/SKILL.md")
    copilot = text(".github/copilot-instructions.md")
    workflow = text(".github/workflows/research-agent-validation.yml")

    ensure(source["source_commit"] == active["autoResearcher"]["source_revision"],
           "AutoResearcher source version drift")
    ensure(active["autoResearcher"]["default"] is True and
           active["autoResearcher"]["automatic_mode_selection"] is True,
           "Default AutoResearcher instruction routing missing")
    ensure(active["autoResearcher"]["active_source_manifest"] ==
           "research/agent_protocol/ACTIVE_SOURCE_MANIFEST.json",
           "Different active upstream source manifest")
    for path, expected_sha in source["git_blob_sha1"].items():
        full = ROOT / path
        ensure(full.is_file() and git_blob_sha1(full.read_bytes()) == expected_sha,
               "Pinned AutoResearcher source changed: " + path)

    ensure(all(x in agent for x in (
           "AutoResearcher is the default", "Information", "Decision", "Compute",
           "research/ACTIVE_RESEARCH.json", "research/agent_protocol/CURRENT_CONTEXT_PACKET.md")),
           "Root agent does not route to the default scientific loop")
    ensure("Default to AutoResearcher" in copilot and
           "ACTIVE_RESEARCH.json" in copilot,
           "Copilot does not route to the current science")
    ensure("information" in skill and "net decision value" in skill,
           "FFNDS-specific skill lacks Q1/Q2 context")

    ensure([q["id"] for q in active["questions"]] == ["Q1", "Q2", "Q3"],
           "Research question IDs drifted")
    ensure([q[:2] for q in contract["scientific_questions"]] == ["Q1", "Q2", "Q3"],
           "Objective contract no longer follows information/decision/compute")
    ensure(triage["policy"]["goal_semantics"][0].startswith("Q1/Q2/Q3"),
           "Legacy triage policy has recaptured science priority")
    ensure(current["current_research_source"] ==
           triage["current_research_source"] == "research/ACTIVE_RESEARCH.json",
           "Multiple competing active research sources")
    ensure(current["next_active_research_task"]["id"] ==
           triage["next_active_research_task"]["id"] == active["id"],
           "Current decision/triage task mismatch")

    ensure(pause["status"] == "PAUSED_BY_EXPLICIT_USER_REQUEST" and
           pause["resume_requires_explicit_user_instruction"] is True and
           current["experiment_pause"]["all_experiments_paused"] is True,
           "Explicit user experiment pause was changed")
    ensure(current["eligible_model_method_ids"] ==
           pause["eligible_model_method_ids"] ==
           active["experiment_authority"]["eligible_model_method_ids"] == [],
           "Experimental model method accidentally eligible")
    ensure(active["experiment_authority"]["automatic_followups"] is False and
           current["next_active_research_task"]["automatic_followup"] is False,
           "Automatic experimental continuation enabled")
    ensure("workflow_dispatch:" in workflow and not re.search(
           r"^\s*(?:push|pull_request|schedule):", workflow, flags=re.MULTILINE),
           "Static validation workflow has an automatic trigger")

    behavior = record("research/agent_protocol/DEFAULT_BEHAVIOR_CASES.json")
    ensure(behavior["source_revision"] == source["source_commit"] and
           len(behavior["cases"]) >= 5 and
           any(case["id"] == "DEFAULT_RESEARCH_WITHOUT_TRIGGER"
               for case in behavior["cases"]),
           "Behavior corpus is not bound to default AutoResearcher")
    for path in ("AGENTS.md", "README.md", ".github/copilot-instructions.md",
                 ".agents/skills/ffnds-paper-discovery/SKILL.md",
                 "research/agent_protocol/README.md", "research/RESEARCH_NAVIGATION.md",
                 "research/agent_protocol/CURRENT_CONTEXT_PACKET.md"):
        check_relative_markdown_links(path)
    for linked in (active["active_document"], active["context_packet"],
                   active["research_navigation"]):
        ensure((ROOT / linked).is_file(), "Broken active research evidence path: " + linked)
    ensure(not (ROOT / "docs/94_AutoResearcher_算子优先重新探索_20261024.md").exists() and
           not (ROOT / "research/20261024/operator_first_autoresearcher_reexploration.json").exists(),
           "Superseded algorithm-first task is present")

    print("PASS: pinned upstream AutoResearcher source, compact live Q1/Q2/Q3 route,")
    print("      Markdown links, behavior corpus, user experiment pause and manual-only workflow.")
    print("Scope: static structure only. No numerical experiment or agent behavioral run.")


if __name__ == "__main__":
    main()
