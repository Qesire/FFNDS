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
    # Semantic/creative writing quality is evaluated using blinded case reviews,
    # not keyword quotas or mandated prompt length.
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

    # Presence, source routing and permissions are objectively checkable.
    # Whether an agent truly discovers mechanisms requires paired trajectories.
    for entry in ("AGENTS.md", ".github/copilot-instructions.md",
                  ".agents/skills/ffnds-paper-discovery/SKILL.md"):
        ensure((ROOT / entry).is_file(), "Missing default agent entry: " + entry)
    ensure(active.get("question_relationship") ==
           "COUPLED_LENSES_NON_SEQUENTIAL_INFORMATION_DECISION_COMPUTE",
           "Information/decision/compute research became an artificial stage gate")
    interfaces = active.get("research_interfaces", {})
    ensure(all(interfaces.get(key) for key in
           ("information_first", "computation_first", "bidirectional_update", "priority_selection")),
           "Lost a constructive information- or operator-first research entry")

    ensure([q["id"] for q in active["questions"]] == ["Q1", "Q2", "Q3"],
           "Research question IDs drifted")
    ensure([q[:2] for q in contract["scientific_questions"]] == ["Q1", "Q2", "Q3"],
           "Objective contract no longer follows information/decision/compute")
    ensure(contract["information_first_research_contract_20261024"]["status"] ==
           "COUPLED_QUESTIONS_NO_SEQUENTIAL_GATE",
           "Objective contract differs from bidirectional science contract")
    ensure(triage["policy"]["current_decision"] == "research/CURRENT_DECISION.json",
           "Historical method triage is no longer connected to the decision record")
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
    case_ids = {case["id"] for case in behavior["cases"]}
    ensure(behavior["source_revision"] == source["source_commit"] and
           len(behavior["cases"]) >= 8 and
           {"DEFAULT_RESEARCH_WITHOUT_TRIGGER",
            "OPERATOR_FIRST_DISCOVERY_IS_VALID",
            "INFORMATION_FIRST_DISCOVERY_IS_VALID"}.issubset(case_ids),
           "Behavior case portfolio lost its symmetric discovery questions")
    ensure("only case.user_input" in behavior.get("evaluation_input_rule", ""),
           "Behavior-case answers may be leaked to candidate-agent evaluation")
    for path in ("AGENTS.md", "README.md", ".github/copilot-instructions.md",
                 ".agents/skills/ffnds-paper-discovery/SKILL.md",
                 "research/agent_protocol/README.md", "research/RESEARCH_NAVIGATION.md",
                 "research/agent_protocol/CURRENT_CONTEXT_PACKET.md"):
        check_relative_markdown_links(path)
    for linked in (active["active_document"], active["context_packet"],
                   active["research_navigation"]):
        ensure((ROOT / linked).is_file(), "Broken active research evidence path: " + linked)
    # Archived prose remains byte-identical, while its former path is a short
    # navigation stub; results/protocols/raw data are never moved by this rule.
    archive_manifest = record("archive/20261024/historical_handoffs/MANIFEST.json")
    archived_entries = archive_manifest["entries"]
    ensure(len(archived_entries) >= 14 and
           len({entry["path"] for entry in archived_entries}) == len(archived_entries),
           "Archive has duplicate paths or unexpected missing entries")
    for entry in archived_entries:
        ensure(entry["path"].startswith("docs/") and
               entry["archive"].startswith("archive/20261024/historical_handoffs/docs/"),
               "Archive may include a live/experimental path")
        original_blob = ROOT / entry["archive"]
        old_entry = ROOT / entry["path"]
        ensure(original_blob.is_file() and
               git_blob_sha1(original_blob.read_bytes()) == entry["source_blob_sha1"] and
               original_blob.stat().st_size == entry["bytes"],
               "Archive content does not match original Git blob: " + entry["path"])
        ensure(old_entry.is_file() and
               len(old_entry.read_bytes()) < 1600 and
               entry["archive"] in text(entry["path"]),
               "Historical entry is not a concise working redirect: " + entry["path"])
        check_relative_markdown_links(entry["path"])
    check_relative_markdown_links("ARCHIVE.md")
    check_relative_markdown_links("archive/20261024/historical_handoffs/README.md")

    ensure(not (ROOT / "docs/94_AutoResearcher_算子优先重新探索_20261024.md").exists() and
           not (ROOT / "research/20261024/operator_first_autoresearcher_reexploration.json").exists(),
           "Superseded algorithm-first task is present")

    print("PASS: pinned upstream AutoResearcher source, compact live Q1/Q2/Q3 route,")
    print("      Markdown links, content-addressed archived handoffs, pause and manual workflow.")
    print("Scope: static structure only. No numerical experiment or agent behavioral run.")


if __name__ == "__main__":
    main()
