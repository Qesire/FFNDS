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

    # Search-augmented upstream lock: synchronize pinned WARD files, study and agents.
    required_ward = {"core/WEB_RESEARCH_DISCOVERY.md",
                     ".agents/skills/web-research-discovery/SKILL.md"}
    ensure(required_ward.issubset(set(source["git_blob_sha1"])),
           "Search-augmented upstream files missing from pinned manifest")
    upstream = active.get("autoResearcher", {})
    ensure(upstream.get("protocol_version") == "2.4-search-augmented" and
           upstream.get("web_research_protocol") == "core/WEB_RESEARCH_DISCOVERY.md" and
           "web-research-discovery" in upstream.get("skills", []) and
           active.get("web_research", {}).get("upstream_revision") ==
           source["source_commit"] and
           active["web_research"].get("upstream_skill") ==
           ".agents/skills/web-research-discovery/SKILL.md",
           "Live FFNDS study is not pinned to search-augmented AutoResearcher")
    ensure("web-research-discovery/SKILL.md" in text("AGENTS.md") and
           "web-research-discovery/SKILL.md" in text(
               ".agents/skills/ffnds-paper-discovery/SKILL.md"),
           "Default and domain agents must inherit the upstream search Skill")
    ensure("core/WEB_RESEARCH_DISCOVERY.md" in text(
               "research/agent_protocol/UPSTREAM_SYNC.md"),
           "Upstream synchronization procedure is missing generic WARD core")

    # Broad/deep web-research routing is an instruction integrity property,
    # never evidence that any agent actually performed live searches.
    web = active.get("web_research", {})
    ensure(web.get("default") is True and
           web.get("mode") ==
           "PROACTIVE_BROAD_DEEP_ITERATIVE_WEB_SEARCH_FOR_SUBSTANTIVE_SCIENTIFIC_WORK" and
           web.get("protocol") == "research/agent_protocol/DEEP_WEB_RESEARCH.md",
           "Live deep web-search research default is missing")
    ensure((ROOT / web["protocol"]).is_file(),
           "Deep web research protocol is missing")
    for entry in ("AGENTS.md", ".github/copilot-instructions.md",
                  ".agents/skills/ffnds-paper-discovery/SKILL.md",
                  "research/method_discussion/AGENTS.md",
                  "research/method_discussion/README.md",
                  "research/method_discussion/TEMPLATES.md",
                  "research/method_discussion/discoveries/E0/README.md",
                  "research/agent_protocol/CURRENT_CONTEXT_PACKET.md",
                  "research/agent_protocol/E0_STAGE_ANCHOR.md",
                  "research/agent_protocol/README.md",
                  "research/20261024/FIRST_TASK_E0_QUANTIZATION_FFN_DISCOVERY_20261024.md"):
        ensure("DEEP_WEB_RESEARCH.md" in text(entry),
               "Deep web-search prompt routing drift: " + entry)
    local_skill = text(".agents/skills/ffnds-paper-discovery/SKILL.md")
    ensure(re.match(r"\A---\nname:\s*ffnds-paper-discovery\n"
                    r"description:[^\n]+\n---\n", local_skill),
           "Local domain skill YAML frontmatter is invalid")

    # Live-stage routing: static structural consistency, NOT creative grading.
    stage = active.get("current_stage", {})
    ensure(stage.get("id") == "E0_OPEN_DISCOVERY" and
           stage.get("phase") == "PHENOMENON_AND_COMPETING_MECHANISM_EXPLORATION",
           "Live research was silently promoted beyond the user-selected E0 phase")
    ensure(active.get("active_document") == stage.get("task_document"),
           "Active document and sole E0 task disagree")
    ensure(stage.get("multi_agent_mode") ==
           "INDEPENDENT_DISCOVERY_AND_CHALLENGE_DISCUSSION_ONLY" and
           stage.get("new_experiment_authority") == "NONE" and
           stage.get("transition_authority") ==
           "EXPLICIT_USER_DIRECTED_SERIALIZED_COORDINATOR_UPDATE_ONLY",
           "E0 discussion acquired unapproved stage/execution authority")
    for key in ("task_document", "direction_document", "agent_anchor",
                "discovery_surface", "open_problems_document"):
        rel = stage.get(key, "")
        ensure(bool(rel) and (ROOT / rel).is_file(),
               "Missing active E0 instruction/evidence link: " + key)
    for entry in ("AGENTS.md", ".github/copilot-instructions.md",
                  "research/method_discussion/AGENTS.md",
                  "research/agent_protocol/CURRENT_CONTEXT_PACKET.md",
                  ".agents/skills/ffnds-paper-discovery/SKILL.md",
                  "research/method_discussion/TEMPLATES.md"):
        ensure(stage["id"] in text(entry),
               "Agent entrypoint has drifted from live E0 research phase: " + entry)
    # Single FFN-first scientific question; historical E0 gap dossier remains immutable
    # evidence rather than a source of agent task priority.
    focus = active.get("research_focus", {})
    reset_doc = "research/20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md"
    first_task = "research/20261025/E0_FFN_FIRST_TASK_20261025.md"
    historical_gap = "research/20261025/E0_GAPS_REMEDIATION_REQUIREMENTS.md"
    ensure(focus.get("priority") == "INTRINSIC_FFN_MECHANISM_FIRST" and
           focus.get("scientific_reset") == reset_doc and
           focus.get("first_task") == first_task and
           focus.get("quantization_role", "").startswith("OPTIONAL_COARSE_MAGNITUDE") and
           active.get("active_document") == first_task and
           stage.get("open_problems_document") == reset_doc and
           stage.get("direction_document") == reset_doc and
           stage.get("latest_deliverable", {}).get("gaps_and_required_remedies") == historical_gap and
           (ROOT / historical_gap).is_file(),
           "FFN-first active science routing or historical evidence provenance drift")
    for entry in ("AGENTS.md", ".github/copilot-instructions.md",
                  ".agents/skills/ffnds-paper-discovery/SKILL.md",
                  "research/method_discussion/AGENTS.md",
                  "research/agent_protocol/E0_STAGE_ANCHOR.md",
                  "research/agent_protocol/CURRENT_CONTEXT_PACKET.md",
                  "research/agent_protocol/README.md",
                  "research/method_discussion/discoveries/E0/README.md",
                  "README.md"):
        ensure("E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md" in text(entry),
               "FFN-first project entrypoint drift: " + entry)
    for entry in ("AGENTS.md", ".github/copilot-instructions.md",
                  ".agents/skills/ffnds-paper-discovery/SKILL.md",
                  "research/method_discussion/AGENTS.md",
                  "research/agent_protocol/E0_STAGE_ANCHOR.md",
                  "research/agent_protocol/CURRENT_CONTEXT_PACKET.md",
                  "research/agent_protocol/README.md", "README.md"):
        ensure("E0_FFN_FIRST_TASK_20261025.md" in text(entry),
               "New FFN discovery task not visible to agent: " + entry)
    # Historical G4 mechanism audit remains discoverable, but only the
    # most recent G5 audit owns the current latest_* pointers below.
    divergence = "research/20261025/E0_INTRINSIC_DIVERGENCE_AUDIT_20261025.md"
    ensure((ROOT / divergence).is_file() and
           "E0_INTRINSIC_DIVERGENCE_AUDIT_20261025.md" in text(first_task),
           "Historical intrinsic divergence audit lost from research navigation")
    ensure("MULTIPLE_COMPETING_MECHANISMS" in
           focus.get("search_space_policy", "") and
           len(stage.get("current_open_forks", [])) >= 4 and
           stage.get("new_experiment_authority") == "NONE",
           "Research handoff narrowed or created an implicit experiment queue")
    # Graph↔Agent index must resolve to real Mermaid node labels, source/evidence
    # files and link to current E0 study. This is a purely static integrity check.
    from graph_agent_bridge import load as load_graph_registry, validate as validate_graph_registry
    graph = active.get("graph_agent_collaboration", {})
    ensure(graph.get("status") == "ACTIVE_SOURCE_GRAPH_AND_EVIDENCE_DISCUSSION" and
           graph.get("canonical_stage") == "E0_OPEN_DISCOVERY" and
           graph.get("model_execution_authority") == "NONE" and
           graph.get("node_registry") == "research/agent_protocol/GRAPH_AGENT_BINDINGS.json" and
           graph.get("protocol") == "research/agent_protocol/GRAPH_AGENT_COLLABORATION.md" and
           graph.get("visual_computation_graph") == "docs/98_端到端计算图与证据注解_20261025.md" and
           stage.get("graph_agent_protocol") == graph.get("protocol"),
           "Graph-Agent collaboration missing from current E0 control plane")
    result = validate_graph_registry(load_graph_registry())
    ensure(result["nodes"] >= 20 and result["edges"] >= 18 and
           "GRAPH_AGENT_COLLABORATION.md" in text("AGENTS.md") and
           "GRAPH_AGENT_BINDINGS.json" in text(".agents/skills/ffnds-paper-discovery/SKILL.md") and
           "graph_annotations/E0/README.md" in text("research/method_discussion/AGENTS.md"),
           "Graph Agent registry / multi-agent discovery routing drift")
    # Latest G5 evidence is a scientific expansion; it must not silently
    # promote the particular affine/micro16/HiF4 branch as the sole task.
    review = "research/20261025/E0_POST_AFFINE_DIRECTION_REVIEW_20261025.md"
    ensure(focus.get("latest_mechanism_audit") == review and
           stage.get("latest_direction_review") == review and
           (ROOT / review).is_file() and
           "E0_POST_AFFINE_DIRECTION_REVIEW_20261025.md" in text(first_task),
           "Post-affine science handoff is not current")
    graph_data = load_graph_registry()
    g5 = {n["id"]: n for n in graph_data["nodes"]}
    ensure({"P15", "P16", "P17", "P18"}.issubset(g5) and
           g5["P15"]["evidence_type"] == "OBS" and
           g5["P16"]["evidence_type"] == "OBS" and
           g5["P17"]["evidence_type"] == "SYS" and
           g5["P18"]["evidence_type"] == "ALG" and
           "G5" in graph_data["graph_roles"],
           "G5 full-depth function evidence and exact dense speed got conflated")
    ensure(stage.get("new_experiment_authority") == "NONE" and
           "MULTIPLE_COMPETING_MECHANISMS" in focus.get("search_space_policy", ""),
           "Latest progress narrowed research or granted experiment authority")
    # New research-wide video evidence obligation and genuine 40x4 propagation.
    # Earlier frozen video drivers and metrics are historical, not silently retrofitted.
    contract = "research/agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md"
    depth_template = "config/e0_full_depth_quality_discriminator_TEMPLATE_20261026.json"
    ensure((ROOT / contract).is_file() and (ROOT / depth_template).is_file()
           and (ROOT / "tools/ffnds_video_metrics.py").is_file()
           and (ROOT / "tools/ffnds_video_metric_gate.py").is_file(),
           "Missing mandatory video metric and full-depth native analysis gate")
    ensure(stage.get("video_quality_evidence_contract", {}).get("metric_report_required_for_new_videos") is True
           and stage.get("full_depth_trajectory_analysis", {}).get("sites_per_video") == 160
           and stage.get("full_depth_trajectory_analysis", {}).get("status") == "DESIGNED_NOT_EXECUTED"
           and stage.get("new_experiment_authority") == "NONE",
           "Video evidence or full40x4 trajectory scope drifted / silently authorized GPU work")
    ensure({n["id"]: n["evidence_type"] for n in graph_data["nodes"]}.get("P23") == "OPEN"
           and {n["id"]: n["evidence_type"] for n in graph_data["nodes"]}.get("P24") == "OPEN"
           and {n["id"]: n["evidence_type"] for n in graph_data["nodes"]}.get("P25") == "POLICY",
           "New native student-prefix trajectory or metric completeness must not be reported as measured")
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
    discussion = active.get("multi_agent_method_discussion", {})
    method = discussion.get("default_methodology", {})
    ensure(discussion.get("mode") == "DISCUSSION_ONLY" and
           method.get("name") == "AutoResearcher" and
           method.get("automatic") is True and
           method.get("source_revision") == source["source_commit"] and
           method.get("source_manifest") ==
           "research/agent_protocol/ACTIVE_SOURCE_MANIFEST.json" and
           method.get("scoped_instructions") ==
           "research/method_discussion/AGENTS.md",
           "All discussion agents must inherit pinned AutoResearcher defaults")
    ensure(set(method.get("apply_to_all_roles", [])) ==
           {"PROPOSER", "INDEPENDENT_REVIEWER", "CHALLENGER",
            "REFORMULATOR", "INTEGRATOR"},
           "An agent discussion role lost default AutoResearcher guidance")
    ensure(discussion.get("execution") ==
           "NO_IMPLEMENTATION_NO_CPU_GPU_MODEL_EXPERIMENTS_NO_TASK_SCHEDULING",
           "Method discussion was promoted into experimental execution")
    discussion_agents = text("research/method_discussion/AGENTS.md")
    # Verify real pointers and collaboration architecture, not defensive wording.
    ensure("AutoResearcher" in discussion_agents and
           "../agent_protocol/ACTIVE_SOURCE_MANIFEST.json" in discussion_agents and
           "E0_OPEN_DISCOVERY" in discussion_agents and
           "discoveries/E0/README.md" in discussion_agents and
           "reviews/<method-id>/vN/<review-id>.json" in discussion_agents,
           "Discussion-scoped AutoResearcher research routes are incomplete")
    for required in ("AGENTS.md", "README.md", "TEMPLATES.md", "QUEUE.md"):
        ensure((ROOT / "research/method_discussion" / required).is_file(),
               "Method discussion entrypoint missing: " + required)

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
    # Preserve the completed passive receipt even when a later user instruction
    # creates a separate bounded scope. Neither completion grants future work.
    authority = active["experiment_authority"]
    bounded = authority.get("resume_scope", {})
    ensure(authority.get("completed_authorization_is_not_future_permission") is True and
           authority.get("new_model_experiment_requires_separate_explicit_user_instruction") is True and
           authority.get("automatic_followups") is False and
           bounded.get("execution_status") == "COMPLETE" and
           bounded.get("pending_model_jobs") == 0,
           "Completed E0 bounded observation vs fresh experiment permission is ambiguous")
    completed = [*authority.get("completed_resume_scopes", []), bounded]
    passive_path = "config/e0_observation_replication_20261025.json"
    passive = [scope for scope in completed if scope.get("protocol") == passive_path]
    ensure(len(passive) == 1, "Missing or duplicated original passive completion receipt")
    passive = passive[0]
    ensure(passive.get("execution_status") == "COMPLETE" and
           passive.get("pending_model_jobs") == 0 and
           passive.get("protocol_sha256") ==
           hashlib.sha256((ROOT / passive_path).read_bytes()).hexdigest(),
           "Historical passive completion or protocol identity changed")
    budget = passive.get("actual_budget", {})
    strata = passive.get("actual_source_strata", {})
    ensure(budget.get("source_clusters") == 6 and
           budget.get("full_generations") == 12 and
           budget.get("peak_simultaneous_GPU_cards", 99) <= 4 and
           strata.get("CAL24_subject") == 3 and
           strata.get("CAL24_background") == 3 and
           strata.get("camera") == 0,
           "Frozen E0 result scope or cohort description changed unexpectedly")
    if bounded.get("protocol") != passive_path:
        protocols = bounded.get("protocols", [])
        ensure(bool(protocols), "Later bounded scope lacks frozen protocols")
        generation_cap = 0
        for item in protocols:
            path = ROOT / item["path"]
            ensure(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() ==
                   item["sha256"], "Later bounded protocol identity changed")
            generation_cap += record(item["path"])["max_full_generations_total"]
        actual = bounded.get("actual_budget", {})
        ensure(bounded.get("max_full_generations_total") == generation_cap and
               0 < actual.get("full_generations", 0) <= generation_cap and
               0 < actual.get("peak_simultaneous_GPU_cards", 99) <=
               bounded.get("max_gpu_cards", 0) <= 4 and
               bool(bounded.get("completed_result")) and
               (ROOT / bounded["completed_result"]).is_file(),
               "Later completed execution is outside its explicit frozen bounds")
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
            "INFORMATION_FIRST_DISCOVERY_IS_VALID",
            "E0_CONTINUATION_DOES_NOT_PICK_OLD_OPERATOR",
            "E0_FP4_METADATA_REQUIRES_PHENOMENON_FIRST",
            "E0_IDEA_CREATIVITY_NOT_CENSORED",
            "E0_MULTIAGENT_INDEPENDENT_SCIENCE",
            "PROACTIVE_BROAD_WEB_SEARCH_ON_E0_RESEARCH",
            "MULTI_PASS_CITATION_AND_OPPOSITION_SEARCH",
            "SEARCH_REAL_PRIMARY_IMPLEMENTATION",
            "WEB_TOOL_UNAVAILABLE_NO_FICTION",
            "E0_NEW_FINDING_REQUIRES_INDEPENDENT_CHALLENGE",
            "E0_COHORT_AND_FORMAT_NONOVERCLAIM",
            "E0_COMPLETED_AUTHORIZATION_IS_NOT_FUTURE_RESUME",
            "E0_COMPLETE_FIELDS_NEXT_MECHANISM",
            "E0_QUANTIZER_CLOSURE_NOT_UNIVERSAL",
            "E0_POSTHOC_CONCENTRATION_VS_REAL_VALUE",
            "E0_BROAD_INDEPENDENT_REVIEW_AFTER_LATEST",
            "GRAPH_AGENT_NODE_CONTEXT_DEFAULT",
            "GRAPH_AGENT_PROPOSE_REVIEW_UPDATE",
            "GRAPH_AGENT_QUERY_IS_NOT_EXECUTABLE_TRACE",
            "E0_AFFINE_FOLLOWTHROUGH_NOT_SINGLE_ROUTE",
            "E0_EXACT_RESIDENCY_NOT_FFN_COMPUTE_REDUCTION",
            "E0_GRAPH_EXTENDS_AFTER_NEW_EVIDENCE",
            "E0_EVERY_GENERATED_VIDEO_REQUIRES_METRIC_REPORT",
            "E0_ALL_LAYER_APPROXIMATION_REQUIRES_REAL_CUMULATIVE_TRACE",
             "WARD_PINNED_WEB_RESEARCH_DEFAULT"}.issubset(case_ids),
           "Behavior case portfolio lost its symmetric discovery questions")
    ensure("only case.user_input" in behavior.get("evaluation_input_rule", ""),
           "Behavior-case answers may be leaked to candidate-agent evaluation")
    for path in ("AGENTS.md", "README.md", ".github/copilot-instructions.md",
                 ".agents/skills/ffnds-paper-discovery/SKILL.md",
                 "research/agent_protocol/README.md", "research/RESEARCH_NAVIGATION.md",
                 "research/agent_protocol/CURRENT_CONTEXT_PACKET.md",
                 "research/method_discussion/AGENTS.md",
                 "research/method_discussion/README.md",
                 "research/method_discussion/QUEUE.md",
                 "research/method_discussion/TEMPLATES.md"):
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
    check_relative_markdown_links("research/20261025/E0_GAPS_REMEDIATION_REQUIREMENTS.md")
    check_relative_markdown_links("research/20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md")
    check_relative_markdown_links("research/20261025/E0_FFN_FIRST_TASK_20261025.md")
    check_relative_markdown_links("research/20261025/E0_INTRINSIC_DIVERGENCE_AUDIT_20261025.md")
    check_relative_markdown_links("research/20261025/E0_POST_AFFINE_DIRECTION_REVIEW_20261025.md")
    check_relative_markdown_links("research/agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md")
    check_relative_markdown_links("research/agent_protocol/GRAPH_AGENT_COLLABORATION.md")
    check_relative_markdown_links("research/method_discussion/graph_annotations/E0/README.md")
    check_relative_markdown_links("docs/98_端到端计算图与证据注解_20261025.md")
    check_relative_markdown_links("research/20261025/E0_FFN_INTRINSIC_PRIOR_ART_20261025.md")
    check_relative_markdown_links("research/20261025/E0_FFN_KNOWLEDGE_GAPS_AND_DISCRIMINATORS_20261025.md")
    check_relative_markdown_links("core/WEB_RESEARCH_DISCOVERY.md")
    check_relative_markdown_links(".agents/skills/web-research-discovery/SKILL.md")
    check_relative_markdown_links("research/agent_protocol/UPSTREAM_SYNC.md")
    check_relative_markdown_links("research/agent_protocol/DEEP_WEB_RESEARCH.md")
    check_relative_markdown_links("research/agent_protocol/E0_STAGE_ANCHOR.md")
    check_relative_markdown_links("research/method_discussion/discoveries/E0/README.md")
    check_relative_markdown_links("research/20261024/FIRST_TASK_E0_QUANTIZATION_FFN_DISCOVERY_20261024.md")
    check_relative_markdown_links("ARCHIVE.md")
    check_relative_markdown_links("archive/20261024/historical_handoffs/README.md")

    # Evidence priority is a read-order aid. It must never authorize model
    # work or promote a filename-classified report into a native outcome.
    priority = record("research/experiment_evidence/PRIORITY_RANKING.json")
    inventory = record("research/experiment_evidence/FILE_INVENTORY.json")
    ensure(priority["execution_authority"] == "NONE" and
           active["experimental_evidence_navigation"]["experiment_authority"] == "NONE",
           "Evidence ranking became an execution queue")
    rows = priority["ranking"]
    ensure(priority["ranked_count"] == len(rows) and len(rows) >= 25 and
           [row["rank"] for row in rows] == list(range(1, len(rows) + 1)),
           "Curated experiment evidence ranking has duplicate/invalid ranks")
    ensure(len({row["path"] for row in rows}) == len(rows) and
           {row["evidence_maturity"] for row in rows} == {"T1", "T2", "T3"},
           "Evidence ranking lost primary/raw/synthetic/protocol separation")
    for row in rows:
        original = ROOT / row["path"]
        ensure(original.is_file() and
               original.stat().st_size == row["bytes"] and
               git_blob_sha1(original.read_bytes()) == row["git_blob_sha1"],
               "Ranked evidence SHA/path mismatch: " + row["path"])
    ensure(inventory["file_count"] == len(inventory["files"]) and
           len(inventory["files"]) > 1000 and
           inventory["all_unranked_role_labels"].startswith("Filename/extension"),
           "Experiment inventory incomplete or inferred outcome claims")
    full_inventory = {row["path"] for row in inventory["files"]}
    ensure(all(row["path"] in full_inventory for row in rows),
           "Curated experiment missing from full inventory")
    check_relative_markdown_links("research/EXPERIMENT_EVIDENCE_INDEX.md")
    check_relative_markdown_links("research/experiment_evidence/README.md")

    ensure(not (ROOT / "docs/94_AutoResearcher_算子优先重新探索_20261024.md").exists() and
           not (ROOT / "research/20261024/operator_first_autoresearcher_reexploration.json").exists(),
           "Superseded algorithm-first task is present")

    print("PASS: exact AutoResearcher v2.4 WARD source, shared skills and intrinsic FFN discovery route,")
    print("      Markdown links, E0 issues, completed permission scope and historic pause.")
    print("Scope: static structure only. No numerical experiment or agent behavioral run.")


if __name__ == "__main__":
    main()
