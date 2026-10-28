#!/usr/bin/env python3
"""Read-only FFNDS scientific computation-graph index for Agent research.

No model, network, GPU, or job execution. Registry edges include hypotheses:
a returned path is a research dependency sketch, not verified causal proof.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research/agent_protocol/GRAPH_AGENT_BINDINGS.json"

def load():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))

def load_crosswalk(data):
    metadata = data.get("graph_synthesis_20261027")
    if not metadata:
        return None
    path = ROOT / metadata["crosswalk"]
    assert path.is_file(), f"Missing research mechanism crosswalk: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_crosswalk(data):
    """Validate existing research mechanism metadata, not numerical/model validity."""
    cw = load_crosswalk(data)
    if cw is None:
        return {}
    assert (ROOT / cw["document"]).is_file(), "Missing current FFN science synthesis"
    index_path = ROOT / cw["paper_index"]
    assert index_path.is_file(), f"Missing primary paper registry: {index_path}"
    papers = json.loads(index_path.read_text(encoding="utf-8"))["records"]
    paper_ids = [p["id"] for p in papers]
    assert len(paper_ids) == len(set(paper_ids)), "Duplicate primary paper IDs"
    paper_set = set(paper_ids)
    node_ids = {n["id"] for n in data["nodes"]}
    edge_keys = {(e["from"], e["to"], e["relation"]) for e in data["edges"]}
    family_ids = set()
    for f in cw["families"]:
        assert f["id"] not in family_ids, f"Repeated mechanism family: {f['id']}"
        family_ids.add(f["id"])
        for field in ("home", "source", "comparison"):
            for identifier in f[field]:
                assert identifier in node_ids, f"Unknown {field} node {identifier} in {f['id']}"
        for paper in f["papers"]:
            assert paper in paper_set, f"Unknown paper {paper} in {f['id']}"
        assert f["information"] and f["cost"] and f["falsifier"], f["id"]
    for edge in cw["synthesis_edges"]:
        key = (edge["from"], edge["to"], edge["relation"])
        assert key in edge_keys, f"Crosswalk relation missing in main graph: {key}"
        assert edge["edge_type"] == "SCIENTIFIC_COMPARISON_OR_CONDITIONAL", key
        assert (ROOT / edge["evidence_document"]).is_file(), key
    assert len(cw["synthesis_edges"]) == data["graph_synthesis_20261027"]["new_typed_comparison_edges"], "Synthesis edge count mismatch"
    return {
        "mechanism_families": len(family_ids),
        "primary_papers": len(paper_ids),
        "crossgraph_comparison_edges": len(cw["synthesis_edges"]),
        "warning": "Literature and conditional comparisons are research mappings, not native model execution.",
    }



def load_paper_focus(data):
    ref=data.get("paper_method_focus_v2")
    if not ref: return None
    file=ROOT/ref["overlay"]
    assert file.is_file(), "Missing registered literature focus graph"
    return json.loads(file.read_text(encoding="utf-8"))


def validate_paper_focus(data):
    """Validate every collected paper against the actual G14 diagrams and indexed evidence."""
    lens=load_paper_focus(data)
    if lens is None: return {}
    spec=data["paper_method_focus_v2"]
    assert lens["schema"]=="ffnds.paper_method_focus_graph.v2"
    original=json.loads((ROOT/lens["primary_index"]).read_text(encoding="utf-8"))
    catalogue={p["id"]:p for p in original["records"]}
    papers=lens["paper_links"]; other=lens["supplemental_links"]
    assert len(papers)==spec["primary_papers"]==len(catalogue)
    assert len(other)==len(lens["supplemental_papers"])==spec["supplemental_papers"]
    assert {p["id"] for p in papers}==set(catalogue), "All indexed primary paper IDs must be linked"
    assert len({p["id"] for p in other})==len(other)
    assert not ({p["id"] for p in other}&set(catalogue))
    additions={p["id"]:p for p in lens["supplemental_papers"]}
    assert {p["id"] for p in other}==set(additions)
    stages=lens["focus_nodes"]
    stage_ids={s["id"] for s in stages}
    assert len(stage_ids)==len(stages)==spec["focus_nodes"]
    ids={node["id"] for node in data["nodes"]}
    edges={(e["from"],e["to"],e["relation"]) for e in data["edges"]}
    for s in stages:
        assert s["id"] in ids and s["native_anchor"] in ids-stage_ids
        assert (s["id"],s["native_anchor"],"literature_focus_mapping_not_native_execution") in edges
        assert s["execution_status"]=="NOT_RUN_FOR_PAPER"
    assert len(stages)==spec["typed_analogical_edges"]
    document=(ROOT/lens["graph_document"]).read_text(encoding="utf-8")
    for paper in papers+other:
        pid=paper["id"]
        targets=paper["focus_nodes"]
        assert targets and set(targets)<=stage_ids and len(set(targets))==len(targets),pid
        assert (paper["source_url"].startswith("https://") and paper["review_depth"]
            and paper["mechanism"] and paper["intervention"] and paper["decision_time"]
            and paper["wan_transfer_question"]),pid
        assert document.count(pid+'["')==1,f"Missing or repeated Mermaid paper node {pid}"
        for target in targets:
            assert f'{pid} -. "关注/作用" .-> {target}' in document,(pid,target)
        src=catalogue.get(pid,additions.get(pid))
        assert paper["source_url"]==src["url"] and paper["review_depth"]==src["reading_depth"]
        assert paper["mechanism"]==src["observed_mechanism"]
        assert paper["intervention"]==src["physical_action"]
        assert paper["decision_time"]==src["decision_time"]
        assert paper["wan_transfer_question"]==src["wan_specific_discriminator"]
    return {"paper_focus_primary":len(papers),"paper_focus_supplemental":len(other),
            "paper_focus_stages":len(stages),
            "paper_to_stage_links":sum(len(p["focus_nodes"]) for p in papers+other),
            "paper_overlay":"LIT_STATIC_NOT_EXECUTED"}


def inspect_paper(data, paper_id):
    lens=load_paper_focus(data)
    if lens is None: raise SystemExit("No G14 literature overlay")
    paper=next((p for p in lens["paper_links"]+lens["supplemental_links"]
               if p["id"]==paper_id),None)
    if paper is None: raise SystemExit(f"Unknown literature ID: {paper_id}")
    index=json.loads((ROOT/lens["primary_index"]).read_text(encoding="utf-8"))
    original=next((p for p in index["records"]+lens["supplemental_papers"]
                   if p["id"]==paper_id),None)
    stages={p["id"]:p for p in lens["focus_nodes"]}
    return {"paper":original,"method_graph":paper,
            "native_stage_analogies":[stages[s] for s in paper["focus_nodes"]],
            "warning":"Source-tiered literature mapping, not verified native Wan execution."}


def inspect_focus(data, focus_id):
    lens=load_paper_focus(data)
    if lens is None: raise SystemExit("No G14 literature overlay")
    focus=next((f for f in lens["focus_nodes"] if f["id"]==focus_id),None)
    if focus is None: raise SystemExit(f"Unknown stage: {focus_id}")
    candidates=[p for p in lens["paper_links"]+lens["supplemental_links"]
                if focus_id in p["focus_nodes"]]
    return {"focus":focus,
            "primary_papers":[p["id"] for p in lens["paper_links"] if focus_id in p["focus_nodes"]],
            "supplemental_papers":[p["id"] for p in lens["supplemental_links"] if focus_id in p["focus_nodes"]],
            "method_comparisons":[{"id":p["id"], "author_method":p["mechanism"],
                                  "decision_time":p["decision_time"],
                                  "physical_or_training_action":p["intervention"],
                                  "reading_depth":p["review_depth"],
                                  "wan_transfer_discriminator":p["wan_transfer_question"]}
                                  for p in candidates],
            "warning":"This relation is a scientific lens, not original model execution."}



def validate_method_operation_crosswalk(data):
    """Check five-operation methods against G14 provenance, not native execution."""
    pointer = data.get("current_method_prior_constraint_view")
    if not pointer:
        return {}
    for key in ("method_plan", "source_graph", "typed_prior_edges", "visible_graph_section"):
        assert pointer.get(key), f"Missing G14 method pointer: {key}"
    for key in ("method_plan", "source_graph", "typed_prior_edges"):
        assert (ROOT / pointer[key]).is_file(), f"Missing method reference {key}: {pointer[key]}"
    graph_path, _, section = pointer["visible_graph_section"].partition("#")
    assert section, "G14 method diagram requires explicit section anchor"
    assert (ROOT / graph_path).is_file(), "Missing G14 method diagram"
    assert section in (ROOT / graph_path).read_text(encoding="utf-8"), "Missing G14-C method section"

    method_map = json.loads((ROOT / pointer["typed_prior_edges"]).read_text(encoding="utf-8"))
    lens = json.loads((ROOT / pointer["source_graph"]).read_text(encoding="utf-8"))
    assert method_map["schema"] == "ffnds.method_operation_crosswalk.v2"
    assert method_map["current_method_plan"] == pointer["method_plan"]
    assert (ROOT / method_map["next_agent_task"]).is_file()
    assert method_map["next_agent_current_stage"] == "M-B_BH_BF_TERMINAL_COMPLETE__RAW_BF16_CROSS_DEPTH_NUMERATOR_NORM_STATE"
    assert method_map["one_turn_next_agent"] == "M-B"

    operations = set(method_map["operation_definitions"])
    assert operations == {"O-A", "O-B", "O-C", "O-D", "O-E"}
    assert set(method_map["candidate_aliases"]) == {"M-A", "M-B", "M-C"}
    methods = method_map["project_methods"]
    names = [item["name"] for item in methods]
    assert len(names) == len(set(names)), "Duplicate project method name"
    for method in methods:
        assert set(method["operations"]) <= operations, method["name"]

    valid_papers = {x["id"] for x in lens["paper_links"] + lens["supplemental_links"]}
    valid_stages = {x["id"] for x in lens["focus_nodes"]}
    literature = method_map["literature_examples"]
    literature_ids = [x["source_id"] for x in literature]
    assert len(literature_ids) == len(set(literature_ids)), "Duplicate literature example"
    for entry in literature:
        assert entry["source_id"].startswith("EXTERNAL_") or entry["source_id"] in valid_papers
        assert entry["operations"] and set(entry["operations"]) <= operations

    progress = method_map["methods_progress"]
    assert {item["method"] for item in progress} == {"M-A", "M-B", "M-C"}
    for item in progress:
        assert (ROOT / item["evidence"]).is_file(), f"Missing project evidence: {item['evidence']}"
    b_progress = next(item for item in progress if item["method"] == "M-B")
    assert b_progress["status"].startswith("B0_CAL24_LABELS_B1_FIXED_POOL_COMPLETE__")
    for recent in b_progress["recent_completed_results"]:
        recent_path = (ROOT / recent["path"]).resolve()
        assert recent_path.is_relative_to(ROOT.resolve()), "Recent report must stay in repository"
        recent_raw = recent_path.read_bytes()
        assert len(recent_raw) == recent["bytes"] and hashlib.sha256(recent_raw).hexdigest() == recent["sha256"], "Changed recent result binding"
    binding = b_progress["native_result"]
    result_path = (ROOT / binding["path"]).resolve()
    assert result_path.is_relative_to(ROOT.resolve()), "Native report must stay in repository"
    raw = result_path.read_bytes()
    assert len(raw) == binding["bytes"] and hashlib.sha256(raw).hexdigest() == binding["sha256"], "Changed native report binding"
    report = json.loads(raw)
    assert report["evidence"] == "NATIVE_CAPTURE_AND_SAVED_HEAD_DIAGNOSTIC"
    assert report["b0"]["launch_exit"] == report["b1"]["launch_exit"] == 0
    assert report["b0"]["historical_raw_fp32_poststate_match"] == 4
    assert report["b0"]["FFN_calls"] == 160 and report["b0"]["independent_sources"] == 2
    assert report["b1"]["observed"]["head_calls"] == 72
    assert report["b1"]["observed"]["video"] == report["b1"]["observed"]["fit"] == 0
    latest = b_progress["latest_completed_result"]
    latest_path = (ROOT / latest["path"]).resolve()
    assert latest_path.is_relative_to(ROOT.resolve()), "Latest report must stay in repository"
    latest_raw = latest_path.read_bytes()
    assert len(latest_raw) == latest["bytes"] and hashlib.sha256(latest_raw).hexdigest() == latest["sha256"]
    completed = json.loads(latest_raw)
    assert completed["labels"]["totals"]["sources"] == 24 and completed["labels"]["totals"]["cases"] == 96
    assert completed["paired_CPU_M256"]["head_calls"] == 448
    assert completed["paired_CPU_M256"]["full_M_native_calls"] == 0
    assert completed["B0_full_M_neutral"]["calls"] == 16 and not completed["B0_full_M_neutral"]["full_state_reproduction"]

    allowed_relations = set(method_map["relation_types"])
    assert allowed_relations == {
        "REQUIRES", "CONSTRAINS", "COMPETES_WITH", "PREDICTS_DIFFERENTLY"
    }
    edges = method_map["method_prior_constraint_edges"]
    unique = set()
    for edge in edges:
        assert edge["method"] in {"M-A", "M-B", "M-C"}
        assert edge["relation"] in allowed_relations
        assert edge["paper_ids"] and set(edge["paper_ids"]) <= valid_papers
        assert edge["focus_nodes"] and set(edge["focus_nodes"]) <= valid_stages
        assert edge["constraint"] and edge["decision"] and edge["status"]
        identity = (edge["method"], edge["relation"], tuple(edge["paper_ids"]),
                    tuple(edge["focus_nodes"]))
        assert identity not in unique, f"Duplicate method prior relation {identity}"
        unique.add(identity)
    assert len(edges) >= 12, "Active method prior relations are incomplete"
    return {
        "method_operation_types": len(operations),
        "method_literature_examples": len(literature_ids),
        "method_project_routes": len(methods),
        "method_prior_constraint_relations": len(edges),
        "method_next_agent": method_map["one_turn_next_agent"],
        "method_next_stage": method_map["next_agent_current_stage"],
        "method_validation_scope": "Static references only; no model or peer review",
    }


def validate(data):
    assert data.get("schema_version") == "1.0", "Unsupported graph registry schema"
    graph = ROOT / data["graph_document"]
    assert graph.is_file(), f"Missing graph document: {graph}"
    text = graph.read_text(encoding="utf-8")
    required_types = {"SRC","ALG","OBS","NATIVE-OBS","NATIVE-ACT","SYN","VIDEO","SYS","OPEN","POLICY","LIT","SPEC"}
    nodes = data.get("nodes", [])
    ids = [n["id"] for n in nodes]
    assert len(ids) == len(set(ids)) and ids, "Duplicate/missing graph node IDs"
    for n in nodes:
        node_id = n["id"]
        # Match a Mermaid quoted NODE LABEL, rather than passing on mere prose.
        pattern = r'[\[\{]"' + re.escape(node_id) + r'(?=[\s,:：])'
        assert re.search(pattern, text), f"Node not found as Mermaid label: {node_id}"
        assert n["graph_section"] in data["graph_roles"], f"Unknown graph section: {node_id}"
        assert n["evidence_type"] in required_types, f"Bad evidence type: {node_id}"
        assert n["meaning"] and n["availability"] and n["open_question"], node_id
        assert n["evidence"], f"Unbound evidence: {node_id}"
        for p in n["evidence"]:
            assert (ROOT / p).is_file(), f"Missing node evidence {node_id}: {p}"
    for e in data.get("edges", []):
        assert e["from"] in ids and e["to"] in ids, f"Unknown edge endpoint: {e}"
        assert e["relation"], f"Unlabeled edge: {e}"
    for key in ("claim_protocol", "annotation_template", "current_scientific_audit"):
        assert (ROOT / data[key]).is_file(), f"Missing graph collaboration reference: {key}"
    synthesized = validate_crosswalk(data)
    paper_map = validate_paper_focus(data)
    method_map = validate_method_operation_crosswalk(data)
    return {"status":"STATIC_GRAPH_BRIDGE_VALID", "nodes":len(nodes), "edges":len(data["edges"]),
            "scope":"Graph labels, local evidence and mechanism-crosswalk references only; no executed kernel or agent evaluation.",
            **synthesized, **paper_map, **method_map}

def inspect(data, node_id):
    nodes = {n["id"]:n for n in data["nodes"]}
    if node_id not in nodes:
        raise SystemExit(f"Unknown node: {node_id}")
    return {"node":nodes[node_id],
            "incoming":[e for e in data["edges"] if e["to"]==node_id],
            "outgoing":[e for e in data["edges"] if e["from"]==node_id],
            "graph_document":data["graph_document"],
            "warning":"Edges marked as candidate/posthoc are not executed mechanisms."}

def trace(data, start, end):
    ids={n["id"] for n in data["nodes"]}
    if start not in ids or end not in ids:
        raise SystemExit("Unknown start/end graph node")
    adj={key:[] for key in ids}
    for e in data["edges"]:
        adj[e["from"]].append(e)
    todo=deque([(start,[])])
    visited={start}
    while todo:
        current,path=todo.popleft()
        if current == end:
            return {"start":start,"end":end,"edges":path,
                    "warning":"This is a registry path, not proof of a causal or executable candidate."}
        for e in adj[current]:
            if e["to"] not in visited:
                visited.add(e["to"])
                todo.append((e["to"],path+[e]))
    return {"start":start,"end":end,"edges":[],
            "note":"No currently registered path; the actual source may support other mechanisms."}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=["validate","nodes","inspect","trace","mechanisms","mechanism","papers","paper","focus"])
    ap.add_argument("ids",nargs="*")
    args=ap.parse_args()
    data=load()
    validation = validate(data)
    if args.command=="validate":
        result=validation
    elif args.command=="mechanisms":
        cw = load_crosswalk(data)
        result = [{"id": f["id"], "name": f["name"], "tier": f["tier"]}
                  for f in cw["families"]] if cw is not None else []
    elif args.command=="mechanism" and len(args.ids)==1:
        cw = load_crosswalk(data)
        if cw is None:
            raise SystemExit("No mechanism crosswalk registered")
        family = next((f for f in cw["families"] if f["id"]==args.ids[0]), None)
        if family is None:
            raise SystemExit(f"Unknown mechanism family: {args.ids[0]}")
        matched = [e for e in cw["synthesis_edges"]
                   if e["from"] in family["home"] or e["to"] in family["home"]]
        result = {"family": family, "crossgraph_edges": matched,
                  "warning": "Scientific comparison, not verified native execution"}
    elif args.command=="papers":
        lens=load_paper_focus(data)
        result=[{"id":p["id"],"focus_nodes":p["focus_nodes"],"review_depth":p["review_depth"],
                 "source_url":p["source_url"]} for p in lens["paper_links"]+lens["supplemental_links"]]
    elif args.command=="paper" and len(args.ids)==1:
        result=inspect_paper(data,args.ids[0])
    elif args.command=="focus" and len(args.ids)==1:
        result=inspect_focus(data,args.ids[0])
    elif args.command=="nodes":
        result=[{"id":n["id"],"graph_section":n["graph_section"],"meaning":n["meaning"],
                 "open_question":n["open_question"]} for n in data["nodes"]]
    elif args.command=="inspect" and len(args.ids)==1:
        result=inspect(data,args.ids[0])
    elif args.command=="trace" and len(args.ids)==2:
        result=trace(data,*args.ids)
    else:
        ap.error("inspect/mechanism/paper/focus requires one ID; trace requires start and end nodes")
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
