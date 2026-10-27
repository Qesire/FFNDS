#!/usr/bin/env python3
"""Read-only FFNDS scientific computation-graph index for Agent research.

No model, network, GPU, or job execution. Registry edges include hypotheses:
a returned path is a research dependency sketch, not verified causal proof.
"""
from __future__ import annotations

import argparse
from collections import deque
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
    return {"status":"STATIC_GRAPH_BRIDGE_VALID", "nodes":len(nodes), "edges":len(data["edges"]),
            "scope":"Graph labels, local evidence and mechanism-crosswalk references only; no executed kernel or agent evaluation.",
            **synthesized}

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
    ap.add_argument("command",choices=["validate","nodes","inspect","trace","mechanisms","mechanism"])
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
    elif args.command=="nodes":
        result=[{"id":n["id"],"graph_section":n["graph_section"],"meaning":n["meaning"],
                 "open_question":n["open_question"]} for n in data["nodes"]]
    elif args.command=="inspect" and len(args.ids)==1:
        result=inspect(data,args.ids[0])
    elif args.command=="trace" and len(args.ids)==2:
        result=trace(data,*args.ids)
    else:
        ap.error("inspect/mechanism requires one ID; trace requires start and end nodes")
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
