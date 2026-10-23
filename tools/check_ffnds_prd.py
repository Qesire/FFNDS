#!/usr/bin/env python3
"""Verify exact upstream AutoResearcher PRD source and FFNDS exploratory ledger."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from validate_representation import validate_ledger
m=json.loads((ROOT/"research/agent_protocol/PRD_SOURCE_MANIFEST.json").read_text())
assert m["source_commit"]=="0684b560f94504cf458c42e594dc6522e6f61e27"
assert len(m["copied_paths"])==8
for p in m["copied_paths"]:
    raw=(ROOT/p["path"]).read_bytes()
    blob_sha=hashlib.sha1(("blob "+str(len(raw))+"\0").encode()+raw).hexdigest()
    assert blob_sha==p["git_blob_sha"],p["path"]
ledger=json.loads((ROOT/"research/20261023/FFNDS_PRD_representation_ledger.json").read_text())
validate_ledger(ledger)
assert len(ledger["representations"])>=5
assert len(ledger["candidates"])>=3 and all(c["status"]=="SEED" for c in ledger["candidates"])
decision=json.loads((ROOT/"research/CURRENT_DECISION.json").read_text())
assert decision["eligible_model_method_ids"]==[]
assert json.loads((ROOT/"research/20261023/R17E0_execution_summary.json").read_text())["actual_debits"]["metric_evaluator_calls"]==0
print("FFNDS PRD exact upstream source, valid representation ledger, zero model authorization: OK")
