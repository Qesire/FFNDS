"""Validate registered FFNDS multisource evidence design; no model execution."""
import json
import re
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"config/e0_multisource_directional_evaluation_20261026.json"
def check(d):
    errors=[]
    src=d["phase1"]["source_records"]
    ids=[s["index"] for s in src]
    if sorted(ids)!=[0,4,8,12,16,20] or len(set(s["source_id"] for s in src))!=6: errors.append("Six distinct verified sources required")
    if sorted(d["phase1"]["add_source_indices"])!=[4,8,16,20]: errors.append("Four new exposed sources required")
    for s in src:
        if not re.fullmatch("[0-9a-f]{64}",s["source_id"]): errors.append("Invalid source SHA")
        if s["index"] in [4,8,16,20] and (s["role"]!="EXPOSED_AFFINE54_CALIBRATION" or s["paired_native_completed"]): errors.append("Calibration leak/native completion incorrectly marked")
    if d["phase1"]["expected_new_sites"]!=4*40*4 or d["phase1"]["expected_new_videos"]!=8: errors.append("Incorrect cost scope")
    if d["phase2"]["minimum_additional_distinct_sources"]<4 or d["phase2"]["minimum_total_distinct_sources"]<10: errors.append("Independent source coverage insufficient")
    if d["evaluation"]["global_SSE"]!="SECONDARY_DIAGNOSTIC_ONLY": errors.append("Global SSE should not drive ranking")
    if d["status"]!="DESIGNED_NOT_NATIVE_EXECUTED" or d["execution"]["this_registration_is_not_run"]!=True: errors.append("Unfounded native execution claim")
    return errors
if __name__=="__main__":
    d=json.loads(P.read_text())
    errors=check(d)
    if errors: raise SystemExit("; ".join(errors))
    print("PASS source and metric design; native expansion not yet executed")
