"""Check that every new video in a local run bundle has one complete metric report.

This tool has no model access; register actual MP4 and quality JSON output paths.
A finished generation alone cannot pass this evidence gate.
"""
import argparse
import json
import hashlib
from pathlib import Path
from ffnds_video_metrics import validate, digest


def inspect(bundle_path, require_acceptance=False):
    bundle=json.loads(Path(bundle_path).read_text())
    if bundle.get("schema")!="ffnds.video.bundle.v1":
        raise ValueError("Wrong video evidence bundle schema")
    entries=bundle.get("videos")
    if not isinstance(entries,list) or not entries:
        raise ValueError("Video bundle must contain at least one actual generated video")
    base=Path(bundle_path).resolve().parent
    errors=[]
    seen=set()
    for index,entry in enumerate(entries):
        key=(entry.get("source_id"),entry.get("seed"),entry.get("variant"))
        if key in seen or any(v in ("",None) for v in key):
            errors.append(f"{index}: duplicate/incomplete source-seed-variant")
        seen.add(key)
        path=base/entry.get("video_path","")
        evidence=base/entry.get("metric_report_path","")
        if not path.is_file() or not evidence.is_file():
            errors.append(f"{index}: missing video or metric report")
            continue
        if digest(path)!=entry.get("video_sha256"):
            errors.append(f"{index}: video digest mismatch")
        if digest(evidence)!=entry.get("metric_report_sha256"):
            errors.append(f"{index}: metric report digest mismatch")
        try:
            report=json.loads(evidence.read_text())
            errors.extend(f"{index}: {message}" for message in validate(report,require_acceptance))
            if report.get("candidate",{}).get("sha256")!=entry.get("video_sha256"):
                errors.append(f"{index}: report/video artifact digest mismatch")
            identity=report.get("identity",{})
            for k in ("source_id","seed"):
                if identity.get(k)!=entry.get(k):
                    errors.append(f"{index}: identity mismatch {k}")
        except (ValueError,TypeError) as exc:
            errors.append(f"{index}: unreadable report {exc}")
    if bundle.get("expected_video_count")!=len(entries):
        errors.append("expected video count differs from registered entries")
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--require-acceptance",action="store_true")
    args=p.parse_args()
    failures=inspect(args.bundle,args.require_acceptance)
    if failures:
        raise SystemExit("VIDEO_EVIDENCE_INCOMPLETE:\n"+"\n".join(failures))
    print("ALL_VIDEOS_HAVE_CORE_METRIC_REPORTS; QUALITY_ACCEPTED_ONLY_IF_SEPARATELY_REVIEWED")


if __name__=="__main__":
    main()
