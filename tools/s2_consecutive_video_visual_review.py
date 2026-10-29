"""CPU-only synchronized screenshot + difference-sheet extraction of REAL FFNDS videos.

This is not image generation, new VBench scoring, or automatic visual diagnosis.
Inputs must be original HF output.mp4 and registered generation receipts; each
input video is SHA-256 verified against exact repository research evidence.
Outputs are chronological 9-frame strips, each at 2-frame cadence, for
same-source Dense/value10/dcac10. Never modify the historical videos.

Example on asset-owning worker:
  CUDA_VISIBLE_DEVICES='' python tools/s2_consecutive_video_visual_review.py \
    --hf-root /home/xiang/.cache/huggingface \
    --manifest config/s2_consecutive_video_visual_review_20261029_draft.json \
    --out /home/xiang/.cache/huggingface/ffnds-derived/s2_consecutive_visual_review/attempt1
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("dense", "value10", "dcac10")
EXPECTED_FPS = 16
EXPECTED_FRAMES = 81
EXPECTED_WH = (832, 464)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def contained(hf_root, name):
    p = Path(name)
    target = (p if p.is_absolute() else hf_root / p).resolve()
    require(target.is_relative_to(hf_root.resolve()), "An input video or receipt is outside HF root")
    return target


def bind_source(hf_root, source):
    require(source["seed"] == 42 and source["source"] >= 0, "Source/seed mismatch")
    require(set(source["arms"]) == set(ARMS), "Require original Dense and both fixed candidates")
    bound = {}
    for arm in ARMS:
        spec = source["arms"][arm]
        receipt = spec["generation_receipt"]
        p = contained(hf_root, receipt["path"])
        require(p.is_file() and p.stat().st_size == receipt["bytes"]
                and file_sha256(p) == receipt["sha256"], "Generation receipt identity mismatch")
        gen = json.loads(p.read_text())
        require("video" in gen and isinstance(gen["video"], dict), "Generation video binding missing")
        v = gen["video"]
        require(v.get("sha256") == spec["video_sha256"], "Generation video SHA mismatch")
        video_path = contained(hf_root, v["path"])
        require(video_path.is_file() and file_sha256(video_path) == spec["video_sha256"],
                "Video missing or SHA mismatch")
        require(v.get("frames") == EXPECTED_FRAMES and v.get("fps") == EXPECTED_FPS
                and list(v.get("size", [])) == list(EXPECTED_WH), "Recorded real video geometry mismatch")
        bound[arm] = video_path
    return bound


def decode_selected(path, selected, cv2):
    cap = cv2.VideoCapture(str(path))
    require(cap.isOpened(), "Could not decode "+str(path))
    try:
        meta = (round(cap.get(cv2.CAP_PROP_FRAME_COUNT)), round(cap.get(cv2.CAP_PROP_FPS)),
                round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        require(meta == (EXPECTED_FRAMES, EXPECTED_FPS, *EXPECTED_WH), "MP4 frame/fps/size mismatch")
        frames = {}
        for i in range(EXPECTED_FRAMES):
            success, frame = cap.read()
            require(success and frame.shape[:2] == (EXPECTED_WH[1], EXPECTED_WH[0]),
                    "Incomplete real 81-frame decode")
            if i in selected:
                frames[i] = frame.copy()
        require(not cap.read()[0], "More than 81 decoded frames")
        require(set(frames) == set(selected), "Selected frame coverage missing")
        return frames
    finally:
        cap.release()


def draw_sheet(frames_by_arm, frame_numbers, cv2, np, *, absolute_difference):
    cell_w = 368
    cell_h = 205
    header_h = 32
    rows = []
    metrics = []
    for j, frame_index in enumerate(frame_numbers):
        cells = []
        changes = []
        for arm in ARMS:
            frame = cv2.resize(frames_by_arm[arm][frame_index], (cell_w, cell_h),
                               interpolation=cv2.INTER_AREA)
            if absolute_difference:
                if j == 0:
                    shown = np.zeros_like(frame)
                    change = None
                else:
                    prior = cv2.resize(frames_by_arm[arm][frame_numbers[j-1]],
                                       (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                    diff = cv2.absdiff(frame, prior)
                    change = float(np.mean(diff))
                    shown = np.clip(diff.astype(np.float32)*3.0, 0, 255).astype(np.uint8)
            else:
                shown = frame
                change = None
            region = np.full((cell_h+header_h,cell_w,3), 24, dtype=np.uint8)
            region[header_h:] = shown
            kind = "abs RGB delta x3" if absolute_difference else "source video RGB"
            cv2.putText(region, f"{arm} | F{frame_index:02d} t={frame_index/16:.3f}s",
                        (8,21), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255),1,cv2.LINE_AA)
            cells.append(region)
            changes.append(change)
        rows.append(np.concatenate(cells, axis=1))
        metrics.append({"frame":frame_index,"mean_abs_rgb_delta_from_previous_selected":
                       dict(zip(ARMS,changes))})
    return np.concatenate(rows,axis=0), metrics


def run(args):
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    hf = args.hf_root.resolve()
    require(hf.is_dir(), "Real asset HF root not present")
    plan_path = args.manifest.resolve()
    plan = json.loads(plan_path.read_text())
    require(plan["schema"] == "ffnds.s2.consecutive_video_screenshot_sample.v1"
            and plan["kind"] == "DESIGN_ONLY_NOT_EXTRACTED", "Unexpected/modified source manifest")
    require(plan["rules"]["fps"] == EXPECTED_FPS
            and plan["rules"]["frames"] == EXPECTED_FRAMES
            and plan["rules"]["size_wh"] == list(EXPECTED_WH)
            and plan["rules"]["all_arms"] == list(ARMS), "Source protocol mismatch")
    sources = plan["sources"]
    require(len(sources)==7 and len({s["source"] for s in sources})==7,
            "Exactly seven preselected different image clusters expected")
    if args.only_source is not None:
        sources = [s for s in sources if s["source"] == args.only_source]
        require(len(sources)==1,"Selected source not in frozen list")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Explicitly hide GPUs for passive CPU screenshot task")
    out = args.out.resolve()
    require(out.is_relative_to(hf) and not out.exists(), "Fresh output within HF is required")
    # Verify every video and hash before creating any output.
    bound = {s["source"]:bind_source(hf,s) for s in sources}
    out.mkdir(parents=True)
    started = time.monotonic()
    report={"schema":"ffnds.s2.real_screenshot_visual_evidence.v1",
            "status":"RUNNING_REAL_VIDEO_SHA_VERIFIED",
            "source_manifest":str(plan_path),"manifest_sha256":file_sha256(plan_path),
            "method":"true decoded video frames, same time across arms, stride=2",
            "source_selection":"PRESELECTED_EXTREMES_NOT_POPULATION_INFERENCE",
            "effects":"RGB difference is not motion ground truth; no causal attribution",
            "model_calls":0,"gpu_calls":0,"video_generations":0,"sources":[]}
    def save():
        (out/"RUN.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    save()
    try:
        for source in sources:
            numbers = set()
            for w in source["windows"]:
                require(w["frames"] == list(range(w["start"],w["start"]+17,2))
                        and w["start"]>=0 and w["start"]+16<=80,
                        "Expected 9 consecutive 2-frame screenshots from 81-frame source")
                numbers.update(w["frames"])
            videos = {arm:decode_selected(bound[source["source"]][arm],numbers,cv2) for arm in ARMS}
            subdir=out/f"ord{source['source']}_seed42"
            subdir.mkdir()
            details={"source":source["source"],"source_id":source["source_id"],
                     "image_sha256":source["image_sha256"],"prompt_sha256":source["prompt_sha256"],
                     "stratum":source["stratum"],"seed":source["seed"],
                     "reason":source["reason"],
                     "true_video_sha256":{a:source["arms"][a]["video_sha256"] for a in ARMS},
                     "windows":[]}
            for w in source["windows"]:
                rec={"time_seconds":w["seconds"],"frames":w["frames"]}
                for mode in ("RGB","ABS_DIFF"):
                    sheet, changes=draw_sheet(videos,w["frames"],cv2,np,
                                              absolute_difference=(mode=="ABS_DIFF"))
                    dst=subdir/f"frames_{w['start']:02d}_{w['start']+16:02d}_{mode}.jpg"
                    success=cv2.imwrite(str(dst),sheet,[cv2.IMWRITE_JPEG_QUALITY,91])
                    require(success,"Contact-sheet write failed")
                    rec[mode]={"file":str(dst.relative_to(out)),
                               "sha256":file_sha256(dst),"bytes":dst.stat().st_size,
                               "dimensions_wh":[int(sheet.shape[1]),int(sheet.shape[0])]}
                    if mode=="ABS_DIFF":
                        rec["selected_step_mean_abs_rgb_delta"]=changes
                details["windows"].append(rec)
            report["sources"].append(details)
            save()
        report.update(status="COMPLETE_REAL_VIDEO_CONTACT_SHEETS_NOT_HUMAN_ACCEPTANCE",
                      "wall_seconds":time.monotonic()-started,
                      "next":"Human must inspect original RGB and ABS_DIFF sheets across all arms; score subject motion, background, camera translation, object geometry, flicker and temporal persistence; do not equate RGB deltas with motion.")
    except Exception as e:
        report.update(status="FAILED_NO_RETRY",error_type=type(e).__name__,error=str(e))
        raise
    finally:
        save()


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-root",required=True,type=Path)
    ap.add_argument("--manifest",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--only-source",type=int)
    run(ap.parse_args())
