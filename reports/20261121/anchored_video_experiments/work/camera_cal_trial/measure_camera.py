"""Check realised image motion on Dense teacher videos without object-specific masks."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "outputs/FFNDS_camera_cal_trial"


def fit_background(previous, current):
    height, width = previous.shape
    mask = np.full_like(previous, 255)
    mask[height//4:3*height//4, width//4:3*width//4] = 0
    first = cv.goodFeaturesToTrack(previous, 400, 0.01, 8, mask=mask)
    if first is None or len(first) < 8:
        return None
    last, valid, _ = cv.calcOpticalFlowPyrLK(previous, current, first, None)
    back, reverse, _ = cv.calcOpticalFlowPyrLK(current, previous, last, None)
    keep = valid.ravel().astype(bool) & reverse.ravel().astype(bool) & (np.linalg.norm(back-first, axis=2).ravel() < 1)
    if keep.sum() < 8:
        return None
    affine, inliers = cv.estimateAffinePartial2D(first.reshape(-1,2)[keep], last.reshape(-1,2)[keep],
                                                  method=cv.RANSAC, ransacReprojThreshold=2)
    if affine is None or inliers.sum() < 8:
        return None
    return affine, float(inliers.mean())


def clip_motion(path):
    capture = cv.VideoCapture(str(path))
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Cannot decode {path}")
    previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height, width = previous.shape
    center = np.array([width/2, height/2])
    offsets = np.array([[0,0],[-.32,-.32],[.32,-.32],[.32,.32],[-.32,.32]])
    probes = center + offsets * [width,height]
    composite = np.eye(3)
    step_rms, support = [], []
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        count += 1
        current = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        fit = fit_background(previous, current)
        previous = current
        if fit is None:
            continue
        matrix, fraction = fit
        displacement = probes @ matrix[:,:2].T + matrix[:,2] - probes
        step_rms.append(float(np.sqrt(np.square(displacement).sum(1).mean())))
        support.append(fraction)
        composite = np.vstack([matrix,[0,0,1]]) @ composite
    capture.release()
    if count != 80:
        raise ValueError(f"Expected 80 transitions in {path}, got {count}")
    complete = len(step_rms) == count
    shift = composite[:2,:2] @ center + composite[:2,2] - center
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), transitions=count, valid=len(step_rms),
                mean_rms_px=float(np.mean(step_rms)) if step_rms else None,
                median_inlier_fraction=float(np.median(support)) if support else None,
                net_center_x_px=float(shift[0]) if complete else None,
                net_center_y_px=float(shift[1]) if complete else None,
                net_scale=float(np.linalg.norm(composite[:2,0])) if complete else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=OUT/"dense", type=Path)
    parser.add_argument("--output", default=OUT/"dense_motion.json", type=Path)
    args = parser.parse_args()
    protocol = json.loads((HERE/"protocol.json").read_bytes())
    cv.setNumThreads(2)
    cv.setRNGSeed(42)
    rows = {}
    for t in protocol["trajectories"]:
        ident = t["trajectory_id"]
        rows[ident] = clip_motion(args.directory/(ident+".mp4"))
    pairs = []
    for source in protocol["sources"]:
        key = source["record_id"]
        seed = 43 if source["split"] == "calibration" else 44
        a, b = [rows[f"{key}_{cond}_s{seed}"] for cond in ("base","camera")]
        pairs.append(dict(source=key, clause=source["clause"], seed=seed, base=a, camera=b))
    result = dict(definition="Outer-frame robust similarity fit. Positive x is apparent image movement to the right. Whole-clip shift/scale require all 80 fits. Subject motion is not measured.", trajectories=rows, pairs=pairs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
