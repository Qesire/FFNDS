"""Summarise apparent horizontal background motion, with support and temporal segments."""

import hashlib
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from measure_camera import fit_background, OUT


def describe(rows):
    selected = [r for r in rows if r is not None and r["inliers"] >= .60]
    dx = np.asarray([r["dx"] for r in selected])
    return dict(total=len(rows), supported=len(selected), median_dx=float(np.median(dx)) if len(dx) else None,
                mean_dx=float(np.mean(dx)) if len(dx) else None,
                screen_right_over_quarter_px=int(np.sum(dx>.25)),screen_left_over_quarter_px=int(np.sum(dx<-.25)),
                subquarter_px=int(np.sum(np.abs(dx)<=.25)))


def score(path):
    cap = cv.VideoCapture(str(path))
    ok, frame = cap.read()
    assert ok
    previous = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    height,width = previous.shape
    center = np.array([width/2,height/2])
    rows, composite = [], np.eye(3)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            fit = fit_background(previous,current)
            previous = current
            if fit is None:
                rows.append(None)
                continue
            mat, fraction = fit
            shift = mat[:,:2]@center+mat[:,2]-center
            composite = np.vstack((mat,[0.,0.,1.])) @ composite
            rows.append(dict(dx=float(shift[0]),dy=float(shift[1]),inliers=fraction))
    finally:
        cap.release()
    assert len(rows)==80
    net = (composite[:2,:2]@center+composite[:2,2]-center).tolist() if None not in rows else None
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), full=describe(rows),
                thirds=[describe(rows[:27]),describe(rows[27:53]),describe(rows[53:])],
                net_center_xy_all_fits=net, frames=rows)


def run(files, output):
    cv.setNumThreads(2)
    scores = {}
    for key,path in files.items():
        cv.setRNGSeed(42)
        scores[key] = score(path)
        control = json.loads(path.with_suffix(".json").read_bytes())
        assert scores[key]["sha256"] == control["video"]["mp4_sha256"]
        print(key, scores[key]["full"], scores[key]["net_center_xy_all_fits"])
    result = dict(definition="Signed adjacent-frame translation of the image centre under an outer-feature robust similarity fit. Positive x means image moves right. The support filter retains fits with >=60% inliers; the direction counts use a 0.25px dead band. The composed net transform uses all fitted transitions. Segments include 27,26,27 transitions. Neither subject motion nor 3D camera pose is measured.",trajectories=scores)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right-0905",type=Path)
    parser.add_argument("--right-1014",type=Path)
    args=parser.parse_args()
    assert (args.right_0905 is None) == (args.right_1014 is None)
    files={}
    for source, extra in (("0905",args.right_0905),("1014",args.right_1014)):
        for cond,label in (("base","base"),("camera","pan_left")):
            files[source+"_"+label]=OUT/"dense"/f"vbi_{source}_{cond}_s44.mp4"
        if extra is not None:
            files[source+"_pan_right"]=extra
    run(files,OUT/("background_direction_checks.json" if args.right_0905 is not None else "original_left_background.json"))
