"""Compare four-frame foreground point displacements at common tracked positions."""
import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from measure_camera import fit_background, OUT


# Fixed by inspecting the first Dense source frame before any A/B video exists.
# Coordinates are in the 416 x 232 preview. All inputs are 832 x 464.
REGIONS = {
    "0905": [[(162,50),(175,47),(185,50),(187,56),(177,64),(165,63)],
             [(163,77),(183,73),(201,86),(212,106),(229,149),(233,176),
              (202,182),(183,169),(169,132),(161,104)]],
    "1014": [[(207,31),(214,35),(218,47),(229,61),(221,73),(205,67),(200,50)],
             [(138,54),(152,56),(166,73),(181,86),(188,106),(182,119),(158,104),(143,80)],
             [(180,110),(204,120),(215,109),(234,90),(269,91),(290,98),(308,111),
              (311,132),(296,147),(273,144),(246,137),(214,145),(180,138)]],
}


def features(path, source):
    cap = cv.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (464,832,3)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    for polygon in REGIONS[source]:
        cv.fillPoly(mask,[np.asarray(polygon, dtype=np.int32)*2],255)
    pts = cv.goodFeaturesToTrack(gray, 140,.005,5,mask=mask)
    assert pts is not None and len(pts)>=10
    return pts.reshape(-1,2)


def tracks(path, seeds, filename):
    cap = cv.VideoCapture(str(path))
    ok, first = cap.read()
    assert ok
    prev = cv.cvtColor(first,cv.COLOR_BGR2GRAY)
    points = seeds.copy()
    frames, bg, support = [points.copy()], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        curr = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        affine = fit_background(prev,curr)
        bg.append(None if affine is None else np.vstack([affine[0], [0.,0.,1.]]))
        support.append(0. if affine is None else affine[1])
        idx = np.flatnonzero(np.isfinite(points[:,0]))
        next_points = np.full_like(points, np.nan)
        if len(idx):
            pts = points[idx].reshape(-1,1,2)
            nxt, valid, _ = cv.calcOpticalFlowPyrLK(prev,curr,pts,None)
            back, reverse, _ = cv.calcOpticalFlowPyrLK(curr,prev,nxt,None)
            h,w = prev.shape
            x,y = nxt[:,0].T
            keep = valid.ravel().astype(bool) & reverse.ravel().astype(bool) & (np.linalg.norm(back[:,0]-pts[:,0],axis=1)<.65)
            keep &= (x>=3)&(x<w-3)&(y>=3)&(y<h-3)
            next_points[idx[keep]] = nxt[keep,0]
        points = next_points
        frames.append(points.copy())
        prev = curr
    cap.release()
    assert len(frames) == 81 and len(bg) == 80
    pos = np.asarray(frames)
    np.savez_compressed(filename, xy=pos, background_inliers=support, affine=np.asarray([np.full((3,3),np.nan) if m is None else m for m in bg]))
    residual = np.full((77,len(seeds),2),np.nan)
    for t in range(77):
        if min(support[t:t+4]) < .60:
            continue
        transform = np.eye(3)
        for h in bg[t:t+4]:
            transform = h @ transform
        residual[t] = pos[t+4] - (pos[t]@transform[:2,:2].T + transform[:2,2])
    valid = np.isfinite(residual[:,:,0])
    med = [np.median(np.linalg.norm(residual[t,valid[t]],axis=1)) for t in range(77) if valid[t].sum()>=4]
    return residual, dict(initial_features=len(seeds), tracked_counts_t0_t20_t40_t60_t80=[int(np.isfinite(pos[t,:,0]).sum()) for t in (0,20,40,60,80)],
                         outer_affine_success=sum(v is not None for v in bg), matched_transition_count=sum(s>=.60 for s in support),
                         median_background_inliers=float(np.median(support)), supported_four_frame_windows=len(med),
                         median_four_frame_residual_px=float(np.median(med)) if med else None)


def paired(vectors, threshold=4):
    truth = vectors["Dense"]
    common = np.logical_and.reduce([np.isfinite(v[:,:,0]) for v in vectors.values()])
    counts = common.sum(1)
    common[counts < threshold] = False
    output = dict(four_frame_windows=int((counts>=threshold).sum()), surviving_points_per_window=counts.tolist())
    if not common.any():
        return output
    # Each available time window receives equal total weight across its survivors.
    weight = np.broadcast_to((1/np.maximum(counts,1))[:,None],common.shape)[common]
    true = truth[common]
    den = float((weight * np.sum(true*true,axis=1)).sum())
    if den == 0:
        return output
    for name, v in vectors.items():
        if name == "Dense": continue
        pred = v[common]
        dot = float((weight*np.sum(true*pred,axis=1)).sum())
        gain = dot / den
        def energy(array):
            return float((weight*np.sum(array**2,axis=1)).sum())
        output[name] = dict(projected_gain=gain, orthogonal_relative_rms=(energy(pred-gain*true)/den)**.5,
                            relative_rms=(energy(pred-true)/den)**.5,
                            displacement_rmse_px=(energy(pred-true)/weight.sum())**.5,
                            direction_cosine=dot/(den*energy(pred))**.5 if energy(pred)>0 else None)
    return output


def main():
    p = argparse.ArgumentParser()
    flag = p.add_mutually_exclusive_group()
    flag.add_argument("--closed-loop",action="store_true")
    flag.add_argument("--right",action="store_true")
    p.add_argument("--available",action="store_true")
    args = p.parse_args()
    cv.setNumThreads(2)
    directory = OUT/"evaluation"
    (directory/"lk").mkdir(exist_ok=True,parents=True)
    report = {"definition":"First-Dense-frame foreground polygons with common seed corners; every four-frame displacement subtracts outer-frame similarity motion. Require forward-backward error <0.65 px and at least 60% RANSAC inliers in each contributing background fit. Survival does not certify object identity. Pooled vector projections weight supported time windows equally and use identical surviving tracks for all compared arms.",
              "first_frame_polygons_at_416x232":REGIONS, "clips":{}}
    for source in REGIONS:
        seeds = features(OUT/"dense"/f"vbi_{source}_base_s44.mp4",source)
        for condition in (("right",) if args.right else ("base","camera")):
            stem = f"vbi_{source}_{condition}_s44"
            base_folder = OUT/"right_control" if args.right else OUT
            paths = {"Dense":base_folder/"dense"/(stem+".mp4")}
            if args.right and args.available and not paths["Dense"].is_file():
                continue
            if args.closed_loop:
                paths.update({k:OUT/"render_ab"/k/(stem+".mp4") for k in ("A","B")})
                if args.available and not all(v.is_file() and v.with_suffix(".json").is_file() for v in paths.values()):
                    continue
            vectors,coverage = {},{}
            for name,path in paths.items():
                cv.setRNGSeed(42)
                vectors[name],coverage[name] = tracks(path,seeds,directory/"lk"/(stem+"_"+name+".npz"))
            r = dict(coverage=coverage)
            if args.closed_loop: r["paired_against_dense"] = paired(vectors)
            report["clips"][stem] = r
    filename = "right_subject_motion.json" if args.right else "closed_loop_subject_motion.json" if args.closed_loop else "dense_subject_motion.json"
    file = directory/filename
    file.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
