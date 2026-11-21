"""Measure adjacent-frame fine texture at matched foreground tracks in the three videos."""

from pathlib import Path
import hashlib
import json

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs/FFNDS_camera_cal_trial"
PATCH = 17
ARMS = ("dense", "A", "B")


def patches(frame, points):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)
    band = gray - cv.GaussianBlur(gray, (0, 0), 1.4)
    height, width = gray.shape
    inside = np.isfinite(points).all(axis=1)
    inside &= (points[:,0] >= 9) & (points[:,0] < width-9)
    inside &= (points[:,1] >= 9) & (points[:,1] < height-9)
    out = np.full((len(points), PATCH*PATCH), np.nan, dtype=np.float32)
    for j in np.flatnonzero(inside):
        tile = cv.getRectSubPix(band,(PATCH,PATCH),tuple(map(float,points[j])))
        out[j] = (tile - tile.mean()).ravel()
    return out


def metrics(x, y):
    ex, ey = np.mean(x*x,1), np.mean(y*y,1)
    # Stabilisation is symmetric and small compared with the Dense eligibility
    # threshold (RMS >= 2 grey levels). A flat estimate therefore scores zero.
    corr = np.mean(x*y,1) / np.sqrt((ex+0.01)*(ey+0.01))
    return dict(correlation=float(np.median(corr)), contrast=float(np.median(np.sqrt((ex+ey)/2))))


def measure(tracks, paths):
    readers = {arm: cv.VideoCapture(str(paths[arm])) for arm in ARMS}
    assert all(rd.isOpened() for rd in readers.values())
    assert all(tracks[a].ndim == 3 and tracks[a].shape == tracks["dense"].shape for a in ARMS)
    assert tracks["dense"].shape[0] == 81
    previous, rows = {}, []
    try:
        for t in range(81):
            current = {}
            for a in ARMS:
                ok, frame = readers[a].read()
                assert ok
                current[a] = patches(frame, tracks[a][t])
            if t:
                keep = np.logical_and.reduce([np.isfinite(z[a]).all(1) for z in (previous,current) for a in ARMS])
                keep &= np.mean(previous["dense"]**2,1) >= 4
                keep &= np.mean(current["dense"]**2,1) >= 4
                n = int(keep.sum())
                result = {a:metrics(previous[a][keep],current[a][keep]) for a in ARMS} if n >= 3 else None
                rows.append(dict(to_frame=t,common_texture_points=n,arms=result))
            previous = current
    finally:
        for reader in readers.values():
            reader.release()
    good = [r for r in rows if r["arms"] is not None]
    summary = {a: {k: float(np.median([r["arms"][a][k] for r in good])) for k in ("correlation","contrast")} for a in ARMS} if good else None
    return dict(usable_transitions=len(good), possible_transitions=80,
                median_point_count=float(np.median([r["common_texture_points"] for r in good])) if good else None,
                median_over_transitions=summary, rows=rows)


def controls():
    rng = np.random.default_rng(11051)
    x = rng.normal(128,30,(150,180)).clip(0,255).astype(np.uint8)
    bgr = lambda im: cv.cvtColor(im,cv.COLOR_GRAY2BGR)
    p = np.array([[30,30],[70,70],[90,85],[100,110],[135,30]],np.float32)
    shift = np.float32([3,1])
    warped = cv.warpAffine(x,np.float32([[1,0,3],[0,1,1]]),(180,150))
    base = patches(bgr(x),p)
    rigid = metrics(base,patches(bgr(warped),p+shift))
    blurred = cv.GaussianBlur(warped,(0,0),1.5)
    blur = metrics(patches(bgr(cv.GaussianBlur(x,(0,0),1.5)),p),patches(bgr(blurred),p+shift))
    random_frame = rng.normal(128,30,(150,180)).clip(0,255).astype(np.uint8)
    noise = metrics(base,patches(bgr(random_frame),p+shift))
    assert rigid["correlation"] > .99 and blur["contrast"] < .3*rigid["contrast"] and noise["correlation"] < .3
    return dict(rigid=rigid, blurred=blur, unrelated=noise)


def main():
    cv.setNumThreads(2)
    out = OUT / "evaluation"
    result = dict(definition="Median across adjacent video frames of median registered high-band patch correlation and contrast on shared surviving first-frame foreground tracks. At least 3 common matched 17x17 patches per transition and Dense contrast >= 2 grey levels at both ends. Contrast is a grey-level RMS. A fixed matched support is used across methods within a condition.", controls=controls(), trajectories={})
    for t in json.loads((HERE/"protocol.json").read_bytes())["trajectories"]:
        if t["split"] != "validation":
            continue
        ident = t["trajectory_id"]
        paths, tracks, bindings = {}, {}, {}
        for a in ARMS:
            directory = OUT/"dense" if a == "dense" else OUT/"render_ab"/a
            video = directory/(ident+".mp4")
            receipt = json.loads(video.with_suffix(".json").read_bytes())
            signature = hashlib.sha256(video.read_bytes()).hexdigest()
            assert signature == receipt["video"]["mp4_sha256"]
            trackfile = out/"lk"/f"{ident}_{'Dense' if a=='dense' else a}.npz"
            with np.load(trackfile,allow_pickle=False) as data:
                tracks[a] = data["xy"]
            paths[a] = video
            bindings[a] = dict(video_sha256=signature,tracks_sha256=hashlib.sha256(trackfile.read_bytes()).hexdigest())
        checks=measure(tracks,paths)
        result["trajectories"][ident] = dict(bindings=bindings,**checks)
        print(ident,checks["usable_transitions"],checks["median_over_transitions"])
    (out/"foreground_texture.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
