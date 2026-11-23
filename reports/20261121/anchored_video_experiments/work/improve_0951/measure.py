"""Measure the 0951 improvements on raw, receipt-verified 81-frame clips."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent/"vbench_scale24"), str(HERE.parent/"diagnose_mechanism_0951")]
import diagnose0951 as diag
import patch_transport as patch

OUT = ROOT/"outputs/FFNDS_0951_improvement"
HISTORY = ROOT/"outputs/FFNDS_0951_mechanism"
EVAL = OUT/"evaluation"
ARMS = ("dense", "old", "low_only", "fold_high", "fold_low_native_high", "h20", "compact_low20")
CONDS = ("pan_left", "base", "pan_right")
LABELS = dict(dense="Dense", old="原双步锚点", low_only="仅低步近似",
              fold_high="高步折叠", fold_low_native_high="高步原生＋低步折叠", h20="高步半数层原生",
              compact_low20="高步原生＋仅低步20层近似")


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    with NamedTemporaryFile(mode="w",encoding="utf8",dir=path.parent,delete=False) as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=1, allow_nan=False)
        tmp=Path(stream.name)
    tmp.replace(path)


def files(cond, arm):
    name = "vbi_0951" + ("" if cond=="base" else "_"+cond) + ".mp4"
    current = OUT/arm/name
    if current.is_file() and current.with_suffix(".json").is_file():
        return current
    standalone=OUT/"compact_dense"/name
    if arm=="dense" and standalone.is_file() and standalone.with_suffix(".json").is_file():
        return standalone
    baseline = diag.ROOT if cond=="base" else diag.ROOT/"camera7"
    if arm in ("dense", "old"):
        return baseline/("candidate" if arm=="old" else arm)/name
    if arm=="low_only":
        historical = HISTORY/"runtime"/arm/name
        if historical.is_file():
            return historical
    return None


def trajectory(path):
    """Return warm-foreground centroid and area after registering the outer background."""
    cv.setRNGSeed(42)
    cap = cv.VideoCapture(str(path))
    points, area, support, dx = [], [], [], []
    total = np.eye(3)
    mask = np.full((diag.H, diag.W), 255, dtype=np.uint8)
    mask[diag.H//4:3*diag.H//4, diag.W//4:3*diag.W//4] = 0
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if prev is not None:
            corners = cv.goodFeaturesToTrack(prev, 400, .01, 8, mask=mask)
            nxt, valid, _ = cv.calcOpticalFlowPyrLK(prev, gray, corners, None)
            rev, valid2, _ = cv.calcOpticalFlowPyrLK(gray, prev, nxt, None)
            good = valid.ravel().astype(bool) & valid2.ravel().astype(bool)
            good &= np.linalg.norm(rev-corners, axis=2).ravel()<1
            a, b = corners[good,0], nxt[good,0]
            affine, inliers = cv.estimateAffinePartial2D(a, b, method=cv.RANSAC, ransacReprojThreshold=2.)
            if affine is None or inliers.sum()<12:
                raise ValueError(f"background registration failed: {path} frame {len(points)}")
            support.append(float(inliers.mean()))
            centre = np.array([diag.W/2, diag.H/2])
            dx.append(float((affine[:,:2]@centre + affine[:,2]-centre)[0]))
            total = np.vstack((affine, [0.,0.,1.])) @ total
        fg, _, _ = diag.region(frame)
        y, x = np.nonzero(fg)
        points.append(np.linalg.solve(total, [x.mean(), y.mean(), 1.])[:2].tolist())
        area.append(float(len(y)/np.linalg.det(total[:2,:2])))
        prev = gray
    cap.release()
    assert len(points)==81
    return dict(centroid=points, area=area, background_inliers=support, background_dx=sum(dx))


def compare(path, ref):
    xy, base = (np.asarray(x["centroid"]) for x in (path,ref))
    # A single largest-warm-component switch on the first frame contaminates every
    # subsequent displacement if used as the origin. Both tracks already live in
    # the same input-image coordinate system; no per-arm origin is estimated.
    def smoothed_position(points):
        return np.stack([np.median(points[max(0,i-2):i+3],axis=0) for i in range(len(points))])
    d, truth = smoothed_position(xy), smoothed_position(base)
    err = np.linalg.norm(d-truth, axis=1)
    v, t = d[4:]-d[:-4], truth[4:]-truth[:-4]
    proj = float((t*v).sum()/np.square(t).sum())
    cos = float((t*v).sum()/(np.linalg.norm(t)*np.linalg.norm(v)))
    logarea = np.log(np.array(path["area"])/np.array(ref["area"]))
    return dict(displacement_median_error_px=float(np.median(err)), displacement_p90_error_px=float(np.percentile(err,90)),
                signed_motion_gain=proj, motion_cosine=cos, median_abs_log_area_ratio=float(np.median(abs(logarea))))


def measure(cond, arm, path):
    rec = read(path.with_suffix(".json"))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert rec["video"]["mp4_sha256"]==sha and len(rec["step_ms"])==4
    asset = EVAL/"raw"/f"{cond}_{arm}_{sha[:12]}.json"
    if asset.is_file():
        scores = read(asset)
        assert scores["sha256"]==sha
    else:
        mapped = "candidate" if arm=="old" else arm
        old = (HISTORY/"factorial" if arm=="low_only" else diag.OUT)/f"{cond}_{mapped}.json"
        if old.is_file() and read(old)["sha256"]==sha:
            diag_data = read(old)
        else:
            cv.setRNGSeed(42)
            diag_data, _ = diag.run(path)
        pold = HISTORY/"patches"/f"{cond}_{mapped}.json"
        track = read(pold) if pold.is_file() and read(pold)["sha256"]==sha else patch.score(path)
        scores = dict(sha256=sha, diagnostic=diag_data, patch=track, trajectory=trajectory(path))
        write(asset, scores)
    times = dict(wall_sec=rec["wall"], step_ms=rec["step_ms"], job=rec["job"],
                 allocated_gib=rec["peak_allocated"]/2**30,
                 anchor_gib=rec.get("counters",{}).get("anchor_gpu_capacity_bytes",0)/2**30)
    return rec, scores, times


def evaluate(require_new):
    cv.setNumThreads(2)
    cv.setRNGSeed(42)
    result = {}
    for cond in CONDS:
        rows, observations, receipts = {}, {}, {}
        for arm in ARMS:
            path = files(cond, arm)
            if path is None or not path.is_file():
                continue
            rec, scores, times = measure(cond, arm, path)
            rows[arm] = dict(**scores["diagnostic"]["all"], patch_ncc=scores["patch"]["all"]["best"],
                             background_dx=scores["trajectory"]["background_dx"], timing=times, path=str(path.relative_to(ROOT)))
            observations[arm] = scores
            receipts[arm] = rec
        ref = "low_only" if "low_only" in rows else "dense"
        for arm in rows:
            rec = receipts[arm]
            for key in ("latents","flows"):
                assert rec[key][0]==receipts["dense"][key][0] and rec["initial"]==receipts["dense"]["initial"]
                if arm in ("fold_low_native_high","compact_low20"):
                    assert rec[key][:3]==receipts["dense"][key][:3]
            legacy=None
            if arm=="old" and (OUT/"old"/Path(rows[arm]["path"]).name).is_file():
                legacy_dir = diag.ROOT if cond=="base" else diag.ROOT/"camera7"
                legacy=read(legacy_dir/"candidate"/(Path(rows[arm]["path"]).stem+".json"))
            if arm=="low_only" and (OUT/"low_only"/Path(rows[arm]["path"]).name).is_file():
                legacy=read(HISTORY/"runtime"/"low_only"/(Path(rows[arm]["path"]).stem+".json"))
            if arm=="dense" and Path(rows[arm]["path"]).parent.name=="compact_dense":
                legacy_dir = diag.ROOT if cond=="base" else diag.ROOT/"camera7"
                legacy=read(legacy_dir/"dense"/(Path(rows[arm]["path"]).stem+".json"))
            if legacy:
                match={k:rec[k]==legacy[k] for k in ("initial","flows","latents")}
                assert all(match.values()),(cond,arm,match)
                for field in ("mp4_sha256","decoded_rgb24_sha256"):
                    match[field]=rec["video"][field]==legacy["video"][field]
                if not match["decoded_rgb24_sha256"]:
                    archived=EVAL/"raw"/f"{cond}_{arm}_{legacy['video']['mp4_sha256'][:12]}.json"
                    if archived.is_file():
                        hist=read(archived)["diagnostic"]["all"]
                        match["new_minus_archived_video_metrics"]={k:rows[arm][k]-hist[k] for k in
                            ("body_band_correlation","body_compensated_flicker","body_area_pct","edge_normal_accel_p75")}
                rows[arm]["historical_comparison"]=match
            rows[arm]["motion_reference"] = ref
            rows[arm]["motion"] = compare(observations[arm]["trajectory"], observations[ref]["trajectory"])
        result[cond] = rows
    if require_new:
        assert "fold_high" in result["pan_left"] and "h20" in result["pan_left"]
    write(EVAL/"实测指标.json", result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--require-new",action="store_true")
    evaluate(parser.parse_args().require_new)
