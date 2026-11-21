"""Compare foreground shapes after analytically removing translation and isotropic size."""
from pathlib import Path
import hashlib
import sys

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "improve_0951"))
from measure import ARMS, CONDS, EVAL, diag, files, read, write


DIRECTIONS = ((1,0), (2,1), (1,1), (1,2), (0,1), (-1,2), (-1,1), (-2,1))


def describe(mask):
    y, x = np.nonzero(mask)
    area = len(y)
    shape = []
    for dx, dy in DIRECTIONS:
        projected = x*dx + y*dy
        offset = projected.min()
        hist = np.bincount(projected - offset).cumsum()
        low, high = np.searchsorted(hist, np.array([.025, .975])*area) + offset
        center = projected.mean()
        scale = (area*(dx*dx + dy*dy))**.5
        shape.extend([(center-low)/scale, (high-center)/scale])
    # The warm-region estimator has explicit boundaries. Count contact per frame.
    margin = np.array([x.min()-.03*diag.W, .82*diag.W-x.max(), .90*diag.H-y.max(), y.min()])
    return shape, float(margin.min())


def extract(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == read(path.with_suffix(".json"))["video"]["mp4_sha256"]
    dest = EVAL / "silhouette" / (digest + ".json")
    if dest.exists():
        return read(dest)
    cap = cv.VideoCapture(str(path))
    rows, margins = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        shape, margin = describe(diag.region(frame)[0])
        rows.append(shape)
        margins.append(margin)
    cap.release()
    assert len(rows) == 81
    data = dict(shape=rows, boundary_margin=margins, sha256=digest)
    write(dest, data)
    return data


def compare(a, ref):
    def temporal_rms(x):
        return np.sqrt(np.mean(x*x, axis=1))
    x, y = (np.asarray(v["shape"]) for v in (a, ref))
    valid = np.minimum(a["boundary_margin"], ref["boundary_margin"]) > 2
    err = temporal_rms(x-y)
    second = temporal_rms(np.diff(x, n=2, axis=0))
    return dict(shape_deviation_median=float(np.median(err)),
                shape_deviation_interior=float(np.median(err[valid])) if valid.any() else None,
                joint_interior_frames=int(valid.sum()),
                shape_acceleration_median=float(np.median(second)),
                shape_acceleration_p90=float(np.quantile(second,.9)),
                frames_touching_window=int((np.array(a["boundary_margin"])<=2).sum()))


def controls():
    # Exact translation, large change in isotropic scale, and a true shape deformation.
    def make(center, axes):
        mask = np.zeros((diag.H, diag.W), np.uint8)
        cv.ellipse(mask, center, axes, 20, 0, 360, 1, -1)
        return np.array(describe(mask)[0])
    ref = make((260,190),(50,35))
    sets = [make((280,160), axes) for axes in ((50,35), (80,56), (80,35))]
    errors = [float(np.sqrt(np.mean((v-ref)**2))) for v in sets]
    assert errors[0] < 1e-9 and errors[1] < .01 and errors[2] > errors[1]*5
    return dict(zip(("translation_rms","isotropic_resize_rms","anisotropic_resize_rms"),errors))


def main():
    cv.setNumThreads(2)
    result = {}
    for cond in CONDS:
        items = {}
        for arm in ARMS:
            path = files(cond, arm)
            if path is not None and path.is_file():
                items[arm] = extract(path)
        ref = "low_only" if "low_only" in items else "dense"
        result[cond] = dict(reference=ref, arms={arm:compare(v, items[ref]) for arm,v in items.items()})
    known = controls()
    warm = EVAL.parent/"attempt1/warm_old/vbi_0951_pan_left.mp4"
    if warm.is_file():
        base = diag.ROOT/"camera7/candidate/vbi_0951_pan_left.mp4"
        known["same_latent_different_encoding"] = compare(extract(warm),extract(base))
    write(EVAL/"轮廓方向尺度复核.json", dict(controls=known, cases=result))
    explain(result["pan_left"], known)


def explain(left, known):
    labels = {"dense":"Dense","old":"原双步近似","low_only":"仅低步近似（动作参照）",
              "fold_high":"高步折叠","h20":"高步半数原生",
              "fold_low_native_high":"原生高步＋低步折叠","compact_low20":"原生高步＋低步半数近似"}
    lines = ["# 去除平移与尺度后的左摇轮廓", "",
             "每帧从既有暖色熊身选区内的像素计算八个平面方向上的投影。取 2.5%、97.5% 两个端点，各自扣除该方向质心，再除以主体面积平方根和方向向量长度，得到 16 个无量纲数。与仅低步近似版逐时刻比较，取这 16 个差的 RMS 后汇总时刻中位数。该量在连续平面上对平移与等比缩放不变，仍会响应姿态和暖色选区变化。", "",
             "“双方未触边”要求两条片该帧的暖色选区都距同一预设测量窗口边界大于 2 像素。分母均为 81 个时刻，子集因视频配对而不同，逐项不宜硬排优劣。时间二阶差是在每条片自身连续三帧上计算，反映形状随时间变化的曲率，正确的非匀速动作也会产生数值。", "",
             "| 版本 | 形状偏差 | 双方未触边时的偏差 | 双方未触边帧数 | 形状时间二阶差 | 自身触边帧数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, row in left["arms"].items():
        interior = row["shape_deviation_interior"]
        inside = "—" if interior is None else f"{interior:.3f}"
        lines.append(f'| {labels[arm]} | {row["shape_deviation_median"]:.3f} | {inside} | '
                     f'{row["joint_interior_frames"]} | {row["shape_acceleration_median"]:.3f} | {row["frames_touching_window"]} |')
    lines.extend(["", f'栅格椭圆平移的形状偏差小于 1e−12，等比放大 1.6 倍为 {known["isotropic_resize_rms"]:.4f}，只沿长轴方向放大为 {known["anisotropic_resize_rms"]:.4f}。固定暖色测量窗口仍可能裁到熊身，暖色对口鼻和阴影的覆盖也会改变。与动作参照的相似度不能替代运动审片。',
                  "", "[逐组原数](轮廓方向尺度复核.json) · [复算实现](../../../work/improve0951/silhouette.py)"])
    same = known.get("same_latent_different_encoding")
    if same:
        lines.extend(["",f'首轮预热与归档旧片所有潜变量相同，x264 编码线程不同。两者的中位形状偏差为 {same["shape_deviation_median"]:.3f}，双方同时未触边的 {same["joint_interior_frames"]} 帧为 {same["shape_deviation_interior"]:.3f}。这是此对片的测量敏感度，不能直接用作其他生成条件的误差上界。[原片指纹与编码对照](../attempt1/编码复核.json)'])
    (EVAL/"轮廓方向尺度复核.md").write_text("\n".join(lines)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
