"""Measure texture retention and axis-aligned structure in nine original, decoded MP4s."""
import json
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT.parents[1] / "2026-11-18/gen-j/work/state_coupled/blind_first_v1/private/baseline"
cv2.setNumThreads(1)


def measure(path):
    cap = cv2.VideoCapture(str(path))
    rows = []
    axis_profiles = [[], []]
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        assert y.shape == (464, 832)
        scale = float(np.std(y))
        # Adjacent pairs that do not straddle 4, 8, or 16 pixel block boundaries.
        dx = np.abs(np.diff(y, axis=1))
        dy = np.abs(np.diff(y, axis=0))
        inside_x = np.arange(1, y.shape[1]) % 4 != 0
        inside_y = np.arange(1, y.shape[0]) % 4 != 0
        contrast = (dx[:, inside_x].mean() + dy[inside_y, :].mean()) / (2 * scale)
        # Band-pass residual with length scales defined in pixels.
        blur1 = cv2.GaussianBlur(y, (0, 0), 1)
        blur4 = cv2.GaussianBlur(y, (0, 0), 4)
        rows.append([float(contrast), float(np.std(blur1-blur4)/scale), scale])
        for ax, arr in enumerate([dx.mean(axis=0), dy.mean(axis=1)]):
            axis_profiles[ax].append(arr/max(float(arr.mean()), 1e-6))
    cap.release()
    assert len(rows) == 81
    # Rank periodic modulation of gradient magnitudes in the first and second
    # spatial axes, using the mean profile over generated frames 17..80.
    spectra = {}
    for ax, profiles in zip(("x", "y"), axis_profiles):
        p = np.mean(profiles[17:], axis=0)
        win = np.hanning(p.size)
        p -= np.average(p, weights=win)
        amp = 2*np.abs(np.fft.rfft(win*p))/win.sum()
        idx = np.arange(1, len(amp))
        idx = idx[(p.size/idx >= 5) & (p.size/idx <= 80)]
        top = sorted(idx, key=lambda i: -amp[i])[:8]
        spectra[ax] = [[round(float(p.size/i), 2), round(float(amp[i]), 4)] for i in top]
    return dict(inside_edge_midband_luma_std=rows, common_axis_period_and_amplitude=spectra)


def main():
    result = {}
    for src in ("0392", "0800", "1013"):
        s = "vbi_"+src
        files = {"dense": BASE/s/"Dense/video.mp4", "joint": ROOT/f"outputs/FFNDS_joint_DEV3/{s}.mp4",
                 "zero": ROOT/f"outputs/FFNDS_zero_fused_DEV3/{s}.mp4"}
        result[src] = {key: measure(path) for key, path in files.items()}
    (HERE/"raw_grid.json").write_text(json.dumps(result, indent=1), encoding="utf8")
    for src, rec in result.items():
        print(src)
        for key, val in rec.items():
            a = np.asarray(val["inside_edge_midband_luma_std"])
            print(key, np.round([a[0,0],a[4,0],a[8,0],a[16,0],np.median(a[17:49,0]),np.median(a[49:,0]),np.median(a[17:,1])], 4),
                  {ax: ranks[:4] for ax,ranks in val["common_axis_period_and_amplitude"].items()})


if __name__ == "__main__":
    main()
