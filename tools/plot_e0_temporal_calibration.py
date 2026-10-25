"""Plot source-held-out saved-operand temporal controls, not video quality."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["reuse", "fixed_scale", "fixed_affine", "input_row_RMS_calibrated"]
LABELS = ["Reuse", "Fixed scale", "Fixed affine", "Input row RMS"]
COLORS = ["#929292", "#4477aa", "#228833", "#ee7733"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / "temporal_calibration"
    if output.with_suffix(".png").exists() or output.with_suffix(".pdf").exists():
        raise FileExistsError("Preserve the original figure")
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.3), layout="constrained")
    blocks = [0, 10, 20, 30, 39]
    positions = np.arange(len(blocks))
    for k, (name, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
        values = [100 * result["by_block"][str(b)]["predictors"][name]["mean_relative_SSE"]
                  for b in blocks]
        axes[0].bar(positions + (k - 1.5) * .18, values, width=.17, label=label, color=color)
    axes[0].set_xticks(positions, blocks)
    axes[0].set_xlabel("Block")
    axes[0].set_ylabel("SSE / full target energy (%)")
    axes[0].set_title("Fixed controls explain much of the drift")
    axes[0].legend(fontsize=8)
    for k, name in enumerate(MODELS[1:]):
        color = COLORS[k + 1]
        for x, step in enumerate([1, 3]):
            rows = [r for r in result["records"] if r["block"] == 39 and r["target_step"] == step]
            values = [100 * r["predictors"][name]["relative_SSE"] for r in rows]
            loc = x + (k - 1) * .22
            axes[1].scatter(loc + np.linspace(-.04, .04, len(values)), values, color=color, s=18, alpha=.65)
            axes[1].plot([loc - .075, loc + .075], [np.mean(values)] * 2, color=color, linewidth=3,
                         label=LABELS[k + 1] if x == 0 else None)
    axes[1].set_xticks([0, 1], ["High expert 0→1", "Low expert 2→3"])
    axes[1].set_ylabel("SSE / full target energy (%)")
    axes[1].set_title("Block39: expert transitions differ")
    axes[1].legend(fontsize=8)
    rows = [r for r in result["records"] if r["block"] == 39]
    for k, name in enumerate(MODELS[1:]):
        full = np.mean([r["predictors"][name]["relative_SSE"] for r in rows])
        centered = np.mean([r["predictors"][name]["centered_error_over_target_energy"] /
                            (1 - r["description"]["current_spatial_mean_energy_fraction"]) for r in rows])
        axes[2].bar(k - .17, 100 * full, width=.32, color=COLORS[k + 1])
        axes[2].bar(k + .17, 100 * centered, width=.32, color=COLORS[k + 1], hatch="///", alpha=.55)
    axes[2].set_xticks(range(3), LABELS[1:])
    axes[2].set_ylabel("SSE / corresponding target energy (%)")
    axes[2].set_title("Block39: variation error stays substantial")
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Patch(facecolor="#777777", label="Full output"),
                            Patch(facecolor="#aaaaaa", hatch="///", label="Position-centered output")], fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("Wan2.2 saved dense prefixes · 6 exposed sources · 32 rows · no executed cache / quality claim")
    fig.savefig(output.with_suffix(".png"), dpi=170)
    fig.savefig(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
