"""Plot saved radial pairing controls; percentages are local geometry, not speed."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairing", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    r, t = json.loads(args.pairing.read_text()), json.loads(args.template.read_text())
    output = args.outdir / "radial_pairing"
    args.outdir.mkdir(parents=True, exist_ok=True)
    if output.with_suffix(".png").exists() or output.with_suffix(".pdf").exists():
        raise FileExistsError("Preserve the first figure")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), layout="constrained")
    states = ["pre_FFN_state", "actual_post_FFN_state", "decoded_leave_group_out_state"]
    x = np.arange(3)
    for ax, representation, title in zip(axes[:2], ["raw", "position_centered"],
                                        ["Raw alignment survives wrong pairing",
                                         "Position-centered relation retains pairing"]):
        for offset, key, label, color in [(-.18, "paired_radial_fraction", "Correct row", "#4477aa"),
                                          (.18, "deranged_radial_fraction", "Wrong row: exact mean", "#ee7733")]:
            values = [100*r["by_block"]["0"]["states"][s][representation][key] for s in states]
            bars = ax.bar(x+offset, values, .34, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set_xticks(x, ["Pre FFN", "Post FFN", "Post minus group"])
        ax.set_ylabel("Radial energy / corresponding direction energy (%)")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 29 if representation == "raw" else 11)
        ax.legend(fontsize=8, loc="upper right")
    sources = list(r["block0_by_source"])
    for key, label, color, marker in [
        ("actual_paired_radial_fraction", "Actual paired rows", "#4477aa", "o"),
        ("own_mean_radial_fraction", "Own-source mean direction", "#228833", "s"),
        ("other_five_mean_radial_fraction", "Other-five-source mean", "#aa3377", "x")
    ]:
        values = [100*t["block0_by_source"][i]["actual_post_FFN_state"][key] for i in sources]
        axes[2].plot(sources, values, marker=marker, color=color, label=label, linewidth=1)
    axes[2].set_ylim(0, 29)
    axes[2].set_xlabel("Held source (all seed42)")
    axes[2].set_ylabel("Post-FFN radial fraction (%)")
    axes[2].set_title("Common direction transfers across sources", fontsize=11)
    axes[2].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Block0 · 6 exposed sources · 4 steps · 5 groups · 32 rows · saved FP64 geometry only", fontsize=12)
    fig.savefig(output.with_suffix(".png"), dpi=170)
    fig.savefig(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
