"""Standalone scientific overview of the saved CAL24 descriptive observations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--temporal", type=Path)
    ap.add_argument("--geometry-only", action="store_true")
    args = ap.parse_args()
    x = json.loads(args.result.read_text())
    if args.geometry_only:
        fig, ax = plt.subplots(figsize=(7.3, 4.2), layout="constrained")
        blocks = [0, 10, 20, 30]
        states = [("pre_FFN_state", "Before FFN", "o"),
                  ("actual_post_FFN_state", "After FFN", "s"),
                  ("decoded_leave_group_out_state", "After removing own group", "^")]
        for key, label, marker in states:
            ys = []
            for block in blocks:
                cases = [s for s in x["cases"] if s["block"] == block]
                ys.append(100 * np.mean([s[key]["total_directional_energy_removed_fraction"] for s in cases]))
            ax.plot(range(4), ys, marker=marker, label=label)
        ax.axhline(100 * 2 / 5120, color=".4", linestyle=":", label="Isotropic reference: 2 / 5120")
        ax.set(xticks=range(4), xticklabels=blocks, xlabel="FFN block (final block39 excluded)",
               ylabel="Direction energy removed by ideal LN geometry (%)", ylim=(0, 28),
               title="Radial alignment survives self-inclusion controls")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.suptitle("Six exposed sources; five fixed groups; saved native operands, FP64 analysis", fontsize=9)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=170)
        plt.close(fig)
        return
    if args.temporal:
        temporal = json.loads(args.temporal.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        colors = plt.get_cmap("tab10").colors
        for k, block in enumerate([0, 10, 20, 30, 39]):
            sites = [s for s in x["scopes"] if s["block"] == block]
            pairs = [s for s in temporal["pairs"] if s["block"] == block and s["same_expert"]]
            pre = [100*s["profiles"]["q2_pre"]["row_amplitude_times_static_profile_relative_SSE"] for s in sites]
            post = [100*s["profiles"]["q2_post"]["row_amplitude_times_static_profile_relative_SSE"] for s in sites]
            axes[0].scatter(pre, post, s=18, color=colors[k], alpha=.7, label=f"block {block}")
            axes[1].scatter([s["raw_Frobenius_cosine"] for s in pairs],
                            [s["row_mean_centered_Frobenius_cosine"] for s in pairs],
                            s=22, color=colors[k], alpha=.7)
        axes[0].plot([0, 25], [0, 25], "k:", linewidth=1)
        axes[0].set(xlabel="Pre-Q2 profile SSE (%)", ylabel="Post-Q2 profile SSE (%)",
                    title="Most coarse-profile variation precedes Q2", xlim=(0, 25), ylim=(0, 25))
        axes[1].plot([-1, 1], [-1, 1], "k:", linewidth=1)
        selected = [s for s in temporal["pairs"] if s["same_expert"]]
        axes[1].set(xlabel="Raw FFN temporal cosine", ylabel="Row-mean-centered temporal cosine",
                    title="Common means change temporal similarity",
                    xlim=(min(0, min(s["raw_Frobenius_cosine"] for s in selected) - .05), 1),
                    ylim=(min(0, min(s["row_mean_centered_Frobenius_cosine"] for s in selected) - .05), 1))
        axes[0].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle("Accepted native D-Q observations: six exposed sources; 32 uniform rows/site", fontsize=10)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=170)
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for index in x["source_indices"]:
        scopes = [s for s in x["scopes"] if s["source_index"] == index]
        zeros, residuals = [], []
        for step in range(4):
            subset = [s for s in scopes if s["step"] == step]
            zeros.append(100 * np.mean([s["V2"]["zero_element_fraction"] for s in subset]))
            residuals.append(100 * np.mean([s["V2"]["row_amplitude_times_static_profile_relative_SSE"] for s in subset]))
        axes[0].plot(range(4), zeros, "o-", alpha=.8, label=f"source {index}")
        axes[1].plot(range(4), residuals, "o-", alpha=.8)
    axes[0].plot(range(4), [0] * 4, "k--", label="all-zero groups of 64")
    axes[0].set(title="Element zeros do not yield whole zero groups", ylabel="Zero fraction (%)", ylim=(-.5, 15))
    axes[1].set(title="Group profiles vary beyond row amplitude", ylabel="Relative profile SSE (%)", ylim=(0, 23))
    for ax in axes:
        ax.set(xlabel="Denoising step (expert changes at step 2)", xticks=range(4))
        ax.axvline(1.5, color=".65", linestyle=":")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("CAL24 dense D-Q: six exposed sources, 120 scopes, 256 sampled rows/scope", fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
