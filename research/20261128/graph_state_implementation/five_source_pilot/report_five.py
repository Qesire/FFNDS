"""Write a repository-safe report from the verified five-source receipts."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


LABELS = {
    "00_frozen": "冻结 centered + wide8192",
    "01_staged_ffn": "分阶段 FFN",
    "02_partition_anchor": "单遍 low 锚点",
    "03_attention_stream": "attention 流式",
    "04_block_stream": "整块流式",
    "05_high_output_fold": "high 输出投影折叠",
    "06_projection_precompose": "投影预合成",
    "07_ffn_factor": "FFN 因子 rank 512",
    "08_attention_joint": "attention 联合基 rank 64",
    "09_combined_graph": "组合计算图",
    "10_combined_state": "组合状态",
    "11_ffn_gauge": "FFN 规范变换",
}
BASE = "00_frozen"
ASSET_PATHS = {"attention_basis_dir", "ffn_factor_dir", "ffn_gauge_dir"}
DIMS = ("imaging_quality", "aesthetic_quality", "motion_smoothness", "subject_consistency",
        "background_consistency", "i2v_subject", "i2v_background")


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def scalar(values, fmt):
    if any(value is None for value in values):
        return "—"
    return format(fmean(values), fmt)


def report(results, scores, videos, out):
    assert results["status"] == "complete" and not results["missing_profiles"]
    profiles = results["profiles"]
    assert set(profiles) == set(LABELS) and len(results["sources"]) == 5
    assert all(row["exact_frozen_trace"] for row in profiles[BASE]["quality"])
    assert set(scores["scores"]) == set(DIMS)
    dimensions = {name: scores["scores"][name] for name in DIMS}
    for dimension in dimensions.values():
        assert set(dimension) == set(profiles)
        assert all(set(row["rows"]) == set(results["sources"]) for row in dimension.values())
    for name, configuration in profiles.items():
        assert [q["source_id"] for q in configuration["quality"]] == results["sources"]
        for key in ASSET_PATHS:
            if configuration["profile"][key]:
                configuration["profile"][key] = "verified_school_asset"
        receipts = read(videos / name / "candidate.json")["records"]
        assets = receipts[results["sources"][0]]["asset_identities"]
        configuration["asset_sha256"] = {
            asset: {site: entry["sha256"] for site, entry in sites.items()}
            for asset, sites in assets.items()
        }
        for row in configuration["quality"]:
            source = row["source_id"]
            receipt = receipts[source]
            assert receipt["record"]["video"]["mp4_sha256"] == row["mp4_sha256"]
            assert receipt["asset_identities"] == assets
    out.mkdir(parents=True, exist_ok=True)
    save(out / "PILOT_RESULTS.json", results)
    save(out / "VBENCH_SUMMARY.json", scores)

    lines = [
        "# 五源端到端优化结果",
        "",
        "学校 A100 四步推理，冻结的 centered + BF16 wide8192 作对照。12 个配置各完成五条 832×464、81 帧、16 FPS 的真实视频。固定种子为 42。五个来源包含上一轮的三个成像退化样本、一个运动退化样本和一个 latent 误差接近中位数的样本。",
        "",
        "FFN 因子 rank 512、attention 联合基 rank 64 和 16 模态 FFN 规范变换均来自独立 CAL 数据。整块流式适配器修复了原生调制接口所需的 batch/token 轴，使用修复后的同一组来源完成整块流式、high 输出折叠、组合计算图和组合状态。",
        "",
        "每项优化均在同一块 GPU 上另跑五条冻结参考，每个进程先热身一条。表中报告五个同源速度比的几何均值。速度比范围取五源配对的最小值和最大值。四步 GPU 时间为四次 CUDA 步计时之和，完整请求从调用生成入口计到返回并完成 CUDA 同步，生成入口负责写视频。模型冷加载、轨迹及视频验算、终点 latent 归档和独立评分位于该窗口外。质量误差使用独立冻结组的同源原片和终点，经已签收的 MP4/RGB 和终点 hash 校验。",
        "",
        "| 配置 | 完整请求速度比 | 四步 GPU 速度比 | 平均耗时，候选 / 同卡冻结（s） | 峰值显存（GiB） | 轨迹一致 | 终点 latent 相对 L2 | RGB RMSE | 帧间差分 MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table = []
    for name, label in LABELS.items():
        result = profiles[name]
        timing = result["timing"]
        quality = result["quality"]
        exact = sum(row["exact_frozen_trace"] for row in quality)
        if name != BASE:
            assert [row["source"] for row in timing["sources"]] == results["sources"]
            comparison = [
                f"{timing['wall_speedup']:.3f}×", f"{timing['gpu_speedup']:.3f}×",
                f"{timing['candidate_mean_seconds']:.2f} / {timing['reference_mean_seconds']:.2f}",
                f"{timing['peak_candidate_gib']:.2f}",
            ]
        else:
            comparison = ["—"] * 4
        values = [
            f"{name[:2]} {label}", *comparison, f"{exact}/5",
            scalar([row["final_latent_relative_l2"] for row in quality], ".3%"),
            scalar([row["rgb_rmse_0to1"] for row in quality], ".5f"),
            scalar([row["temporal_delta_rgb_mae_0to1"] for row in quality], ".5f"),
        ]
        table.append("| " + " | ".join(values) + " |")
    lines.extend(table)
    fastest = max((a for a in LABELS if a != BASE), key=lambda a: profiles[a]["timing"]["wall_speedup"])
    smallest = min((a for a in LABELS if a != BASE), key=lambda a: profiles[a]["timing"]["peak_candidate_gib"])
    time = profiles[fastest]["timing"]
    speed = [row["wall_speedup"] for row in time["sources"]]
    memory = profiles[smallest]["timing"]
    lines.extend([
        "",
        f"完整请求速度比最高的是 {LABELS[fastest]}，五源配对几何均值 {time['wall_speedup']:.3f}×，逐源范围 {min(speed):.3f}–{max(speed):.3f}×。"
        f"{LABELS[smallest]}的峰值显存最低，从同卡冻结的 {memory['peak_reference_gib']:.2f} GiB 降到 {memory['peak_candidate_gib']:.2f} GiB。",
    ])
    for name in ("09_combined_graph", "10_combined_state"):
        trials = profiles[name]["timing"]["sources"]
        candidate = fmean(x["candidate_steps_ms"][2] for x in trials) / 1000
        reference = fmean(x["reference_steps_ms"][2] for x in trials) / 1000
        lines.extend(["", f"{LABELS[name]}在第三步 low 锚点平均耗时 {candidate:.2f} s，同卡冻结为 {reference:.2f} s。"])
    frozen_motion = {row["source_id"]: row["mean_adjacent_rgb_mae_0to1"] for row in profiles[BASE]["quality"]}
    motion = [
        row["mean_adjacent_rgb_mae_0to1"] / frozen_motion[row["source_id"]] - 1
        for row in profiles["10_combined_state"]["quality"]
    ]
    lines.extend(["", f"组合状态相对冻结组的相邻帧 RGB 平均变化量，逐源从 {min(motion):+.1%} 到 {max(motion):+.1%}。该指标包含画面运动、纹理变化和亮度变化。"])
    lines.extend([
        "",
        "RGB 指标在 [0,1] 像素尺度上归一化。速度比大于 1 表示候选更快。该五源筛查描述这组固定来源，尚需独立来源估计总体质量变化。[端到端数值、视频哈希和配对计时](PILOT_RESULTS.json)",
        "",
        "## 七项 VBench",
        "",
        "冻结配置一行列出实际得分。各候选配置列出相对冻结的差值，单位为百分点，同一套固定指标模型逐片评分。共有 420 个分数。",
        "",
        "| 配置 | 成像 | 美学 | 运动平滑 | 主体一致 | 背景一致 | 输入主体 | 输入背景 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    vtable = []
    for name, label in LABELS.items():
        if name == BASE:
            values = [f"{100 * metric[name]['score']:.3f}" for metric in dimensions.values()]
        else:
            values = [f"{100 * (metric[name]['score'] - metric[BASE]['score']):+.3f}" for metric in dimensions.values()]
        vtable.append("| " + " | ".join([f"{name[:2]} {label}", *values]) + " |")
    lines.extend(vtable)
    imaging = dimensions["imaging_quality"]
    clip_delta, arm, source = min(
        (imaging[arm]["rows"][source] - imaging[BASE]["rows"][source], arm, source)
        for arm in LABELS if arm != BASE for source in results["sources"]
    )
    factor_delta, factor_source = min(
        (imaging["07_ffn_factor"]["rows"][source] - imaging[BASE]["rows"][source], source)
        for source in results["sources"]
    )
    lines.extend(["", f"{LABELS[arm]}在 `{source}` 的成像分下降最多，差值为 {100 * clip_delta:+.2f} 个百分点。"
                  f"FFN 因子 rank 512 的最大单片下降出现在 `{factor_source}`，差值为 {100 * factor_delta:+.2f} 个百分点。"])
    lines.extend(["", "[逐源 VBench 数值](VBENCH_SUMMARY.json) · [逐源指标与计时 CSV](SOURCE_METRICS.csv)", ""])
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    with (out / "SOURCE_METRICS.csv").open("w", encoding="utf-8", newline="") as file:
        fields = ["profile", "source_id", "exact_frozen_trace", "final_latent_relative_l2", "rgb_rmse_0to1",
                  "temporal_delta_rgb_mae_0to1", "mean_adjacent_rgb_mae_0to1", "wall_speedup",
                  "gpu_speedup", "candidate_wall_s", "reference_wall_s", "candidate_peak_gib",
                  "reference_peak_gib", *dimensions, "mp4_sha256"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name, profile in profiles.items():
            timing = {x["source"]: x for x in profile["timing"]["sources"]} if profile["timing"] else {}
            for row in profile["quality"]:
                source = row["source_id"]
                score = {dim: metric[name]["rows"][source] for dim, metric in dimensions.items()}
                merged = dict(row, profile=name, **timing.get(source, {}), **score)
                writer.writerow({k: merged[k] for k in fields if k in merged})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("scores", "videos", "out"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    report(read(args.scores / "result/PILOT_RESULTS.json"), read(args.scores / "VBench_VERIFIED.json"),
           args.videos, args.out)
