"""Show paired A/B effects for each held-out trajectory. Layers are repeated measures."""

import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "outputs/FFNDS_camera_cal_trial"


def main():
    raw = json.loads((OUT / "fit/RESULT.json").read_bytes())
    protocol = json.loads((HERE / "protocol.json").read_bytes())
    assert raw["status"] == "PASS"
    validation = [r for r in protocol["trajectories"] if r["split"] == "validation"]
    table = []
    lines = [
        "# 镜头混合校准的独立来源回放",
        "",
        "A 使用普通提示。B 在同样的来源、种子与校准行预算下，替换每张图种子 43 的一条教师轨迹，分别追加拉近、左摇或右摇。两组在高噪声专家的相同四层，各自拟合可折叠的剩余项。验证使用原生教师端点、命中层真实门值及当前轨迹的 64 个原生响应。",
        "",
        "每行对应一条留出轨迹，先在各层单独归一化，再对第 0、13、26、39 层等权。联合误差为点值、空间差、潜视频时间差三项相对误差能量的均值。时间投影理想值为 1。投影偏差是逐层投影与 1 的绝对差再平均，避免正负抵消。正交 RMS 针对与该层真实时间差垂直的误差分量。最后一列是 B 相对 A 使时间差相对 RMS 降低的层数。",
        "",
        "| 来源 | 提示 | 联合误差 A → B | 时间差相对 RMS A → B | 时间投影 A → B | 时间投影偏差 A → B | 时间正交 RMS A → B | 改善层数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"
    ]
    for trajectory in validation:
        rows = [r for layer in raw["layers"] for r in layer["validation"] if r["trajectory_id"] == trajectory["trajectory_id"]]
        assert sorted(r["layer"] for r in rows) == protocol["layers"]
        by_arm = {}
        for arm in ("frozen", "A", "B"):
            vals = [r["arms"][arm] for r in rows]
            by_arm[arm] = dict(joint=mean(v["joint_relative_energy"] for v in vals),
                               time_rms=mean(v["time"]["relative_rms"] for v in vals),
                               time_gain=mean(v["time"]["projected_gain"] for v in vals),
                               time_gain_deviation=mean(abs(1-v["time"]["projected_gain"]) for v in vals),
                               time_orthogonal=mean(v["time"]["orthogonal_rms"] for v in vals))
        win = sum(r["arms"]["B"]["time"]["relative_rms"] < r["arms"]["A"]["time"]["relative_rms"] for r in rows)
        numbers = " | ".join(f"{by_arm['A'][k]:.5f} → {by_arm['B'][k]:.5f}" for k in by_arm["A"])
        cond = "原提示" if trajectory["condition"] == "base" else "左摇"
        lines.append(f"| {trajectory['source_id']} | {cond} | {numbers} | {win}/4 |")
        table.append(dict(trajectory=trajectory["trajectory_id"], metrics=by_arm, time_layers_improved=win))
    lines.extend(["", "四层、448 个非探针位置和 224 条时间边都是同一张图上的重复观测。两种提示同用一个验证种子。结果仅衡量这些教师状态上的恢复能力；评估近似在完整去噪链中对动作、轮廓和纹理的影响，需要配对成片。", "", "[实验与来源登记](实验登记.md) · [逐层、分来源、分条件的完整结果](fit/RESULT.json) · [拟合与评分代码](../../work/camera_cal_trial/fit_camera.py)"])
    (OUT / "独立来源回放.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    (OUT / "paired_summary.json").write_text(json.dumps(table, indent=2, allow_nan=False)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
