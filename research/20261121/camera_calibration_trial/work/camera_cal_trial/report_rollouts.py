"""Rebuild the 0905/1014 closed-loop results table and review index from receipts."""
import json
from pathlib import Path
from statistics import median

OUT=Path(__file__).resolve().parents[2]/"outputs/FFNDS_camera_cal_trial"


def read(name):
    return json.loads((OUT/name).read_bytes())


def main():
    subject=read("evaluation/closed_loop_subject_motion.json")["clips"]
    texture=read("evaluation/foreground_texture.json")["trajectories"]
    background=read("background_direction_checks.json")["trajectories"]
    receipt=read("render_ab/RESULT.json")
    assert len(receipt["attempts"])==8 and receipt["nfe_this_job"]==32
    lines=[
        "# 两种离线校准的成片结果与镜头方向核查",
        "",
        "两张留出图分别为海边的企鹅和马球骑手，均使用种子 44。用户观察到，原提示本身就有向左运动，追加左摇后的观感与原提示相近。以下分别记录在相近运动条件下换校准权重的效果，以及在 Dense 上追加反方向指令的结果。",
        "",
        "## 原提示、左摇与右摇",
        "",
        "| Dense 来源 | 指令 | 可用帧间 / 全部 | 背景向右 / 向左 / 近静止 | 单帧水平像移中位数（px） |",
        "|---|---|---:|---:|---:|"
    ]
    for s in ("0905","1014"):
        for tag,label in (("base","原提示"),("pan_left","左摇"),("pan_right","右摇")):
            m=background[s+"_"+tag]["full"]
            lines.append(f"| {s} | {label} | {m['supported']}/{m['total']} | {m['screen_right_over_quarter_px']} / {m['screen_left_over_quarter_px']} / {m['subquarter_px']} | {m['median_dx']:+.2f} |")
    lines.extend([
        "",
        "表中方向指屏幕上的背景纹理。只保留外圈相似变换内点率至少 60% 的帧间，水平位移绝对值至多 0.25 像素归为近静止。背景向右与镜头向左摇相符。企鹅三条片均有完整覆盖。马球三条片的有效覆盖为 27–42 个帧间，尚不能确定未通过筛选的时段如何运动。两条右摇指令都改变了首步预测，但在可可靠估计的时段均未实现稳定的背景反向。因此这一批尚未提供成功切换镜头方向的独立校准验证。没有为这两条右摇提示运行候选权重。",
        "",
        "三路对照按原提示、左摇、右摇从左到右排列。[0905 企鹅](review/0905_dense_base_left_right.mp4) · [1014 马球](review/1014_dense_base_left_right.mp4) · [逐帧方向统计](background_direction_checks.json)",
        "",
        "## 四个高噪声层的真实生成",
        "",
        "A 是普通校准，B 是镜头混合校准。两者在其余 76 个专家层站点上共用同一算法和权重。每个条件有自己的 Dense 参照，每对两条视频都复现其初始噪声及首个原生步。下表是解码视频统计；盲片中的 Variant 1、Variant 2 排列因组而异。",
        "",
        "| 来源 | 条件 | 可比运动窗 / 77 | 主体四帧位移投影 A / B | 正交相对 RMS A / B | 前景高频纹理相关 Dense / A / B | 盲审片 |",
        "|---|---|---:|---:|---:|---:|---|"
    ])
    for s in ("0905","1014"):
        for c,label in (("base","原提示"),("camera","左摇")):
            k=f"vbi_{s}_{c}_s44"
            v=subject[k]["paired_against_dense"]
            t=texture[k]["median_over_transitions"]
            links=f"[Dense、Variant 1、Variant 2](review/{s}_{c}_blind.mp4)"
            lines.append(f"| {s} | {label} | {v['four_frame_windows']}/77 | {v['A']['projected_gain']:.4f} / {v['B']['projected_gain']:.4f} | {v['A']['orthogonal_relative_rms']:.4f} / {v['B']['orthogonal_relative_rms']:.4f} | {t['dense']['correlation']:.4f} / {t['A']['correlation']:.4f} / {t['B']['correlation']:.4f} | {links} |")
    costs=receipt["attempts"]
    warm=[r["wall"] for r in costs if not r["cold"]]
    cold=[r["wall"] for r in costs if r["cold"]]
    assert len(warm)==7 and len(cold)==1
    lines.extend([
        "",
        "运动统计在三路均能追踪的首帧主体特征上计算，先减去外圈变换，再以 Dense 的位移为参照做有符号投影。投影理想值为 1，正交 RMS 越低表示偏离参考运动方向的分量越小。企鹅两条件有 77 个有效四帧窗；马球只应按实际覆盖的局部窗口解读。",
        "",
        "纹理统计来自同一组三路均存活的前景点，相邻帧各取 17×17 像素的高频块，中心随点移动。仅用 Dense 两端均有足够纹理且至少三点可比的帧间；逐帧间取点的中位数，再跨时刻取中位数。高频对比度也同时落盘。高相关可以与模糊并存，应结合对比度和原片判读。合成平移、模糊、独立噪声和同视频对照均已复核，方法不对遮挡后新出现的表面给出评分。",
        "",
        "镜头混合校准在两张图的左摇教师面板上使时间差相对 RMS 分别下降约 0.45%、0.38%，收益主要在第 0 层，普通提示也有相近改善。在四层实际生成中，企鹅左摇的运动正交误差略增，马球左摇在有限的可追踪窗口内略降，两片的纹理相关也朝不同方向变化。当前证据未显示一致的运动或纹理改善，成片视觉判定留给审片者。[逐站回放](独立来源回放.md) · [前景运动统计](evaluation/closed_loop_subject_motion.json) · [纹理连续性、对比度及控制](evaluation/foreground_texture.json)",
        "",
        f"作业 182638 在一张 A100 上生成 8 条视频，执行 32 次去噪。含首次权重打包的第一条整片为 {cold[0]:.2f} 秒，七条热运行范围为 {min(warm):.2f}–{max(warm):.2f} 秒，中位数 {median(warm):.2f} 秒。两组计算图相同，运行次数不足以估计稳定的速度差。补充的右摇作业 182672 使用 8 次去噪、实耗 10 分 54 秒。[闭环回执](render_ab/RESULT.json) · [右摇回执](right_control/RESULT.json) · [人审记录与右摇实验登记](人审记录与反向镜头对照.md)"
    ])
    (OUT/"四层成片与方向核查.md").write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
