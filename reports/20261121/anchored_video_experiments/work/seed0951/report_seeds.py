"""Combine paired foreground, contour, camera and coarse action results by seed."""
import json
from pathlib import Path
from statistics import median


OUT=Path(__file__).resolve().parents[2]/"outputs/FFNDS_0951_seeds"


def read(path):
    return json.loads(path.read_bytes())


def main():
    pairs=read(OUT/"scores/paired.json")["seeds"]
    assert [p["seed"] for p in pairs]==[42,43,44,45,46]
    notes=[]
    rows=[]
    for p in pairs:
        s=p["seed"]
        d=p["dense"]
        a=p["candidate"]
        pos=read(OUT/"motion"/f"pair_s{s}.json")
        w=pos["metrics"]["world"]
        row=dict(seed=s, background_dense_px=d["background_center_x_px"],background_anchor_px=a["background_center_x_px"],
                 texture_dense=d["body_band_correlation"],texture_anchor=a["body_band_correlation"],
                 texture_gap=d["body_band_correlation"]-a["body_band_correlation"],
                 flicker_delta=a["body_compensated_flicker"]-d["body_compensated_flicker"],
                 normal_accel_delta=a["edge_normal_accel_p75"]-d["edge_normal_accel_p75"],
                 gain=w["gain"], cosine=w["cos"], orthogonal_rms=w["orthogonal_relative_rms"],trajectory_p90_px=w["p90"],
                 dense_initial=pos["initial"])
        rows.append(row)
        if s!=42:
            aa=read(OUT/"candidate"/f"vbi_0951_pan_left_s{s}.json")
            dd=read(OUT/"dense"/f"vbi_0951_pan_left_s{s}.json")
            assert aa["seed"]==dd["seed"]==s and dd["initial"]==aa["initial"]==pos["initial"]
            assert aa["flows"][0]==dd["flows"][0] and aa["latents"][0]==dd["latents"][0]
        notes.append(f"| {s}{'（参照）' if s==42 else ''} | {row['background_dense_px']:+.0f} / {row['background_anchor_px']:+.0f} | {row['texture_dense']:.3f} / {row['texture_anchor']:.3f} | {row['flicker_delta']:+.2f} | {row['normal_accel_delta']:+.2f} | {w['gain']:.3f} / {w['cos']:.3f} | {w['p90']:.1f} | [同步盲片](review/s{s}_blind.mp4) |")
    assert len({r["dense_initial"] for r in rows})==5
    runs=rows[1:]
    gap=[r["texture_gap"] for r in runs]
    def count(key,threshold):
        return sum(r[key]>threshold for r in runs)
    lines=[
        "# 0951 左摇换种子后的真实生成",
        "",
        "固定原图、左摇提示、四步调度与旧锚点权重，种子 43、44、45、46 各生成 Dense 和高低两步均近似的旧锚点版。种子 42 使用既有原片。五个种子的初始噪声指纹各不相同，四对新片均复核了各自相同的初始噪声和首个原生步。全部新增轨迹均纳入表中。",
        "",
        f"与各自 Dense 成片相比，新增种子中 {count('texture_gap',0)}/4 的熊身运动补偿后纹理相关降低，{count('flicker_delta',0)}/4 的补偿后亮度闪烁增大，{count('normal_accel_delta',0)}/4 的轮廓法向运动波动增大。新种子的纹理相关降幅为 {min(gap):.3f} 至 {max(gap):.3f}，中位数 {median(gap):.3f}，原种子 42 为 {rows[0]['texture_gap']:.3f}。方向计数描述四组实际输出，最终可见的严重程度以盲审为准。",
        "",
        "| 种子 | 背景横移，Dense / 锚点（px） | 熊身纹理相关，Dense / 锚点 | 闪烁差 | 轮廓法向加速度差 | 主体位移投影增益 / 余弦 | 主体轨迹差 P90（px） | 视频 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        *notes,
        "",
        "种子 42 在这五对中的纹理相关降幅、补偿后闪烁增量和主体轨迹 P90 偏差均最大。种子 43 的纹理和粗轨迹更接近 Dense；种子 44 的轮廓增量只有 0.013 px/帧²；种子 45、46 的四帧位移投影增益只有约 0.51、0.60。背景均向符合左摇的方向移动，两版在每个新种子上的净水平像移相差不到 7 像素。因而同图左摇下反复出现了纹理延续的损失，其严重程度及主体动作对齐程度随种子改变。种子同时改变了 Dense 自身的动作、镜头幅度和可见表面，不能单凭这五对将差异归结为镜头速度。",
        "",
        "闪烁差和轮廓法向加速度差均取旧锚点减 Dense。背景在屏幕上向右横移记为正，与镜头向左相符。主体位移取连续暖色区域质心，消除各自的背景变换后，在四帧间隔上与同种子的 Dense 比较，投影增益与方向余弦理想值均为 1。暖色代理可能受姿态、毛色、遮挡影响；轮廓法向加速度还会响应真实非匀速运动。主体、纹理、轮廓、背景和可跟踪覆盖分别判读。",
        "",
        "本实验有四个新增噪声种子，一个图像来源和一个提示。它检验已经发现的 0951 现象对种子的敏感性，不能估计其他来源上的失败率，也不能把新种子中的差异逐一归因到某一去噪步。42 上的分步因果干预是另一项实验。",
        "",
        "[跨种子主体与背景明细](scores/跨种子主体与背景.md) · [完整逐种子数据](paired_motion_texture.json) · [实验登记](实验登记.md) · [审片入口](review/审片.html)"
    ]
    (OUT/"换种子结果.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    (OUT/"paired_motion_texture.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
