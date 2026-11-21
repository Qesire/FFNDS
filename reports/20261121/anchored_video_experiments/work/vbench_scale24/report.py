"""Combine measured, source-matched VBench-I2V results on the fixed FFNDS cohort."""
import csv
from hashlib import sha256
import io
import json
from math import isclose, isfinite
import os
from pathlib import Path
from statistics import fmean, median

from scope import HERE, OUT, QUALITY, I2V, aggregate


DEST = OUT / "vbench"
ARMS = ("candidate", "dense")
CAMERA_BASE = "vbi_0951"
NATIVE = [d for d in (*QUALITY, *I2V) if d not in ("dynamic_degree", "camera_motion")]
TEMP = "temporal_flickering"
LABEL = {
    "subject_consistency": "跨帧主体一致性",
    "background_consistency": "跨帧背景一致性",
    "motion_smoothness": "运动平滑度",
    "dynamic_degree": "动态程度",
    "aesthetic_quality": "美学质量",
    "imaging_quality": "成像质量",
    "i2v_subject": "首帧条件图与视频的主体一致性",
    "i2v_background": "首帧条件图与视频的背景一致性",
    "camera_motion": "镜头指令命中率",
    TEMP: "时序闪烁（补充）",
}
GROUPS = {"total": "本批综合分", "quality": "视频质量", "i2v": "视频与条件一致性"}
CAMERA_LABEL = {"pan_left": "左摇", "pan_right": "右摇", "tilt_up": "上仰", "tilt_down": "下俯",
                "zoom_in": "拉近", "zoom_out": "拉远", "static": "固定", "None": "未识别"}
NOTE = "我肉眼基本无法区分运动幅度。认可"


def check(condition, msg):
    if not condition:
        raise ValueError(str(msg))


def close(x, y):
    check(isclose(float(x), float(y), abs_tol=1e-10, rel_tol=1e-10), (x, y))


def digest(path):
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return Path(os.path.relpath(path, DEST)).as_posix()


def collect():
    files = {}

    def read(path):
        raw = path.read_bytes()
        files[relative(path)] = sha256(raw).hexdigest()
        return json.loads(raw)

    scope = read(HERE / "scope.json")
    masks = scope["base_source_ids_by_dimension"]
    cohort = read(OUT / "COHORT.json")
    camera_dir = OUT / "camera7"
    cameras = read(camera_dir / "COHORT.json")
    check(len(cohort) == 24 and len(cameras) == 7, "Incomplete cohort")
    check({r["base_source"] for r in cameras} == {CAMERA_BASE}, "Unexpected camera cohort")
    check(files[relative(OUT / "COHORT.json")] == scope["base_cohort_sha256"], "Cohort changed")
    check(len({r["source_cluster_id"] for r in cohort}) == 24, "Repeated base source")
    sources = {r["record_id"]: r for r in cohort + cameras}
    check(len(sources) == 31 and {r["seed"] for r in sources.values()} == {42}, "Source identity")
    base_ids = {r["record_id"] for r in cohort}
    cam_ids = {r["record_id"] for r in cameras}
    sha = {}
    for s, r in sources.items():
        folder = OUT if s in base_ids else camera_dir
        check(sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"], s)
        check(digest(OUT / "inputs" / (r.get("base_source", s) + ".jpg")) == r["image_sha256"], s)
        start = []
        for arm in ARMS:
            video = folder / arm / (s + ".mp4")
            receipt = read(video.with_suffix(".json"))
            vid = receipt["video"]
            sha[arm, s] = digest(video)
            check(sha[arm, s] == vid["mp4_sha256"] and receipt["source"] == s, video)
            check((vid["width"], vid["height"], vid["frames"], vid["fps"]) == (832, 464, 81, 16), s)
            start.append(receipt["initial"])
        check(start[0] == start[1], "Initial noise differs: " + s)

    pairs = {}

    def record(d, arm, s, score, media_sha, evidence):
        check(s in sources and media_sha == sha[arm, s] and isfinite(float(score)), (arm, s, d))
        meta = sources[s]
        is_camera = s in cam_ids
        eligible = is_camera if d == "camera_motion" else s in masks[d]
        pair = pairs.setdefault((d, s), dict(dimension=d, source=s, base_source=meta.get("base_source", s),
            source_cluster_id=meta["source_cluster_id"], sample_group=meta["cohort_group"], prompt=meta["prompt"],
            seed=meta["seed"], image_sha256=meta["image_sha256"], prompt_sha256=meta["prompt_sha256"],
            dimension_eligible=eligible, upstream_dimension_tag=eligible and not is_camera,
            contributes_to_official_formula=eligible and d != TEMP))
        check(arm not in pair, ("Duplicate score", d, arm, s))
        pair[arm] = float(score)
        pair[arm + "_mp4_sha256"] = media_sha
        pair[arm + "_evidence"] = relative(evidence)

    for d in NATIVE:
        for arm in ARMS:
            path = DEST / "quality" / (d + "_" + arm + ".json")
            results = read(path)
            rows = results["rows"]
            check((results["dimension"], results["arm"], results["arm_count"]) == (d, arm, 24), path)
            check(len(rows) == 24 and {r["source"] for r in rows} == base_ids, path)
            masked = [r for r in rows if r["official_source_dimension"]]
            check({r["source"] for r in masked} == set(masks[d]) and len(masked) == results["official_subset_n"], path)
            scale = 100 if d == "imaging_quality" else 1
            for r in rows:
                record(d, arm, r["source"], r["video_results"] / scale, r["sha256"], path)
            for records, field in ((rows, "all24_native_score"), (masked, "official_subset_score")):
                mean = fmean(r["video_results"] / scale for r in records)
                if d == "background_consistency":
                    close(mean, sum(r["video_sim"] for r in records) / sum(r["cnt_per_video"] for r in records))
                close(mean, results[field])
    for s in sorted(base_ids):
        path = OUT / "raft" / (s + ".json")
        raft = read(path)
        check(s == raft["source"], path)
        for arm in ARMS:
            value = raft["dynamic_degree"][arm]
            check(isinstance(value, bool), path)
            record("dynamic_degree", arm, s, value, raft["sha"][arm], path)

    flicker_path = DEST / "temporal/temporal_flickering.json"
    flicker = read(flicker_path)
    check(flicker["cohort_sha256"] == scope["base_cohort_sha256"] and len(flicker["records"]) == 24, flicker_path)
    for item in flicker["records"]:
        for arm in ARMS:
            record(TEMP, arm, item["source"], item[arm], item[arm + "_mp4_sha256"], flicker_path)
    camera_path = DEST / "i2v/camera_motion.json"
    camera = read(camera_path)
    check(camera["official_dimension"] == "camera_motion", camera_path)
    check(CAMERA_BASE not in scope["source_camera_metadata_rows"], "Check whether camera inputs are official")
    official_commands = set(next(iter(scope["source_camera_metadata_rows"].values())))
    check({r["instruction"] for r in cameras} == official_commands, "Expected all seven official commands")
    camera_rows = {}
    for arm in ARMS:
        branch = camera["arms"][arm]
        check(len(branch["videos"]) == 7, camera_path)
        for item in branch["videos"]:
            s = item["source"]
            meta = sources[s]
            check(s in cam_ids and meta["image_sha256"] == item["image_sha256"] and meta["prompt"] == item["prompt"], s)
            parent = sources[meta["base_source"]]
            check(parent["image_sha256"] == meta["image_sha256"] and
                  parent["prompt"] + ", " + meta["instruction"] == meta["prompt"], s)
            target = item["target"]
            check(target in CAMERA_LABEL and target != "None", target)
            close(item["score"], target in item["predicted"])
            close(item["score"], branch["by_instruction"][target])
            record("camera_motion", arm, s, item["score"], item["mp4_sha256"], camera_path)
            result = camera_rows.setdefault(s, dict(source=s, instruction=meta["instruction_zh"], target=target))
            check(target == result["target"], result)
            result[arm] = item["predicted"]
        close(branch["accuracy"], fmean(item["score"] for item in branch["videos"]))
    check(len({i["target"] for i in camera_rows.values()}) == 7, "Camera targets")

    summary = {}
    for d in LABEL:
        per_dim = [row for (dim, _), row in pairs.items() if dim == d]
        check({row["source"] for row in per_dim} == (cam_ids if d == "camera_motion" else base_ids), d)
        for row in per_dim:
            row["candidate_minus_dense_pp"] = 100 * (row["candidate"] - row["dense"])
        use = [row for row in per_dim if row["dimension_eligible"]]
        deltas = [row["candidate_minus_dense_pp"] for row in use]
        item = {arm:fmean(row[arm] for row in use) for arm in ARMS}
        item.update(pairs=len(use), distinct_images=len({row["image_sha256"] for row in use}),
            candidate_minus_dense_pp=fmean(deltas), median_paired_delta_pp=median(deltas),
            higher=sum(x > 0 for x in deltas), tied=deltas.count(0.0), lower=sum(x < 0 for x in deltas))
        summary[d] = item
        close(100 * (item["candidate"] - item["dense"]), item["candidate_minus_dense_pp"])
    check([summary[d]["pairs"] for d in LABEL] == [15, 9, 15, 15, 24, 24, 15, 9, 7, 24], "Dimension masks")
    for arm in ARMS:
        close(summary[TEMP][arm], flicker["aggregate"][arm])
        close(summary["dynamic_degree"][arm], scope["dynamic_degree_registered_mean"][arm])
    scores = {arm:aggregate({d:item[arm] for d,item in summary.items()}) for arm in ARMS}
    groups = {group:{**{arm:scores[arm][group] for arm in ARMS},
                     "candidate_minus_dense_pp":100 * (scores["candidate"][group] - scores["dense"][group])}
              for group in GROUPS}
    for weights in (QUALITY, I2V):
        total_group_weight = sum(w for _, _, w in weights.values())
        for d, (low, high, w) in weights.items():
            summary[d]["normalized"] = {arm:scores[arm]["normalized"][d] for arm in ARMS}
            summary[d]["total_delta_contribution_pp"] = summary[d]["candidate_minus_dense_pp"] / (high-low) * w / total_group_weight / 2
    close(sum(summary[d]["total_delta_contribution_pp"] for d in (*QUALITY,*I2V)), groups["total"]["candidate_minus_dense_pp"])
    for path in (Path(__file__).resolve(), HERE / "scope.py"):
        files[relative(path)] = digest(path)
    ledger = sorted(pairs.values(), key=lambda r:(r["source"], list(LABEL).index(r["dimension"])))
    check(len(ledger) == 223 and sum(r["contributes_to_official_formula"] for r in ledger) == 133, "Paired rows")
    return dict(version=1, suite="VBench-I2V (VBench++), 0.1.5", difference="candidate minus Dense, percentage points",
        protocol=dict(base_pairs=24, camera_instruction_pairs=7, independent_camera_images=1, seed=42,
                      camera_base_source=CAMERA_BASE, camera_base_prompt=sources[CAMERA_BASE]["prompt"],
                      camera_base_has_native_camera_prompts=CAMERA_BASE in scope["source_camera_metadata_rows"],
                      camera_review=relative(camera_dir / "镜头同步核查.html"), camera_scores=relative(camera_path),
                      upstream_base_prompts=355, upstream_camera_prompts=763, upstream_videos_per_prompt=5,
                      upstream_full_videos_per_arm=5590, subject_base_pairs=15, background_base_pairs=9),
        normalization_and_weight=scope["normalization_and_weight"], references=scope["references"],
        official_metadata_sha256=scope["metadata_sha256"], measured_files_sha256=files,
        groups=groups, dimensions=summary, camera_by_instruction=list(camera_rows.values()), paired_rows=ledger,
        user_observation=dict(quote=NOTE, scope="此前已经看过的原批视频的运动幅度。",
                              camera_scope="已观看的新增 0951 镜头片段",
                              camera_description="用户观察到锚点版持续闪烁、抖动，轮廓变平滑且不稳定。"))


def render(data):
    dims = data["dimensions"]
    groups = data["groups"]
    p = data["protocol"]
    lines = ["# 保留锚点版与 Dense 的 VBench-I2V 完整评测", "",
             f"本批共 31 对视频。原批 24 对各用不同输入图，补充的 7 对取同一张 {p['camera_base_source'].removeprefix('vbi_')} "
             "泰迪熊翻越木栅栏图，逐一改变镜头指令。配对内共用输入图、提示词、种子 42 和初始噪声。所有视频均为 832 × 464、16 fps、81 帧。", "",
             "| 综合指标（百分制） | 保留锚点版 | Dense | 差（百分点） |", "|---|---:|---:|---:|"]
    for g, name in GROUPS.items():
        x = groups[g]
        lines.append(f"| {name} | {100*x['candidate']:.3f} | {100*x['dense']:.3f} | {x['candidate_minus_dense_pp']:+.3f} |")
    cc = {d:dims[d]["total_delta_contribution_pp"] for d in (*QUALITY,*I2V)}
    parts = []
    if max(cc.values()) > 0:
        d = max(cc, key=cc.get)
        parts.append(f"最大的正向贡献来自{LABEL[d]}（{cc[d]:+.3f} 个百分点）")
    if min(cc.values()) < 0:
        d = min(cc, key=cc.get)
        parts.append(f"最大的负向贡献来自{LABEL[d]}（{cc[d]:+.3f} 个百分点）")
    lines += ["", "按官方权重分解本批综合分差，" + "，".join(parts) + "。", "",
              "## 九项官方维度与一项补充诊断", "",
              "两版列为官方原分（1 表示 100%），差值为保留锚点版减 Dense 的百分点。配对计数按同一个来源或同一条镜头指令比较，两臂得分使用保存的全精度数值。", "",
              "| 维度 | 对数 | 保留锚点版 | Dense | 均值差（百分点） | 配对差中位数（百分点） | 锚点较高 / 相等 / 较低 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for d, name in LABEL.items():
        x = dims[d]
        lines.append(f"| {name} | {x['pairs']} | {x['candidate']:.6f} | {x['dense']:.6f} | {x['candidate_minus_dense_pp']:+.3f} | "
                     f"{x['median_paired_delta_pp']:+.3f} | {x['higher']} / {x['tied']} / {x['lower']} |")
    refs = data["references"]
    motion = dims["dynamic_degree"]
    n_motion = motion["pairs"]
    positive = {arm:round(motion[arm] * n_motion) for arm in ARMS}
    lines += ["", "主体和运动维度覆盖 15 个主体来源，背景维度覆盖 9 个背景来源，美学与成像覆盖全部 24 个基础来源。"
              f"动态程度用冻结的 RAFT 与官方阈值逐片判定，过阈值的来源在锚点版为 {positive['candidate']}/{n_motion}，"
              f"在 Dense 为 {positive['dense']}/{n_motion}，两臂逐例判定相同 {motion['tied']}/{n_motion}。"
              "补充的时序闪烁列使用全部相邻帧的像素变化计算，正常物体运动、镜头移动或亮度变化都能改变分数。细微动作是否闪烁须结合画面判断。[时序闪烁逐源结果](temporal/判读.md) · [方向与幅度的 RAFT 复核](../RAFT复核.md)", "",
              "## 七类镜头指令", "",
              f"0951 的官方原提示为 *{p['camera_base_prompt']}*。官方元数据为这张图登记了主体和质量维度，"
              "镜头数据集中没有这张图。本次将官方七类指令分别接到原提示末尾，每类各生成一对。CoTracker2 可以给一条片段返回多个镜头类型，包含指定类型即计为命中。", "",
              "| 指令 | 保留锚点版识别结果 | Dense 识别结果 | 命中（锚点 / Dense） |", "|---|---|---|---:|"]
    for r in sorted(data["camera_by_instruction"], key=lambda x:list(CAMERA_LABEL).index(x["target"])):
        c, d = (r[a] for a in ARMS)
        tags = lambda s:"、".join(CAMERA_LABEL.get(x,x) for x in s) or "无"
        lines.append(f"| {r['instruction']} | {tags(c)} | {tags(d)} | {int(r['target'] in c)} / {int(r['target'] in d)} |")
    lines += ["", "七种指令共享一张条件图和一个种子。镜头命中率只描述这个场景的指令覆盖，场景间和种子间的泛化仍待验证。"
              f"[七组同步视频]({p['camera_review']}) · [镜头逐片判定]({p['camera_scores']})", "",
              "## 对分数与人眼记录的解释", "",
              "计算主体和背景的条件图一致性时，条件图与最相似视频帧的特征相似度占 40%，相邻帧相似度的均值和最小值各占 30%。"
              "即使画面在后半段缓慢偏离输入图，早期最相似帧仍可抬高第一项，相邻帧的小变化也可维持后两项。"
              "[主体项实现](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/i2v_subject.py) · "
              "[背景项实现](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/i2v_background.py)", "",
              f"用户此前对已观看的原批视频给出的判断是「{NOTE}」。这句话针对运动幅度。"
              "例如，1013 中两臂均通过运动幅值门槛的格点仅占 6.4%，这些格点的 RAFT 平均角差为 33.9°。这个角差描述局部方向偏移，"
              "全片动态程度则只记录是否跨过运动阈值。[原批视频逐帧核查](../同步核查.html) · [方向性测量与覆盖率](../RAFT复核.md)", "",
              "在已观看的新增 0951 镜头片段中，用户指出保留锚点版持续闪烁、抖动，轮廓变平滑且不稳定。镜头项只判断预测集合里是否出现指定运动，"
              "即使命中也可能伴有轮廓失真。上表的时序闪烁和运动平滑度来自原批基础提示，不能代表这些新增镜头的画质。"
              f"[0951 镜头同步核查]({p['camera_review']}) · [熊身纹理、边缘与运动补偿的定量分析](../camera7/diagnosis/0951主体不稳定分析.md)", "",
              "逐维原分先用官方上下界线性归一化。视频质量的六项中，动态程度权重 0.5，其余各 1。视频与条件一致性的三项中，首帧主体和背景各 1，镜头命中率权重 0.1。两个加权组均值各占综合分的一半，时序闪烁单独报告。"
              f"[归一化与权重]({refs['weights']}) · [官方合分脚本]({refs['total']})", "",
              "公开全量方案使用 355 个基础图文条件（246 个主体、109 个背景）和 763 个镜头条件，每个条件生成 5 条，每个模型共 5,590 条。"
              "本次每臂 31 条，用官方九维公式汇总，其中镜头项使用自定义的 0951 七指令。这里的分数只支持本批配对比较，不能直接用于公开榜单排名。"
              f"[官方采样规范与分维清单]({refs['protocol']})", "",
              "逐源 CSV 保存每项实测的配对值，共 223 行。筛选 `contributes_to_official_formula=True` 可得按官方权重进入本批综合分的 133 行，"
              "另有 24 行时序闪烁和 66 行官方标签范围外的基础来源测量。每行附两版视频 SHA-256 和评分回执位置。", "",
              "[逐源配对分数与原片指纹](逐源配对.csv) · [全精度汇总、得分贡献与测量回执哈希](汇总.json) · [合表程序](../../../work/vbench_scale24/report.py)", ""]
    return "\n".join(lines)


def main():
    data = collect()
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(data["paired_rows"][0]))
    writer.writeheader()
    writer.writerows(data["paired_rows"])
    table = buffer.getvalue().encode("utf-8-sig")
    data["paired_csv_sha256"] = sha256(table).hexdigest()
    outputs = {"逐源配对.csv": table, "汇总.json": (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(),
               "VBench完整评测.md": render(data).encode()}
    for name, raw in outputs.items():
        file = DEST / name
        tmp = file.with_suffix(file.suffix + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(file)
    print(json.dumps(data["groups"], indent=2))


if __name__ == "__main__":
    main()
