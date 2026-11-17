# 学校集群鱼鳞纹：完整视频人工评审

状态：**USTC_SCHOOL_ONLY / TEMPLATE_ONLY / NO_REVIEW_YET**。所有 A/B 视频都来自已验证的学校 source pair 或新学校 Dense capture，不使用远端 worker 的旧视频作为本次实验对象。完整实验规格见 [实验主方案](EXPERIMENT_PLAN.md)。

## 必须保留的可播放视频

每个新 arm 同时保存完整 81 帧、832×464、16fps 的 RGB8 FFV1 无损 MKV 和 H264 MP4 预览，生成学校绝对路径与 SHA 的 `human_review/REVIEW_INDEX.json`。原始学校 Dense MP4 只读保存。无损 MKV 保护 RGB8，不代表保留了 VAE FP32 原始浮点精度；FP32 边界由单独数值记录说明。

学校 Slurm 产物必须真正可读并能通过学校允许的文件服务/SCP/SFTP 下载到观看设备；仅有 JSON、拼图、关键帧或频谱不构成交付完成。不得把旧 S2 报告中的 `root=hf` 路径当作本轮的可访问视频链接。

## 用户优先的 A/B 观看流程

1. **A0**：观看学校实际生成或复用的完整原始 Dense 视频，人眼确认有目标“鱼鳞纹”、可见时段、空间范围后，才开始对该学校 source 做 A1。
2. **A1–B3**：同一真实 source/seed 的控制/候选先盲标 A/B，完整单独播放，再交替播放；用户可以按 100% 原像素观察、循环及放大 ROI。首次记录后揭盲。画面尺寸、帧序、FPS 不可因预览缩放而改变源文件。
3. 关闭播放器自动插帧、增强、锐化、降噪/HDR，记录播放器及显示缩放。对于无法分辨的差异，允许重复审看或要求新完整视频，不强迫评分。
4. 记录 severity 0–4（0 无、1 轻微、2 明显局部、3 大面积持续、4 严重破坏画面）、global/broad/localized、persistent/periodic/intermittent、可见秒数/ROI、清晰度/运动代价。
5. A/B 判定 `clearly_better / slightly_better / unchanged / worse / uncertain`；按用户的原话记录判断依据和下一动作。自动高频频谱、VBench 和 AI 观察为 `NON_VOTING`；若不同意人眼，研究哪一个指标测错现象，而不改写用户评价。

## 用户评审记录模板（非真实结果）

```json
{
  "schema":"FFNDS_FISHSCALE_SCHOOL_HUMAN_REVIEW_V2",
  "status":"TEMPLATE_ONLY",
  "school_source_id":null,
  "pair_id":null,
  "A":{"video_path":null,"video_sha256":null,"severity":null,"scope":null,"pattern":null,"time_ranges_sec":[],"roi_xyxy_px":[]},
  "B":{"video_path":null,"video_sha256":null,"severity":null,"scope":null,"pattern":null,"time_ranges_sec":[],"roi_xyxy_px":[]},
  "fishscale_change":null,
  "visual_preference":null,
  "fidelity_tradeoff":null,
  "confidence":null,
  "reviewer":"user",
  "human_next_action":null,
  "notes":null,
  "blind_revealed_after_first_review":false
}
```

实际记录在学校输出的 `human_review/REVIEWS.jsonl` 追加，不覆盖旧审阅；没有 user A0 阳性则不得擅自声称 A1 必要，没有 A/B 用户判断就保持 `AWAITING_HUMAN_REVIEW`。任何视觉改善需同时检查是否靠模糊、细节或真实动作损失达成；根因声明还要有单变量可定位的数值/视频边界证据。
