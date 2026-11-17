# Wan 四步鱼鳞纹：人眼评审协议与模板

**状态：TEMPLATE_ONLY。** 用户尚未观看本诊断新生成的配对视频，也没有任何实际 A/B 分数。

本协议服务于 [完整方案](EXPERIMENT_PLAN.md)。**人眼视觉判断是主判据。** 所有新解码的视频必须同时交付完整 81 帧 RGB8 FFV1/MKV 无损视频和同尺寸、同 16fps 的 H264 MP4 预览；原历史 Dense MP4 始终保留。机器谱、光流、VBench、AI 分析只提供非投票解释。

## 观看顺序

1. **A0**：观看原始 Dense 的 ord898、ord96，确认哪个视频真实存在目标鱼鳞纹。若均不明显，则从剩余 12 条历史视频中选择。不要让自动高频分数替你挑阳性。
2. **A1 及后续**：每个来源的完整视频采用盲标 A/B，逐条原尺寸播放，再允许交替、循环、放大特定 ROI；第一次判断录入后才揭盲。完整时长不得改、视频帧不得抽样，不能只看联系表。
3. 建议关闭播放软件的自动插帧、降噪、锐化、HDR 增强，记录显示倍率与色彩处理。如果缩小窗口后纹理看不到，改为 100% 原始像素查看。
4. 视频生成后先写 REVIEW_INDEX（真实可访问完整文件和 SHA），再邀请用户观看；状态 AWAITING_HUMAN_REVIEW 时，不可自动推进新 GPU scope。
5. 所有人眼评审分别追加到 REVIEWS.jsonl，后一次不覆盖前一次；AI 分析另外记录为 NON_VOTING。

## 最小反馈

- **严重程度** severity：0 无；1 仔细看有；2 明显局部；3 大面积持续；4 严重破坏观感。
- **空间** scope：global / broad / localized / uncertain。
- **时间** temporal_pattern：persistent / periodic / intermittent / none / uncertain；给出可见的秒数及原 832×464 像素坐标 ROI。
- **质量代价**：细节、清晰度、主体/相机动作是否保留，是否产生偏色、块状和模糊。
- **比较**：A / B / tie / uncertain；fishscale_change 为 clearly_better / slightly_better / unchanged / worse / uncertain。
- **后续决策**：continue / stop_and_explain / repeat_viewing / request_targeted_pair。用户自己的文字观察优先。

## REVIEWS.jsonl 一行格式（模板，不代表结果）

~~~json
{
  "schema": "FFNDS_FISHSCALE_HUMAN_REVIEW_V1",
  "status": "TEMPLATE_ONLY",
  "review_id": null,
  "reviewer": "user",
  "pair_id": null,
  "first_review_blinded": true,
  "A": {"file_sha256": null, "severity": null, "scope": null, "temporal_pattern": null, "time_ranges_s": [], "roi_xyxy_px": [], "detail_loss": null},
  "B": {"file_sha256": null, "severity": null, "scope": null, "temporal_pattern": null, "time_ranges_s": [], "roi_xyxy_px": [], "detail_loss": null},
  "preference": null,
  "fishscale_change": null,
  "fidelity_tradeoff": null,
  "confidence": null,
  "notes": null,
  "human_next_action": null,
  "blind_revealed_after_first_review": false
}
~~~

**所有文件须可回看。** 评审 UI 或 Review Index 不得只提供缩略视频或截图；容器无损不代表 VAE 的 FP32 浮点没有量化，数值边界另由原生运行记录判定。

## 结束条件

观察到明显改善且不存在通过模糊掩盖伪影的明显代价，称为 human_visual_improvement；只有对应首个异常边界经过对照支持才称 root_cause_localized。人眼 uncertain 时保持 INCONCLUSIVE，不由机器分数计算出 success。
