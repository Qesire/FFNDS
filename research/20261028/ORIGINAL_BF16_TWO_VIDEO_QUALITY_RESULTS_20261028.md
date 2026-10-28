# 原始 BF16 两视频：八轴官方评分实测

两条既有原始 BF16 Dense 视频已在原版评测环境中完成八轴评分，actual exit0；14个视频维度结果、80次RAFT／1600 iterations，零新生成、零训练。

| 来源 | MUSIQ | aesthetic | smoothness | 连续 RAFT | dynamic binary |
|---|---:|---:|---:|---:|---|
| 0 | 74.2573 | 0.576321 | 0.983478 | 4.836444 | False |
| 12 | 77.5607 | 0.715613 | 0.992380 | 2.885609 | False |

两来源等权 MUSIQ=75.909002。原始基线有可测连续运动；同为False的二值动态结果不区分4.84／2.89与旧6016候选的约0.36–0.64。接触表中运动看起来细微不足以判定 motion collapse，正式连续评分提供了更合适的定量对照。

完整 JSON 保留其余主体／背景／I2V轴、两来源各40对RAFT原分数及五个固定8-pair窗。原始BF16与旧HiF4 Dense都是各自版本的观测；模型数值和算子路径差异不允许将差异归为单独的量化因果效应。

耗时 81.765s，outer 83.725s，峰值reserved 1.379GiB，26份执行源码SHA复核、monitor 0违规。CPU新字节proof实际exit0，复用的原v4八模块imports proof有真实独立exit0。

N=2已暴露开发图像簇，每来源一个seed42；官方五独立采样、人工完整播放、camera和flicker仍未验收。本结果建立当前数值路径的开发基线。

协议 `7c7b8f424df9ed5a1425c07ad5b63a8edf20f65253656508ec23efe727cdffbf`；RUN `06a05dddf8bffdc30f9e340337278572637add314eec1a314e5de94f1656759a`；actual exit `87273b13052a9caacc09695da0831de20bd192870c6a43fcb25f98b9c8f49e75`。

[完整机器结果](ORIGINAL_BF16_TWO_VIDEO_QUALITY_RESULTS_20261028.json)
