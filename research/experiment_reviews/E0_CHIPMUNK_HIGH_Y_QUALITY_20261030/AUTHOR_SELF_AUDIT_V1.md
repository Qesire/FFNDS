# Y-only 三臂质量范围作者自审

两来源 898/96，均为暴露开发来源、每源单 seed：新生成 NONE 与 HIGH_Y_REUSE，复用已真实闭合 factorial HIGH_ONLY 视频作为 HIGH_P64；NONE 与旧 Dense 视频字节相同。Y 父和旧 P64 父均要求原 GPU 与独立 VERIFY actual EXIT0。每源三视频逐视频调用原版八维共24结果；其中原版 dynamic degree 的 RAFT 和独立连续 RAFT 分别真实调用，各120 forward、2400 iterations。连续 RAFT 固定81帧 stride2 的40相邻采样对，不描述为全部80相邻帧。各轴分开报告，按来源等权；重复旧视频评分不是新样本。

新视觉范围只解码 Y 候选每源81帧、两接触表；旧 NONE 和 HIGH_P64 已有之前的视觉资产；本 scope 不绑定其 manifest 或验证回执，后续汇总必须独立核实旧视觉闭合与视频 SHA 后才可复用，接触表只供独立 AI 查看，不是完整播放或人工验收。所有输入、source、两级父闭合和真实视频绑定在 scope；metric 子进程内额外小型父身份 hash 作为 UNKNOWN，不归入主进程 HashReads。preflight、GPU、独立 VERIFY 均须真实 EXIT0，失败保留、无重试。新 scope 不生成视频，不训练。

每源 quality 与独立 RAFT 各一 GPU，视觉 CPU；全局最多四 GPU。各 phase 按实际 hash 次数和字节先预扣、完成后记账；snapshot 和 worker 额外输出预算分别保留。复用原指标环境及权重，不下载新模型。旧 frozen producer 均未改。四个 CPU 合成测试覆盖三臂身份/负例、24/120计数与八轴/RAFT汇总；六个隔离帮助入口通过，未实际评分或解码。protocol/jobs 当前待真实不同 Agent 独立初审、回应及 recheck，ACTIVE 由 root 登记。
