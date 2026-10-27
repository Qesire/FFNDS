# UB1 consumer disjoint20：首次执行的部分结果

**实际执行（NATIVE/VIDEO）**：原冻结范围四个 part 的 launcher 退出码为 `[1,0,0,0]`。part1–3 完成30条真实视频及15个 fresh Dense/candidate 配对；part0 完成 source0 两条生成及一对，source1 Dense 另已生成、保存四步和完整81帧 core 记录，但该 part 在 core 后的资源检查因 GPU 监测查询超时而失败。全范围共能核对33份完成生成的原始 receipt 和132个已保存 flow step；这不是40视频/20配对的完整验收。

原协议 SHA256 `ad81fffb73256c14e65c4221d255080072bcc9902cf420b541e4da4f61254ca6`。逐part RUN/launcher SHA、全部33份 generation/core SHA和实际账本在[机器记录](UB1_CONSUMER_DISJOINT20_INITIAL_EXECUTION_20261027.json)。part0 原 RUN SHA256 `916917c2a657884dece12c46fda2930f8a8d1b9250a6342a67397db7adb8a706`，launcher SHA256 `74253a96bfe727775d065c5407641611b75b3da29d7fa9443ab50ab77d619ece`、退出1；其他三个均退出0。原冻结文件和失败收据保持原身份。

独立只读核验33个 GENERATION：每个记录四步、160个 FFN 时序位置、1600次 native linear 和1600次 activation QDQ，合计132 flow step、52800次 native linear、52800次 activation QDQ；每个对应 core 记录有81个完整解码帧。33次 generation 也已在part账本中预扣，但预扣本身是预算身份，完成数量由上述实际生成收据确认。part0 source1 Dense 的 GENERATION 与 core 文件存在，最终 PHASE 状态仍是 `FAILED_CORE_METRICS_PRESERVED_NO_RETRY`，不能将它升级成通过的完整 part 或配对。

part0 的失败文本为 `Assigned GPU monitor failed: TimeoutExpired`：监测查询超时，238个成功采样中没有记录 foreign PID，但超时段的隔离状态未知，不能声称整个运行过程始终隔离。part1/2/3分别记录449/705/714次采样，均无监测违规或采样到的 foreign PID；采样也不是连续证明。该失败是运行监测故障，不是 consumer 数值或视频质量的反例。

part0 整体保持隔离，不纳入本次全20来源验收；原失败和已产生文件不覆盖、不自动重试。source0–4 的后续补全属于独立新范围，目前仅准备，尚未运行。本报告不分析视频质量分数，也不声明官方评分、人类验收或算法泛化通过。
