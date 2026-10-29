# TEAL Wan sampled-row 可执行性 CPU 阶段 A：独立初审 REVISE

2026-10-30；评审 `/root/sm120_full_quality_build_terra`，作者 `/root/teal_full_llm_calibration_build_terra`。精确检查 producer SHA `ea2f0b4e24843a534b287ed93062231bf143a93c389a0a9d2d4b28979216d3fb`、tests `1aaa99bc287b070e3985881db0ce8d4185116282b8f03561e9b0305b556b322b`、protocol `4c7ee816fb4a8e53af713a54b919bbfe4d9caf0e4e979ead90e1a2fbd5dd4059`、私有 spec `7fd01f16ec3a8ecddb7d6b69968beedcba773b43d8205839ee2aa2d3736890a0`、fingerprint `aeb9ef7a1b8407a1b74a7e07edb5cd3445ae43858c234c15e257ea7c6e8516ce`。三项非实验合成测试通过；未读取真实源tensor、运行模型或GPU。

数学与身份核查：`keep=abs(h)>t` 的补集严格对应视频 V3 的 BF16 比较/含等号删除；2:4 沿每个实际 token 行的hidden K连续4列计算，`≤2` 是存储资格必要条件，不保证layout或速度。采样行中全冷列集合包含真实全M冷列集合，故样本冷列数是同状态全M冷列数上界。反例：每列在采样行都为零、但未采样行至少有一个值大于阈值时，上界可为100%，真实全M冷列为0。每行2:4资格只作用于实际采样行，不代表完整M或连续16行tile。两条source独立、160位置/256行嵌套，源内逐site等权；角色和暴露身份正确。

原始父 RUN/EXIT及manifest经既有闭合门检查；每个payload完整SHA后按实际文件h/rows读取，raw tensor SHA 与metadata原行坐标匹配。哈希、h和rows各自字节及packet操作预扣且中断当前读记UNKNOWN；总读不含metadata/source/包身份与内存扫描，已明确。固定独立supervisor、私有worker、CPU-only、source闭包、实际EXIT和软RSS/输出监控有明确边界；输出退出时才扫描，不是硬连续aggregate quota。现有三测试核实际V3 BF16阈值边界、冷列和K4基本情况。

**需修订：** 640次 `compute(h,t,torch)` 只在返回后递增 `completed.site_rate_evaluations`，并且两个rate之后才persist；第一次调用完成、第二次调用中断时，现有RUN不能证明哪些实际评估已发生。逐rate开始前持久预扣/尝试并校验≤640，成功后立即记录完成，增加合成调用中断及超预算拒绝测试；保留真实失败partial。该问题不要求对每个底层tensor归约逐kernel计数，但不能将未观察的内部计算写为已知。作者回应与修订后提交确切协议/spec/source再复核。

初审结论 **REVISE**。不是执行批准；真正独立recheck后还须ACTIVE登记、身份复核及实际 EXIT0/closed。该范围不测W2、GPU、视频质量或系统速度。
