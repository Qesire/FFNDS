# 计算图与缓存优化：独立实现记录

**五源 A100 实机测试已完成；当前新增优化未改善原基线的整体取舍。** 组合计算图完整请求速度比为 **0.816×**，组合状态为 **0.794×**；后者峰值下降 **36.80%**，同时成像／美学平均下降 **6.435／3.334 个百分点**。当前结束将这些组合推进为主方法升级，保留原冻结参照与全部实测记录。[结果后的重新判断](five_source_pilot/review/REASSESSMENT.md) · [原始结果](five_source_pilot/RESULTS.md)。

2026-11-28。用户要求“明确区分并分别实现尽可能的优化”。本轮将上一轮推导落实为分开的代码路径、12 个配置、离线资产编译器和现有四步生成器适配入口。

- [实现、数学身份与当前结果](IMPLEMENTATION_RECORD.md)
- [运行接口与全部配置](../../../tools/ffnds_optim/README.md)
- [机器可读记录](IMPLEMENTATION_RECORD.json)
- [CPU 测试原始结果](evidence/CPU_TEST_RESULTS.xml) · [Triton 解释器结果](evidence/TRITON_INTERPRETER_RESULTS.xml)
- [BF16 sm80 编译证据](evidence/ATTENTION_KERNEL_COMPILE.json) · [CUDA 设备状态](evidence/ATTENTION_CUDA_STATUS.json)
- [文件身份清单](MANIFEST.json)

初次实现提交 `a366f689` 的结果是 **50 项 CPU 测试通过、1 项独立 Triton FP32 解释器测试通过、2 个 BF16 内核完成 A100 sm80 离线编译**。完整生成入口已连接并通过 CPU 集成核验；该初次实现阶段没有 GPU 设备、Wan 前向、新视频、评分或计时；后续实机结果以本页顶部链接为准。此前 50 源评分与 108 次计时仍归属于原冻结版本。

## 五源实机测试

[执行方案与学校作业](five_source_pilot/EXECUTION_PLAN.md)已完成12配置、每配置5条质量视频，以及55个候选／同GPU参考配对。[结果](five_source_pilot/RESULTS.md) · [独立复算与研究取舍](five_source_pilot/review/REASSESSMENT.md)。
