# 计算图与缓存优化：独立实现记录

2026-11-28。用户要求“明确区分并分别实现尽可能的优化”。本轮将上一轮推导落实为分开的代码路径、12 个配置、离线资产编译器和现有四步生成器适配入口。

- [实现、数学身份与当前结果](IMPLEMENTATION_RECORD.md)
- [运行接口与全部配置](../../../tools/ffnds_optim/README.md)
- [机器可读记录](IMPLEMENTATION_RECORD.json)
- [CPU 测试原始结果](evidence/CPU_TEST_RESULTS.xml) · [Triton 解释器结果](evidence/TRITON_INTERPRETER_RESULTS.xml)
- [BF16 sm80 编译证据](evidence/ATTENTION_KERNEL_COMPILE.json) · [CUDA 设备状态](evidence/ATTENTION_CUDA_STATUS.json)
- [文件身份清单](MANIFEST.json)

当前结果是 **50 项 CPU 测试通过、1 项独立 Triton FP32 解释器测试通过、2 个 BF16 内核完成 A100 sm80 离线编译**。完整生成入口已连接并通过 CPU 集成核验；本轮没有 GPU 设备、Wan 前向、新视频、评分或计时。此前 50 源评分与 108 次计时仍归属于原冻结版本。

## 五源实机测试

[执行方案与学校作业](five_source_pilot/EXECUTION_PLAN.md)已固定五个配对来源，学校 A100 正在执行新配置。
