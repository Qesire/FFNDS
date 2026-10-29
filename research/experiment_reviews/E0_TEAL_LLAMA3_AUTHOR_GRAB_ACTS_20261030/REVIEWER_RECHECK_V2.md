# 原作者全32层校准：V5 独立复核 APPROVE

2026-10-30，独立评审 `/root/sm120_full_quality_build_terra`，作者 `/root/teal_full_llm_calibration_build_terra`。V4 的批准因实际 import 阻断已撤回。本次仅针对 producer SHA `d7eafd1c950406a24f750827c7301c20da3a172c81da2a20624ad41b24a371bd`、tests SHA `b871352ed2c6d7d6cdaecbb6529a3aa596233fd851c35a5ed4e91c7cc08fe241`、protocol SHA `2662652d8cc03637746bc2777cc77c2e4335c2be42ad2d253608c31ff28fa939`、私有 GPU_JOB_V5 SHA `04ea596cd1f8293eb7f2d4728caee5b452210627eedcb8b8c1a5d0b797df3d1c`，fingerprint `7fd4071e4eaa4594f1d87bf9f8a9658970a8a912e7b0ecf2c3712a70750e919a`。

V5 将作者根目录置首而不将 `utils/` 放在首位，绑定原作者 `utils/__init__.py`，真实 CPU-only 子进程在目标 venv 中导入六个作者模块并核实际文件路径；六项合成/工程测试实际通过。Parquet 与13个模型文件均采用单次完整 hash 中 fd/path 身份核验及执行前后 stat 对照。V3 对初审五项的修订及上一复核的其余分析适用：真实 author token IDs/mask 比较、32层前后 hook、全部有限性 observer 扣账、源码导入身份、GPU PID监控和CDF 检验都保留。

**反例与边界：** stat 检查和采样式资源/PID监控不能证明两次采样间或非合作写者下无竞争；启动时仍须核输入无其他写者、空卡与完整确切身份。FA2 wheel / torch / A100 实际兼容和32 GiB host预算可能导致真实失败，失败保留账目、不自动重试；GPU、host和输出峰值是采样观测。该固定前300行采集不证明完整 LLM 质量或性能。没有执行模型、GPU、校准或读取数据行。

结论 **APPROVE**，只允许协调者据这个精确版本登记并再做启动前身份和资源核验；不替代实际 EXIT0 和 closed 校验。
