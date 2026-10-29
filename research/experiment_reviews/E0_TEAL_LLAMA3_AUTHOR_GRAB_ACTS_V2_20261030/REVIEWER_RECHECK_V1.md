# TEAL Llama3 校准 V2：确切修订版独立复核 APPROVE

2026-10-30，独立评审 `/root/sm120_full_quality_build_terra`，作者 `/root/teal_full_llm_calibration_build_terra`。仅核 producer `1dd31c32a2b7c215927739e015bc295abe78efbb0a9fa72ca427c2c0ab1cbf78`、tests `6b13b76295f298eeb3db71c8eb9e6ee0e5d596026c455e26e8ffc6c334c340b3`、protocol `e7652d7d5cefe199425066232deba5bba9a028d821e5a6e11df6276e05f07fdf`、私有 spec `f3bad024d3315a1a8e3c8d465325433742117b8ad5f541e87fa69e3a59b617ef`、fingerprint `2037109d9b82cdc9190c6d40c2bccc064f7b468ec8ba5a11bc5a683db3f53ffb`。

初审的环境反例已修正：实际 supervisor argv 包含 `-I -B`，执行检查 `sys.flags.isolated`，exact argv/非隔离拒绝测试通过；相同目标 venv 实际无 GPU 强制 FA2 分支及作者模块导入测试通过。新 protocol 对 V1 实际失败 RUN/EXIT 字节与状态作绑定，检查真实 EXIT1、零模型载入、零 layer forward/sort/hist，不冒充自动重试或清除历史。8项工程/合成测试通过；目标 venv inspect 得到上述 fingerprint。作者提前导入造成的 `utils.data.load_dataset` 绑定按实际源码显式替换，保留原有数据选择与作者 token/层校验。此前评审逐项机制和账本限定继续适用。

**反例及条件：** 实际 CUDA 可见状态下仍可能出现不同惰性导入、FA2 ABI/kernel 不兼容、内存/时间超限或其他错误；本次 CPU 强制分支导入不证明 GPU 兼容。采样式显存、RSS、输出、外部进程监控不证明瞬时峰值和无竞争；asset stat 不能约束非合作写者。失败须保留真实 EXIT 和 partial 且不自动重试。成功后的64 histogram还需真实 closed 与后续消费者独立重 hash，不能凭计划称实际校准成功。

结论 **APPROVE**：仅允许此确切新范围进入协调者 ACTIVE 登记和启动前身份/资源核验；无模型、数据、GPU计算由本 reviewer 执行，实际结果仍须 EXIT0 与 closed 验证。
