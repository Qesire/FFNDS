# V3 target-worker monitor identity correction

作者 `/root/sm120_full_quality_build_terra`。Root 启动前在目标 SM120 worker 实查 `/usr/bin/nvidia-smi` 为 1,243,808 B、SHA-256 `384896f294f80e8e5b59c11dbb04d3fdc028cf9bb01d7839154b96f94e966f4e`，V2 协议误绑定生成 metadata 的另一 worker 文件。V2 两 scope 仅 metadata 登记，没有启动 CPU 或 GPU；保留旧协议和注册身份，不覆盖旧回执。

现已在实际 SM120 worker 生成新的 V3 draft、freeze 和 CPU/GPU/VERIFY 私有 jobs；V2→V3 深度比较每个 part 仅 `execution_monitor_binary` 改变。算法、source、parent、资源、限制和执行步骤未改变；V3 使用新协议 SHA、fingerprint 与 job SHA，等待真实独立 recheck 与 root 后续 ACTIVE 重绑。Target-worker metadata 命令不运行模型、GPU、评分或视频。原 38 项合成测试与 [V1作者回应](AUTHOR_RESPONSE_V1.md)的科学、失败与预算边界仍适用；此前 V2 评议不自动授权 V3。
