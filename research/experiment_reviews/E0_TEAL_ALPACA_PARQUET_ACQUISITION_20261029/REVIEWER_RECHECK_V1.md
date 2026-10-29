# 独立复核：Alpaca 固定 parquet 获取 — APPROVE

2026-10-30，独立评审 `/root/sm120_full_quality_build_terra`，作者 `/root/teal_full_llm_calibration_build_terra`。本次针对明确版本：producer `ee86a8e9f0ed0aa48da701e3ac0bc4664bb7ba4b184caf97868f129636f53c23`；test `a1343ad0f7a778e7852f2e6977e7ce0522ecee8458ee09ecae45eb27c82418a2`；protocol `config/e0_teal_alpaca_parquet_acquisition_20261030_pending_review_v3.json` SHA `3a952002c027aafe809c31a57304a86895a14549235f2ec1428c20678fb4e1af`；scope fingerprint `c6dc8c9288799bff569657a61808a6d4a77dd0b52fba8554ab759cde1f0220ea`；私有 `CPU_JOB_V3.json` SHA `0245e82a01462d19d53218230c9758a7af0d034a5800d5e1120d2d6da62c36d1`。初审 [V1](INDEPENDENT_REVIEW_V1.md) 的 REVISE 保留历史。此结论只审核所列单个有限资产范围，不是 ACTIVE 或启动授权。

- 固定公开 revision、唯一 parquet 与 LFS SHA/size 已由本 reviewer 独立只读 HF 元数据核对。真正获取后同进程两次完整哈希、总预扣 48,493,276 字节，不额外增加第三次校验读；固定 revision 是项目选择，不假称作者历史快照。
- 后端调用及每次 hash 预扣；被中断时未完成进度 UNKNOWN。wall、CPU、峰值 RSS、实际 worker 使用和异常状态分别记录；Torch/Triton/Transformers 在 backend 导入前后无导入。固定匿名、fresh scoped cache、未启动模型或读取数据记录。
- 输出扫描只跳过可变严格后代 ENOENT 并计数，root 身份和其他异常失败；超预算落失败 RUN 并抛错。128 MiB 为退出时统计及每文件限制下的范围，非实时采样或硬 aggregate filesystem quota；SIGKILL/不可写时实际 supervisor EXIT 和目录盘点保留，未报告的网络字节 UNKNOWN。24 MB 单文件、一顺序调用场景接受该有限范围；不得将它升级为瞬时峰值证明。
- `inspect-closed` 检查真实 supervisor child EXIT0、RUN/protocol/spec SHA、两次完整 hash 账、调用计数、当前固定 cache blob 路径/regular file/大小。闭合不重读全文件；因此若有不遵循私有目录所有权的其他写者在两次哈希后以同尺寸改写，单靠此 stat 不能检测，启动前确认 scoped cache 无并发写者、下游再各自绑定/校验实物。
- 逐项修订见 [作者回应](AUTHOR_RESPONSE_V2.md) 和后续 V3 更新；本 reviewer 实际执行 7 项非实验合成测试，全部通过。测试及静态检查不构成数据获取结果。

独立 recheck 结论 **APPROVE**：注册前仍须绑定真实两人 review 记录、确切代码/config/job 与私有 worker 身份；实际启动及其 EXIT/closed 按 ACTIVE 单独处理。未下载数据、未改变 ACTIVE、未运行模型/GPU。
