# 原作者全32层校准：独立复核 APPROVE

2026-10-30，独立评审 `/root/sm120_full_quality_build_terra`，作者 `/root/teal_full_llm_calibration_build_terra`。本复核只针对 producer SHA `9647065627b3af8dba90a561ad476ec9cb979c44155a85c8b6750934cab90d76`、tests SHA `360b9287d5a1702ca95c5860e528c0699ebe365180c6da810daf9b6fe9bb786d`、protocol SHA `ce9f2db923218708178e4dc3c5b030d5950ddc9c15a1cc1e482e3ea219648d7f`、私有 GPU_JOB_V4 SHA `f5fd1878d2a9451c8a541fb8b908dea87bb876a7cc29a5e1e8e0662b1a540cf3`，fingerprint `b1ceecdd32e6c9371a9a24117e4d74bb03a13679dd902724470c49bd5b734d1c`。V3 作者回应适用，V4 增加13输入 blob 的 hash 前后 fd/path 身份和执行前后 stat 对照。

初审五项逐项核验：作者实际 tokenizer 的前10段 IDs 与 attention mask 同独立预检逐字节哈希比较且首层执行前验证；32个层有顺序、形状和 pre/post 完成记录，64次全量 finite 归约及 5,368,709,120 元素明确扣账，sort/histogram/save 记录 attempt 与成功完成；实际模块来源、完整 supervisor 路径和资源 identity 被校验；CUDA 上下文在资产复验前建立，UUID 监控外部 compute PID，全部参数和设备映射受检；64个 CDF 文件检查键、形状、有限性、正数量及有序中心并预扣 hash/读取。原源码路径和条件、输入实际 EXIT 及代码绑定均须在登记和启动前再次检查。五个非实验合成测试实际通过；inspect 输出 fingerprint 与上列一致。

**反例和限制：** 在两次采样之间出现并消失的外部 GPU 使用、进程内瞬时 RSS/GPU 峰值和子进程 RSS 均非完整观测；输出和资源监控是采样式软限制。输入 stat 比对防止通常的可观察变化，不能提供针对非合作写者的强制不可变性，且 hash 完成与后续 stat 之间仍有竞争窗口；资产闭合后的单写者条件是必要运行前提。FA2 wheel / torch / GPU 的兼容与32 GiB host预算仍需真实执行才能判断，失败保留真实账目且不得自动重试。校准样本来自固定公开数据前300行，是该有限采集的身份；本批准不代表生成质量、性能、完整 LLM 或 Wan 复现。

结论 **APPROVE**：仅批准这个精确版本进入协调者的 ACTIVE 登记及启动前复核；不自行登记、执行模型或 GPU，也不替代实际 EXIT0 和闭合核验。

## 撤回：REVISE

本复核的 APPROVE 已撤回。协调者随后在 CPU-only 工程导入中实际得到 `utils is not a package`：`AUTHOR/utils` 被插到 `sys.path` 首位，`utils.py` 可遮蔽 `utils` 包，且镜像包初始化文件需核对。此反例发生在模型启动前，是实质启动阻断；上述五个合成测试没有覆盖真实导入。作者应修复固定来源包初始化和搜索路径，并在相同环境实际导入六个作者模块后，提交精确新版本重审。该工程检查不是模型或校准实验，旧批准不得用于 ACTIVE 或启动。
