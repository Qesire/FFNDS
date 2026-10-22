# R12 D0：等预算原生动作价值检验

已完成 E0 后，用户明确批准直接执行固定 D0。CPU 已重新核对原始两层的 108/162/216 支持、实际权重与 Smooth 哈希：三种动作全局主 MAC 均为 358628091494400，严格等于 dense 的一半。这里尚未测得新动作收益。

本轮单独登记 [D0 协议](R12_D0_registered_protocol.json) 与 [授权范围](R12_D0_authorization.json)，原 [E0 交接](HANDOFF_R12_E0.md)、[草案](R12_E1_native_value_DRAFT.json) 和旧路线关闭记录保持历史原样。主问题是固定三动作的来源条件 oracle 能否同时胜过最佳固定动作及冻结 UB1，各至少 1%，并超过重复噪声的十倍。

原生执行器使用真实 packed GEMM。step3/block30–31 的 N0=162/162、NA=216/108、NB=108/216 全部使用原始偏置、alpha=1、零 R256、零训练；其余 158 个位置保持 UB1 数值路径。冻结 UB1 单独作为带原补偿的工程对照。block31 接收真实改变后的状态，没有中途恢复；只在完整步骤之间恢复原 DQ latent。生成视频因此不是自由学生质量结果。

严格按三阶段执行：4 DQ + 4 UB1；4 N0 + 4 N0 重复；前 16 项全部通过后执行 4 NA + 4 NB。每个完整四步必须有 1600 次模型 linear/QDQ、320 次真实 FFN 主 GEMM、正确的 0/160/158 补偿位置及零 raw fallback。每次完整 216 支持另做原教师同输入 Y/Q1/Q2 逐位控制，并单独计费。总上限 24 次、最多两卡、HF 新增不超过 0.5GiB、节点始终保留 80GiB；不落盘完整 M 激活或复制权重。

当前两张卡均被其他任务占用；没有抢占、终止或启动 D0 模型任务。执行前通过项目 GPU 锁和原生资源检查；数值控制失败即保留记录并停止本轮。D0 未过门槛只关闭这个零补偿三动作探针，不扩宽、不换来源、不调阈值。即使过关也不自动启动 D1、预测器或 rollout。

入口：`tools/run_paired_budget_value.py`；分析：`tools/analyze_paired_budget_value.py`；仅 CPU 的已执行准备：[preflight](R12_D0_CPU_preflight.json)。运行前必须通过 `python tools/check_research_routes.py --method-id R12E1_D0_EQUAL_BUDGET_NATIVE_VALUE`，并核对 [冻结源绑定](R12_D0_before_execution.json)。

调度入口 `python3 tools/execute_paired_budget_value.py` 只排队这一个冻结 attempt。它等 GPU 空闲后再调用既有 worker 锁，逐阶段核对完整对照，不覆盖任何 RUN；遇控制失败保留 STOP 并停止后续生成。执行状态在忽略目录 `runs/r12d0_native_value_attempt1/EXECUTOR.json`。完整结果产生前不自动提交或推送。

补充准备：[原始模型/输入/完整 TF2 张量身份](R12_D0_asset_preflight.json) 在 CPU 上通过，四来源原图与 seed42 完全一致；原生函数签名用源码 AST 检查，未伪造 CPU 上的 CUDA 可用性。[准备检查记录](R12_D0_pre_execution_checks.json)保留一次零模型生成的 CPU import 失败及修复。50 项相关离线测试通过；原生 QDQ/Y/writeback 仍须由实际 GPU 对照证明。
