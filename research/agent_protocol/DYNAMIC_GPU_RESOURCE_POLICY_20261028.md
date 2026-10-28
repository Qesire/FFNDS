# FFNDS · GPU 动态资源调度原则（2026-10-28，现行）

**用户最新要求**：使用 GPU 的数量以实际能够调用并投入工作的最大量为准，**不预先设置全局卡数上限**。本文解释资源调度；科研路线、数据来源、实验范围、模型身份、评测要求由原有权威文件管理。

## 1. 唯一现行规则：实际可调用量，而非 2/4 的固定整数

- **全局硬上限：无预置值（`null`）**。2026-10-27 及早前「最多2张」「最多4张」属于过去指令和已冻结实验预算的时间点；它们不是现行全局限额。
- **实际调度目标**：在当前权限、账号/项目配额、Slurm 分区及 QOS、可成功分配的空闲 GPU、每个 job 的显存/CPU/RAM/时长、可并行且相互独立的 READY 工作数量之内，充分使用**实际能够申请并持续正常运行**的资源。资源数可以在各次调度/释放后变化；不得把历史机器总量或看到的空闲量直接当成已获分配量。
- **科学问题决定producer与并行任务**：按[开放式证据采集合同](EVIDENCE_DRIVEN_ACQUISITION_20261029.md)优先启动已明确需要且已注册的native source扩充、时空tubelet、student状态、动态量化信息或完整视频生产者。A100 BF16和实际可调用的PRO6000原生FP4可按模型/算子条件并行，旧M-B/B1等历史优先级不再控制全局任务。资源分配服务于决定性的科研证据，不单纯为了占满GPU重复运行。
- **每任务资源**：需要 1 张 GPU 的单个分片仍可记录 `GPU_per_part=1`；这是实际算法资源需求，**不是全局GPU上限**。同理各任务真实工作量、数据/磁盘预算及单任务时间限制仍要冻结。

## 2. 启动前和运行期间如何获得实际数量

1. 检查真实可访问节点、Slurm queue/partition/QOS/account、账号配额、已占用的同账号 GPU 与其他任务，并结合 `sinfo`、`squeue -u "$USER"`、`scontrol show job` 和工作环境下实际可用的管理接口。检查可用显存、CPU/RAM、已有模型实例、数据及权重驻留条件；拒绝把「节点显示空闲」直接当「可调用」。
2. 先列出**已经准备完毕、输入及预算可核对**的独立 job/part，核实当前 `ACTIVE_RESEARCH.json` 的实验身份、冻结代码/资产/参数及对照。允许按用户新资源指令创建新版本的注册范围；禁止隐式修改旧协议哈希或重写原失败/完成凭据。
3. 在调度系统许可范围内，尽量同时提交/启动所有有用且独立的 READY 作业，使用真实的 job 配额/分配结果递进扩容；调度器拒绝、OOM、系统占用或权益限制时按**实际可调用**资源回退、按依赖重排。对于同一轮准备完成但不足以使用全部空闲卡的情况，实际并发即为工作量上限，不制造伪任务。
4. 每次扩容/失败/作业退出后，重新检查可分配资源与 READY 队列，及时释放已完成或失败作业的占用。每轮 RUN/EXIT 记录实际 GPU UUID/节点、申请和获批数量、并发峰值、task id、单job CPU/RAM/显存、起止/耗时、失败和重试（默认不自动重试）、冻结 protocol SHA 及实际产物。
5. 不在不知道可调用数量的情况下填入默认数字；若当前无法访问集群，记录 `ACTUAL_CALLABLE_GPUS_NOT_OBSERVED`，而不是虚构 GPU 数或把旧 2/4 当兜底值。

## 3. 与旧范围及真实实验控制的关系

`research/ACTIVE_RESEARCH.json` 的 `resource_scheduling_policy_20261028` 是当前**资源数**的权威指针。此前记录的 2/4 或每个 frozen scope 的 `global_max_GPU` 继续作为**历史协议/资源收据**保存，说明当时怎么执行。原 `user_continuation_authorization_20261027.max_simultaneously_assigned_idle_GPUs=2` **保留原值以使已冻结 V2 启动器能够验证其历史范围**，最新全局资源策略以本文件及 ACTIVE 的 `resource_scheduling_policy_20261028` 为准，后者 `global_gpu_card_cap=null`。若未来某些未执行分片必须突破其冻结的 scope 并发预算，须产生新注册协议/范围及完整 SHA 绑定；不能直接把冻结 JSON 中的 2/4 改掉后仍声称原协议身份不变。旧 V2 每个已登记作业可以遵循其精确成本执行，其他独立的新注册作业可以在账户实际允许的资源内与之同时调度。

模型/视频执行仍需要真实 ACTIVE scope、源/种子/对照与资产冻结、完整质量指标和可追踪 RUN；「动态最大可调度」只是**资源利用策略**，不是「任意启动未登记实验」。前一批任务成功不构成新的模型执行记录或已分配 GPU 证据。

## 4. 快速检核

- 阅读 [ACTIVE](../ACTIVE_RESEARCH.json) → [任务台账](CURRENT_TASK_LEDGER_20261028.json) → [当前 B1 交接](NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)。
- 新登记范围的全局资源数使用 `dynamic_actual_callable` / `global_gpu_card_cap: null`，已冻单任务需求与分片协议保留其精确成本。
- 静态检查 `python tools/check_autoresearcher_binding.py` 和 `python tests/test_parallel_research_scope.py` 验证权威指针；它们不会查询 Slurm、申请 GPU 或替代真实调度验证。
