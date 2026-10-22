# R12 D0 已完成：固定零补偿三动作未过门槛

当前决定：[CURRENT_DECISION](../CURRENT_DECISION.json)；说明：[docs/68](../../docs/68_R12_D0等预算原生动作检验.md)；原始结果：[native_results](R12_D0_native_results.json)。**模型资格为空；本轮24次已经用完，不能重跑或自动扩大。**

## 实测结果

四个已暴露TF2来源0/5/10/15，固定step3/block30–31；N0=162/162、NA=216/108、NB=108/216，raw偏置、alpha1、pair零R256/零训练，其余158位置为UB1。三动作真实主GEMM都为dense的50%。完整步骤之间恢复DQ输入，层间不恢复。

先8项DQ/UB1，再8项N0/重复，全部通过后执行8项NA/NB。**24/24完成且数值与成本控制通过**：原四步DQ/UB1逐位复现、N0重复逐位相同、16个完整216原教师Y/Q1/Q2身份、每次1600模型linear/QDQ、320个真实主GEMM、正确补偿数量、零raw fallback。32次额外教师linear/QDQ单独计费。

汇总SSE：UB1=181764.492633；N0=187459.333938；NA=186876.716781；NB=188395.528387。最佳固定为NA，逐来源oracle为NB/NA/N0/NA，SSE=186443.298254。条件收益 **0.231928%<1%**；相对UB1反而 **增加2.574103%**。η=0，只通过噪声门槛。不存在新阈值、扩大q/宽度/来源或后续阶段触发。

## 失败归因的新增证据

[预算扣回复算](R12_D0_budget_debit.json)在原R7和新D0的同源原生操作数SHA上核对：NA完整block30输出和随后block31输入完全一致；其将block31改为108组raw的合计代价7954.605357，超过历史完整恢复收益2842.381210。NB合计代价9496.708923，也超过收益2865.673169。代价包含缩宽和移除原偏置/gain/R256组合，不能称为纯宽度效果。N0仅移除两层校准组合已增加5694.841305 SSE。

这关闭固定raw实例，未关闭所有补偿、空间、跨步、信息或动态FFN机制。R4/R7/R9/R11的原协议和有限拒绝原样保留。没有新预测器、自由学生质量或完整FFN测速；诊断视频按构造为DQ。

## 节点、存储与执行记录

用户另行指定空闲节点后，镜像原模型/HiF4缓存，93项SHA全部通过；原排队attempt1为零模型调用并保留。实际仅在新节点GPU0/1执行attempt2，环境/权重/输入不变，四卡目前均空闲。新实验输出加协调端逐位收据/flow镜像共439223752字节（0.4091GiB），最低记录空闲123.27GiB；峰值GPU分配65.53GiB、保留78.10GiB。见[完整资源账本](R12_D0_completed_resources.json)。

协调器本地阶段屏障原先读不到节点本地HF收据，因此仅SIGSTOP协调器，保持模型任务运行，逐位镜像已完成JSON/flow后SIGCONT；两个屏障均保持原校验，没有修改冻结源码。最终48份收据/flow镜像校验通过，零额外模型生成。

私有日志、全部RUN与收据在忽略目录`runs/r12d0_native_value_attempt2`；flow/视频在HF派生缓存，公开结果使用中性相对locator。原执行协议、源码SHA和授权见[R12_D0_before_execution](R12_D0_before_execution.json)及其链接。当前路线检查会拒绝再次启动此方法。

## 已完成的不同方向工作与复核入口

[HiF4完整64组/行边界](../../docs/69_HiF4分组边界与空间动作的可执行布局.md)的实际CPU参考检查与组内重排反例、21帧真实布局算术已经保存。[新增信息/成本复算](R12_information_budget_discriminators.json)构造了相同均值但相反非线性方向的反例，并重核双专家密集冷启动和两GEMM代理成本；[docs/70](../../docs/70_条件信号与跨步方法的真实预算.md)核对Prox、Chipmunk和RAPID³原文。这些是源/数学/CPU结果，不是新原生tile或路由结果。

```
python3 tools/check_research_routes.py
python3 tools/check_r12_d0_artifacts.py
python3 tools/analyze_r12_budget_debit.py --output /tmp/r12_debit_review.json
python3 tools/analyze_r12_information_budget.py --output /tmp/r12_information_review.json
```

当前开放研究资料库来自远端`40a5e49`，已合并且不是排他的实验队列。主目标仍为同一冻结D-Q模型的可部署输入/状态条件计算、真实完整FFN速度与新来源自由视频质量联合达标。下一项模型实验需要自己的实质机制、注册和授权；D0拒绝既不自动触发它，也不构成全局不可能结论。

[最终检查记录](continuation_checks_R12_D0.json)：68项相关测试通过；不含`.runtime`/`runs`的干净Git导出再次通过68项、5个证据CLI和合成反事实smoke。公开元数据扫描覆盖两个私有节点标识并通过；关闭后的D0方法选择返回退出码2。
