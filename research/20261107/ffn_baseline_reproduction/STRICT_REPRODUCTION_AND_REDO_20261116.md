# ICC / Chipmunk：完整复现与下一步实验

2026-11-16。当前下一阶段方案已统一到 [CONTROLLED_REPRODUCTION_PLAN.md](strict_reproduction_20261116/CONTROLLED_REPRODUCTION_PLAN.md)，执行者从该文按阶段0–5推进。本文保留入口与证据定位，不再维护第二份实验顺序或预算。

## 本轮要完成什么

先完整验收作者FFN算法与Wan接口，再比较Native Dense、Q1r2、作者四状态ICC、作者式Chipmunk及Global-P64。ICC保留SVD/CA/CD三家族；主规模11个真实视频来源，按作者clean编码→随机时间前向加噪→CA单次/CD真实相邻两次取得统计。主配方与旧native rollout端点区分。

必要的VAE编码和空文本映射在CAL11内预选3源做配对诊断，最多30次模型调用。主作者式校准66次，加诊断最多96次；可选CAL22连同同一诊断累计最多162次。主五臂55视频槽保持；完整视频基本68、条件全开74，与校准/诊断共享3次技术重试。精确预算、局部回放与离线分解只以[EXPERIMENT_BUDGET.json](strict_reproduction_20261116/EXPERIMENT_BUDGET.json)为准。

## 阅读与实施入口

| 文件 | 用途 |
|---|---|
| [完整下一步方案](strict_reproduction_20261116/CONTROLLED_REPRODUCTION_PLAN.md) | 目的、数据、阶段、对照、验收、解释与停止/推进条件 |
| [机器预算](strict_reproduction_20261116/EXPERIMENT_BUDGET.json) · [数据角色](strict_reproduction_20261116/COHORT_PROPOSAL.json) | 唯一计数与来源角色；Dense捕获属于主55，不重复计账 |
| [ICC作者校准](strict_reproduction_20261116/ICC_AUTHOR_CALIBRATION.md) · [机器配方](strict_reproduction_20261116/ICC_CALIBRATION_RECIPE.json) | CA/CD输入、时间、条件、统计与CAL3配对适配 |
| [ICC完整参考](strict_reproduction_20261116/ICC_STRICT_REFERENCE.md) · [Chip作者审计](strict_reproduction_20261116/CHIPMUNK_UPSTREAM_AUDIT.md) | 四状态、原selector/核/状态机、必要宿主接口 |
| [代码引述](strict_reproduction_20261116/CODE_CITATIONS.json) · [引述核验](strict_reproduction_20261116/CITATION_VERIFICATION.json) | 45项主张、91处摘录、42份固定源码，保持原来源身份 |
| [真实clean输入资产核查](strict_reproduction_20261116/ICC_CALIBRATION_ASSET_AUDIT.md) | 现有图文/FFN面板与尚缺真实clip/x0的区别 |
| [独立审阅](strict_reproduction_20261116/INDEPENDENT_REVIEW.md) | 当前整合方案回核及其证据范围 |

## 当前状态

本次完成设计整合和仓库更新，新模型/GPU运行0。真实clean输入、CAL3 raw μ/logvar访问、actual producer/权重/配置/资源和数值执行仍待绑定。缺CA/CD时不能以普通SVD冒充完整三家族选择。ICC/Q1校准来源未配平，局部选择和主视频使用已暴露DEV；本轮结论保留这些限定。

A/BW/BWS已完成结果按原身份继续使用。原参考核读和此前修订保留在[固定历史提交5a5b84a](https://github.com/Qesire/FFNDS/tree/5a5b84a04f71aa1fa6c426febf614a1a4c1d2492/research/20261107/ffn_baseline_reproduction)，不构成额外活动队列。当前工作按已有执行前复核与ACTIVE流程落实，无需重新询问已授权的研究方向。
