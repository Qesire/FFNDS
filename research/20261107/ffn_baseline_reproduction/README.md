# G19 / Wan2.2 四步 FFN-only 前沿基线复现

**2026-11-07 · DESIGN_ONLY / REVIEW_REQUIRED / NO_NEW_RUN**

本目录面向已有 G19 的公平论文基线，不直接追逐原论文跨硬件、跨模型或 50 步推理的最佳加速倍数。固定研究对象为 Wan2.2-I2V-A14B-LightX2V-Distill-4step、HHLL、无CFG、BF16、A100 80GB、832×464×81帧。Attention 和其他非FFN路径一律保持原生，方法差异限于 FFN。

## 顺序与责任

1. [G19 最新基线、旧研究资产与证据对齐](G19_BASELINE_AND_ASSETS.md)：说明**Chipmunk 的 Wan FFN-only 已有全160实际结果**，避免重复占用GPU。
2. [此前指南的仓库版](REPRODUCTION_GUIDE.md)：ICC、Chipmunk、TARDIS 的论文/源码/公式/缓存/依赖/原版命令。
3. [ICC 完整移植规格](ICC_FFN_PORT_SPEC.md)：当前第一项真正缺失的匹配四步基线。
4. [Chipmunk 资产复用与同源对照](CHIPMUNK_REUSE_AND_COMPARISON.md)：先复核历史，补新的匹配数据时单独立项。
5. [TARDIS 与候选方法分层](TARDIS_AND_ALTERNATIVES.md)：静态折叠与完整动态回退、已有 TEAL/SharQ/RT-Lynx 证据。
6. [统一科学协议与质量—成本验收](EXPERIMENT_CONTRACT.md)：前置独立评议→冻结scope→S0–S5→退出。
7. [机器可读计划](PLAN.json)：方法和历史状态、参数扫描、指标、文件身份、下一步审查，不含运行授权。

## 当前关键裁决

| 方法 | 在FFNDS的状态 | 推荐动作 |
|---|---|---|
| ICC FFN | 官方DiT源码已查，尚无同G19四步配置实测 | **第一项新增移植**，优先建立 rank/缓存/速度 Pareto |
| Chipmunk MLP | 2026-10-30已有 packed P64 真M、全160视频、八轴及RAFT、ABBA等正式证据 | **先资产复用**，再看是否需要新的7源匹配 |
| TARDIS | 原论文/数学先例已查，作者完整源码未确认 | **先忠实算法实现审核**，分开静态fold与oracle/真实predictor |
| TEAL、SharQ、RT-Lynx | 本仓已有多项原生/有限负面实验 | 作为历史强对照，不再重跑已闭合scope |

## 证据与合规

本文档提交不修改 `ACTIVE_RESEARCH.json`，不创建GPU/CPU模型任务，不声称独立评审已经通过。任何新实验须遵守 [前置独立复审协议](../../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)，并在真实可用资源/当前会话额度下建立新scope、身份SHA、RUN/EXIT/独立核验。作者代码仅以Git链接定位，不复制许可证未知源码或原始权重、视频和敏感路径。

[当前G19方法与质量](../G19_METHOD_DERIVATION_AND_RESULTS_20261107.md) · [2026-11-07证据目录](../../../reports/20261107/g19/README.md) · [项目科研规程](../../../AGENTS.md)
