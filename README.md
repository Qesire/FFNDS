# FFNDS · 视频生成 FFN 的信息、函数与实际计算研究

研究目标：理解 Wan 视频生成 FFN **如何组织信息、函数作用和时空行为**，并构造可在完整视频质量与真实系统成本之间取得竞争力的低成本表示和执行方式。科学入口为 **I（信息与表示）/ F（函数与动力学）/ C（计算与执行）**；O-A～O-E 是可组合的方法动作，“稀疏”不是唯一方法范畴。详见[统一科学目标](research/RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)。

## 从这里开始

**首要入口：[P0–P4分级阅读导航](research/RESEARCH_NAVIGATION.md)。** 根据工作类型选最短链路：

| 角色 / 目标 | 第一份该读的材料 | 下一份 |
| --- | --- | --- |
| **接手当前任务的 Agent** | [任务状态与依赖](research/agent_protocol/CURRENT_TASK_LEDGER_20261028.json)＋[当前最小上下文](research/agent_protocol/CURRENT_CONTEXT_PACKET.md) | [现行M-A/M-B/M-C方案](research/20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md) → [M-B B0/B1实际交接](research/agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md) |
| **机制/数学研究者** | [五操作×I/F/C方法分类](research/20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md) | [G14反向先验审计](research/20261028/G14_PRIOR_CONSTRAINT_AUDIT_FOR_NEW_FFN_METHODS.md) → [真实计算图](docs/98_端到端计算图与证据注解_20261025.md) |
| **原文与源码研究者** | [G14论文→真实计算焦点](research/agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json) | [源论文索引](research/literature/20261026/PRIMARY_LITERATURE_INDEX.json) → [方法先验关系](research/agent_protocol/METHOD_OPERATION_CROSSWALK.json) |
| **实验/数据/质量研究者** | [实验专用Skill](.agents/skills/ffnds-experiment-evidence/SKILL.md) | [数据协议](research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md) → [执行手册](research/agent_protocol/EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md) |
| **多Agent讨论和审阅** | [仓库AGENTS](AGENTS.md)与[领域Skill](.agents/skills/ffnds-paper-discovery/SKILL.md) | [真实独立方法讨论队列](research/method_discussion/QUEUE.md) |
| **历史结果复核** | [证据索引](research/EXPERIMENT_EVIDENCE_INDEX.md)与[实验发现谱系](research/EXPERIMENT_FINDINGS_ATLAS.md) | [归档索引](ARCHIVE.md)及对应冻结RUN/原始报告 |

## 当前研究状态 · 截至 2026-10-28

| 研究对象 | 已完成范围 | 当前下一步与证据界限 |
| --- | --- | --- |
| **M-A0**（普通6016维紧凑FFN） | [4位置真实原生Q2局部恢复pilot](research/20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md)，已暴露train16/heldout8；相同更新预算比较Energy94/Random94 | 为M-B/M-C提供强基线；全80FFN和完整自由生成视频**OPEN** |
| **M-B**（gate/suffix敏感函数） | [配对敏感度和block39 LN/head的连续65统计](research/20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)，**ALG＋SRC** | 当前用户指定优先：先[B0原生u/head/flow采集准备及neutral重放](research/agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)，再B1有限响应；B-H/B-F可执行学生**OPEN** |
| **M-C**（完整非线性函数编码求值） | [输入充分性、输出秩和大M查表成本](research/20261027/COMPACT_FFN_ABC_INDEPENDENT_METHOD_REVIEW_20261027.md)，**ALG＋LIT** | 完成数学/保存资产判别，再进入合法原生执行；尚无Wan实际速度/视频 |
| **旧固定支持与执行竞争器** | [consumer20真实视频质量](research/20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)，[A100合成2:4 C5](research/20261027/E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md) | 分别支持视频功能失配与受限kernel物理证据，不能替代新学生质量/整端速度 |

**原生数值研究对象**：LightX2V Wan2.2 I2V A14B，high/high/low/low四个去噪step、两套expert、每套40个Transformer blocks；典型FFN
\[
\mathrm{norm2/AdaLN}\to Q1\to W_1\to\mathrm{GELU}_{\tanh}
\to\mathrm{HiF4\ Q2}(64\to8\to4)\to W_2
\to e_5\text{ gated residual}\to\text{downstream flow}.
\]
原实现使用**HiF4 fake-QDQ+BF16 GEMM**；packed FP4/NVFP4/MXFP4需另立数值/系统证据。旧10 source/12 source×seed/1920 sites、consumer20的20个开发来源与A0的训练/诊断来源，是**不同研究数据范围**；seed、row、block/step和多prompt不当作独立image-cluster。

**论文/图状态**：94项主文献＋10项补充，16个G14算子焦点；主图209个稳定节点/431条关系。新增12条typed方法先验约束在[独立机器交叉表](research/agent_protocol/METHOD_OPERATION_CROSSWALK.json)登记，不算作新原生因果结论。G14图来自文献/源码映射，非实际视频部署结果。

## 科研与运行约束

- **方法研究**：按[AutoResearcher领域Skill](.agents/skills/ffnds-paper-discovery/SKILL.md)、[机制Skill](.agents/skills/mechanism-research/SKILL.md)由计算流→等价数学/信息条件→最强竞争解释→可区分证据形成方案。当前方法权威正文仅以[整合方案v2](research/20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)为准。
- **数据与评价**：按[Calibration V2](research/20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md)区分source image-cluster与token/行、teacher与student轨迹、开发与独立确认、VBench自定义函数与官方采样协议；每个真实视频候选记录连续运动、主体/背景、相机行为、原生latency/VRAM。
- **授权与协作**：[ACTIVE_RESEARCH.json](research/ACTIVE_RESEARCH.json)是执行/资源唯一权威，历史已完成范围不自动成为下轮许可；[QUEUE](research/method_discussion/QUEUE.md)仅登记独立Agent实际提交/评议的方法，不是作业队列。实时/流式GPU实验仍暂停。
- **证据语言**：公式或源图ALG/SRC，作者文献LIT，真实原生运行NATIVE，完整视频VIDEO，物理成本SYS，尚未产生数据OPEN；静态CI通过只表明合同/引用一致。

## 只读复核（不启动GPU或模型）

```bash
python tools/graph_agent_bridge.py validate
python tools/graph_agent_bridge.py paper L34
python tools/graph_agent_bridge.py focus T06
python tests/test_graph_agent_bridge.py
python tools/check_autoresearcher_binding.py
```

原始实验收据、hash、历史错误与候选失配的证据保留在[ARCHIVE](ARCHIVE.md)及[原始报告](reports/)；新研究资料优先更新对应现行方法正文、证据谱系和P0–P4索引，不创建第二份并列“当前状态总览”。

M-B已有[4个真实Dense捕获与同状态head有限结果](research/20261028/B_TERMINAL_NATIVE_CAPTURE_AND_FINITE_RESPONSE_RESULTS_20261028.md)；当前源配对判别与学生任务按唯一任务台账继续。F线专题见[G13](research/20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)，运行权限定于[ACTIVE](research/ACTIVE_RESEARCH.json)。
