# 研究归档

当前接手资料位于[项目首页](README.md)、[接手说明](research/agent_protocol/CURRENT_CONTEXT_PACKET.md)和[机器可读入口](research/CURRENT_HANDOFF.json)。下表按要追溯的对象给出历史资料的保存位置。

| 要追溯的对象 | 保存方式与入口 |
|---|---|
| 2026-11-21 交接更新前的首页、agent 指引、协议、研究导航和证据索引 | [交接入口原件](archive/20261121/handoff/README.md)逐一映射 13 个文件。字节数、Git blob 和 SHA-256 见[清单](archive/20261121/handoff/MANIFEST.json)。[原提交](https://github.com/Qesire/FFNDS/tree/22d60b645e070ed528b3b435ec0cbe4b26467108)保留了当时的目录和相对链接。 |
| 2026-10-24 首次移出的历史交接材料 | [原归档目录](archive/20261024/historical_handoffs/README.md)及其[清单](archive/20261024/historical_handoffs/MANIFEST.json)。 |
| 分期研究、运行与技术背景 | [按问题查历史研究](research/RESEARCH_NAVIGATION.md)、[技术文档](docs/README.md)、[日期报告目录](reports/)。原表、脚本、哈希与 dated scope 在原路径归档，便于旧读者和校验脚本复算。 |
| 旧阶段机器状态与研究队列 | [ACTIVE_RESEARCH.json](research/ACTIVE_RESEARCH.json)、[CURRENT_DECISION.json](research/CURRENT_DECISION.json)、[OBJECTIVE_CONTRACT.json](research/OBJECTIVE_CONTRACT.json)及[方法登记队列](research/method_discussion/QUEUE.md)保留各自阶段的字段、资源条件和决定。整仓现行方法由[接手记录](research/CURRENT_HANDOFF.json)定位。 |
| 更早的主页路径、设备与发布说明 | 从[上一版归档页](https://github.com/Qesire/FFNDS/blob/22d60b645e070ed528b3b435ec0cbe4b26467108/ARCHIVE.md)可继续沿原路径查找。 |

旧首页曾将以下文件列为接续入口。它们在各自日期下保留完整上下文。

- 10 月的[任务台账](research/agent_protocol/CURRENT_TASK_LEDGER_20261028.json)、[执行手册](research/agent_protocol/EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md)、[B 路线交接](research/agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)和[设备调度记录](research/agent_protocol/DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)。
- 11 月的 [G19 接续记录](research/20261107/G19_NEXT_CHAT_20261107.md)和 [G20 双路线交接](research/20261114/G20_TWO_CORE_EXECUTION_HANDOFF_20261114.md)。这两处保留当时的核查条件和未完成事项。

日期目录、源数据和已绑定哈希的回执保留原位。某份文档被归档，表示它已退出当前交接入口；其中的测量、反例和复现实验条件仍可在原有范围内引用。

**历史校验的已知状态。** 归档源提交的图谱测试有两处将文献数写定为 98，而原始索引已有 110 篇。本次交接改为按原始索引核对主文献及其算子映射。旧任务台账、候选门槛和旧入口三组测试分别还有 1、1、18 个跟归档源提交相同的失败断言。图谱工具输出的 `method_next_agent=M-B` 属于当时的研究阶段。现行交接有独立的[校验命令](AGENTS.md)；复查旧协议时请同时核对当时的源码和数据版本。

部分历史测试直接核对执行时的文件字节。Windows 检出若自动将换行改为 CRLF，可能因执行摘要不符而报告失败。需要重现 CI 时，在新的检出使用 `git -c core.autocrlf=false clone`，或直接取 Git blob 的原始字节。
