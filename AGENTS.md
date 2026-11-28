# FFNDS 仓库工作说明

## 先确定要延续的结果

从[当前交接](research/agent_protocol/CURRENT_CONTEXT_PACKET.md)读取默认方法、最新证据和待验证事项。程序读取 [research/CURRENT_HANDOFF.json](research/CURRENT_HANDOFF.json)。[研究导航](research/RESEARCH_NAVIGATION.md)按问题链接到实际代码、结果表和复现实验所需的外部资产。研究分支与默认分支可能处在不同提交，引用实验请记录证据提交和路径。

四步成片默认使用 [`wide8192`](tools/ffnds_fourstep/README.md)、固定 64 个真实探针、四步高高低低和逐位置锚点。记录任何改动发生在哪个专家、层、步、位置和张量，说明哪些信息能在生成时取得。研究原始论文和实现时可用[论文与方法技能](.agents/skills/ffnds-paper-discovery/SKILL.md)，复算已有结果或安排配对时可用[实验与成片证据技能](.agents/skills/ffnds-experiment-evidence/SKILL.md)。登记方法版本时使用[讨论约定](research/method_discussion/AGENTS.md)。

推导跨步增量、邻边误差或缓存折叠条件时，可调用 [`$operator-tensor-analysis`（算子与张量分析）](.agents/skills/operator-tensor-analysis/SKILL.md)。从失败案例重新选择观测量、状态表示或求解目标时，可调用 [`$active-abstraction`（主动抽象与问题重构）](.agents/skills/active-abstraction/SKILL.md)，写明新构造保留哪些动作与纹理信息，以及哪组配对能检验这个判断。

## 哪些证据支持哪些结论

- 复算区分校准来源、反复使用的开发来源、评测来源、同一来源的提示词或种子变体。镜头指令属于生成条件，需记录完整提示和种子。同场景多次生成仍是一个来源。
- 写明比较对象。等价优化要对照原锚点实现，检查四步预测、潜变量及解码输出；对 Dense 的墙钟和画面比较另记。记录设备、预热、运行次数、计时边界、探针成本、锚点槽与峰值显存。组件计时只支持所测组件。
- 拟合对象同时涉及位置上的跨步 FFN 变化及邻边之差。在相同可观测支撑上核查方向投影、正交余能与共同／差分误差，再核对动作、轮廓和纹理。局部 MSE、综合分和轨迹与 Dense 的距离各有适用范围。保留异常样例与用户的原始判读，终局视觉判断交给用户。
- 每次报告成片对照，同时交付原片链接和逐来源配对指标。同源同种子比较默认报告最终 latent 相对 L2 与解码视频 RGB RMSE，涉及动作时加入帧间差分或光流等时间指标。写明参照臂、指标方向、配对差值和已有缺项。局部指标用于定位机制，成片结论结合端到端指标与视频。
- 研究结果按实际证据登记为推导、组件、端到端或人审。提案和讨论回执保留作者、日期和版本，不能替代真实运行记录。更新默认方法时同步更新交接 JSON、接手说明与证据入口，留下原结论的固定版本。[实验证据索引](research/EXPERIMENT_EVIDENCE_INDEX.md)

## 接续任务与历史约束

根据用户当前任务及实际环境确定可执行范围。沿用学校机器、模型、数据或历史执行器前先核对相应资产、有效的资源约束和访问权限。归档中的作业编号、审批状态、排队记录、设备槽和历史“下一步”均按原任务解读。检查已有用户修改，按本次任务范围提交文件。

旧版 `ACTIVE_RESEARCH.json`、`CURRENT_DECISION.json`、`OBJECTIVE_CONTRACT.json` 继续服务各自的历史读取者和校验程序。查询其中字段先确定所属 scope。归档保留原始文件字节、原提交及原目录中的证据链。[归档与兼容入口](ARCHIVE.md)

在仓库根目录运行以下两条命令，可离线检查当前入口、固定证据链接、锚点步骤和归档摘要。使用 Python 3.11 即可。

```sh
python tools/check_autoresearcher_binding.py
python tests/test_live_research_entrypoints.py
```
