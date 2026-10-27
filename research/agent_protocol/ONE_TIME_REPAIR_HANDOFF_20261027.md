# FFNDS · 下一位Agent的一次性修正验收（2026-10-27）

**仅下一位接手Agent执行一次。** 任务是修正前几轮文档/契约调整的真实遗留问题，完成后马上回到科学内容。本文不是永久方法规定，也不修改E0的长期研究目标、ACTIVE运行权限或CI自动触发规则。

## 目标与当前边界

研究目标已更新为I（信息与表示）/F（函数与动力学）/C（计算与执行）并行。最近三项提交为：834e1eaf88f03082be835cc7efef96b8bb7bbc5c（目标测试及Agent行为案例）、8f6f22640b12cce4b3ccf6efe8a8687dab960d06（机制证据组合）、57dfb1e447c5e0bb7cbfc431d0b3c3be4c42227d（研究导航和静态检验）。上一位Agent对JSON和相对链接做过静态检查，**新增Python unittest以及完整源绑定检验尚未实际在本地完整仓库执行**。

开始时读取最新main、[统一目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)、[研究导航](../RESEARCH_NAVIGATION.md)、[四组机制证据](../20261027/PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)、[领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)和[ACTIVE](../ACTIVE_RESEARCH.json)。以实际当前HEAD为准，不复用历史完成状态当作现在的许可。

## 一次性修正任务

### A. 实际执行纯静态回归并记录退出码

在完整仓库检出中运行：

~~~bash
python tools/check_autoresearcher_binding.py
python tools/graph_agent_bridge.py validate
python -m unittest discover -s tests -p 'test_objective_contract.py' -v
python -m unittest discover -s tests -p 'test_parallel_research_scope.py' -v
python -m unittest discover -s tests -p 'test_graph_agent_bridge.py' -v
~~~

这些是文档、静态源图和契约验证。禁止把源码检查或JSON解析的成功写成Python unittest已经通过。按需扩展**确认不会载入模型或触发外部任务的纯静态测试**。

### B. 修正真实问题，不用形式通过掩盖缺陷

- 对照[OBJECTIVE_CONTRACT](../OBJECTIVE_CONTRACT.json)、[目标测试](../../tests/test_objective_contract.py)、[I/F/C回归](../../tests/test_parallel_research_scope.py)、[行为案例](DEFAULT_BEHAVIOR_CASES.json)与[方法模板](../method_discussion/TEMPLATES.md)，检查新旧目标是否一致：科学发现可独立成立；G13仅为F线专题；真实方法需强匹配成本和完整视频；旧0.5/1.5/0.02只是历史作用域验收阈值。
- 对照[AutoResearcher绑定检查](../../tools/check_autoresearcher_binding.py)与[固定上游manifest](ACTIVE_SOURCE_MANIFEST.json)，检查YAML Skill头、原文检索引导、哈希和当前研究状态。检查[图谱校验](../../tools/graph_agent_bridge.py)的源码/证据身份及内部链接。
- 修复现行README、AGENTS、Copilot、Skill、研究导航及模板的错链、旧任务误路由、矛盾权限或过时的必须先F再I/C表达。保留真实历史原始RUN、SHA、质量身份和失败边界，避免制造另一个重复的审查文档。
- [ACTIVE](../ACTIVE_RESEARCH.json)的新CPU/GPU/模型/视频执行仍暂停。仅进行源、数学、文献、文档及上述无模型的Python静态验证；不运行GPU测试、视频生成、训练或后台工作。

### C. 验收后立刻恢复科学研究

验收记录至少包含：实际运行的命令、PASS/FAIL/NOT_RUN和退出码；具体修复、修改路径、最终提交SHA；历史数值阈值、执行暂停及AutoResearcher来源未被改变的核对。若环境无法运行Python，写明NOT_RUN和阻塞原因，不声称修正已全部完成。

**验收通过后，在同一轮收尾提交中彻底删除：**

1. 本文件 research/agent_protocol/ONE_TIME_REPAIR_HANDOFF_20261027.md；
2. 根AGENTS.md顶部指向本文件的一次性提示；
3. .github/copilot-instructions.md顶部的一次性提示；
4. .agents/skills/ffnds-paper-discovery/SKILL.md中YAML frontmatter后的一次性提示。

移除后复核这些入口和git diff，并在本次交接内**重新进入I/F/C正常科学研究**。如果静态修正被具体故障阻塞，留下失败证据与下一步，而不是虚报完成；后续修复完成后仍按上面步骤清除一次性指针。

**一次性含义**：仅本次未验收的修正任务先于下一个科学内容任务；任务结束后，未来Agent不会长期先运行这套修正清单，不新增任何永久的修复优先门槛。
