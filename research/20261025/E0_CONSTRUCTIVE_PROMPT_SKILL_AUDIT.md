# E0 · AutoResearcher 驱动的提示词表达审查

**日期：2026-10-25** · **研究范围：** 当前 E0 问题文档、Agent 入口、独立科研讨论模板、深度 Web Search 引导、静态一致性核验。  
**方法依据：** [research-expansion](../../.agents/skills/research-expansion/SKILL.md)、[problem-reformulation](../../.agents/skills/problem-reformulation/SKILL.md)、[mechanism-research](../../.agents/skills/mechanism-research/SKILL.md)，与仓库固定的 [AutoResearcher 来源](../agent_protocol/ACTIVE_SOURCE_MANIFEST.json)。

## 审查得到的结构性问题

之前的引导常以“不要设计算子、不能泛化、不得启动、不是已经证明”开头，导致真正有信息量的研究动作被放在大量限定之后。E0 最新进展的最大价值是 **Block 0 gated FFN 输出与 residual 径向对齐**以及量化/时空/异常位置等并行发现，而提示词容易把它表述成“尚不能干什么”。

同时，原 [静态验证](../../tools/check_autoresearcher_binding.py) 甚至要求局部 Agent 说明中出现英文短语 `do not` 才视为完整。这是语句表面形式约束，与 AutoResearcher 的“从科学结构出发、创造性扩展、挑选区别性检验”目标相反。

## 修订原则与具体体现

| 以前的表达形态 | 当前正向研究表述 | 对应技能 |
| --- | --- | --- |
| “目前不能据此删组，不能推出视频质量”反复出现 | “从 19.82% 理想 LN 几何投影出发，研究真实 residual/bypass 与未来 flow 中哪一种解释成立” | mechanism-research：源流与关键数学 |
| “不要开始设计算子，不要固定 K/N panel” | “E0 交付真实现象、不同解释和决定性观察；算子创意可以保存为 idea seed” | research-expansion：扩大而后判别 |
| “不能只看已有论文摘要、不能假装查重” | “主动多领域深搜，沿引用扩展、读原始方程及实现，输出最接近竞争机制与新科学认识” | research-expansion + PRD：跨领域和机制重表述 |
| “10 项问题都必须弥补”式审批清单 | “每位 Agent 选择最能改变当前解释的一个未知，用一个强反例或证据推进” | research-expansion：开放探索 |
| “证据不足、权限受限”等多段重复警告 | “以证据类型解释结论，以一段明确的实验授权记录界定执行范围” | mechanism-research：实证范围 |
| 静态验证检索 `do not` 文本 | 静态验证检查实际的 AutoResearcher source manifest、E0 阶段、独立发现目录及 review 文件路径 | problem-reformulation：将流程从语言形式转换为可验证结构 |

## Agent 输出应当怎样展开

**新发现 / 未解张力 → 最强机制解释 → 最有竞争力的第二种解释 → 关键源码/数学/论文 → 有判别力的下一观察。**

提出大胆构想、跨领域类比和理论恒等式是探索价值的一部分。每条结论旁边自然注明它来自实际 Wan 保存操作数、条件推导、论文与官方实现，还是尚待取得的观测。任务阶段与原生资源权限在入口文件中各维护一处清楚的说明。

本轮保留了原始 6 来源、3 主体/3 背景/0 相机、12 次完整生成、Block0 pre/post/leave-out 的数字，保留固定 AutoResearcher 来源与已有历史提案；改变的是**科研表达及默认 Agent 行为组织方式**。

## 检查标准

静态层面：所有 E0 入口链接到同一活跃阶段、当前问题、深度文献协议及独立讨论空间；本地 Skill YAML 有效；权限与完成观测的记录匹配；提示词校验关注结构与真实文件路径。

科研行为层面：需要用不同 Agent 的实际输出评估是否发现更多非同构机制、更准确数学、更多有用反例和更好的下一观察。仓库静态文件一致性只是开始，真实行为质量留给独立轨迹审查。
