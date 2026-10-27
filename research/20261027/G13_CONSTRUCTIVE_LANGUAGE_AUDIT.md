# FFNDS · G13科研表述与文档整理审计（2026-10-27）

**审计目标**：让科学问题、数学结构和竞争性解释直接可见，保留准确的失败数据与证据成熟度，使后续科研Agent主动提出新功能、新表示和新的判别，而不是围绕限制性前言或单一技术格式进行局部调整。

**方法依据**：[领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)与[AutoResearcher机制探索](../../.agents/skills/mechanism-research/SKILL.md)、[research-expansion](../../.agents/skills/research-expansion/SKILL.md)、[problem-reformulation](../../.agents/skills/problem-reformulation/SKILL.md)，及[扩展协议](../../core/EXPANSION_PROTOCOL.md)、[Prompt Playbook](../../core/PROMPT_PLAYBOOK.md)。按“值得理解的结构 → 数学/源码 → 竞争解释 → 有区分度的预测 → 低成本下一证据”判断语言价值。证据级别、明确失败范围、实际资源权限仍独立呈现。

## 发现的真实文档问题与修正

| 原文组织问题 | 已采用的构造性表达与任务作用 |
|---|---|
| **领域Skill最前面堆叠历史更新，YAML元数据位于中部**；同一文件夹带着过时的“实时研究暂停”“先读33篇”等阶段指示 | 现在以有效的YAML frontmatter开头，只读取一次现行G13科学任务和ACTIVE授权；实时文献仍为可用研究方向，模型实验权限与历史工作分开 |
| **“不是/不能/不要”反复定义研究边界**，但没有指出这个观察打开了哪一类函数结构 | 改为先陈述真实发现，例如M16×K64 tile满占 → token稀疏与物理工作单位不匹配 → 研究联合signed功能、重新分组和时空共享的必要性 |
| **将误差指标、量化信息、SAE概念与“重要函数”反复作负面对照** | 主线改为行为轴/无标签函数 → normal-forward计算承担者 → 必要性/充分性/补偿 → 物理成本；各指标成为不同可观察角度，扩大而非收窄函数发现空间 |
| **多次强调“尚无证据”“本轮不运行”等相同声明** | 只在FUNCTIONAL-OBS、CAUSAL、IMPLEMENTABLE的证据身份旁贴合适边界；整份文档末尾集中记录模型实验暂停。数学推导、原论文与真实RUN分别按自己的价值交付 |
| **同一个研究状态在README/AGENTS/导航/首次任务重复且互相可能过时** | 现行任务读G13，原生事实读计算图和原RUN，科研语言与Agent工作流读skill，研究导航按问题索引；ACTIVE为运行权限源 |
| **概念先例被当作新的限制条款** | 每篇原文按其可迁移的数学/机制启示记录，同时明确其实际模型和源码身份；用最强先例产生新的竞争预测，而不是提前排除创新机制 |

## 当前研究表达的标准实例

**积极、严格的负例**：
> “既有token-level后验稀疏在原M16×K64 tile上仍保持100%占用，说明当前表示未改变那些tile的实际执行。下一研究变量是输出函数的联合组织、共享时间状态和合法物理执行单位。”

**开放函数定位**：
> “来源425的同prompt跨seed轨迹改变后三步signed P/N cross作用方向。这个现象提示联合输出函数可能受当前state、门控和后续残差消费者共同调节；最有辨识力的下一证据是同状态下真实gate×W2输出与下游signed响应的路径对应。”

**外部机制的正面可迁移性**：
> “CodeGEMM/LUT-GEMM公开实现的单向量查表机制展示了编码域计算改写；Wan full-M需要单独建立多行核和访存成本。SharQ的双GEMM展示了定点低位疏密配合的物理结构，其稠密残差揭示需要恢复的函数信息。”

**证据范围一句写清楚**：
> “G13目前登记了来自既有native结果的行为现象与SRC计算路径；具体功能–结构的因果归因处于OPEN，模型新实验由用户后续授权。”

## 合并范围及维护契约

本轮对领域skill、AGENTS、G13科学文档、现行综合、E0第一任务、README、研究导航、Graph-Agent协作、文献阅读指南与权威可读图G13文字进行重写/有界修订；**原生源码、真实实验RUN/视频、数值数据、计算图机器节点与关系、85论文索引及ACTIVE授权字段均保持原样**。现行语言的科研重点为**找出有意义的输出函数与真实计算承担结构**，然后才研究quant/proxy/稀疏/缓存等实现。

以后新增论文或科研实验时，以当前结构定位到具体文档：
- 新函数与行为：G13科学报告、FUNCTION_STRUCTURE_RESEARCH_OBJECTS；
- 新数学与外部来源：单篇原文账本及Graph Bindings有类型比较；
- 新物理方法：现行机制综合、Cost/Quality与对应原生RUN；
- Agent提示词：域Skill/AGENTS只维护当前稳定科研原则，新的增量科研结果由来源文档记录。

**验证说明**：本轮只运行独立静态检查（文档引用、Skill元数据、机器图一致性、研究权限字段），未运行新的CPU/GPU模型实验、训练或视频生成。形式化/原生数值与视频质量仍按其既有证据和后续正式评价划分。


## 第二轮：实际会被自动加载的Agent入口和检验契约（2026-10-27）

**这轮核查的中心发现**：最容易使科研重新偏移的，是下层仍具有“当前任务”效力的旧说明。高层Skill/G13已统一后，Copilot、Scoped AGENTS、E0 Context/Stage Anchor、方法讨论目录仍可能把Agent带回过时的first-51论文调查、G6实时暂停、P23/P24 OPEN或旧paired-native运行范围。这些入口已统一到**真实行为函数 → 正常执行承担结构 → 竞争预测与经济实现**。

| 入口与改动 | 修正后由它负责的稳定职责 |
|---|---|
| [.github/copilot-instructions](../../.github/copilot-instructions.md) | 载入G13、根AGENTS、真实Wan FFN源码与AutoResearcher；给出当前研究方法和ACTIVE权限指针 |
| [Scoped discussion AGENTS](../method_discussion/AGENTS.md) | 多Agent只做独立功能/结构/原论文讨论与源码审阅；旧paired-native v2/v3按历史完成范围引用 |
| [CURRENT_CONTEXT_PACKET](../agent_protocol/CURRENT_CONTEXT_PACKET.md) | 统一当前原生证据，确认10个source/12轨迹、1920个嵌套sites及P23/P24的NATIVE-OBS身份，G13 OPEN具体因果承担者；实时文献研究保持开放 |
| [E0_STAGE_ANCHOR](../agent_protocol/E0_STAGE_ANCHOR.md) | 原真实Wan模型、重要科学现象、G13函数/结构三层思路和研究权限；早期E0研究保留历史证据 |
| [agent_protocol/README](../agent_protocol/README.md) | 明确85/171/354/14及研究导航，使用**固定上游8b88a52d**（以前页面误写5bd7ec53） |
| [讨论README](../method_discussion/README.md)、[独立发现](../method_discussion/discoveries/E0/README.md)、[QUEUE](../method_discussion/QUEUE.md) | 行为/函数/结构首先作为开放发现，成熟后进入独立评议方法版本；历史门控商空间等旧提案保持历史身份 |

**真正有机制价值的旧结果保留**：Affine候选原生FFN速度收益与自由视频功能失配；source425跨seed signed P/N交叉项改变；原HiF4 Q2组尺度与实际tile占用；原生整步\(P,N\)有限响应。新入口直接用它们提出联合signed函数、跨层补偿、时空职责及其可能的计算承担路径。

### 对静态检验器作结构级修正

[check_autoresearcher_binding.py](../../tools/check_autoresearcher_binding.py)之前把旧文件名/精确自然语言短语作为“当前Agent已正确引导”的代理检查，导致要消除过时表述时可能引起测试失败。现将**现行指令检查**改为核对ACTIVE中登记的G13文档/Function–Structure对象、真实存在的文件、Skill YAML、默认WARD科研能力与上下文索引，保持现行E0及实验暂停身份。

仍保留可客观验证的核心不变量：**固定AutoResearcher upstream文件Git blob、真实ACTIVE科学与实验权限、原协议哈希及冻结原生RUN、Graph-Agent SRC/OBS/OPEN与真实图谱、方法讨论独立文件路径、单独记录的视频质量身份**。语言质量通过独立Agent实际轨迹评价。

[DEFAULT_BEHAVIOR_CASES.json](../agent_protocol/DEFAULT_BEHAVIOR_CASES.json)由36个扩充至**40个**项目专用案例，新增4类：自然行为函数优先、signed整步响应与局部W1函数的区别、联合计算结构先于量化算法、积极表述与旧Agent任务入口校准；静态检查会检验这4个案例已登记，但其科研表现等待真正独立Agent行为评估。

**本次静态核对**：共检查14个实际生效的指令、工具及研究目录文件中的251条Markdown本地相对链接，未发现失效链接；现行入口的旧G6暂停、旧文献总数、P23/P24旧等级及旧paired-native授权提示均已消解。ACTIVE文件与原main Git blob一致；机器图维持171节点/354关系、论文索引85项；上游pin 8b88a52d保持原样。本轮更改属于文档/静态检查/项目专用行为案例层级，CPU/GPU模型、视频生成、重型CI及Agent真实行为对照均未执行。
