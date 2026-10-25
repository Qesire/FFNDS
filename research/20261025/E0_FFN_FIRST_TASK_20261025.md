# E0 当前唯一第一任务：FFN 本征结构发现，不从量化尺度选择算子

**依循纲领**：[E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md](E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)。**研究阶段**：`E0_OPEN_DISCOVERY`。此文是新科研任务，不是 GPU / 模型执行协议。

## 要首先回答的科学问题

> 在真实 Wan 四步学生模型中，是否存在能够区别「FFN 激活小」「FFN 净作用小」「下游影响小」「物理可省算」四件事的结构？最先让我们改变判断的是哪一种已可获得的证据？

不能用已做完的 Block0 径向配对/四臂 star-null 复算重复充当新发现。它们作为挑战下游功能结论的专项证据，在需要时调用即可。

## 第一轮真实工作（三条相互独立的科学问题）

### I. FFN 内部：贡献分布和有符号交互

- 从固定源重建 `z→W1→GELU→W2→gate` 的真实张量边界及 available saved fields，指出现有 assets 哪些是完整 h、Q2-only、W2 sampled panels、gate、bias。区分已存 32 行 × 五组和完整全网络 hidden；保存资产不够时以精确字段需求完成这一步。
- 理论上先区分 `|h_j|`、`||h_j W2[:,j]||`、净增量、signed cross-term；独立提出 **集中** 和 **分散/高相关** 两种机制的相反预测。
- 重新考虑最有效的定义单位：每通道、成组、按 token 的 tile、输入敏感子空间。先观察支持集与方向，后选择是否有任何方法应以某单位实现。
- **交付**：源码/字段覆盖地图、至少一条有符号交互恒等式或反例、最廉价的现有数据检验。

### II. FFN 的空间与跨步变化

- 检查同一 token 坐标、不同 step 与 expert 的 FFN pre-activation/hidden/output 的绝对值 vs innovation/变化；严格分离共同模式、局部变化及真实时空对象追踪。
- 明确 Wan 四步双专家调度和保存资产限制；和 Chipmunk 的 cached-MLP delta、少步 DiffSparse 的“计算时点”对比。
- 竞争假说：可共享的跨步状态 vs 表面 cosine 高但仍存在不可忽略幅度/局部变化；后者能解释为何光照/细小背景更容易出问题？
- **交付**：来源/step/expert 分层的已有证据可回答程度，以及一条将改变计算重用判断的反例或判别式。

### III. 下游功能价值与计算自由度

- 用实际 gate/residual/nextnorm/attention/scheduler 图刻画 FFN 的有限动作如何改变视频轨迹；将历史观察到的主体外光照/细节波动作为机制发现线索。
- 比较输出能量、时间稳定性、空间低频/高频、条件方向对后续真实作用排序的预测；对小局部误差但大终端效应给出条件性解释。
- 探索新的静态/动态、精确/近似、改变执行顺序/减少重复工作的计算机制；对每个种子列出最早可以作选择的计算节点和可消去的 GEMM 工作，**不要求立即定案**。
- **交付**：功能机制竞争图、能改变下一行动的一项精确反例/条件推导。

**交叉审查者**：独立依据文献与源代码挑战 I/II/III，不能把最接近的已发表原语直接看作完整重合；优先找到真正同构的行动、时序、代价与训练假设。

## 当前资产使用清单

| 资产 | 可以先做的判断 | 不可跳跃的推论 |
|---|---|---|
| `tools/analyze_e0_radial_pairing.py`、E0 保存组面板 | 32 sampled rows 上的 decoder W2 group + gate 方向与几何 | 整体 hidden 稀疏率、真实 GEMM tile 可省算 |
| `E0_TEMPORAL_CALIBRATION.json`、`E0_TEMPORAL_STRUCTURE.json` | 特定跨 step FFN 输出/共同分量/专家分层 | 训练自由缓存策略或其真实速度 |
| `E0_FOLLOWTHROUGH.json` 对应的完成报告与 `E0_FINITE_GROUP_DIAGNOSTIC.json` | 固定两组有限干预的 BF16 本地和四步 downstream 作用 | 通道一般性、硬件速度、视频质量门槛 |
| 旧 R13/R16/R17/A5+HWG 证据 | 对以前具体动作和主体外视频失稳提出机制问题 | “稀疏必然失败”或“必须 dense completion” |

本任务只引用已在仓库核实的研究报告与现有产物；新字段的可用性由 Agent 在来源审计中登记。

## 方法行动与报告格式

每位 Agent 沿 [DEEP_WEB_RESEARCH.md](../agent_protocol/DEEP_WEB_RESEARCH.md) 开展深入原论文与原代码检索，并依据固定 AutoResearcher v2.4 的 `mechanism-research`、`research-expansion`、`problem-reformulation` 和 `web-research-discovery`，**主动大范围检索原论文和原始源码**，挑战先入为主的静态/量化尺度/局部几何解释。记录：`新观点 → 源码与精确数学 → 两个相异机制及其不同预测 → 最强反例/先例 → 最廉价下一检查 → 潜在省算自由度`。

各 Agent 只在其自己的发现/挑战文件记录，汇总者串行维护讨论队列。可读现有获授权的保存数据，做数学/离线复核，但新模型 / GPU / 生成 / 训练 / 扩展原生干预需单独用户授权；旧实验授权全部已完成，零 pending。新的最小实验可以**提出合同**，但不会因为本任务自动运行。

**一次成功的 E0 迭代** = 至少一个比现有描述更有解释力且可证伪的 FFN 内部或功能结构，连同一个来源可靠的判别路径。没有要求预先选定 Proxy、分组、稀疏率、缓存、补算或量化格式。
