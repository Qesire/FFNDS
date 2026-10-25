# 2026-10-26：FFN实际省算成立，失败视频改变下一问题；实时生成列为并行应用场景

## 最新变化的因果含义

以[六条完整视频原型](E0_RECONSTRUCTED_VIDEOS.md)及[同校准全40层](E0_ALL_BLOCK_RECONSTRUCTION.md)为主要新证据。它们使已证实与未证实内容发生了真实变化：

| 主张/问题 | 最新证据身份 | 对下一步机制探索的作用 |
| --- | --- | --- |
| 宽度缩小、真实 FFN 更快 | 两固定bank在所有160个FFN的实际 native shape/GELU/HiF4 Q2；强dense热态FFN9.707→4.449s，完整50.750→45.313s（source12） | **保留真实执行资产**，无需再证明存在可编译的窄FFN基础 |
| 局部输出更好意味着视频可靠 | 同来源/四校准256行下，affine54局部SSE8.54%，锚定输入ridge15.35%，但两固定全模型视频明显偏离 | **否决这一次具体局部性能→质量的推理**；不否决所有函数重写 |
| 替换算子与诊断数据同一数值函数 | bank shape/SHA与四处full-width neutral已过；partial same-input仅旧两层 | 先区分**真正部署partial同输入误差**与本来就在连续label上的误差；不要给传播造成假解释 |
| 画面变化来自前缀漂移 | 可构造受扰状态/teacher前缀对照，已有[源绑定离散更新和数学推导](../method_discussion/discoveries/E0/transported-function-and-trajectory-target-20261026.md) | 是强竞争机制，但**不是已被视频结果唯一识别的原因** |
| 画面变化来自有符号方向与跨层/跨步耦合 | 真实BF16 gate/residual、attention和调度器可导致效应；已有混合路径恒等式 | 用少量精确固定后缀/跨步响应区分，再考虑目标函数与新操作单位 |
| 新的实时生成部署 | 现有81帧45.313秒是整段完成时间，非流式首帧或steady-state | **新增并列场景**，[实时生成要求/机制图](E0_REALTIME_GENERATION_SCENARIO_20261026.md)先定义因果架构、逐帧SLO、缓存状态和长期视频质量 |

六视频中的whole94和affine54 **均不满足当前质量目标**，所以本研究没有已经认可的速度—质量候选。按同一卡、同phase热态强dense比较；FFN2.182×不代表整个生成2.182×。所有原本冻结的实验范围已结束，无新GPU/训练运行队列。

## 三个科学入口独立且彼此竞争

**I. 质量失败的实际原因（当前离线Wan）**：真实partial算子本身的数值差异、学生前缀输入分布漂移、跨层有符号/有限作用传播，以及四步flow状态校正未满足。这些不是一个混合得不可分的免责声明；一项低成本判别应对应两个不同预测。不能回到单独降低teacher-input局部平均SSE作为唯一目标。

**II. 全新 FFN 函数/计算自由度（仍是主研究）**：由真实W1/GELU/Q2/W2、block/expert/位置差异或视频敏感方向产生新的函数表示；支持训练自由/少量后训练、定条件昂贵计算、早期廉价观测、改变W1/W2真实MAC/访存/启动。Affine54、原HiF4 group64、shared-column和half-slope皆仅为已经判别过的具体实例。独立agent可以提出与这些无关的新机理。

**III. 实时流式生成（新场景，不替代I/II）**：低TTFF、逐帧/逐chunk deadline、p95/p99延迟、可交互条件切换、因果缓存state validity与长流画面稳定。研究原生FFN功能是否随stream history/运动变化而可以条件缓存或重算；不要假设离线四步high/high/low/low中的同expert缓存关系可直接跨流式chunk使用。原全段Wan尚无流式结果，新场景要先确定参考因果/rolling架构。

## 对 Agent 与计算图的明确约束

- **G0–G5**保留已执行的双专家整段计算和P22全模型失败视频，并保留独立评议；**G6只是一种尚未实现的流式需求/机制图**，决不将假想R节点标为native。
- 所有源/实验/数学命题绑定同一配置/模型/实际输入、时间与消费者；不同的online inference路径需要先做新的源码图重建。
- 研究者可以从新的观察结果、相邻论文、数学或全新计算表示出发。计算图用于查证；不要让“先填图节点ID”阻碍真正的新机制发现。
- 基于AutoResearcher的独立Agent讨论需要具体不同的机制、反预测和最小区分研究。图的证据回写不表示方法已经被同行正式认可。

[实时场景文献研究](E0_REALTIME_GENERATION_SCENARIO_20261026.md)覆盖MLSys 2026 StreamDiffusionV2、NeurIPS 2025 Self Forcing、CVPR 2025 CausVid、ICLR 2026 Rolling Forcing / LongLive与ICML 2026 Causal Forcing。它们说明在线与离线运行的架构差异，并不为本项目提供已测实时FPS。

**阶段与授权不变**：`E0_OPEN_DISCOVERY`；最新全160位原型已完成而画质不通过，模型任务全部已结束，无自动新执行授权。
