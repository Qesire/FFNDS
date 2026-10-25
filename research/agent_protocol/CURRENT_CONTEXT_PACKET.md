# FFNDS · E0 当前科学上下文（2026-10-26）

**刚刚确定的科研质量底线（2026-10-26）**：[每次生成必须有完整质量/物理指标和全40层×四步真实前缀诊断](VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)。上一轮两层partial-native仅0和30，**不包含末层39**；160处完整模型替换已执行，但尚未采集160处Student原生同输入/离轨输入的配对误差。现有六视频只有RGB/亮度/时间差等部分指标，正式VBench-I2V、光流/细节、真实背景闪烁及人工视频审阅不齐，不符合新的完整质量验收。新产生的每条视频必须有全帧逐帧指标、SHA/运行成本、未执行官方指标的明确状态和证据bundle校验。工具：[全帧分析](../../tools/ffnds_video_metrics.py) · [全视频SHA门禁](../../tools/ffnds_video_metric_gate.py)；新160位研究仍仅静态模板未授权GPU。实时场景用户已明确暂停。

**最新全模型更新 2026-10-26**：[全40层校准](../20261026/E0_ALL_BLOCK_RECONSTRUCTION.md)、[完整160处FFN与六条视频](../20261026/E0_RECONSTRUCTED_VIDEOS.md)以及[新的方向审查](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)。affine54在强resident dense同卡热态FFN9.707→4.449s、全生成50.750→45.313s，但视频变模糊、动作减弱，whole94严重重影，**新原型无画质验收**。后续优先区分same-input部署数值差、受扰前缀输入分布及原生下游有符号/跨步传递，而不是继续调旧局部SSE。跨来源局部8.54%也不是视频指标。正式方法选择未发生，所有注册实验结束。

**并列实时场景（OPEN）**：[实时交互式生成研究](../20261026/E0_REALTIME_GENERATION_SCENARIO_20261026.md)，对应G6新节点。原81帧整段离线完成时间不是逐帧stream FPS/TTFF，后者需确定causal/rolling模型源码、逐帧p95/p99 deadline、KV/FFN缓存有效条件及long-horizon漂移。用户新增场景不自动变更原Wan主线、执行授权或算法选择。

**方向再审（当前首先阅读）**：[E0后affine科学分歧](../20261025/E0_POST_AFFINE_DIRECTION_REVIEW_20261025.md)与新增 [G5图层](../../docs/98_端到端计算图与证据注解_20261025.md)。affine表征、奇偶/热区竞争解释、旧导数层间差异、真实权重常驻属于不同科学对象。当前活跃任务允许由下游画质、时空组织、早期信息、W1/W2可删计算甚至新表示直接提出新机制；不要把下一任务等同于固定1/2斜率或affine+27组。Graph-Agent节点是来源与论证索引，不构成科学问题全集。

**最新实际推进：** [全模型三臂六视频](../20261026/E0_RECONSTRUCTED_VIDEOS.md)已在全部160处FFN真实执行。affine54热态FFN边界9.707→4.449s（2.182×）、生成50.750→45.313s（1.120×），setup另计；两个原型仍明显损坏画面，未满足完整视频质量目标。[全40层相同四来源/256行控制](../20261026/E0_ALL_BLOCK_RECONSTRUCTION.md)中，affine锚定输入ridge15.35%对affine54局部SSE8.54%，后者320/320较低；因此旧覆盖混杂不再是这一固定对照的充分解释，但更低局部误差没有转成可用视频。四次被动补采、另冻的生成前失败恢复、160个新BF16参数包与六视频范围均结束，GPU已释放；旧失败回执不删、历史12/40及旧协议不改。用户上限四张空卡，新实验无自动权限。[功能/受扰轨迹数学](../method_discussion/discoveries/E0/transported-function-and-trajectory-target-20261026.md)区分输入偏移、实际后缀传递和下一状态目标；[独立审查](../method_discussion/discoveries/E0/independent-matched-calibration-and-runtime-audit-20261026.md)保留反解释。[前序四项检查](../20261026/E0_FUNCTION_RECONSTRUCTION.md)仍为原始范围，P14通用假说开放。

**上一轮已完成证据：** [全层函数表示结果](../20261025/E0_AFFINE_FFN_DISCOVERY.md)保存两来源每步40层中心128/原random32及旧full-M导数均值。该两来源字段勿重复采集；四个其它校准来源先前只有五层。full-M不是误差评价行数。上一轮提出的半斜率解释现已用上面的实际控制检验，历史文件保留。

**可复用的运行时成果：** [相同视频的权重常驻](../20261025/E0_DENSE_WEIGHT_RESIDENCY.md)在单张80GB A100同卡热态约75→50秒，18视频/72latent SHA与原dense相同；显存物理采样峰值77.08GiB。没有减少FFN MAC，后续算法应对照这个更强dense参考并报告冷态/显存。两来源实际seed42/54；此前“seed42”简称指base seed，并非source12同seed。全部本轮固定范围完成、0作业，实际峰值2卡、用户上限4卡；旧12/40次冻结回执不变。

**唯一活跃科学入口**：[FFN 本征机制重置](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md) → [首个 FFN 科学发现任务](../20261025/E0_FFN_FIRST_TASK_20261025.md) → [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)。本文件是阅读索引，不增加实验权限。**从 FFN 的输入系数与 W2 value vectors、有符号组合、时空变化、下游功能和可替代计算开始；量化尺度属于辅助幅值线索。**

**最新连续token/前置观察跟进：** [报告](../20261025/E0_SHARED_TILE_PRE_W1.md)、[共享列数据](../20261025/E0_SHARED_TILE_PRE_W1.json)、[原量化组依赖](../20261025/E0_SHARED_TILE_CLOSURE.json)。再次明确续推后，两张空闲A100完成source0/12两次相同dense生成，连续M128向量已保存；另120次W1、2280次W2诊断。25%任意共享列的前置micro16 GELU/value排名：Block39创新SSE6.55%，中间观测层42%–47%。块内共同创新能量首尾65%/78%、中间10%–14%。所有25%共享集合触及全部原64组；原始Python量化合成反例只改一项却改变64项输出。下一判别需要联合函数/执行单位，分别记W1/W2成本；不是继续微调topk。两视频、8latent、40处五字段控制通过，无待执行作业；E0/候选选择未变。

**最新已完成原生/GPU观察：** 用户随后明确指示“同步最新，然后继续推进，不要害怕用gpu”，据此冻结并完成[完整hidden与原生支持分析](../20261025/E0_INTRINSIC_FULL_HIDDEN.md)：六来源180个绝对/同专家创新案例，全部13824通道；另有source0/12两次原生被动四步生成、40处完整M支持观察。25%单元value对角排名的创新相对SSE为2.44%–7.07%，连续64组的后验组能量排名为32.42%–54.67%；两者同通道预算。原布局M16/64/128×K64支持块在12.5%保留率也全部非空。两视频、八latent、40处FFN采样哈希与历史dense一致。实际一张空闲A100，已释放、无待运行作业；旧12/40次范围未改。新的结论是“创新集中”和“原布局可整块跳过”不同，并不排除共享任意列/重表示；没有执行稀疏候选或取得加速/质量结论。

## 真实对象与解释口径

- 当前已完成观测环境：Wan2.2 I2V 蒸馏四步，LightX2V D-Q/HiF4 QDQ→BF16 GEMM，5120→13824→5120 双专家 FFN，含 AdaLN、gate、残差和后续 attention；真实源码身份见 [C0 源绑定](../20261023/C0_coupling_offline_results.json) 与 [runtime 重建](../20261022/R12_E0_runtime_reconstruction.json)。
- 量化元数据在此研究默认视为可选的粗略幅值/范围统计；HiF4 现有 QDQ 并非原生 MXFP4/NVFP4 数据布局或 FP4 GEMM，方法发现不再预设 scale/code、K panel、64 hidden group 或 QDQ 是算法核心。
- 强调四对象分离：`|GELU(a)|` 强度；`Σ_j GELU(a_j)W2[:,j]` 的净作用；后续 norm/attention/多步生成的功能价值；可真实跳过或共享的运算和读取。
- 最近已保存的径向对齐 **19.794% raw 正确对 20.542% 错排、8.839% centered 正确对 3.590% 错排**及同 seed 六来源共有方向 cosine 0.99645，属于特定 32 行×5组 ideal epsilon-LN 表征结果。**不是 FFN 整体功能冗余率**；见 [E0_RADIAL_PAIRING.md](../20261025/E0_RADIAL_PAIRING.md) 与 [复审](../20261025/E0_SKILL_WEB_REAUDIT_20261025.md)。
- [原生四臂跟进](../20261025/E0_FOLLOWTHROUGH.md) 提示 BF16 写回与下游连续传播差异，属于可供未来功能机制分析的既有证据。star-null 是特定相关误差零模型，不是动作可加的充要条件。详见 [E0_SHARED_REFERENCE_MOMENTS.md](../20261025/E0_SHARED_REFERENCE_MOMENTS.md)。
- 历史 A5+HWG 发现主体外光照/细节不稳定；将其转为时间/位置/频谱重要性的竞争解释，研究目标仍是同等设置下全视频质量与实际 FFN 计算的 Pareto 改善。

**当前新问题**：[最新机制分歧与研究空间](../20261025/E0_INTRINSIC_DIVERGENCE_AUDIT_20261025.md)已将完整hidden、原布局支持和连续M128字段归为**已取得证据**。待研究的核心关系是：何种条件函数或时空表示保留逐token创新的集中？这种作用如何传递到视频？能在哪一个真实计算边界减少W1/W2和总成本？研究可从输入、输出函数、下游感知或物理执行任意开始，不让shared-column、原group64或micro16固定所有Agent的假说。

**时点与数值对象**：`q2_post`是实际W2输入，而`h=GELU(a)`是Q2前的对象。micro16先于完整token W1，但使用**额外全宽W1和GELU**投影。原HiF4尺度组依赖是该数值实现的约束，与新格式/条件近似/函数重写所允许的计算不同。保存的全通道sample与全token仅占用统计也不是同一范围。

**计算图协作**：同步查阅 [G0–G6](../../docs/98_端到端计算图与证据注解_20261025.md)、[Graph↔Agent协议](GRAPH_AGENT_COLLABORATION.md)和 [机器可读节点](GRAPH_AGENT_BINDINGS.json)。独立研究者选与问题相关的若干节点沿原生执行、历史条件、下游功能及成本路径核对证据；新机制/反例另入 [E0图注解](../method_discussion/graph_annotations/E0/README.md)，交叉核查后再由整理者更新图/谱系。图的候选路线为待证实边，不自动代表可运行算子；允许提出新的数学节点和执行路径。

## 当前第一组开放科学探索

**A FFN函数与非线性**：利用现存完整W1/W2和q2_post，从signed贡献、GELU特别结构、Jacobian和特征表征中找新的计算单位，而不是再检索相同的hidden字段。

**B FFN空间—时间组织**：逐token集中而raster共享列失效，是真实token异质、空间分组失配还是函数基错误？用竞争预测区分，四步双专家历史按实际可用性审查。

**C FFN视频功能与计算重构**：从残差/attention、背景光照及小物体时序敏感性，或真实GEMM系统约束反推必要函数作用。分别核算W1/W2、观察器和QDQ依赖；欢迎第四种新解释。

独立挑战者通过 [扩展研究文献地图](../20261025/E0_FFN_INTRINSIC_PRIOR_ART_20261025.md) 对照 Geva/FFN weighted-values、superposition、Chipmunk、RT-Lynx、AdaCache、VBench 与 GPU tiling 等原始工作，核对对象、信息时序、训练负担和实际 kernel。默认 AutoResearcher v2.4 的 mechanism-research、research-expansion、problem-reformulation、[WARD 深度检索](DEEP_WEB_RESEARCH.md)。所有 Agent 在 [E0 独立讨论](../method_discussion/discoveries/E0/README.md) 保存自己有实质差异的科学贡献。

## 可使用的证据与能力边界

[完整实验证据导航](../EXPERIMENT_EVIDENCE_INDEX.md) 与 [量化优先时代旧 E0 第一任务](../20261024/FIRST_TASK_E0_QUANTIZATION_FFN_DISCOVERY_20261024.md) 都继续保存，以服务确切新机制而非统领下一任务。原六来源被动12次、有限四臂30次、跨节点技术重复10次、先前完整M被动2次及最新连续向量被动2次均完成；无待运行的新模型任务。原范围峰值2张、先前完整M1张、最新连续采集2张，用户上限仍为最多4张空闲卡。当前可以进行源码/原论文/数学/获准的保存资产复算；另一个 native model/GPU/视频/训练范围由新的明确指令与具体方案确定。现阶段 `E0_OPEN_DISCOVERY`，未选择任何新算子，同行认可仍由真正的独立评议产生。
