# FFNDS 新结果评议与路线修正 · 2026-10-29

**审阅基线：a11ddcf5658a4179da42681fb31da11426b1c4cc。** 本文是对新实测、现行方法和此前 QCFB 主张的作者复核，不冒充新的独立 Agent 评议。使用 mechanism-research、web-research-discovery、ffnds-paper-discovery 与 ffnds-experiment-evidence。本轮读取公开结果 JSON、报告及执行源码，重算已发布标量与有限数学反例；没有重读远程大张量、播放视频、运行模型或分配 GPU。历史 RUN/失败/哈希及 ACTIVE 不变。[可重跑审计](../../tools/research_route_review_oracle.py)与[机器结果](RESEARCH_ROUTE_REVIEW_SCALAR_AUDIT_20261029.json)分别限定 STATIC/ALG 与已发布实测的来源。

**研究判断：M-B 应继续优先，但问题要具体化为“保留实际读出分子、归一化分母和学生状态下的响应，同时降低物理计算”。原始 BF16 科学线与 RTX PRO 6000 原生 NVFP4 线并行。QCFB 保留为可证伪的联合动作候选，当前证据不足以把它升为主方法。** 这沿用仓库最新双轨政策，纠正仍残留的“A100 only”、重复 B-H/A0 待办及通用 group64 执行指令。

## 1. 新结果如何改变判断

| 结果与真实对象 | 本次评议 | 路线修正 |
| --- | --- | --- |
| [A0 全80局部恢复](../20261028/COMPACT6016_FULL80_RECOVERY_RESULTS_20261028.md)：energy94 局部平均 gated relative SSE 0.314906→0.129766，优于 random 的位置73/80 | 局部函数可训练；80个函数不是80个独立来源，局部均值不等于完整轨迹风险 | 保留其 checkpoint、匹配强基线和失败机制，不把当前6016配方作为已通过的视频方案 |
| [六视频质量](../20261028/COMPACT6016_CONSUMER_QUALITY_RESULTS_20261028.md)：Dense/energy/random MUSIQ 76.018/46.944/26.664；energy 两来源 RAFT 比0.135/0.145；smoothness 约0.987仍高 | 本版两个已暴露来源上有强负结果；平滑度和全为False的二值dynamic不足以判断动作保持。尚未人工完整播放 | 学习目标/状态、容量、量化路径均需区分；不能将其归因给所有稀疏或整个M-B |
| [CAL24 配对判别](../20261028/B_CAL24_SOURCE_PAIRED_DISCRIMINATION_RESULTS_20261028.md)：paired/shuffle/marginal/PCA/head-only同选PCA | 在这个候选池上，各机制的实际干预相同，缺乏判别力；既未证实也未否定一般配对收益 | 先验证候选间决策与损失确有分歧，再冻结检验；不继续扫同一个shared-span的rank |
| [B-H vs 同head A0](../20261028/BH_A0_MATCHED_TERMINAL_HEAD_RESULTS_20261028.md)：高/低 B-H误差0.003958/0.003615，A0为0.000609/0.000661，B-H均0/8胜 | 当前H128不是准确率优胜者；其低算量仍有Pareto价值。不同结构/目标且非系统成本匹配，不能宣称整体被A0支配 | 作为极低成本比较器保留；不把重复B-H训练列为默认下一步 |
| [B-F 固定6016/r512](../20261028/BF_FROZEN6016_TERMINAL_RESULTS_20261028.md)：四学生head误差比A0降低约3.1–4.3%，逻辑MAC为A0的59.26% | 有限局部正结果；仅两末层函数。最终误差是相应teacher投影参考的约45–98倍，不能单独归因于信息丢失或优化不足；该参考也不是全方法最优下界 | 分开“表示可容纳”“输入可预测”“目标/训练能到达”；实测带量化/布局的FFN代价，再决定推广 |
| [原始BF16 low39四角](../20261028/ORIGINAL_BF16_LOW39_FOUR_CORNER_RESULTS_20261028.md)与[固定范数转移](../20261028/ORIGINAL_BF16_FIXED_NORM_TRANSFER_RESULTS_20261028.md)：source0冻结增益，source12同状态误差降93.67%/82.45%，own-state末步降69.59% | 这是目前最强的定点干预证据：这个保留headspan的投影有可纠正的分母偏差。只涉及一个已暴露转移来源、low39和两个step | 提升“原始BF16跨深度分子/分母机制判别”，固定gain是强便宜基线；不得直接声称全深度范数补全已成立 |
| [四步轨迹](../20261028/COMPACT6016_TRAJECTORY_DRIFT_RESULTS_20261028.md)与[step-weight审计](../20261028/COMPACT6016_STEP_WEIGHT_SCALAR_AUDIT_20261028.json) | 误差范数增长约18–21倍，但恒定同向flow偏差在既有步长下也会增长16倍；按累计步长归一后仅约1.13–1.32倍 | 保留局部偏差累积、Dense状态传播、近似函数随状态变化三种解释；不将增长曲线直接命名为不稳定反馈 |

两个原始 BF16 Dense 视频已建立原始数值基线。与旧 HiF4 视频的差异不能自动归因为单独“量化误差”，必须锁定模型、scheduler、attention、seed和插入路径后再作因果比较。以上统计主要来自历史暴露开发来源；本次没有增加独立来源数。

## 2. 必须区分两种瓶颈：分子错误与分母错误

固定一行与conditioning，令 C=I−11ᵀ/D，z=Cu，σ(u)=sqrt(||z||²/D+ε)。把原head权重及固定调制的线性部分记为K，原shift与bias合并为c。连续末层读出为

$$F(u)=q(u)/\sigma(u)+c,\qquad q(u)=KCu.$$

于是有限差而非仅Jacobian满足

$$F(u_S)-F(u_D)=
\underbrace{(q_S-q_D)/\sigma_S}_{\text{分子项}}+
\underbrace{q_D(1/\sigma_S-1/\sigma_D)}_{\text{分母项}}.$$

平方误差保留交叉项，不能把两项平方当作可相加的“误差占比”；真实BF16还有LN、调制、gain、GEMM与写回余项。

- **原始BF16 headspan投影**：同状态分子项平方约3.4–4.2e−7，分母项约3.9e−4–1.66e−3。固定gain具有明确作用靶点。
- **实际 B-F 学生**：PCA512 高/低分子项为5.55e−4/6.48e−4，分母项为5.25e−5/3.75e−5；两个HEAD128_PCA384学生也类似。输出基包含headspan，并不保证学生预测的系数保留真实分子。**不能用原始投影的径向主导解释，替代B-F系数预测问题。**

### 2.1 固定gain应先作为强基线，动态修正必须证明额外价值

现行源码 [raw_bf16_fixed_norm.py](../../tools/raw_bf16_fixed_norm.py)只对新产生的BF16 LN结果乘FP32倒数，再copy回BF16，保留原shift与head bias；κ=1通过存储身份检查后不新增tensor算术。该范围限制κ≥1是本次收缩假设，不是所有层或候选的定理；新候选可能需要放大，必须另定义正gain合同。

在连续域，令a_S=q_S/σ_S，a_D=q_D/σ_D。给每个来源等权、来源内行与step按预定权重计w，统一固定gain的最优值是

$$\alpha^*=\frac{\sum w\,a_S^\top a_D}{\sum w\,\|a_S\|^2}.$$

若目标是等来源的相对SSE，w还应包含相应Dense能量分母。它一般不等于未加权σ比均值的倒数。该式是**未来训练/校准集上的匹配强基线**，本轮不重新拟合，也不在source12结果上追调。正gain限制下需考虑边界，分母为零单列退化情形。

标量gain只能消除沿a_S的误差；正交分量仍在。因此B-F分子方向预测不准时，“再补一个标量”没有普遍救济能力。动态gain或随机草图必须在冻结新来源上超过固定gain，并抵消额外读取、归约与kernel成本。

目前全M×D FP32临时量约648.6 MB，不能因gain只有两个参数便称其“免费”。可以研究将增益融入已有LN/modulation写出，但融合改变舍入顺序；缩放完整head输出还会错误地缩放shift/bias，必须作同路径no-op与数值比较。原W2预取仍支付的实验不证明系统加速。

### 2.2 末层充分统计不等于全深度充分统计

终端固定64维head的(q,v)是连续65量表示。中间block后仍有跨token attention与多个残差消费路径，仅保留这个终端headspan或只改下一LN分母，不保证整个后缀相同。下一步应先标明各深度的实际消费者、跳连和可观察边，再判断可压缩统计。跨深度实验要比较同一个原始BF16候选的bare、固定gain、合法动态修正和同成本分子修正；若分子项占优，应停止把该位置视为纯范数问题。

## 3. QCFB需要把归一化交互纳入联合动作

此前 $$m^\top\Gamma m-2b^\top m$$ 在固定线性读出及相应条件矩下成立，仍是有效代理；本轮结果说明它不能自动替代经过LN的有限损失。

令B(x)各列是相同状态、gate之后的待删组对残差u的贡献，m是删除指示。先忽略native余项，u_S=u_D−Bm，则

$$q_S-q_D=-KCBm,$$
$$\boxed{v_S-v_D=-\frac{2}{D}(Cu_D)^\top CBm+
\frac{1}{D}m^\top B^\top CBm.}$$

**方差改变自身就是带有符号线性项与组间交互的二次函数。** 即使某束满足KCBm≈0，方差仍可能改变，继而整体缩放可见读出。取两个中心化正交单位向量e₁,e₂，u_D=e₁+e₂、u_S=e₁、K=e₁ᵀ；线性投影误差严格为0，归一化读出却不同。此反例由审计脚本直接检验。

因此T0应并列评估：完整state贡献抵消、固定投影贡献抵消、有限LN/head损失、native余项和来源尾部。在线目标可以预测**有限动作间的风险差**，或(q变化,v变化)等动作相关量；没有必要强制在线构造完整216×216Gram。归一化会令最终损失不再一般是m的二次多项式，增加一个Gram不能自动恢复精确性。

这对原命题是具体修正：高能组可能联合可删仍正确；“投影中负交互→全生成中安全可删”没有被证明。若是完整状态误差也小且后缀在该区域受控，才有更强的有限风险依据。小σ、跨token和离群来源需单列。

### 3.1 不把缺失packed Q1当作T0的全局阻断

[Q0资产审计](../20261028/P1_Q0_SAVED_SIGNED_ASSET_AUDIT_20261028.json)表明：原始packed Q1码/scale未在已查包中持久化，但真实有符号V2及gate存在。绑定正确版本的decoded W2、Smooth/gain/cast与行坐标后，可以离线形成旧HiF4的完整组贡献；先核验所读weight面板/资产哈希，不把header检查升为完整payload验证。形成标签并未在本轮执行。

这足以研究旧HiF4 T0的联合作用；它不能证明pre-W1预测、原始BF16或NVFP4收益。T1缺失的真实前置观测、producer与代价另列，不必先运行全模型采集才能开始所有T0工作。

## 4. 元信息时点与有效利用：按数值后端重建

| 信息 | 首次真正可用与可节省的范围 | 本轮修正 |
| --- | --- | --- |
| W1/W2量化码、scale、权重侧草图 | 离线固定；可构建字典、投影、成本与条件权重 | 是先验，不是本次输入的动态观测 |
| BF16原始输入与已有pre-FFN LN统计 | W1前；可用于前置选择 | 原始BF16没有天然Q1 FP4码。人为增加量化/草图必须计价，并与直接BF16摘要比较 |
| Q1/A1 scale与signed位置草图 | 动态量化producer生成后、W1开始前；可能节省W1及其后工作 | scale不能区分符号反转；producer内部可得不等于低成本跨CTA可消费。优先融合小摘要和有限动作风险差 |
| GELU / Q2早期归约 | W1/GELU已支付；满足生产时序才可省后续编码、写出和W2 | A2归约的新增统计可复用，但不能追认W1省算；不同前置动作需要实际动作后的Q2 |
| Q2码与scale | Q2已支付；可选择剩余W2动作 | 以late oracle衡量早期信息损失时标清后验身份；不得进入pre-W1在线特征 |
| post-write残差范数/最终LN统计 | 此次W1/W2已经完成 | 适合损失分解与下一次合法历史信息，不能降低本次已支付算量 |

**新增关键条件：NVFP4 group16本身并不保证完整组可交换。** 设G(h)是动态tensor全局scale，Q_G是给定G的局部编码。对于保留完整局部组的S，

$$Q_{G(h)}(Sh)=S Q_{G(h)}(h)$$

仍需局部数值、布局、输入与舍入条件；若重新估全局scale，则一般

$$Q_{G(Sh)}(Sh)\ne S Q_{G(h)}(h).$$

官方Transformer Engine当前scale recipe使用tensor amax；仓库NVFP4 RTN则接收校准后的activation global scale。两者不能混称。若Dense Q2的global scale依赖被省略组，在线重获它可能要求先计算那些W1/GELU；可以选择固定校准global scale的**匹配Dense与候选recipe**、使用有证明的合法上界或把scale变化并入真实动作，但不能一边依赖Dense未知amax一边宣称早删算。离线生成的权重scale保持原值，不在每个子矩阵上重PTQ。

量化组闭包、GEMM K/N物理tile、row tile共享mask、scale swizzle和残差数值合法性是不同条件。HiF4的64组结论不能复制成NVFP4的通用执行协议。R0/R1先固定真正packed W4A4和当前设备/checkpoint；BF16科学线无需等待它。

净效益仍按实际
$$T_{\rm dense}-(T_{\rm observe}+T_{\rm select}+T_{\rm pack/sync}+T_{\rm retained}+T_{\rm correction})$$
判断，明确Dense也已支付的量化成本只抵扣一次。比较静态/step字典、同成本输入摘要、scale-only、signed位置摘要，以及late Q2上界；仅信息分类准确率不能证明行动价值。

## 5. 论文先验复核与竞争解释

- **L06 SliceGPT**：[原论文](https://spcl.ethz.ch/Publications/.pdf/2024_iclr_sliceGPT.pdf)与仓库已有[归一化深读](../literature/20261028/SLICEGPT_NORMALIZATION_AND_TRUNCATION_RECHECK_20261028.json)区分完整正交变换的范数不变性与随后截断的近似。它支持检查分母，不能证明Wan任意hidden旋转穿过GELU/量化合法，也不能给本次固定gain提供全深度保证。
- **L56 Self Forcing**：[原论文§3.2–3.3及限制](https://self-forcing.github.io/static/self_forcing.pdf)支持使用学生实际到达状态与生成级目标。它的AR帧/KV自回归与Wan四步双向去噪不是同一时间轴；不能用该先验宣布本次误差增长已由exposure bias单独解释。对FFNDS应比较teacher-state与student-state同预算拟合，并保留scheduler、局部偏差和state交互。
- **L31 NVFP4**：[官方格式/scale/布局](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)与[仓库固定RTN实现](../../upstream_audit/source_deltas/Wan2.2-I2V-A14B-W4A4/files/quant/nvfp4_rtn_backend.py)约束上节两种global-scale语义。训练recipe、推理后端和历史SM120小矩阵证明不能相互替代。

本次广域检索覆盖normalization invariance、student rollout/exposure bias与NVFP4 producer/scale，随后回读上述原始材料。没有新增论文LID或凭摘要提高其他论文的阅读级别。复核出处与迁移区别同步到文献索引和G14。更强的新颖性结论仍需独立先例检索与真实机制证据。

## 6. 修正后的研究顺序及最小判别

以下是研究设计，**不是新增ACTIVE作业**。优先级表示各线内部顺序，不让两个硬件lane互相等待。

| 工作 | 最小可区分对照 | 改变决定的证据 |
| --- | --- | --- |
| **BF16 / 下一科学判别** | 先核CalibrationV2跨来源/深度覆盖及真实consumer；同一候选bare、固定gain、合法动态范数修正、同成本分子修正；固定新source确认集 | 逐位置direct/radial/signed/native余项和尾部；动态修正只在超过冻结固定gain且成本合算时保留。分子主导则转向系数/特征/目标判别 |
| **B-F / 区分预测与目标** | 冻结基底与输入特征，比较当前Y重建和真实head/finite目标的同预算训练；再单独比较更丰富合法输入；teacher投影仅作表示参考 | 哪一改变缩小预测差距，是否提高来源尾部与own-state响应。不得只依据oracle差距宣布信息不可预测 |
| **状态 / 区分累积与反馈** | 用四角a=S(xD)−D(xD)、p=D(xS)−D(xD)、c=[S(xS)−D(xS)]−a；同预算teacher/student-state训练 | a+p+c及交叉项、scheduler舍入闭合，比较步长归一的偏差和paired视频；避免只看relative SSE曲线 |
| **QCFB / 并行T0** | 绑定已有V2/W2/gate形成有限物理字典；相同字典比较diagonal、joint与finite-normalized风险；按来源/step拆分 | oracle动作收益、固定字典覆盖、跨source稳定与尾部。无联合优势则停该字典，不先建昂贵router |
| **QCFB / T1** | 同deadline、同动作字典比较静态、step、付费BF16摘要、Q1 scale、signed位置摘要；late Q2另列 | 真实净风险收益和观察成本；分箱不能当连续信息的不可约下界 |
| **PRO6000 / R0→R4** | 独立核访问/模型/packed数值→完整Dense→同recipe候选→完整视频/系统；沿用现行双轨协议 | group/global-scale/tile闭包、nonzero kernel、真实FFN与整端wall；不将fake-QDQ或W4A16当原生W4A4 |
| **视频验收** | 科学上需要时及时做有界完整生成；最终确认使用独立image-cluster，配对Dense/source/noise/seed和完整动作语义 | 连续motion、画质、主体/背景、camera、时间语义、失败尾部、人工全播和真实速度并列；现有两来源不作正式通过 |

**暂停投入的具体范围**：当前compact6016配方的“已可用视频”叙事、未区分机制的同池rank扫描、将B-H H128当默认准确率突破、在T0/T1未有证据前投入QCFB完整执行器。保留这些资产与可检验的新变体，不关闭整个函数压缩、量化或稀疏研究家族。

## 7. 对此前路线判断的反思

此前强调联合抵消和量化前置观测，数学上成立，但对真实后缀、producer成本和现有学生的强基线仍需要更实在的约束。新结果把三个问题拆开：**组贡献能否抵消、合法信息能否预测动作、有限数值动作能否保住生成行为**。它们各自需要证据，不能由同一个负交叉项替代。

本轮最值得推进的新机制是**在指定消费路径上联合保留分子与分母，并用真实学生状态校验**；它既可以指导B-F构造，也可以改变QCFB的风险标签。固定范数增益的成功是一条有边界的因果线索，B-F分子主导则是阻止过度推广它的反证。最终仍以完整生成行为与实际执行成本决定方法是否成立。

