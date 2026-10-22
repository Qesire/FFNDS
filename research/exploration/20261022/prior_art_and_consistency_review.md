> **FFNDS research-library snapshot · V2 academic comparison and correction review.** Scholarly comparison and hypothesis positioning, not proof of originality; review original primary sources and their latest code before a publication claim.
>
> Evidence policy: cited prior FFNDS results retain their original provenance; suggested operators and training schedules have **not** been executed by virtue of this import. Do not treat archived "next" instructions or "not yet merged" source-state language as current decisions. Read [the authoritative exploration entry](EXPLORATION_INDEX.md), [docs/66](../../../docs/66_R12等预算动作的操作数与条件粒度审计.md), and [CURRENT_DECISION](../../CURRENT_DECISION.json) first.

---

# FFNDS 探索网 V2：学术先例、原创性边界与机制一致性审查

**审阅稿 · 2026-10-22 · 尚未合并 FFNDS**

审查对象：`FFNDS_全历史探索网_V2_调制跨层融合与质量驱动_20261022.md`（845行）；用户上传的 2026-10-21 文献资料包（26篇论文目录、分类、代码链接与证据记录）；FFNDS `docs/08/28/29/34/37/42/48/53/55/57/58/60/62/63/64/65`、R12 研究草案；截至2026-10-22可获取的作者论文、会议正式摘要、官方源码及算子文档。使用 AutoResearcher `mechanism-research` 与 `research-expansion` 的 **发现→演绎→竞争解释→最小判别实验→可执行构造** 组织方式。

本报告的“重复风险”是**研究定位的相似度评估**，不是专利检索、穷尽文献证明，也不是对未知完整实现的断言。对上传的 26 篇逐类审阅其记录，并对高相关先例重新检索核对；从资料包以外补进 PARE、DiT-Pruning、BudCache 和 FlashInfer DiT 融合接口等。未下载/运行第三方代码、未新增 Wan GPU 实验。

## 0. 结论先行：论文应当主张什么

**删除三种潜在“伪新颖性”：**

1. **“视频/蒸馏 + timestep、输入相关 block/MLP 宽度路由”本身不新。** PARE 在 Wan2.1 I2V/T2V 中做时步及视觉内容条件 block 路由，E-DiT 在 DiT 中做样本相关 block skip/MLP 宽度，Dynamic-in-Few-Step 直接在 Wan 视频少步蒸馏中联合结构计算分配。把 P-B 改名为“Hierarchical Counterfactual Budgeting”不能绕过这些先例。
2. **“AdaLN 是免费的样本条件信号”不成立，至少在标准 Wan 的固定 timestep 下不成立。** 官方 Wan2.2 `time_embedding(sinusoidal(t)) → time_projection(6×dim)` 加每 block 固定 `modulation` 参数形成 shift/scale/gate；相同 token 时间输入、同一专家和层时，单独的 AdaLN 值不含视频内容差异。它能定义 **随 step/block 改变的风险度量、输入调制或先验**，但内容动态性主要需要实时的 `post-attention residual / norm2 / modulated FFN input` 等。实际 LightX2V D-Q 还须核查 `t` 是否因 I2V 部分 token 而异；若 `t` 异，也先鉴别属于预设位置/时间类别还是内容变量。
3. **“时间一致性约束、层敏感性、规范的 kernel 融合、新视频质量指标”单独不新。** TAPE 处理背景闪烁与 temporal mask jitter，SODA/SenCache 研究跨时间/层/模块敏感度，FlashInfer 已公开专用 DiT residual+LayerNorm+scale/shift/gate 融合接口。这些应成为合适先例或强对照。

**值得实测争取的核心贡献组合**：在已经完成四步蒸馏、固定主干权重的 **Wan FFN** 中，构造**真实改变两侧 GEMM** 的、HiF4 数值/硬件兼容的有限条件动作；在 **FFN0/W1 前**用内容相关且获取费用极低的状态信号预测其**相对最强等预算固定动作的下游反事实收益**，并用自由四步视频的质量反馈鉴别 signed-flow 标签在哪些时空区域真正有解释力；最后以共同优化的 FFN 执行器兑现收益。这是**当前未见完全相同的证据链组合**，不是已经确认没人发表过，也不是每个子模块都有独立新颖性。

最重要的独立科学问题不是“有没有 router”，而是：**蒸馏后视频 FFN 中，内容条件的可执行计算动作是否产生了最佳固定策略无法获得的增量价值？这种价值能否在决策点及时、廉价地获知？**

## 1. 文献审查的五维比较坐标

任意新路线与论文统一记录：

- **研究对象**：LLM/图像DiT/视频DiT；原模型、已经蒸馏模型还是蒸馏过程中同时改结构；MoE专家、层、FFN hidden、attention head、token、cache 哪一层。
- **信息边界**：当前 W1 前/后、前一步、前一 block、完整教师、离线 oracle；随 timestep/固定层变化与真正随视频输入变化严格区分。
- **动作及硬件**：skip block、动态宽度、静态分组、token tile、N:M activation、输出缓存、低秩表示；两侧 GEMM 是否节省，QDQ/packing/launch/搬运费用是否计入。
- **优化目标**：局部 channel norm/C-score、原生 FFN SSE、同状态 flow、自由学生视频、官方质量与时空缺陷；各自的统计分母是否相同。
- **训练和证据级别**：training-free/小校准/后训练/重新蒸馏；合成 kernel/单线性层/完整 FFN/端到端视频；来源暴露及外推范围。

这五轴同时不同，才可能形成真正独立的“方法差异”；只有把实验放到 Wan 或换评价指标并不够。

## 2. 最相关的先例及其实际覆盖范围

| 论文（正式出处 / 年份） | 直接覆盖了什么 | 与 FFNDS 的关系；真正可拓展点 |
|---|---|---|
| **PARE** ([arXiv 2605.27336](https://arxiv.org/abs/2605.27336), 2026) | Wan2.1-14B I2V/T2V；attention 宽度结构剪枝，timestep+视觉内容条件 block skip，渐进恢复与路由学习；可与 step distillation 组合 | **V2 P-B 最直接的新补充撞车文献**。不能以“视频内容条件跨 block 路由”为独立创新。区分点是 *冻结四步 Wan FFN 内、等总 FFN MAC 的可预测有符号动作价值*，而不是整 block 跳过。须比 PARE 所允许的相近部署预算/质量。
| **E-DiT** ([ICML 2026](https://proceedings.mlr.press/v306/wang26dd.html)) | latent 条件下 block skip + 当 block 被执行时调节 MLP 宽度 + cache；图像与3D资产 | **动态 FFN 宽度本身已覆盖**。我们的作用在 FFN 内的分组/有符号补偿、真实视频四步闭环、HiF4、等预算最强 fixed 对照，而不是新的通用宽度 router。
| **Dynamic-in-Few-Step** ([arXiv 2607.06631](https://arxiv.org/abs/2607.06631)) | 通过联合少步蒸馏和结构稀疏得到 step-specific MoM，Wan-14B 4步基线上额外减少 FLOPs 并加速 | **少步视频与动态结构已有直接先例**。后训练冻结 D-Q 才是训练制度差异，但不能仅凭“无需重新蒸馏”宣称原创；还要有独立动作/信息/目标。
| **WIDE** ([arXiv 2607.28418](https://arxiv.org/abs/2607.28418), 2026) | LLM token级 FFN/head group 的动态宽度、mask reordering、block/intra-block kernel co-design | **tile/group + kernel 联合设计不是新概念**。在视频 FFN 中的时空执行形态和非 C-score 的下游动作价值有不同验证问题。
| **Fast Forward** ([arXiv 2602.00397](https://arxiv.org/abs/2602.00397)) | LLM prefill；预测器 + FFN 分组选择 + 错误补偿 + layer-wise schedule | **P-A + P-B + 修正网络的通用组合已有高度近似**。必须证明源自真实 Wan FFN 动力学的独立可证伪机制，不以三模块拼接作贡献。
| **Prox** ([arXiv 2607.27591](https://arxiv.org/abs/2607.27591)) | W1前廉价 proxy 只供选择，原生选中通道精算；LLM SwiGLU | 旧 A5+HWG/Proxy 的“代理只用于 mask”不是新颖点。差异要放到 GELU视频FFN 的 *signed native flow 目标/可执行动作*。
| **RT-Lynx** ([ICML 2026](https://proceedings.mlr.press/v306/cong26a.html)) | DiT activation N:M、误差补偿和优化 CUDA kernel；报告最高约1.55×**线性层**加速 | “activation 2:4 + compensation + fused kernel”已覆盖；不能把单层加速当完整 FFN。适合做另一种结构执行基线。
| **DiT-Pruning** ([arXiv 2607.00927](https://arxiv.org/abs/2607.00927)) | 针对 DiT 特征的后训练 saliency、权重激活平衡与 cluster-aware granularity | **后训练 + 特色重要性度量 + 聚类分组不自动新**。FFNDS 功能组必须用下游 *条件边际价值* 和硬件部署定义，且证明比静态聚类更好。
| **DiffSparse** ([ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5e66b9f710a8e38351da690f3806cba9-Abstract-Conference.html)) | 包括 Wan2.1 的 layer-wise 可学习 token 稀疏与动态规划、token cache | 层间预算和少步问题非新，但对象是 token 缓存/计算，不是同层 FFN hidden 单元。区别必须在真正算子和评价体现。
| **SODA** ([arXiv 2603.07057](https://arxiv.org/abs/2603.07057), CVPR 2026) | timestep/layer/module sensitivity；pruning+cache 与优化预算；OpenSora 等 | **跨层敏感性、根据动作选择预算已广泛使用**。我们的增量必须是同成本原生 *conditional action advantage + error-direction interactions*，而非一般“敏感层多算”。
| **SenCache** ([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Haghighi_SenCache_Accelerating_Diffusion_Model_Inference_via_Sensitivity-Aware_Caching_CVPR_2026_paper.html)) | 噪声 latent + timestep 扰动敏感性、样本相关缓存 Wan/CogVideoX/LTX | 不能以“每输入跨步敏感性”作为新颖点；可以检验它是否能推出冻结 FFN 层对的 signed-value，不能直接迁移其缓存结论。
| **TAPE** ([arXiv 2605.17837](https://arxiv.org/abs/2605.17837)) | 视频 token 裁剪的 temporal mask smoothing、层间重选、timestep预算；明确处理背景闪烁 | **背景闪烁和时间一致性已被发现**。FFNDS 新意只能在 *FFN 动作产生何种有符号传播/哪些缺陷可被上游信息预测* 的实测机制，而非加平滑。
| **SeaCache** ([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chung_SeaCache_Spectral-Evolution-Aware_Cache_for_Accelerating_Diffusion_Models_CVPR_2026_paper.html)) | 时间谱演化感知的动态 cache，强调结构低/高频 | 若提出“频率敏感度感知 FFN”，必须给出与特定 FFN 删除动作、时空 residual 干预相关的新增机制，而不是复述频谱时间原则。
| **BudCache** ([ICML 2026](https://proceedings.mlr.press/v306/lei26j.html)) | 固定计算预算下离线寻找 cache policy、对最终输出优化 | “预算严格守恒 + 终态目标”本身不是新颖点；FFNDS 需区分实时内容条件动作与离线固定 cache schedule。
| **FlashInfer norm** ([官方 API](https://docs.flashinfer.ai/api/norm.html)) | 已含 `fused_dit_residual_layernorm_scale_shift`、`fused_dit_gate_residual_layernorm_scale_shift`、`fused_dit_gate_residual_layernorm_gamma_beta` 等接口 | **V2 F1/F3 的一般融合已被覆盖**。探索新点应移至当前 HiF4 native QDQ 下跨 decision+Q1 的零额外读特征、action-specific W1/W2 pack 和确实优于现成 fused operator 的全 FFN 速度。

本清单不把原模型 50-step 与已蒸馏 4-step 的任何 speedup 混合比较，不把 RT-Lynx 的线性层倍速、FlashInfer 的现成算子、E-DiT 的非视频结果算成 Wan D-Q 的实验结果。

## 3. V2 的十二个节点：重复性、已知约束与修订决策

标记：**竞争重叠**＝已有工作直接涉及主要动作/概念，不能作为独立论文贡献；**机制保留**＝科学差异存在但须实验辨别；**支撑/系统**＝作为方法的必要对照与工程贡献；**暂缓**＝优先级较低，不代表逻辑上不可行。

| V2 节点 | 最近论文/代码 | 审查后的定位 | 最小支持实验（先廉价，再真实） |
|---|---|---|---|
| **N01 AdaLN 条件价值** | 官方 Wan time_projection；PARE/E-DiT | **大修**：AdaLN-only 主要是 step/layer 静态对照；`gate` 是固定已知的风险加权；保留 `AdaLN × 实时内容` 的交互研究 | 在 **同step、同block、同专家、同分辨率**，核对不同视频 `β/γ/g` 字节哈希与方差；再测试 `step-only`、`AdaLN-only`、`u-sketch`、`gate-weighted u-sketch` 的 out-of-source action-regret |
| **N02 Attention—FFN 内容联系** | PARE、SODA、Fast Forward | **保留为信息增量**，不能只做“加入attention特征”；必须有**额外收益/字节数** | `r_after_attn` 或 native attention delta 低维摘要 vs `u` 同成本投影；控制相同 step/layer 和下游 action，给出条件信息增量 |
| **N03 跨 block 预算守恒** | PARE、E-DiT、SODA、BudCache | **高度重叠**。R12 的 162/162→216/108 等预算层对是判别**具体机制的有用实验**，不是原创算法本身 | 同 native step 输入的 3 臂 + 最佳 static + source oracle；反转若不存在则改动作表达，不训练大 router；反转存在还需 pre-W1 预测 |
| **N04 FFN 内有符号功能组** | WINA、DiT-Pruning、RT-Lynx、WIDE | **高潜力但需要可执行新构造**；group signed Gram 是已知代数，不是论文贡献 | 比静态 C-topK、signed train-only group、匹配补偿组，**两侧主GEMM**原生执行、跨来源 conditional headroom；输出方向抵消不一定意味着可省计算 |
| **N05 量化兼容功能坐标** | RT-Lynx、SparseGPT、DiT-Pruning、量化动态稀疏相关论文 | **独立机制保留**；不是再重复 R8F2 group Hessian 或 R9A1 OLS+贪心 | 同动作比较 BF16/HiF4 native、重编码权重、分组尺度边界、out-of-calibration span 误差、真正 W1/W2 代价 |
| **N06 融合执行器** | FlashInfer、WIDE、RT-Lynx | **系统支撑；新颖性减弱**。通用 fused norm/DiT gate 已有实现 | 先以现成 fused 对照；再测“同步提取 pre-W1 signal + Q1 +选 pack动作”的新增收益，真实 A100/HiF4 ABI 与bitwise守卫 |
| **N07 时空风险机制** | TAPE、SenCache、SeaCache、SODA | **保留为因果诊断**，不是单独“我们考虑闪烁” | 对相同 FFN 局部范数且不同 signed flow 方向干预产生自由视频，遮罩背景、运动补偿，再测时间频谱和主观/官方感知差异 |
| **N08 真实质量直接引导** | Dynamic-in-Few-Step、PARE、SparkDiffusion、STEP-OPD | **必要结果层；非独立新颖性** | 自由四步已暴露开发视频训练机制；最终必须另一组来源冻结后首次评测三轴/时序风险；与 matched static 比较 |
| **N09 原生反事实目标** | SODA、BudCache、STEP-OPD、Fast Forward | **有希望的差异化核心：对可执行 FFN action 的 signed downstream marginal value 直接监督**；恒等式本身是初等代数 | 等总 MAC action matrix；分解误差方向和 native nonlinearity；相对**最优固定动作**的条件 oracle 与可预测 regret；离线 oracle 不称在线算法 |
| **N10 小型后训练** | PARE、E-DiT、Fast Forward、Dynamic-in-Few-Step | **常规手段；训练预算制度需明确** | 冻结 D-Q，只训同参数 tiny router / action-conditional correction，跨来源训练/选择；与同预算局部目标和 step-only 分配差异 |
| **N11 蒸馏条件几何** | Dynamic-in-Few-Step、SparkDiffusion、OD1 + broader distillation | **潜在机制研究主轴**：比较 O/D-BF16/D-Q 的条件动作损失/信息预见性，不预设蒸馏必使冗余消失 | **匹配真实输入与各自自然轨迹要分离**，重算 group output cancellation、action oracle gaps、pre-W1 signal info 与最终质量敏感性；OD1仅两图不支持总体结论 |
| **N12 跨step/tile** | Chipmunk、SenCache、TAPE、WIDE、Sparse Video-Gen | **暂缓/旁枝**。缓存复用、时间平滑、tile mask 已是成熟概念 | 四步 high/high/low/low 冷启动和真正同坐标配对检查；若有新 value 再做同成本 native tile/temporal action，不能把稀疏token/attention当 FFN hidden |

### 3.1 三项必须立刻修正的 V2 语句

**R-01 / 源码事实校正**：将 P-A 标题由“调制条件功能组选择”改为 **“时间调制坐标下的内容条件 FFN 动作价值”**；`AdaLN-only` 在 standard D-Q 同step下作为常数基线，而非独立视频内容 router。其作用是解释 `u=(1+γ)Norm(r)+β` 和 `g·FFN(u)` 的坐标重加权。若 D-Q 实测 beta/gamma/g 会随视频变化，优先追源码：是否 t/帧类别差异或真正条件依赖。

**R-02 / 论文差异校正**：P-B 的论述不再把“sample-wise block budgeting / width routing”写作新算法。改为“**相同总费用、同 native 数值输入的有符号跨 FFN 动作收益与相互作用**”，再决定它是否足以指导 router；PARE 与 E-DiT 是明确直接竞争者。

**R-03 / 融合先例校正**：F1/F3 的通用组合已有 FlashInfer DiT norm/gate 融合。主张应限于 **当前真实 HiF4 QDQ + BF16 Q1/W1/Q2/W2 和动作 bank 的端到端融合**，并以已有 fused DiT 算子作为对照，不可宣称首次融合 residual+LayerNorm+modulation。

### 3.2 三项必须保持的 V2 优势

**K-01**：最终视频是判据，原生 downstream flow 是判别显微镜；R11 原生 flow 的局部对照不能直接宣称视频质量。

**K-02**：同时研究动作空间、提前可得信息、质量传播、真实执行费用，不能只对 mask/top-k/C-score 做复杂预测。

**K-03**：不以“方法家族已有先例”为由停止真正有新定义的算子；用具体 *计算图/实验动作/可用信息/验证对象* 识别等价与差异。

## 4. 反证一致性：哪些说法会直接冲突，哪些只需实验

| 可能的错误说法 | 与现有成果或数学的冲突 | 严格而积极的替代表述 |
|---|---|---|
| “AdaLN gate 直接表示被省掉的 hidden 组” | gate 为 5120 输出维，hidden 是 13824；经过 W2 混合，不能一一对应 | 用 gate 形成输出风险权重，联动 W2 有符号方向，预测一个可执行 hidden action 的真实收益 |
| “同step不同视频的AdaLN就有样本条件变化” | 标准 Wan time_projection 只读 `t`，同 t、同 block 则通常相同 | 首先逐位比对实际 D-Q；如果恒定，让内容信息来自 post-attention/residual；仍研究时间调制交互 |
| “在 W1 之后选 H 再省一半 W2 = 主FFN达到 2x” | 在两侧 GEMM 成本近等且选择零成本时主计算为 `W1 + 0.5W2`，最高 `4/3×`；完整FFN更慢 | 真正 W1 前缩窄或换可省两个 GEMM 的算子，实测原生代价 |
| “C-score/topk 或局部 FFN MSE 足以保证视频质量” | R7/R11 local/flow 排序不同；UB1补充20 imaging 失败；TAPE 等报告时空一致性伤害 | 将局部score作为诊断/辅助变量，并用真实 downstream signed value 与自由视频质量裁决 |
| “视频闪烁说明稀疏算法必然破坏时序一致性” | TAPE揭示特定 token mask jitter，FFNDS 的画面异常未独立归因 | 用同局部范数不同符号传播方向的干预、时序ROI和稀疏决策时序稳定性区分机制 |
| “原始 O 与蒸馏 D 的 group 密度一定不同” | OD1 两图 block20 top108 保留几乎一致，尚不足以推全模型 | 用 O/D 配对原生输入及自然轨迹测 *动作价值几何与预测性*，并报告何处没有区别 |
| “等总主MAC层对意味着完全相同GPU耗时” | 非均匀 GEMM 效率、QDQ、补偿、kernel数、权重bank与路由不同 | 同时报告等MAC、真实 FLOPs 和完整 FFN/端到端 latency；不共享旧UB1速度 |
| “视频端到端 FFN 1.55× 来源 RT-Lynx” | 其1.55×是线性层级报告，不是全 FFN | 实际 A100 native 边界测速，计入所有 FFN 组成，另外列端到端视频 |
| “SODA/SenCache 已经证明 FFN 有样本条件 oracle” | 其对象为敏感性引导 cache/token/block 等，不等同 FFN hidden 分组 | 显式测 native FFN 等预算 action matrix 才可做这项 claim |
| “训练路由网络就能胜过固定方案” | RB2有限bank后验收益小、PARE/E-DiT提供先例，未证明当前 Wan 动作族有 headroom | 先取 `best fixed` vs `oracle` vs `pre-W1 policy`；根据 gap 决定改动作还是改信息 |
| “后训练小微调和无训练是同一资源等级” | 训练条件决定可比性，Dynamic-in-Few-Step需要联合蒸馏 | 分别报告 frozen/校准/小训练、训练GPU小时及完整训练数据暴露 |
| “完成合成融合预检就有实际生产加速” | FlashInfer示范与A100当前HiF4量化路径/硬件差异，R10 synthetic成本不可泛化 | 数值守卫 + 真实 weights + 160FFN latency + 自由四步视频共同成立 |

## 5. 两项数学机制：可以保留，但不得当独立“首次发现”

### 5.1 最优静态—后验oracle—可得信息 policy 的精确分解

在相同输入、相同动作集合、相同成本和共同标定的损失上：

\[
L_\mathrm{fixed}=\min_a\mathbb E L(x,a),\quad
L_\mathrm{oracle}=\mathbb E\min_a L(x,a),\quad
L_\pi=\mathbb E L(x,\pi(\phi(x))).
\]

\[
\underbrace{L_\mathrm{fixed}-L_\pi}_{\text{可部署动态优势}}
=\underbrace{L_\mathrm{fixed}-L_\mathrm{oracle}}_{\text{动作的条件价值}}
-\underbrace{L_\pi-L_\mathrm{oracle}}_{\text{信息与策略 regret}}.
\]

这是精确的决策分解，不是新的定理。若 `oracle≈best-fixed`，应改**动作集合**（比如成对功能组而不是扩大旧R4 q）；若 `oracle` 大而 `policy` 差，应改**前置信息/可执行特征**；若 `policy` 优但视频差，应改**目标/传播模型**；若质量好但速度差，应改**布局与算子**。这样的失败归因很有研究价值。

对 **同一 step/block**，如果 `φ_AdaLN(x)` 对不同视频完全不变，那么在控制步/层后它的信息增量为零；它仍可以改变目标损失的度量，但不能独立降低 `L_π`。

### 5.2 有符号 downstream 反事实价值

固定 D-Q flow `F_D(x)`、静态 UB1 `F_0(x)` 和可执行动作 `F_a(x)`，设 `e=F_0-F_D, δ=F_a-F_0`，在同 native step 状态且共同输出空间下：

\[
G_a=L_0-L_a=-2\langle e,\delta\rangle-\|\delta\|^2.
\]

它是 SSE 恒等式，不依赖 Jacobian，也不是独立方法。**可投稿的区别只会来自**：怎样在 W1 前以廉价观察近似动作的 *有符号收益*，怎样构造比最优静态更优的真实 FFN action 集合，怎样以 QDQ/原生生成证明符号收益对质量有预测力，以及如何使用匹配校准与硬件来兑现。

跨 block 的 `F_11−F_10−F_01+F_00` 同样是有效相互作用恒等式；“有交互”并不自动是新研究成果。需比较 **交互效应能否改变最优动作的输入条件排序**，否则静态预算就足够。

## 6. 建议重排的论文研究主线：三个彼此可取舍的候选

### T1（首选主轴）：Pre-W1 content-conditioned, signed-value FFN *functional actions*

**新内容所在**：定义少量可量化、可打包、可精确减少 W1 与 W2 的功能动作，使得高影响通道的选择由**实际 gate / W2 / 下游方向**而不只由 C-score 控制；用 pre-W1 内容摘要估计这组动作的条件边际收益。

必须同时比较：`C-topK` / `R4已关闭邻域` / `same-budget signed static group` / `best fixed new action` / `offline oracle` / `u-only tiny predictor` / `u+attention conditional predictor` / `actual fused/native executor`。强先例：Prox、WIDE、Fast Forward、DiT-Pruning、RT-Lynx。实质性新意不能落在“代理只供mask”，而必须落在**可执行 signed value 动作**和**动态质量/成本联合验证**。

**第一决定性实验**：使用 OP2 真实原生 saved groups（现有有些是采样行而非完整M）在严格相同的 step/block 下，计算两种互不等价的可打包支持动作的 native masked gated SSE + full-flow SSE；比较每来源最好动作是否改变，保留真正配对来源。若后验 oracle 相对固定不足，转换为新的量化兼容组表示，而不是扩大 MLP。

### T2（竞争主轴）：Cross-block *marginal* compute value at fixed FFN budget

借用 PARE/E-DiT 的“按样本预算分配”作为强基线，**不声称跨 block 路由本身新**；我们的可区别对象是使用 FFN 层对 A/B 同 MAC、同输入路径、原生 BF16/HiF4 的**跨层 signed flow action response**，包含不同层间非线性交互。只有当 `joint policy` 比两个候选中**最优等预算固定分配**及 additive-layer policy 具有真正额外收益，才考虑作为主方法。

**第一决定性实验**：R12 162/162 与 216/108、108/216 同来源的原生 action SSE + 4 个开发自由生成先导；数据同时训练独立-layer sensitivity baseline、PARE-style latent/timestep router 和 signed interaction predictor；新增复杂性必须以跨来源结果证明。

### T3（联合系统支撑）：Router-feature/QDQ/gate *hardware co-design*

FlashInfer 已实现独立的 DiT gate/residual/LayerNorm 融合；WIDE、RT-Lynx 已体现 sparse kernel 与算法协同。当前贡献只能是当 FFN 动作 T1 或 T2 存在内容可预测优势时，证明**额外决策信息与 native HiF4 前处理共享内存遍历和算子 launch** 能够有效抵消选择费用；建立完整 FFN latency 和真实 video latency。若作为独立 kernel paper，需要更广硬件与模型及新 kernel 原语，否则作为 T1/T2 的方法一部分更合适。

**第一决定性实验**：单独测原生 unfused、FlashInfer 同类可用算子、FFNDS 特化融合（全量数值匹配）；至少在 A100 比较有/无 selector+QDQ/pack，不能把已有融合的 speedup 全归新动态算法。

### 共同的蒸馏机制基础实验

用 O-BF16/D-BF16/D-Q 在相同输入的 native action family 下比较：`V_cond`、support preference reversal、signed group-cancellation、pre-W1 信息预测 regret、free-rollout 的质量敏感性（自身轨迹单独评估）。其定位是 *为什么少步蒸馏模型仍保留或改变可执行条件冗余*，不是断言“蒸馏使 sparsity 消失”，也不是仅重复 OD1 top108 能量指标。

## 7. 保证视频质量的验证原则：和现有论文互补而非矛盾

研究目标必须在**真正自由运行的四步学生**上落地：

- **同源 D-Q**：相同 checkpoint、step schedule、输入图像、prompt、seed、精度/QDQ 和视频解码器。不得用 50-step 原始 O 的 latency 替换 4-step D-Q 的 baseline。
- **强静态**：不仅冻结 UB1，还有同一动作族内部最佳静态；PARE/E-DiT 的结构类别可实现相近可比基线时须给予资源匹配。UB1 原20通过、补充20 imaging 下降3.165968%，不能写作跨来源完成。
- **固定研究协议**：历史 calibration24、formal20、supplement20、TF2 来源已暴露，只能作探索；最终另收集真正未见的覆盖多类别来源，重叠审计 prompt/image/source SHA。
- **真实质量**：原设三个官方轴 aesthetic/imaging/motion 每个相对 D-Q 不下降超2%；另做 temporal luminance, motion-aligned background flicker, object micro-motion 和配对人工盲评，检验官方指标遗漏。局部 MSE、latent flow SSE、teacher-forced 和视频评分不可互换。
- **真实速度**：全局主 FFN MAC≤50%，包含 norm、modulation、实际 QDQ、router、W1/GELU/W2、修正、gate/layout 的完整 FFN≥1.5×；另外独立报告视频端到端延迟与 peak VRAM。任何自定义 kernel 先提供与同数值 reference 的位对照或明确数值偏差。
- **必要消融**：固定 versus动态、step-only versus input/attention、signed-flow label versus局部/C-score、独立层分配 versus联合层动作、调制作为特征 versus 作为静态风险度量、fused versus reference、BF16 versus HiF4、O versus D。

注意：原 `50% / 1.5× / 2%-per-axis` 是 FFNDS 自身严格的验收合同，不是其他论文都使用的统一指标。方法对比时同时给出质量—成本 Pareto 曲线，而不是拿不同论文的口径做无条件倍数排序。

## 8. 下一批支持实验：优先级由“查重后剩余的创新空间”决定

| 次序 | 实验及作用 | 主要可证伪机制 | 可用资料 / 关键新输入 | 导向 |
|---|---|---|---|---|
| **A0（立即）** | 从官方源码和真实 D-Q 运行器核对 `AdaLN(t,block,expert)` 与当前内容输入；生成 `feature_timing_ledger` | 反对“固定step下 AdaLN-alone 能区分视频样本”的错误前提 | 官方 Wan/LightX2V 源码与 OP2 manifest；若已有真实调制值先CPU哈希，否则一次最小原生 capture | 选择 u/attention 对应信号，不训练伪动态调制器 |
| **A1（可并行）** | 在保存 OP2 native 数据上构造并核对两组真正不等价、同 W1/W2 成本的 signed functional actions | 动作族本身是否产生新的条件价值 | OP2 保存的是256行样本，完整 downstream 需额外 native state；先 local signed Gram 检验 | T1 有 headroom →原生 full step；否则构造新动作坐标 |
| **B1（可并行）** | 固定层对 `162/162,216/108,108/216` 同原生输入 native flow + matched correction | R12 是否只是找到更好静态 layer schedule | R7D1 的 full-teacher 行为**不能**当等预算新臂；需真正构建该三臂 | T2 若 oracle>bestfixed 且跨来源稳定则训小策略 |
| **A2/B2** | pre-W1 信息可预测性：constant-step+latent / gate-weighted input / attention summary / history | 是否存在比先例更便宜、更解释性的 signed-action predictor | 使用 A1/B1 已取得的动作 loss table；来源交叉验证 | 完成信息缺口定位，不以 MLP 大小调参 |
| **Q1（早期）** | 已暴露 4 视频自由轨迹，同一动作与 dense/UB1，时空误差与官方三轴 | `native flow gain` 能否作为视频质量的可解释代理 | 固定已有 TF2 开发来源，真实视频+latent+明确 ROI 指标 | 修正监督目标；形成论文的真实质量机制证据 |
| **K1（并行）** | 现成 FlashInfer 风格 DiT norm/gate融合 vs FFNDS 原生符号等价 fusion vs 原生 unfused | 是否能节省选择器/quant I/O 所需预算 | A100 同实际 M/H/D，现有 HiF4-QDQ+BF16 GEMM；全 FFN profiler | 如果扩展 bank 开销太大，倒逼动作设计简化 |
| **D1（取胜方案）** | O/D-BF16/D-Q 对相同动作空间的条件价值几何 | 所谓“蒸馏改变了冗余”是否真正可验证 | OD1 两图数据是先导、非全原模型；扩大不同 block/来源 | 形成兼容学术先例的机制解释 |
| **Q2（冻结最后）** | 质量/成本最终外推：新来源首次官方评分 + 全 FFN/端到端计时 | 论文主张是否能相对于强静态同时赢得真实视频质量与加速 | 新冻结来源 + 同一候选 + 资源匹配 | 真正的主文结果，不是前期筛选代理 |

**行动选择**：先执行 A0 + A1 + B1 的最小原生准备，并行 K1；保留 T1/T2 两条独立竞赛。下一轮最有价值的判别量依次是：①新的可执行动作有没有比最佳 static 更多后验价值；②内容信号能恢复多少；③ native gain 能否在自由视频中保留；④融合可否为其支付成本。这个顺序比“先决定做跨层 router，之后再验证”更符合发现先于配方的研究原则。

## 9. 明确不能提前宣称的创新及可以争取的主张

**不能宣称（因为已有论文或仓库内证据）：**

- 首次使用动态 FFN 宽度、按 timestep/内容选择 block、proxy 只确定mask、FFN 后做误差补偿、按层敏感性分预算、token/tile 重排、AdaLN/LayerNorm/gate 融合、temporal smoothing、冻结模型的后处理剪枝；
- 完整 W1 后 mask 能让两个主 GEMM 同时省一半；用已知后验oracle的差值代替实际路由器成绩；用 RT-Lynx 单层1.55×作为原FFN系统速度；在相同时间步以标准 Wan AdaLN-only 实现内容自适应；
- OD1 已经证明蒸馏稀疏空间不变，或 R4/R9 已经排除了所有 functional grouping / 附加后训练方法。

**经真实结果支持后才能主张：**

- 已蒸馏四步视频 FFN 中，一组新的可执行动作有**稳定的输入条件边际计算价值**，而现有 C-score、静态宽度与现有 layer routing 不足以解释/捕获；
- AdaLN 给出了针对 output risk 的静态调制坐标，来自 **post-attention residual/FFN输入** 的廉价实时统计在该坐标下预测 signed action value，并给出可量化的条件信息增益；
- 跨 FFN block 的有符号传播使得同一计算预算下动作的**联合价值与单层 saliency 不同**，这种差异在真实视频质量与时空缺陷上可辨识；
- 对当前软件 HiF4-QDQ、BF16 核，dynamic selection 与 Q1/Q2/GELU/gate/补偿的联合融合形成**实测额外系统收益**，且已经排除现成 fused norm/gate 的解释；
- O/D 配对研究明确蒸馏如何改变（或不改变）可执行条件价值，并用数学解释为什么选出的动作在四步实际生成中有效。

## 10. 当前结论状态与仓库合并建议

- 已复核：上传资料包确有26篇论文，视频及蒸馏分类精细，但存在重要新增强竞争者 **PARE**，不能把上传目录当完备查重清单；E-DiT 已在包内，为 V2 P-B 的现成方法先例。
- 已复核：官方 Wan2.2 `time_embedding/time_projection` 和每block的6组调制参数来自 timestep 及固定模型参数，**标准同step、同层的 AdaLN 本身不提供视频内容信息**；当前 D-Q `t` 生成与广播还应通过 A0 实测固定。
- 已复核：FlashInfer 官方已有 DiT 残差/LayerNorm/scale/shift/gate 融合 API，不能单独声称 F1/F3 作为新融合模式。
- 保留：新 functional action（N04）、native signed counterfactual标签（N09）、O/D条件几何（N11）及前置信息增量/强视频验证的组合，为比较后真正值得争取的论文工作空间。N03 是有价值的原生判别实验，但不能被称为“首次内容条件跨block宽度”。
- 合并前需要修改 V2：N01/P-A 叙事、N03/P-B 先例定位、N06/P-C FlashInfer 对照；将 26 篇基础目录扩展加入 PARE、DiT-Pruning、SeaCache、BudCache 等，逐条记入“覆盖哪个动作+哪个模型+什么实验级别”。
- **本次仅只读审查与本地报告文件产出；不推送 GitHub，不调整 GPU 资格或正式质量门槛。**

## 11. 来源清单（优先学术一手来源）

- [Wan2.2 官方模型源码 — 时间调制与 block 计算](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py)。
- [PARE: Pruning and Adaptive Routing for Efficient Video Generation](https://arxiv.org/abs/2605.27336)。
- [Elastic Diffusion Transformer, ICML 2026](https://proceedings.mlr.press/v306/wang26dd.html)。
- [Dynamic-in-Few-Step](https://arxiv.org/abs/2607.06631)。
- [WIDE](https://arxiv.org/abs/2607.28418)、[Fast Forward](https://arxiv.org/abs/2602.00397)、[Prox](https://arxiv.org/abs/2607.27591)。
- [RT-Lynx, ICML 2026](https://proceedings.mlr.press/v306/cong26a.html)、[DiT-Pruning](https://arxiv.org/abs/2607.00927)。
- [DiffSparse, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5e66b9f710a8e38351da690f3806cba9-Abstract-Conference.html)。
- [SODA](https://arxiv.org/abs/2603.07057)、[SenCache](https://openaccess.thecvf.com/content/CVPR2026/html/Haghighi_SenCache_Accelerating_Diffusion_Model_Inference_via_Sensitivity-Aware_Caching_CVPR_2026_paper.html)、[TAPE](https://arxiv.org/abs/2605.17837)、[SeaCache](https://openaccess.thecvf.com/content/CVPR2026/html/Chung_SeaCache_Spectral-Evolution-Aware_Cache_for_Accelerating_Diffusion_Models_CVPR_2026_paper.html)、[BudCache](https://proceedings.mlr.press/v306/lei26j.html)。
- [FlashInfer 原生 norm/fused DiT API](https://docs.flashinfer.ai/api/norm.html)。
- 本项目的本地归档：FFNDS `docs/08/28/29/34/37/42/48/53/55/57/58/60/62/63/64/65`，`research/CURRENT_DECISION.json`，`research/20261022/POST_TRAIN_DYNAMIC_FFN_PAPER_PLAN.md`。

**下一次审稿查重注意**：在投稿冻结前应重新做新增论文检索，逐篇读取高相似论文的正文方法图与官方代码、补充材料和实现记录，尤其检查其 router 输入时序、真实 FFN width/action 及是否使用最终自由视频损失。本轮主要核对作者摘要、会议正式摘要、官方源码和用户提供的文献二手归档；“未见相同组合”不等于法律或学术意义上的全球排他性证明。