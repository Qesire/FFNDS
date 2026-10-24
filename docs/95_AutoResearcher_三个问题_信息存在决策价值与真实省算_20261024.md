# AutoResearcher：恢复三个根本问题——信息是否存在、可利用、值得省算（2026-10-24）

> 本轮是研究问题重新建模，不从 SVD、MoE、补算、R256 或通道剪枝开始。用户恢复的是**信息—数学—文献层面的探索**；research/20261024/EXPERIMENT_PAUSE.json 的 CPU/GPU/model experiments 与自动重试仍然暂停。所有新方法都保持 exploratory seed。

## 0. 原来三个问题（2026-10-23 信息论/数学路线）

- Q1 **信息是否存在**：某个 FFN/Block **之前**，已计算的 residual、attention update、AdaLN、之前 blocks、视频时空结构是否含有可预测后续 FFN 计算重要性的额外信息？
- Q2 **信息是否可利用**：提取、保存、传输该信息的实际成本，加上路由/打包/新算子的实际执行，是否仍优于同成本最强固定方法？
- Q3 **计算是否值得省略**：在原生 HiF4 QDQ、GELU、gate、residual 和**四步学生自由轨迹**里，去掉哪些运算才不会破坏最终视频？是否产生实测真实 FFN 时间/全生成时间的 Pareto 改进？

**原错误**：旧 Dense/UB1 可以得到“哪些行值得 dense 恢复”的标签，导致 Agent 越过 Q1/Q2，把一个可观察的历史动作冒充总体研究目标。上一轮直接列双侧 SVD/删组/条件专家也跳过了 Q1/Q2。本轮目标先确定信息的种类、决定价值和动作价值之间的因果关系，**随后**再从可行操作集合中生成算子；不以“完全不补算”替换“必须补算”成为新教条。

## 1. 研究对象：源码语义与可见信息时间线

冻结 Wan2.2-I2V-A14B LightX2V 四步蒸馏 D-Q / HiF4 BF16，原 checkpoint 5120→13824→5120，四步与高低噪声专家切换。核查原生源码与历史哈希：research/20261023/C0_coupling_offline_results.json 绑定的 .runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py、HiFloat4.ptq.adapters.wan_i2v；项目实现 tools/ub1_row_dispatch.py 和 tools/packed_ffn_reference.py 证明原生 Q1→W1→GELU(approximate=tanh)→Q2→W2 边界。Wan infer_ffn 在 norm2/modulation/Q1/W1 前已计算本 block attention 加法；后续写回到残差流，下一 block 再归一化并执行 attention/FFN。**精确的源状态变量和舍入位次以后续原生绑定为准，不用普通 pre-LN Transformer 想象替代源码。**

信息轴：Z0=step/expert/block/gate/latent (t,h,w) 几何、可直接引用的条件；Z1=当前已有的残差状态与 attention increment、该次 native normalization 如果可不重复归约则返回的 mean/rstd；Z2=Q1 之后 W1 之前已经存在的完整向量 q 与廉价 sketch(q)；Z3=已存在但需保存/搬运的 earlier-block statistics、source/latent-time temporal memory；Z4=部分 FFN 运算之后的特征（如果改变决策点才有效）。完全计算的 hidden h、teacher/dense output、未来 block/step/video 是**离线标签**，不是无成本在线信息。

注意：latent temporal bins 有21个，并非81个输出视频帧；历史 R17 的 step/block/routing 标签只能支持原动作家族。历史24源 OP2 的3840个文件头仅见 V1/BaseY/TeacherY/StudentY/b2/gate，**没有** pre-norm residual、attention increment、可比较的原生新动作有限价值向量；不得拿这些旧文件计算其实不存在的 Q1 条件信息增量。

## 2. Q1 信息是否存在？两个严格不同的目标

### 2.1 本地 FFN 的确定性信息上界（新研究分界）

固定实际权重/量化/数据类型及确定性执行路径 θ 后，完整 post-Q1 输入 q 充分决定当前 FFN hidden h 和输出 y：

$$h=f_\theta(q),\qquad y=g_\theta(h).$$

因此对任何之前已有状态 Z，**条件于完整 q 和 θ 时**

$$I(h;Z\mid q,\theta)=0,\qquad I(y;Z\mid q,\theta)=0.$$

这不是“上游特征绝对无用”。如果路由器因实际开销只能读取廉价摘要 s(q)，则 $I(h;Z\mid s(q),\theta)$ 可能非零；同样如果目标是 **省略某计算后的下游后果**（而不是本 FFN 的 hidden 本身），历史状态可能穿过 residual bypass 保留重要信息。

**可证伪预测**：若新增 pre-LN/attention 信息只在给定粗 sketch(q) 时预测 hidden，但在给定完整 q 时所有增益消失，说明它服务的是低成本本地计算预测；若对 downstream action value 在给定完整 q 时仍存在增益，则它识别了 FFN 之外的状态依赖。二者对应不同信息来源及部署动作，不能混为“显著相关”。

### 2.2 残差旁路造成的条件不充分

令原归一化前残差状态为 r（具体是否已包含 attention increment 要按同一原生 source 对齐），FFN输入 $q=Q_1(\operatorname{AdaLN}(\operatorname{LN}(r),c))$，候选低成本 FFN 输出 $F_a(q)$，下一残差状态 $r_a^+=\mathcal W(r,F_a(q),g)$。可能有 $r_1\neq r_2$ 而 $q_1=q_2$，此时 $F_a(q_1)=F_a(q_2)$，但 $\mathcal T(r_{1,a}^+)$ 和最终比较损失可能不同。

老 C0 的合成有限反例已验证：相同 post-LN BF16 bits、尺度1时 nextnorm SSE A=0.410533/B=0.155355（B优），尺度4时 A=0.0090549/B=0.0517386（A优）。这是**信息存在可能性的严格反例**，不是 Wan 频率、真实视频或 predictor gain。单个共享尺度乘法若只重标所有动作误差的相同系数，不会改变同 token 动作排名；C0 对此也做过反向控制。必须观察动作方向与残差矢量的关系，而非只比 norm magnitudes。

### 2.3 不要把互信息误认成决策收益

对特定且**冻结**的低成本原生动作集合 $\mathcal A$、动作损失 $L(a)$ 与实测成本 $C(a)$，定义条件 Bayes risk：

$$R(Z)=\mathbb E\big[\min_{a\in\mathcal A}\mathbb E\{L(a)+\lambda C(a)\mid Z\}\big].$$

额外读取信号 S 的零成本信息增量为

$$\operatorname{VOI}(S\mid Z_0)=R(Z_0)-R(Z_0,S)\ge0.$$

真正的 **NetVOI** 还需扣掉 signal extraction、memory traffic、selector、packing、kernel 低效及运行时不稳定性，按总部署实际 latency 衡量，且如 $\mathcal A$ 含整视频相互依赖动作，成本/损失必须联合计量，不能机械地把每行最优求和。

对于两个动作 a0/a1，记
$$D=L(a_1)-L(a_0)+\lambda[C(a_1)-C(a_0)],$$
令 $m(Z)=\mathbb E[D\mid Z]$。选择 a1 当且仅当 $m(Z)<0$，新增 S 的信息价值为
$$\mathbb E[\min(0,m(Z_0))]-\mathbb E[\min(0,m(Z_0,S))].$$

**关键机制**：额外信号即使明显提升对 hidden/SSE 的 R²，甚至与最优动作标签有互信息，若不改变实际动作判断或改变量不足以支付信息成本，就没有可利用价值。适用到不同动作族时需要分别计算，而不能把 Dense-vs-UB1 的最优行标签当作普遍上界。

## 3. Q2 信息是否可利用？最可能改变结论的三个信息入口

### H1 — 归一化丢失的尺度/旁路状态

已存在 norm 计算本来就计算均值/逆方差；当前 torch/native_layer_norm 原始 API 的下游只保留 y，C0 的 tools/prew1_norm_statistics.py 提供一个**同次 native 调用取 y/mean/rstd** 的版本，36个旧 synthetic CPU 条件下输出逐位相同，尚未部署到原生 GPU，也没有 latency/no-op 视频等价验证。

构造 [同一 q 与同一 metadata] 的不同 residual-scale/attention-innovation 任务：如果某些 action-value margin 会翻转，则 cheap read 的 added information 可能有效；如果没有翻转或只作统一损失缩放，则该路径不应进一步建昂贵 predictor。

### H2 — attention innovation 与前层稀疏因果父节点

前一个 attention update $\Delta_{\rm attn}$ 和上一层 FFN output 不只是与本层 q 相关，它们是当前 token state 的来源。优先从真实数据看 $\operatorname{proj}_{r^\perp}(\Delta_{\rm attn})$ 的方向、与残差对齐以及跨block增量是否能预测 **未来 action margin**，并比较在 q-sketch + norm stats 之后还有多少增量价值。

[Sparse Inter-Layer Dependencies](https://arxiv.org/abs/2607.11990) 已证明在研究过的语言模型中 neuron activation 可由少量上游 FFN/attention 父节点解释，是“寻找廉价因果父节点”的先例；它**没有证明**本 Wan checkpoint 同样稀疏、其父节点能预测最终视频价值，或其原本 attribution 能直接被拿来做 GPU FFN 路由。差异必须落在新的 action-value 决策而不是把 sparse predecessor attribution 换名字。

### H3 — 视频时空组织能否降低观察与决策成本

不同 latent temporal bins、spatial tiles、blocks 的动作价值可能有相关结构；和单行独立预测相比，同一稳定区域共享少量 metadata/selector 决策能显著降低物理路由/packing开销。旧 R17 仅证明某旧 dense-row 路由 mask 的 first latent bin 与 later20 差异巨大，和五主体源 mask 几何变动，**不是益处证明**。

[FIS-DiT](https://arxiv.org/abs/2605.11869) 提供了在 few-step 视频使用 frame-interleaved execution/coverage 的真正先例；[MOD-DiT](https://proceedings.mlr.press/v306/liu26bb.html) 从前一步已获取状态预测后续 mask。二者面向的具体操作包括 block/attention，不得直接声称 FFN 的 action-conditional info 继承其加速。我们要验证**超出已知 step/layer/tile metadata 的输入特异增量**，而不是换一种固定时空调度。

### 自适应信息采集而不是固定大型代理

设 Z0 为已有元数据与免费张量统计，Z1 为需扫描 q 的低秩 sketch 或跨block保存状态。可以先求动作 margin 的可信区间：已足够确定的区域直接选择对应低成本算子；**只有边界不确定处**才支付 Z1 的读取成本。其规则是“下一份信号能减少的条件期望决策后悔 > 下一份信号的实际费用”；不是补算策略，也不是固定二级 mask 规则。

这个思路本身属于价值信息/顺序测量经典决策理论。潜在科研贡献只能来自 Wan-specific 可用状态、可判别误差方向、实际物理分组与 full-video Pareto，而不是将 Bayes 公式声称为原创。

## 4. Q3 哪些计算值得省略：输出误差 ≠ 后续价值

固定 input 下 FFN 贡献可表示 $F(q)=b+\sum_gv_g(q)$（只在实际 Q2 独立组、舍入语义许可的范围用实数加性表示）。真正删除集合 $\bar S$ 后的当前输出扰动 $\delta_{\bar S}$ 不包含新增 dense 计算；其未来影响取决于当前完整状态 r 和后续 transformer/denoising 映射 $\mathcal T$：

$$\Delta_{\rm video}(S;r)=\mathcal T(\mathcal W(r,F_S(q)))-\mathcal T(\mathcal W(r,F(q))).$$

一阶仅作解释：若有限状态扰动 $\delta$ 足够小，
$$\|\Delta_{\rm video}\|_H^2 \approx \delta^\top J_\mathcal T^\top H J_\mathcal T\,\delta
=\sum_{g,h\in\bar S}v_g^\top K(r)\,v_h,\quad K=J_\mathcal T^\top HJ_\mathcal T.$$

与旧输出空间 Gram 不同，**K 是 state-dependent 的下游敏感几何**：它可能随 residual scale、attention bypass、step 及空间位置改变；同样局部大幅度在某些方向可无害，在另一些方向会触发背景、光照、微小物体变化。K 不能凭空从 C1 scalar label 得到，也不能把完整大幅度动作偷换为微小 Jacobian 结论。

已有 R7 对固定两个完整 teacher restoration 的旧四来源标量重算：交叉方向项 −11859.538 与有限传播项 +11657.095 大幅互相抵消，净交互只有 −202.443；**净值小不等于耦合小**。R7 不是新方法和全轨迹自由视频验证，但足以阻止继续以单个局部 SSE/组能量选择重要计算。

比计算全部 J 更可行的下一步：对少数**已冻结的全动作**做同前缀 finite intervention，保留真实输出差向量、下一block响应和最终可观测有限价值；再检查从 Z0→Z1→Z2 哪一步真正减少动作regret。选择动作前，**先研究其可执行图与代价**：可以是某一 whole64直接不执行、条件组支持、改变执行单元、或别的算法；补偿/补算只在独立证据支持时成为有成本对照。

## 5. 相互竞争的可证伪机制（不要事先决定方法）

| 机制 | 有区别的预测 | 最廉价下一判别 |
|---|---|---|
| M1 只有局部计算可预测，后续影响由 q 足够解释 | 加入 pre-LN scale/attention 后，本地hidden预测不变且有限动作排序也基本不变 | 来源留出；条件于 q-sketch 后进行增量决策风险比较；完整q条件零MI是数学基线 |
| M2 残差旁路状态改变后续计算价值 | 在相同 post-Q1 / 层时空坐标下，归一化前尺度或attention方向能翻转行动收益符号 | 冻结真实动作定义，再与完整同前缀下一block/step有限差配对；比较shuffle对照 |
| M3 跨层/时空共享信息有增量决策收益 | 前层创新/latent time region 信息改善输入相关动作选择，超过元数据固定策略且节省多行policy cost | 比较 Z0、Z0+norm、+attention、+history 的净VOI及真实 physical tile成本 |
| M4 静态方案已接近条件上限 | 增加信号只能预测 proxy/hidden，不能减少动作regret，或成本超过收益 | 同源同算子/预算 fixed support 与 Bayes actor；若无净改进保留静态更优候选 |

不是从“重要性分数高”推演“省某通道”，而是让不同的机制对**可观测的动作利润和物理成本**作出不同预测。M1/M4 并非研究失败，它们也能决定不要训练路由器。

## 6. 已有数据可用与不可用：停止重复错误实验

- 可直接使用：原生 FFN 源码/哈希与 QDQ、C0 scale/rank reversal synthetic counterexample、R7 旧已暴露有限标量、R16/R17 trace 几何、旧 C1 保存的 nextnorm 与 post SSE 及其各自 population 定义。C0 已完整复核3840文件头；不要继续搜索不存在的 pre-norm/attention 字段。
- 不能直接当作真实 Q1/Q2/第三问题证据：R17 的“选择 dense 的7917行”、6源 C1 top84 capture、有限两交换 0 headroom、R12 混合删 alpha/bias/R256、193对 R13 原始逐行 copy。它们只关闭其准确动作/特征/来源定义。
- 不能从旧数据得到：前置信号与**新**直接稀疏完整动作同样本绑定、相同 q 时的经验条件排名翻转频率、动量/attention创新与真实视频质量的联合分布、跨block因果作用向量、GPU 实际 VOI/分桶时间。**缺字段不伪造“已检验”。**

## 7. 实践准备：下一轮判别，不启动实验

- **I0 只读源码状态图**：记录当前 FFN pre-attention residual、attn_out、post-attn residual、norm2统计、gate、Q1 何时实际存在；标记 q 的 BF16 和 HiF4 边界，避免对不存在的张量“零成本读取”。已知代码 C0_norm_statistics CPU 36 case 不是原生 GPU no-op证明。
- **I1 预先冻结机制可比较的动作原型**：至少一对真正非补算、不同物理算子但可匹配成本的完成 FFN；由于 Q1 的 VOI 依赖动作集合，先定义动作，后采其有限效用。不能从 old C1/UB1 的恢复标签直接给新的无补算动作训练。
- **I2 同前缀分层因果采样**：固定真实 source-group × step × block × 21 latent temporal bins × spatial tiles，分来源和场景留出，不把31668行/81帧/技术重复当独立来源。先测 full-action post、nextnorm、nextblock，之后再检查自由轨迹和视频；每一层目标单独命名。
- **I3 incremental NetVOI 对照**：metadata-only → +已有 norm stats → +attention innovation → +q sketch → +cached upstream。每一步都应与**最佳同预算 static**、条件 signal shuffle、改变源的等预算 policy 比较；重采集/路由/packing费用不能忽略。看 action sign margin、regret、信号付费后的净价值，而非孤立 R²/MI。
- **I4 终局**：若确定有行动上的可经济利用信息，基于该信号寻找/设计实际 low-compute native FFN 物理算子，再取得独立授权后做完整四步自由 I2V paired 视频与官方质量、真实 FFN + generation wall。若在充分可用信息下仍无净收益，则把研究重心移到静态冗余结构/算子，而不是靠 Dense 恢复填补指标。

## 8. 近邻研究与新贡献边界

1. [Sparse Inter-Layer Dependencies of Transformer FFN Neurons, 2607.11990](https://arxiv.org/abs/2607.11990): preceding FFN/attention 中稀疏父节点预测下游 neuron activation；是 Q1/H2 的**信息来源**先例，不是 Wan native动作价值或硬件速度证明。
2. [FIS-DiT, 2605.11869v2](https://arxiv.org/abs/2605.11869): 蒸馏 few-step 视频做 frame-interleaved 结构计算，是 Q2/H3 的**部署粒度**先例；具体 frame/block执行并非 FFN hidden 路由。
3. [MOD-DiT, ICML 2026](https://proceedings.mlr.press/v306/liu26bb.html): 借用先前 diffusion 信息预测注意力模式；不同观测/动作/任务，不能直接拿其 attention 成功作为 FFN 预测收益。
4. [MiTA, NeurIPS 2026, 2602.01219](https://arxiv.org/abs/2602.01219): 将 attention 重表示为动态 fast-weight MLP 并按信息/表示/路由拆解；启发**先转换分析坐标再选算子**，不声称 attention 与 Wan GELU FFN 相同。
5. AutoResearcher [harness @5bd7ec53](https://github.com/Qesire/AutoResearcher/blob/5bd7ec537ae63f2b6a6dd579f0e886aff58a177c/core/AGENT_HARNESS.md): source-grounded concise evidence packet、PRD E/T/R、NetVOI/conditional margin 判别、严格区分证明与 native evidence；不是重复掏一份现成算法清单。

论文层面的可探索增量是“**cheap upstream causal information → downstream finite action-value → viable FFN physical operator under cost-quality constraints**”的关联机制。单独的VOI公式、LayerNorm尺度或跨层相关性，都不是原创方法。正向查重欢迎借用已有模块，仅拒绝在完整机制相同时宣称独立发明。

### Evidence classification & pause
- Proven analytic: frozen deterministic FFN local sufficiency given full q/θ; Bayes conditional risk/VOI identities; first-order K formula with stated differentiability/locality.
- Archived **synthetic evidence only**: 4D post-LN BF16 collision / nextnorm action rank reversal; source C0.
- Archived real native **narrow scopes**: R7 old full teacher restoration scalar interaction; R16/R17 routing and partial video; old C1 label capture.
- **Unverified/open**: Wan true positive NetVOI, conditional upstream action predictability, new no-recovery operator quality, new native latency/free-video Pareto.
- Experiment status: ALL_CPU_GPU_MODEL_EXPERIMENTS_PAUSED; no new forward or video generated; no automatic followup.
