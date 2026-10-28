# QCFB / Q-STAGE：命题检验、完整推导与可执行判别 v1

2026-10-28。基于 FFNDS `8b62fd282e5731ab8824d20c0035907cc1d67ec5`；研究身份 **ALG + SRC复核 + LIT + OPEN**。使用 mechanism-research、problem-reformulation、web-research-discovery 与仓库领域 Skill。研究入口仍是[统一前沿](INTEGRATED_QMETA_FUNCTION_EXECUTION_FRONTIER_20261028.md)；M-B B0/B1 优先级及 ACTIVE 不变。本轮只执行合成代数检查和文档/图验证，未调用 Wan、HiF4 CUDA、训练、GPU、视频或计时实验。

## 1. 判定与这次新增的科学结构

用户提出的组束抵消、量化误差交互和动作依赖观测，均可形成严谨的条件性数学命题。新的研究对象可表述为：**在前置可用信息条件下，选择具有低联合函数风险、且属于真实可执行动作集的完整组集合。** 数学支持这一对象；真实 Wan 中可泛化、可提前预测、具有净速度收益的组束仍为 OPEN。

本次补全六处决定性条件：

1. 完整64组的 Q2 可交换性只保证保留的 Q2 输入相同时的组局部计算；不自动保证改变 shape 后 W1/W2 的逐位结果相同。
2. 条件风险公式要求 mask 是决策信息的函数；后验 oracle mask 不能代入同一个前置信息风险公式。
3. 条件 Gram 是二阶矩，必须包含条件协方差。量化项还要求 **C、b、Γ 联合可实现**，不只 Γ 为 PSD。
4. 负交叉项只表示相对于单删误差之和的折扣。成对匹配不穷尽多组抵消；逐组贪心也未必有效。
5. 行风险求和只适用于行可分读出。真实后缀若混合 token，必须使用含跨行项的整状态读出。
6. Q-STAGE 后段优化的是观测条件下的期望风险，不能在观测前挑选知道隐藏状态的最优动作。

## 2. 原生组分解：源码、数值与标签的边界

对象为 D=5120、H=13824=216×64、四步 high/high/low/low、每 expert 40 blocks。令 x 表示**固定的实际 post-Q1 输入**，W1/W2 使用既有解码权重，所有 bias、Smooth、gain/alpha、cast 均保留原来位置；将实际 post-Q2/post-gain/cast 的64维向量记为 h_j(x)，令 v_j=W2_j h_j。

对固定数值 h_j，定义 decoded 实数层：

$$y_Q=b_2+\sum_{j=1}^{216}v_j,\quad y_{Q,m}=b_2+\sum_j(1-m_j)v_j,\quad y_{Q,m}-y_Q=-Vm.$$

证明只是 W2 按输入列分块的线性展开。b2 加一次，省组不删 b2。若用未量化原始输入或权重代替这些量，便改变了被证明的算子。

**SRC 已复核**：[保存的 CUDA 源](../../upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu) `hifx_quant_cuda_inner` 第229–287行仅访问 `thread_idx*64+i`；4坐标最大值→8坐标层→64坐标层的尺度和舍入都在这一组内。该文件 SHA256 为 `07120a5d2519e3ed3c8782acb8897a44e31103f93dd84d353ae7bb86b043e00c`，本轮重算与[R12记录](../20261022/R12_HiF4_group_separability.json)一致。

因此，对完整有序组选择矩阵 S，在组边界、保留输入值、recipe和逐组参数不变时，Q2(Sh)=S Q2(h)。W1选输出行、GELU逐坐标、Smooth2逐坐标/对应组选择，便形成有条件的早删闭包。R12 的8个保存输入参考案例及4个合成hidden pack案例是历史记录；其中保存输入是 post-Q1 256×5120，并非8份真实完整 FFN hidden组输出。R12亦记录组内交换0/4后401个BF16元素不同。**本轮没有重新执行 R12 参考模型或原CUDA。**

原生删组误差应写成 `−Vm + η_m`：η_m包含改变W1形状导致的保留行差异、Q2阈值跃迁、W2归约/输出cast和门控残差舍入。整个误差经读出后为 `−Dm+ξ_m`，于是

$$|\|-Dm+\xi_m\|^2-\|Dm\|^2|\le2\|Dm\|\|\xi_m\|+\|\xi_m\|^2.$$

R12没有给出ξ_m的统一小上界；尤其量化阈值处，微小上游改变可产生离散跳变。future neutral/selected-action 原生对照需分别测量这部分。

## 3. 联合省略的严格对象与多组自由度

在一个冻结state中，令 g 是该次门控，P是对所有比较动作固定的线性读出，D=[d1,…,dG]，d_j=P diag(g)v_j。P可随state变化但在该state的反事实比较中冻结；若来自suffix Jacobian，它是离线监督，且是局部一阶近似。

$$L(m)=\|Dm\|^2=m^\top D^\top Dm.$$

两组令 a=||d_i||、c=||d_j||，ρ=〈d_i,d_j〉/(ac)，则

$$L_{ij}=a^2+c^2+2ac\rho=(a-c)^2+2ac(1+\rho).$$

所以接近零要求幅值匹配及ρ接近−1。例 d_i=100、d_j=−1，有负交叉项，但联合损失9801仍巨大。比删任意一个都更好还需 `2〈d_i,d_j〉<−max(a²,c²)`。它不是“只要负相关就安全删”的规则。

三组 `d1=(1,0), d2=(−1/2,√3/2), d3=(−1/2,−√3/2)` 各单删和每个双删的风险都是1，而三组联合为0。这证明仅搜索近乎互消的pair会漏掉有效bundle；二次型仍足以描述该三组作用，不需要凭此引入三阶统计。

增加一个省略组 j 的边际风险是

$$L(S\cup\{j\})-L(S)=\Gamma_{jj}+2\sum_{i\in S}\Gamma_{ij}.$$

其符号与已有集合有关。混合正负交叉项时，集合函数一般既非单调，也无普适的次模贪心保证。Γ虽PSD，二元约束及硬件合法集合仍是组合选择；可先比较小型合法动作字典/枚举与同成本diagonal基线，不把连续松弛当成部署解。

## 4. 前置信息条件化：可测性、波动与不可辨识

令 Z 为在 pre-W1 实际获得且计费的信息（Q1真实payload、step/expert及已存在state等），并要求 m=π(Z)。二阶矩存在时：

$$\Gamma(Z)=\mathbb E[D^\top D\mid Z],\quad
\mathbb E[\|Dm\|^2\mid Z]=m^\top\Gamma(Z)m,\quad\Gamma\succeq0.$$

随机策略可额外以独立随机种子R条件化；若seed与隐藏操作数相关，就需纳入真实信息。**反例**：Z恒定，标量d以等概率为0或2，oracle选择m=1[d=0]。真实省略风险为0，而Γ=2；在m=1分支代入 `m²Γ` 得2。缺的是决策条件信息，不是平方展开。

设 μ=E[D|Z]，则

$$\Gamma=\mu^\top\mu+\mathbb E[(D-\mu)^\top(D-\mu)\mid Z].$$

取 S=±1，d1=S,d2=−S，均值皆为0，Γ却是 `[[1,−1],[−1,1]]`。预测平均方向后相乘会错误地把单删风险也报为0。`UᵀU+diag(σ²)`是完整条件二阶矩的受限模型，低rank/对角结构需要校验，不能默认为无损。

静态弱组和条件抵消组可以分别来自稳定均值、共同波动或少数来源巧合。对固定mask，E||Dm||²=||E[Dm]||²+E||Dm−E[Dm]||²；平均输出接近0不够，第二项也要小。跨来源/step稳定性必须在独立image/source簇检验，不能把同图多row/site当独立证据。

G12已给出的signed histogram碰撞说明：即便含正负码计数，缺失坐标配对仍可能无法辨识点积。因此“加符号草图”是可检验候选，不是充分统计定理。可用最优行动价值来定义需求：

$$\mathrm{VOI}(Z\to Z')=\min_m E[L(m)|Z]-E[\min_m E[L(m)|Z']\mid Z]\ge0,$$

其中Z'细化Z、动作集相同且暂不计信息价格；不等式来自细信息策略可以忽略新增信息。加入producer、读码、预测/求解/dispatch成本后，净价值可以为负。oracle差距不等于cheap-Z可学收益。

## 5. 量化误差参与决策：定义、联合PSD与目标分离

固定相同输入、state和读出，定义 `e_Q=P diag(g)(y_Q−y_ref)`，`C=E[||e_Q||²|Z]`，`b=E[Dᵀe_Q|Z]`。y_ref必须登记究竟只去掉Q2、去掉哪些量化，还是独立高精轨迹；最后一种不再是纯当前FFN量化误差。

$$e_{total}=e_Q-Dm,\qquad R_{ref}(m|Z)=C-2b^\top m+m^\top\Gamma m.$$

**证明**：展开平方后利用m对Z可测，将m提出条件期望。符号约定若将e_Q定义成ref−Q，线性项也要随之反号。

相对量化dense的改善条件为 `2bᵀm>mᵀΓm`。例 e_Q=1、d1=1，删除可将ref误差1降到0；但相对原量化dense的改变量为1。保持dense目标时风险是 `R_dense=mᵀΓm`，不能借负b项取消该预算。实际方法可以使用 `min C_native(m)`、`R_dense≤ε`、同时报告R_ref的约束形式；ε由预先冻结的任务指标与原生结果校准。若同时选择bitwidth/量化recipe q，则D_q、e_Q,q、Γ_q、b_q和成本均随q改变，固定一套Γ、b的式子还不是完整精度联合分配算法。

**比Γ PSD更强的必要条件，也是新的估计器设计入口**：

$$H(Z)=E\left[[e_Q,D]^\top[e_Q,D]\mid Z\right]
=\begin{bmatrix}C&b^\top\\b&\Gamma\end{bmatrix}\succeq0.$$

因此 `b∈Range(Γ)`，`C≥bᵀΓ†b`（广义Schur补）；所有binary或连续m的预测平方风险都应非负。例C=Γ=1却b=2会给m=1风险−2，虽Γ PSD仍不可能来自真实二阶矩。可用共同因子 `Ĥ=AᵀA+diag(s²)` 联合预测C、b、Γ，再把 `(1,−m)ᵀĤ(1,−m)`当风险；这保证矩一致性，不保证校准准确或在线低成本。

若每个候选最多删k组，且对同一Z有 `||Γ̂−Γ||op≤εΓ`、`||b̂−b||2≤εb`，那么忽略对所有动作相同的C误差，

$$|\hat R(m)-R(m)|\le k\epsilon_\Gamma+2\sqrt{k}\epsilon_b=\delta_k.$$

在相同合法动作集上精确最小化估计风险，得到 `R(m̂)−R(m*)≤2δ_k`；近似求解再加其优化误差，成本估计误差也需加入。这是条件性regret界，不是现有预测器的误差证书。

## 6. row tile和真实后缀：一次关键纠错

若P为逐row读出且目标为各row平方和，共用mask的tile风险是 `Γ_T=Σ_r ω_r E[D_rᵀD_r|Z_T]`，ω_r≥0。应先合成每一row的组贡献，再平方、再汇总；不能先将不同row相加再平方。两行d=(1,−1)、(1,1)，第一行双删0，tile双删4，负交叉项已消失。

若P跨token混合，用整状态注入 `u_j=vec(diag(g)V_j)`，d_j=P u_j，则

$$\Gamma_{ij}=E[u_i^\top P^\top P u_j\mid Z]
=\sum_{r,s}E[u_{r,i}^\top (P^\top P)_{rs}u_{s,j}\mid Z].$$

只有度量按row分块时才退化为行Gram求和。例两行注入(1,1)，identity平方和2，而P=[1,1]的读出平方4；注入(1,−1)读出则0。真实suffix可能还耦合不同tile，此时独立优化每tile只是近似，不能把tile风险简单相加成全局风险。

对非线性后缀F，`F(u+δ)−F(u)=J_uδ+r(δ)`；若该邻域Jacobian为L-Lipschitz，`||r||≤L||δ||²/2`。于是平方读出误差相对线性项的差不超过 `L||Jδ||||δ||²+L²||δ||⁴/4`。BF16程序需独立finite验证；J零响应不意味着大幅度有限干预零响应。若跨block/step同时省算，各组还会改变以后输入，局部固定state的二次型不能直接加成自由视频总损失。

## 7. Q-STAGE：正确的动作依赖信息递推

对Z1，先选a1，再生产实际Z2(a1)；后段只允许使用I2=(Z1,a1,Z2(a1))。精确的两阶段Bayes决策写为

$$V_1(z_1)=\min_{a_1}\left\{C_1(a_1,z_1)+E_{Z_2|z_1,a_1}\left[\min_{a_2\in\mathcal A_2(a_1,Z_2)} E[L(a_1,a_2)+C_2\mid z_1,a_1,Z_2]\right]\right\}.$$

第二个条件期望不可漏掉，除非Z2已经完全决定损失。反例：隐藏状态S为均匀0/1，Z2恒定，末段动作a2∈{0,1}，损失1[a2≠S]。真实最优风险1/2，`E[min_a L(a,S)]=0`却偷用了S。

三类a1的区别：

| a1 | 后段可用Q2 | 早期可省计算 | 数学地位 |
|---|---|---|---|
| 完整hidden64删组 | 仅保留组；R12条件下与原组相同 | W1对应输出、GELU/Q2组、W2对应K组 | 条件闭包 |
| W1输入K-panel选择 | 必须重新产生，多个预激活/组scale变化 | 部分W1乘积；其它部分另计 | 一般不交换 |
| M-A/M-B学生替代 | 学生自己的hidden与Q2 | 取决于真正替代算子 | 新recipe需另证 |

甚至在whole64闭包中，先删组也**删掉了其信息**：被删组Q2不能作为晚期免费观测。早期动作在省算与继续观察的价值之间做选择，形成主动信息获取问题。对whole64纯省略，若m1为早删、m2仅在保留组上晚删，则总mask `m=m1+(1−m1)⊙m2`；最终decoded输出相同的总mask可有不同的获取信息和W1成本，故仅优化总稀疏率会遗漏Q-STAGE的意义。

## 8. 先例审查与贡献边界

本轮沿前沿文档的官方链接追到原论文方法段，补充G14稳定映射，详情见[原文ledger](../literature/20261028/QCFB_PRIMARY_METHOD_AUDIT.json)。作者代码未逐分支核验，外部结果不作为Wan实测。

| 先例 | 本轮原文核对 | 与QCFB具体关系 |
|---|---|---|
| FANG，AAAI26 / arXiv:2512.23014v1 | Method Eq.(7)按语义context加权、分组重建；附录Eq.(11)组内加权Hessian | 函数组、重建及二阶交互已有先例；不能声称此前只会独立Top-K。QCFB待验的是pre-W1条件跨完整量化组的联合删算 |
| Learnable Permutation，arXiv:2601.22980v1 | Methods Eq.(1)–(3)：相邻权重/激活协同排列，cost matrix与匹配、N:M优化 | 可学习排列已有先例；HiF4需保持整个64组及内部层次，不能照搬任意channel排列 |
| QuantSparse，ICLR26 / arXiv:2509.23681v4 | §3.2–3.4：attention distillation与量化后二阶残差时序恢复 | 联合量化/稀疏已知；FFN完整组与提前信息是待检验差异，不能把平方交叉项称新发明 |
| Veda，ICML26 / arXiv:2605.30325v1 | §4.1 Eq.(4)max-pooled teacher tile分数、§4.2 head-aware tiling、§4.3 skip kernel | 从可学习观测到可执行tile已有闭环先例；QCFB需证明signed bundle目标有额外价值 |

可主张的**候选差异**是“Q1前置信息→联合负交互whole64动作→数值合法的W1/GELU/Q2/W2真实省算→四步动态行为”；四个接口必须有配对证据。尚未完成全球优先权证明，当前更准确的称呼是可证伪机制候选。

## 9. 对齐用户T0–T3与现有五项任务

用户四阶段和旧crosswalk五任务粒度不同，保留旧ID并显式映射，避免覆盖历史：

| 用户阶段 | 原机器ID | 最小判别及目前状态 |
|---|---|---|
| T0 联合结构 | T0_Q2_COMMUTE + T1_SIGNED_BUNDLE | 数学/保存SRC成立；真实same-source/group方向资产待补；现有能量汇总不能恢复交叉符号 |
| T1 提前预测 | T2_Q1_DECISION | 同source/rowtile的真实Q1码与方向标签配对；比较static joint、diagonal、RMS/step、同成本head和sign-shuffle；NOT_MEASURED |
| T2 物理省算 | T3_ACTION_DEPENDENT_Q2 + T4_PHYSICAL_VIDEO的kernel子阶段 | 整组闭包原生neutral、预pack/dispatch/rowtile共用、实际W1/W2执行量及净墙时；NOT_MEASURED |
| T3 全视频 | T4_PHYSICAL_VIDEO的rollout子阶段 | 与固定M-A0及数值匹配dense在同预算比较，全160调用与完整四步source级质量/运动/显存/墙时；NOT_MEASURED |

仅合成脚本通过不能上调上述Wan状态。现有R12证明量化边界、CAL24能量profile证明特定统计，均不提供完整Γ、b或cheap-Z预测效果。研究资产需具备真实source/seed/step/expert/block/row坐标、W1/W2/recipe哈希、postgain组向量、gate和固定P；b另需同输入高精参考。列出producer缺口即可，不将未来采集冒充现成数据。

**可执行新构造（OPEN）**：少量硬件合法whole64动作字典，共同PSD二阶矩预测器，对每个动作计算风险及实成本；与同字典diagonal、static joint和matched-cost普通输入预测器比较。动作字典一开始即包含多组组合，以避免pair-only盲点。先证明heldout来源上存在联合选择收益，再判别Q1是否有净信息价值；成功后才投入原生执行器/全视频，M-B B0/B1继续原优先顺序。

## 10. 本轮可复核产物

- [合成oracle](../../tools/qcfb_algebra_oracle.py)：保留旧检查，新增可测性、均值Gram失败、负相关不足、多组抵消、联合PSD、非线性有限差、跨行读出和阶段信息泄漏等反例。
- [本轮静态与代数检查结果](QCFB_AUDIT_VALIDATION_20261028.json)：注明执行范围；不含模型结果。
- [独立数学审阅](../independent_reviews/QCFB_20261028/INDEPENDENT_MATH_REVIEW.md)：真实子Agent对基础版本的审阅，非正式proposal投票。
- [机器先验/任务表](../agent_protocol/QUANT_FUNCTION_EXECUTION_FRONTIER_CROSSWALK_20261028.json)：增加本轮合同及阶段映射；历史实验与授权未修改。
