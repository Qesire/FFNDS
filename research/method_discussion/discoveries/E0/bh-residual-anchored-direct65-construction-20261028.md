# B-H：以残差为锚、受 head 可达流形约束的 direct65 学生

2026-10-28 · v1.1 独立构造 / **ALG + SRC；训练、数值拟合、矩阵实验和部署均 OPEN**。依据 main `d61557b` 的[统一方案 v2](../../../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)及[B 快照 v2](../../../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)。本次只做符号推导与整数成本计算，未执行 QR、GEMM、模型、数据分析或拟合；未登记运行、未填写独立同行支持。原稿 SHA `f3de5085facf9331e2b9b9ce39123a025e4000e3f4a505ac967ed964730b6ff7` 经[独立可达性评议](b-head-direct65-feasibility-independent-20261028.md)和 root 指出后，将充分性所需的 centered-kernel 维数条件澄清为 D>rank(K)+1；当前 D5120/rank≤64 对象仍满足。

**构造性发现**：最后 block 的学生可以直接预测“相对于已有残差 r 的64维 head 增量＋一个非负方差修正”，同时省去构造完整5120维 y/u。但是这65个输出受到一个此前未显式使用的约束：head 坐标的最小可达方差是一个64维二次型。利用这个约束可以构造始终落在连续 LN/head 可达集合内的学生。与之竞争的“先预测低秩 y，再只算统计”也合法，但精确 residual cross 项会重新引入 D×rank 投影，通常不会额外减少主 MAC。

**操作与范围**：M-B；O-B 替代末层读出函数，O-E 负责未来同状态验证。仅覆盖每个 expert 的 block39；其余39层保留原计算。合法在线输入为原 post-Q1 x、W1 前已经存在的 r、原 gate/time；teacher y/u/head 只作为离线标签。原 Smooth1/Q1、条件编码、attention、前39层、scheduler 与 VAE 均付原价。直接读出需要末层 FFN＋写回＋LN/head 的联合替换入口，仅替换 W2 的 hook 无法实现本图。

## 1. 残差锚的精确连续身份

逐 token 记 D=5120、p=64、P₀=I−11ᵀ/D，固定当前 head 调制 A=diag(1+scale)，T=WA，**K=TP₀**，c₀=W shift+b。定义

\[
q(w)=Kw,\quad v(w)=\|P_0w\|^2/D,\qquad
F_{\mathbb R}(w)=q(w)/\sqrt{v(w)+\epsilon}+c_0,\quad \epsilon=10^{-6}.
\]

K1=0，因此均值已被编译进64个线性读出，不需另预测 mean(u)。令 δ=D_g y、a=P₀r、u=r+δ，则

\[
q(u)=q_r+KD_gy,
\]
\[
v(u)=v_r+\frac{2a^TD_gy+\|D_gy\|^2-(1^TD_gy)^2/D}{D}.
\]

第二式保留 residual cross 与 gate 后二次项；忽略它们、只预测 head 线性投影会遗漏统一分母。这里的 y 包含原 b₂。将 b₂ 移到 anchor 只在此实数模型中成立，原 BF16 bias/GEMM/乘加顺序须另核，因此主构造直接使用原 r。

SRC：原 `infer_ffn` 在 norm2/AdaLN/Smooth1/Q1 前已有 attention 写回后的 r；post_process 执行 gate 后原 BF16 加法。block39 后立即 final LN→时间调制→64维 head，无下一层 attention。实际 `post_infer` 在 unpatchify 后转 FP32 flow。对应源码及 hash 已在[B 源码审计](b-final-block-head-source-audit-20261028.md)和 B 快照绑定；此处不改变其原生身份。

## 2. direct65 的新约束：head 坐标与方差共同可达

令 G=KKᵀ。对任意中心化状态 z，Kz=q。最小范数解 zₘᵢₙ=KᵀG⁺q，因此

\[
q\in\operatorname{range}K,\qquad
v\ge v_{\min}(q):=q^TG^+q/D.\tag{1}
\]

在 **D>rank(K)+1** 时反向也成立：ker(K)∩1⊥ 的维数为 D−1−rank(K)>0，可增加任意所需非负能量。当前维数至少5055，故式(1)对本对象是连续可达性的充要条件。若 centered kernel 为零，方差必须恰等于最小值，不能改用这个不等式充分条件。若学生任意独立预测 q 和正 v，可能输出任何真实 LN/head 都不可达的结果；“正方差”本身不足。

以 r 的 **head-null 方差** 为锚：

\[
s_r=v_r-v_{\min}(q_r)
=\|\operatorname{Proj}_{\ker K}P_0r\|^2/D\ge0.
\]

一个明确的65输出参数化为

\[
\widehat q=q_r+\Pi\Delta q_\phi,
\quad \widehat s=\max(s_r,\tau)\exp\ell_\phi,
\quad \widehat v=v_{\min}(\widehat q)+\widehat s,
\quad \widehat o=\widehat q/\sqrt{\widehat v+\epsilon}+c_0,\tag{2}
\]

Π 为 range(K) 的正交投影；若未来实测 K 满行秩，Π=I。**未计算实际 rank，不能现在删除此条件。** 固定 τ=ε·2⁻¹⁶，只为允许 sᵣ=0 时仍有有限比例标签及可增长的 slack；它不是数据调参。训练标签可定义为 Δq*=q(u)−qᵣ、ℓ*=log[max(sᵤ,τ)/max(sᵣ,τ)]。零增量、零 log 比保持 residual baseline，仅在 sᵣ<τ 时有显式 floor。对真实标签的 floor 影响满足

\[
\|F_{\mathbb R,\mathrm{floor}}-F_{\mathbb R}\|_2
\le \|T\|_2\sqrt D\,\tau/(3\sqrt3\epsilon),
\]

由 Δv≤τ 及 maxᵥ≥₀ √v/(v+ε)³ᐟ²=2/(3√3 ε) 得到。没有将此连续界移植为 BF16 误差界。式(2)还保证

\[
(\widehat o-c_0)^TG^+(\widehat o-c_0)
=D\,v_{\min}(\widehat q)/(\widehat v+\epsilon)\le D.
\]

**更简单的竞争参数化**是 v̂=max(vᵣ,τ)expℓ，q̂=qᵣ+Δq，标签为 log[max(vᵤ,τ)/max(vᵣ,τ)]；它保留直接 v/vᵣ 解释与更低成本，但不保证式(1)。把两者区分，可以判断改进来自低维可预测性还是仅来自可达约束。exp 的溢出、G⁺ 的条件数、sᵣ 的消减误差是未来数值 gate；不能通过静默 clamp 后继续称精确恒等。

## 3. 两个不同的合法在线计算图

**H-R65：预测增量与方差 slack 比。** 首先付费计算 qᵣ=Kr 和 vᵣ，组成 ηᵣ=[qᵣ/√(vᵣ+ε), log(vᵣ+ε)]∈R⁶⁵；然后

\[
z=Q_{2,\phi}[\operatorname{GELU}_{tanh}(U_xx+U_\eta\eta_r+b_{step})],
\qquad (\Delta q,\ell)=Vz+a_{step},
\]

最后执行式(2)。h 为64的倍数。当前原生时间调制/gate 由 step/expert 的已知条件产生，固定4步可以用 step 条件 bias；若将来 gate/调制具有新的源依赖，必须显式输入并计费其投影，不能静默沿用固定表。学生使用自己的窄激活及 Q2，**没有 teacher V2/y 输入**。U/V、固定 Smooth2 子资产及训练版本均尚未产生；不能把该结构图标成可运行权重。

η 是一个有意受限的 residual 摘要。合法的强竞争图是额外 Uᵣr 的 full-r 分支，用同一隐藏宽度；它能观察摘要遗漏的量化信息，但再次付 D×h MAC。把 residual 视为“已在内存里”不等于其学习投影免费。

**H-LR-stat：先预测低秩 y，但不展开 D维输出。** 令 ŷ=Φcφ(x)，Φ∈Rᴰˣᵏ；同样以合法 x→h→k 生成系数。固定当前 gate 后离线编译

\[
B=P_0D_g\Phi,\quad L=KD_g\Phi,\quad H=B^TB.
\]

在线另外计算 t=Bᵀr，则

\[
\widehat q=q_r+Lc,
\qquad \widehat v=v_r+[2t^Tc+c^THc]/D.\tag{3}
\]

式(3)在实数中恰等于该低秩 ŷ 的统计，方差自然非负。有限精度 cross/Gram 消减仍可能为负；如需平方根/QR 型稳定求值，须另付分解及在线代价。它与先 BF16 materialize ŷ、原 gate/add/LN/head **不是逐字节等价图**。

**成本判别**：省掉 Φc 的 Dk 展开，却需付 Bᵀr 的 Dk，二者主 MAC 正好抵消；再多 Lc 的64k和 Hc 的 k²。故 H-LR-stat 对同 h,k 的 materialized B-F＋head，多 **64k+k² MAC/token**，可能主要节省 D维中间量/归一化访存，并不天然减少 GEMM 工作。

## 4. 明确的主算术、存储与启动成本

统一比较“**一个末层 FFN＋head 投影**”，D=5120、M=31668；1MAC=一次乘加。以下是逻辑稠密矩阵 MAC，不是 Tensor Core 实际 tile 指令、QDQ 或墙时。H-R65 使用65维 η 分支及式(2)，满行秩时已加2个64²二次型；rank不足还加64²投影。保留原 pre-W1 Smooth1/Q1，每个学生另有其真实 GELU/Smooth2/Q2；Dense/A0/B-F 另有原 D维 gate/writeback/LN/head 调制。所有图的 reductions/elementwise 均另算。

| 图 | MAC/token | 全 M MAC | GEMM 数 |
| --- | ---: | ---: | ---: |
| Dense H13824＋原 head | 141,885,440 | 4,493,228,113,920 | 3 |
| A0 h6016＋原 head | 61,931,520 | 1,961,247,375,360 | 3 |
| B-F h6016,k512＋原 head | 36,831,232 | 1,166,371,454,976 | 4 |
| B-F h1024,k64＋原 head | 5,963,776 | 188,860,858,368 | 4 |
| H-R65 h1024 | 5,711,872 | 180,883,562,496 | 4＋2小 GEMM |
| H-R65 h6016 | 31,919,872 | 1,010,838,506,496 | 4＋2小 GEMM |
| H-full-r h1024 | 10,888,192 | 344,807,264,256 | 4＋2小 GEMM |
| H-full-r h6016 | 62,330,752 | 1,973,890,254,336 | 4＋2小 GEMM |
| H-LR-stat h1024,k64 | 5,971,968 | 189,120,282,624 | 6，合并 r 投影后5 |

H-R65 公式为 D·64+h(D+65+65)+2·64²；full-r 为 D·64+h(2D+65)+2·64²。**full-r h6016 已略高于 A0＋head**。两个二次型可堆成2M行后一次小 GEMM，但另有布局/暂存。65维 η 和65输出都不齐64 tile；若各 pad128，h1024 增加129,024 MAC/token，实际 backend 的 padding/效率须 SYS 核实。将 [x,η] concat 可少一次 GEMM，但另建 M×5185 BF16 缓冲328,397,160 B，不记免费。

G 的一次离线编译需 D·64²=20,971,520 MAC/step，G⁺/rank 判定另有64³级分解工作；未执行。H-LR-stat 的 B/L/H 需要 O(Dk) centering、64Dk 与 Dk²离线 MAC/step，Φ/gate/调制变化均需重新绑定。未来 B1 的分解只有代码/条件/精度完全相符才可复用，不凭“已经做过 QR”省账。

| 逻辑存储对象 | bytes |
| --- | ---: |
| 单个 M×D BF16 r/x/u | 324,280,320 |
| 单个 M×D FP32 cast/reference | 648,560,640 |
| M×13824 /6016 /1024 BF16 hidden | 875,556,864 /381,029,376 /64,856,064 |
| M×64 /65 FP32 统计 | 8,107,008 /8,233,680 |
| H-R65 h1024 学生 BF16矩阵＋K/G⁺ FP32 | 12,079,104 |
| H-R65 h6016 同上 | 64,495,104 |
| H-full-r h1024 /6016 同上 | 22,431,744 /125,316,864 |
| H-LR-stat h1024,k64：学生BF16＋K/B/L/H FP32 | 13,271,040 |

以上权重是**单个执行条件**，未加 bias、scale、优化器、缓存副本或4步驻留；不是峰值显存。qᵣ/vᵣ 的稳健参考采用 FP32。普通 PyTorch `r.float() @ K.T` 需要上述648,560,640 B的 D宽 FP32 cast；保留 BF16 K、融合 centered reduction、或定制 mixed kernel 都是独立有代价的实现。BF16 round(K) 通常不再精确满足 K1=0，不能既用低成本 BF16 投影又无条件声称连续均值不变。

末层仅占每步1/40 FFN。同形状前提下，即使其自身变得免费，也至多节省全体 FFN 主 MAC 的2.5%；h1024 H-R65 对全部 FFN 的算术下降约2.405%，整系统更小。这使 B-H 更适合作为有限函数坐标的机制锚点；更大覆盖须由能进入前39层的 B-F 等结构另行建立。

## 5. BF16 边界与均值方向的可证伪预测

实际写回是 U(r,y)=RN_BF16[r+RN_BF16(g·y)]；其有效增量不等于实数 Dg y。随后 LN 输出、调制乘加、head GEMM 又有舍入。式(1)–(3)是**连续实数对象**；最终学生直接产生64维 head 输出、再按原布局转 FP32 flow，会改变原离散求值图，须同时比较连续统计标签与实际 native head 标签。

一个无需运行的 BF16 ties-to-even 符号反例：令 h₀=2⁻⁷，

\[
r=(1,1+2h_0,1+4h_0,1+6h_0),\quad r'=r+\mathbf1,
\quad g=1,\ y=h_0\mathbf1.
\]

原 r/r′ 都是 BF16，中心化向量均为 (−3h₀,−h₀,h₀,3h₀)，方差5h₀²。r 附近 spacing=h₀，写回后仍有同一中心化向量；r′ 附近 spacing=2h₀，ties-to-even 给出

\[
U(r',y)=(2,2+4h_0,2+4h_0,2+8h_0),
\]

其中心化向量变成 (−4h₀,0,0,4h₀)，方差8h₀²。将四通道模式重复1280次即可得到 D5120例。**名义上常数方向的更新在离散写回中改变了中心化方向与方差**；它对某些 head 会改变输出，不只是 ε 的径向微效应。

由此得到条件预测：若同一实验还确认原 post-Q1 x 字节相同，则只观察 x 和 gauge-invariant ηᵣ 的学生无法区分这对状态，而 mean-aware 或 full-r 输入可能区分。这里没有声称实际 Wan 已出现这样的 x 碰撞；原 LN 内部 reduction 精度也不能由 Python 无 `.float()` 推定。μᵣ 标量未必充分，逐通道 exponent/ULP 还可能重要。未来可比较 η、η＋μ、full-r 三种**预先冻结输入类**，检验收益是否随 residual 量级/舍入缺陷而变，避免把 mean-aware 收益误归为真实连续 head 的均值敏感性。

## 6. B0 能生产什么、尚缺什么

截至本构造，root 报告 B0 原 V1 在源码扫描时限内停止、0 FFN；修复版另行启动。**此文没有成功 capture 的 OBS。** 已核的两 expert 原 head 小字节资产是合法静态输入；实际 runtime layout/modulation 仍按成功 B0 CASE 绑定。

成功 B0 的 source01/04、seed42、step0/3四个 case，会保存 full u/head、256同步 r/V1/Y/u rows、gate、真实 head调制和算子身份。它们可在另行冻结的分析中产出 qᵣ/vᵣ、qᵤ/vᵤ、Δq、variance/slack ratio 及 BF16 writeback 缺陷标签，**不需重跑模型**，但投影、二次型/分解的真实 CPU/GPU 计算仍要计费。完整 u/head 可用于同 M native neutral；full-r/full-V1 不在保存范围，不能声称已有全 M 对齐学生训练对，更不能从 u−gy 逆出原 BF16 r。

独立来源数是2，不是4个step-case或1024行。两来源已开发暴露，只能诊断、不能制造新盲确认。旧 CAL24 V1/Y/gate 无对齐 r/u/head，不能单独生产 B-H variance 标签。后续 source-cluster 训练/验证、未暴露来源、合法 student-state 轨迹、自己的窄 Q2 scale、优化预算及完整自由视频均 OPEN，不能从 A0 完整 bank 自动继承。

**最小有辨识力的后续**：先以成功 B0/B1 区分连续统计重构误差和 native舍入误差；再冻结一个 h 与 source-level 分割，比对无锚直接预测、r锚比值、r锚可达 slack、同宽 B-F，并给所有方法相同合法输入和拟合预算。可达 slack 若仅消除不可能输出而非改善来源级 head误差，结论就是约束有用但可预测性未获支持；H-LR-stat 若没有低于 materialized B-F 的墙时，其收益限于表示/访存。任何结果继续通过真实四步学生轨迹和完整视频检验，末层局部成功不外推为运动保持或部署速度。
