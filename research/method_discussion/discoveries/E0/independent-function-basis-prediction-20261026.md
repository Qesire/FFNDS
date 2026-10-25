# P14 独立讨论：应找可预测的函数基，而非仅找能量最大的输出基

**新增结论：** 输出后验低秩只降低“表示输出需要几个系数”，没有保证这些系数能在 W1 前廉价取得。最直接的判别不是再报一条输出奇异值曲线，而是把同一持出输出误差严格拆成**基的表示误差＋前置信息的系数预测误差**。进一步，一个高能但不可由便宜信息预测的主成分，可能不如低能但可预测的方向；因此应加入经典 reduced-rank ridge regression（RRR-ridge）的预测基作为强对照。

**身份与范围：** 2026-10-26，独立发现/挑战记录。当前仍为 E0；此次只读根 AGENTS、当前第一任务、Graph↔Agent 协议、相关节点、已有报告和外部原论文/原代码，作纸面推导。没有运行 SVD、ridge、模型或任何新科学计算，没有访问 worker、操作 Git、改共享状态或保存新模型权重。协调者提供的当前仓库修订为 `087893f`；本 Agent 未用 Git 核验。原 P14 claim 仍是 OPEN，不授予正式方法标签。

## 1. 图锚点、标签对象和真正的信息时点

本次审查对象是 `E0_GRAPH_SEED_FUNCTION_BASIS_20261025`，文件 SHA256 为 `37afb7ed79ccdcd8c9ff0c6a8cb1e7b920840ede64fa7134fad99f391a712b02`。读到的节点注册表 SHA256 为 `a2ee383aa019f43541f2c52e549642a0920c3ef96b3a923ac893eac387649905`；当前第一任务 SHA256 为 `94af68c5a1f0607b26df2873b432ef8aad80054f5485ba7ba2bf90d5f28209b9`。它们是此次读取绑定，不是新实验收据。

相关链为 `G02`（high/high/low/low 合法历史）、`G18/F03`（modulation 后实际 post-Q1 输入）→`F04` W1→`F06` GELU→`F08R/F08` 原生 Q2尺度依赖/decoded系数→`F09/F15` signed W2输出诊断→`P06/P14` 同专家创新/输出函数基假说。潜在省算必须回到 `P10/P11/P12`；实际写回和视频仍在 `F12/F13/G21/D09`。F03 是可读计算图中的输入节点；P14 注册为 `posthoc_hypothesis_only`，不是已执行预测器。

以行向量批次记号，令实际 W1 输入为 `X∈R^(N×D)`，`D=5120`、`H=13824`。标签可直接取已保存原生 ungated FFN 输出再乘**当前** gate。时间创新沿用已有诊断语义：

\[
Y=\Delta U=\bigl(f_e(X_t)-f_e(X_{old})\bigr)G_t,
\]

这里只在相同 expert 定义；不把它混同为 `f_t G_t−f_old G_old`，也不混同为真实 residual 的跨步差。绝对输出和创新应分别分析。`q2_post` 是 F08 后真实 W2 输入，不能用 F06 的 GELU 代替。

已经有两来源全部40层/四步的 center128、uniform32 保存输入和输出；source0/42、source12/54 均已暴露。最低成本的新判别不需要重跑 W1/W2，也不需要重新生成视频。128×5120 标签天然 rank≤128，中心化后≤127；“秩远小于5120”本身没有解释力。

## 2. 严格分解：输出可压缩性和前置可预测性是两项量

**MATH_CONDITIONAL，独立推导。** 在训练来源确定均值 `μ` 和正交基 `B∈R^(D×k)`，`BᵀB=I`，持出时均冻结。设 `Y_c=Y−1μᵀ`，合法前置信息为 Z，预测系数为 `P(Z)∈R^(N×k)`。由正交性，逐个持出批次都有

\[
\|Y_c-P(Z)B^\top\|_F^2
=\underbrace{\|Y_c(I-BB^\top)\|_F^2}_{\text{固定基表示误差}}
+\underbrace{\|Y_cB-P(Z)\|_F^2}_{\text{系数预测误差}}.
\]

第一项允许使用已完成的当前 FFN 标签投影，是**后验表示诊断**；第二项才检查提前信息是否足够。用 held 当前 `Y_cB` 作为系数不能叫 W1 前预测。不能对持出来源重新求基后，再把两项归为跨源可部署结果。

对于单 token 的总体平方损失，给定固定 B 和信息 Z，最优可测系数是 `E[Bᵀy_c | Z]`，最小系数风险为 `E tr Cov(Bᵀy_c | Z)`。这不是说完整 post-Q1 输入存在随机不可知性：给定完整 X 与固定模型，FFN 确定。信息损失来自便宜压缩观测 Z；若直接使用完整 X，问题转为**指定低成本函数类能否近似这个确定性映射**，不是条件熵下界。

**精确 rank-one 反例。** 令 `y(x)=v φ(wᵀx)`，v固定，所有输出严格在一维子空间。便宜观测 `z=Px` 若漏掉 w 的方向，存在 `u∈ker(P)` 且 `wᵀu≠0`。选 `x₀±u` 使两个 GELU值不同，即得到相同 z、不同输出；任何只看 z 的预测器都不能同时正确。更一般地，`y=v Σ_j c_j φ(w_jᵀx)` 仍是 rank one，却可能包含许多不同的非线性特征。低维 range 不提供穿过 GELU 合并这些特征的恒等式。这是对给定观测的反例，不是对所有可能算法的复杂度下界。

## 3. 能量主成分还可能选错“值得预测”的方向

**MATH_CONDITIONAL，独立构造。** 取正交 `v1,v2`，完整输入中两个独立对称符号 `s,ε∈{−1,1}`，便宜观测仅含 `z=s`，输出为

\[
y=\sqrt{L}\,\epsilon v_1+s v_2,\qquad L>1.
\]

输出 PCA 的 rank-one 基选 v1，真实系数已知时谱尾误差仅1；但由 z 预测其系数的最佳值为0，部署总风险为 `L+1`。若选择 v2，预测 `s` 即可，风险为 L。因而**最小后验谱尾的基，不一定是给定信息下总预测误差最小的基**。ε 是被观测丢弃的输入分量，不是额外随机模型噪声。

另一竞争解释与 SVD-Cache 直接相关：`F_t=C+εE_t`，C为巨大而稳定的 rank-one 公共状态，E_t为高维独立创新。绝对 `F_t` 的主基可以跨来源非常稳定，但 `ΔF_t=ε(E_t−E_old)` 仍高维、难预测。因此绝对输出谱、中心化谱和创新谱不能互换；“跨prompt稳定主基”不单独支持 P14 的创新重构。

## 4. 精确 RRR-ridge：不能简单对 ridge fitted output 再做 PCA

**经典原理＋自包含推导。** 此处是已知统计方法的迁移，不是新发明。亲读 Mukherjee–Zhu 的原始[作者PDF §2，式4–10](https://dept.stat.lsa.umich.edu/~jizhu/pubs/Mukherjee-SADM11.pdf)：它把 ridge 目标写成增广数据上的降秩回归。令训练 Z、Y 各列按训练均值中心化，截距单独恢复，考虑

\[
\min_{\operatorname{rank}(A)\le k}
\|Y-ZA\|_F^2+\lambda\|A\|_F^2.
\]

记 `C=ZᵀZ+λI≻0`、`T=C^(−1/2)ZᵀY`，配方给出

\[
J(A)=\|C^{1/2}A-T\|_F^2+\text{与A无关项},\qquad
A_* = C^{-1/2}[T]_k.
\]

`[T]_k` 是最佳 rank-k 截断。等价地，让 B取 `YᵀHλY` 的 leading k eigenspace，其中 `Hλ=Z C^(−1) Zᵀ`，则

\[
A_* = C^{-1}Z^\top Y B B^\top.
\]

这里的 B 由**输入可解释且带惩罚的响应**选择；P14 的输出 PCA 则由 `YᵀY` 选择。直接 SVD `HλY` 得到的是 `YᵀHλ²Y`，一般不是上述惩罚目标的精确 RRR-ridge 解；λ=0 且使用正交投影时两者才合并。若实现采用 Cholesky `C=LLᵀ`，等价写法是 `A*=L^(−T)[L^(−1)ZᵀY]_k`。

条件需保留：λ>0保证可逆；λ=0且秩亏时用可识别子空间/伪逆，不能写普通逆。采用平均损失 `N⁻¹||·||²+λ||A||²` 时 C中的惩罚是 `NλI`。特征缩放、λ、k、均值全部仅由训练来源确定。截断点有重复奇异值时子空间/解可不唯一；优化目标不因此改变。

该最优性针对**训练目标＋指定线性函数类**，不保证持出误差优于所有对照。如果 Z直接含5120维输入而训练只有32/128行，无惩罚训练拟合可以近乎插值任意标签；训练 SSE 下降不是 FFN 机制证据。训练 token 错配与来源持出正好检查这种假象。

## 5. 最近原论文/原代码：各自如何真正取得系数

### FLAP：一个重要竞争者是“去掉的部分近似固定均值”

亲读 [FLAP v1 Methodology，式2–5](https://arxiv.org/html/2312.11983v1)，以及作者当前 [prune.py 的 metrics/compress](https://raw.githubusercontent.com/CASIA-LMC-Lab/FLAP/main/lib/prune.py)、[BiasGPT](https://raw.githubusercontent.com/CASIA-LMC-Lab/FLAP/main/lib/layerwrapper.py)。它保留部分真实非线性通道，将删除通道的校准均值折成输出 bias；WIFV 使用输入波动和权重列范数。代码实际裁剪 MLP up/gate 行与 down 列，并写入补偿 bias。

迁移到 Wan 的连续输出解释为 `q=(q_S,q_R)`，`ŷ=W2,S q_S+W2,R μ_R+b2`。余项由 `W2,R(q_R−μ_R)` 决定；原生已有 b2 必须保留。若 P14 的收益主要来自非零均值，FLAP式均值补偿可能已解释大部分，不必预测 token系数。此处的对角波动分数仍不是 signed联合最优；适配 HiF4 时 S/R必须考虑原64组依赖。

更强的独立对照可在训练来源拟合 `q_R−μ_R≈(q_S−μ_S)R`，并将 R吸收到保留 W2 中。这是**新增的保留特征回归重构**，不能冒称 FLAP 原实现。它支付选中 W1/GELU/Q2 的费用，可能优于输入线性预测，但不是“所有W1前已有信息”。

### SliceGPT：通过全网络坐标与权重变换使系数继续被模型计算

亲读 [SliceGPT v2 §3.1–3.4](https://arxiv.org/html/2401.15024v2)，以及原 [rotate.py](https://raw.githubusercontent.com/microsoft/TransformerCompression/main/src/slicegpt/rotate.py) 的 `rotate_mlp_input/output`、`rotate_and_slice_sequential`、`pca_calc`。其核心是 RMSNorm 的正交计算不变性、相邻输入/输出权重吸收旋转、实际裁掉残差 embedding 维；代码用未中心化 `XᵀX`，并修改 shortcut 矩阵和层维度。它不是在完整输出算完后临时投影再免费获得系数。

与 P14 的关系是共享“校准数据决定线性基”的组件，实际动作不同。不能把它视为任意旋转 GELU hidden 的许可。Wan 当前逐通道 AdaLN/gate 在一般旋转后变为稠密 `Qᵀdiag(g)Q`，原 Smooth/QDQ 也有坐标依赖；直接局部移植缺乏保持原图的证明。若只在 full-D FFN输出后投影并原维加回 residual，尚未减少后续 residual 宽度或 W1 费用。

### ASVD / SVD-LLM：activation-aware 的是已有线性层，不是免费越过非线性

亲读 [ASVD v3 §3.3、3.5](https://arxiv.org/html/2312.05821v3)，及原 [SVDLinear.from_linear/forward](https://raw.githubusercontent.com/hahnyuan/ASVD4LLM/main/modules/svd_linear.py) 和 [activation calibration](https://raw.githubusercontent.com/hahnyuan/ASVD4LLM/main/act_aware_utils.py)。ASVD对 `WS` 作低秩分解、逆缩放右因子；实现生成两层新 Linear并实际执行两次投影。其 abs统计是前向校准；可选 Fisher测量调用 backward，但测梯度本身不等于 optimizer 更新。

沿此问题继续读 [SVD-LLM v4 §3.1–3.2](https://arxiv.org/html/2403.07378v4) 和 [SVDLLM.py](https://raw.githubusercontent.com/AIoT-MLSys-Lab/SVD-LLM/main/SVDLLM.py) 的 whitening、`local_update.add_batch_update_u`：其代码收集输入 Gram、Cholesky白化，截断后用 `torch.linalg.lstsq` 更新左因子。这里明确存在**闭式参数更新和新因子权重**，尽管不做完整反向训练。

对 P14 的含义：若只因子化 W2，系数仍来自完整 F08 q；若因子化 W1，仍须产生 H维preactivation并执行 GELU/Q2。它们不由输出低秩自动得到。原理上，线性层的校准误差满足 `||(W−Ŵ)X||²=tr[(W−Ŵ)XXᵀ(W−Ŵ)ᵀ]`；对角 activation 缩放不处理全部相关性。跨非线性的 X→Y 预测需要另一个拟合目标，不能把线性层 SVD 的最优性搬过去。

### SVD-Cache：刷新时投影完整特征，跳步时用合法历史预测

亲读 [SVD-Cache arXiv v1 §3.2–3.5，式4–11](https://arxiv.org/html/2601.07396v1)。其刷新步骤已有完整 F，再用固定参考右基分解；主分量用 EMA、余项复用。按式8–9且参考奇异值非零，σ相消得到 `F_k=F V_C,k V_C,kᵀ`。因此它节省重复SVD，但刷新时取得系数仍先需要 F；跳步省算来自历史可预测性，不是谱尾小本身。

我追到该论文的 [作者仓库](https://github.com/BlackMaple1203/SVDCache)，此次可读主页只列 LICENSE；没有可核验的实现文件。CVF最终PDF/HTML请求403，GitHub tree API也未能由web工具读取。故这里只核验 arXiv公式，**没有声称独立检查其缓存实现、复现配置或最终会议版本**。论文还报告 distilled FLUX设置，不能简单因“少步”排除它；但本项目 high/high/low/low 每expert只有一次旧输出，普通单历史 EMA初始化可能退化为复用，不提供已观测同expert趋势。额外离线先验或跨expert转移须单独说明。

## 6. 对真实工作量的不同预言

**理想投影账本，非计时。** Dense两投影为 `2MDH` MAC。三种“低秩”产生不同动作：

| 路线 | 系数如何得到 | 理想投影成本与真正未解点 |
| --- | --- | --- |
| 输出 PCA oracle | 当前完整 Y投影到B | W1/GELU/Q2已经发生，若Y也已产生则W2亦已付；不构成省算 |
| W2因子化 | 完整q再投影到k维 | `MDH+Mk(H+D)`；只改W2时，两投影账本的极限仍有完整W1那一半，另有全部GELU/Q2 |
| 从完整post-Q1直接预测的rank-k线性映射 | 数据拟合的 `X L Bᵀ` | `2MDk`，理想比为`k/H`；这是替换原非线性函数的新近似，必须验证预测误差、参数与native写回，不能由PCA结果推得 |
| 原整组保留＋均值/回归重构 | 只计算固定选组的非线性q | 可能同时减W1/W2宽度；需包含原Q2闭包、重构新权重及实际GEMM舍入，不能称无额外W1的观察器 |

固定线性 sketch Z=XP 后再线性预测时，静态 `P R` 可以离线合成一个输入因子；不必部署成三次乘法。若Z含非线性或动态观测则不能如此免费合成，需逐项计费。任何速度比较还要包含静态因子的显存、原生QDQ、缓存旧输入/输出和装载成本，并比较已有 exact-output resident dense；MAC比不是实际wall收益。

## 7. 最小保存数据控制：先检验“预测差距”，无需新采集

以下是**尚未执行的建议**，不登记范围、也不把协调者正在考虑的其它范围写成已授权/已有结果。可先选一个中间block、一个同expert更新对、两来源互换；仅用保存 post-Q1 与原生 FFN输出/gate。固定分层于 block/expert/step，避免把不同 W2函数混为一个共享基。

1. 只在训练来源固定一个center面板拟合均值、基和参数；在另一来源 center/uniform 分开评价。若另作同来源空间持出，剔除与训练center重合的uniform行；报告实际唯一行数。固定 k远小于训练行数，λ/缩放只按训练数据选择，不能逐来源挑最佳报告。
2. 同一个训练源 PCA基 B上，比较 mean-only、持出真实系数 `Y_cB`、只看当前 post-Q1的ridge系数预测、训练token错配后的同样预测。四者把均值、谱尾与信息配对逐一分开；错配保留训练Y的谱，破坏输入—标签对应。
3. 加同rank随机正交输出基和精确目标定义下的 RRR-ridge。前者检查特定输出子空间是否必要；后者检查“能量基挑错方向”是否才是系数预测失败的原因。若只用一个便宜输入sketch失败，只排除那个sketch，不能排除完整X或不同代价的非线性观测。
4. 若需要下一层竞争，才加入 FLAP式整组均值补偿与保留组q回归重构。它们读取q标签作保存数据诊断，但部署取得q_S需支付所选W1/GELU/Q2。最初的预测差距控制不必做全W2分解或新W1投影。

**不同结果会改变什么：**

- 均值控制已接近PCA oracle：主要是FLAP式稳定基线，不足以支持token特有低维系数。
- 跨源PCA oracle好，前置预测差且错配后接近：range低维成立，但前置信息/函数类未解释系数；不要继续把谱压得更低当作省算进展。
- RRR-ridge持出显著好于同rank PCA＋ridge：可预测方向与高能方向不同，应转向有监督可预测函数基；仍需检查真实代价和视频。
- 保留非线性组重构优于同投影预算输入线性映射：支持需要少量特定非线性特征，而非完全折叠；不支持“任意rank-k输出都能提前预测”。
- 训练源谱好但来源持出oracle坏：主要问题是基不迁移；先排除样本秩/均值/位置组织，不应把失败归咎于预测器不够复杂。
- 绝对输出好而创新/中心化输出坏：更接近SVD-Cache稳定大状态＋难预测创新的竞争解释，而非P14已找到便宜更新。

本轮没有得到上述任一新数值结果。后验输出能量或预测SSE还不能替代 `F12/F13/D09` 的有限native动作和完整自由四步视频。

## 8. 闭式拟合、训练、权重与授权是不同轴

闭式最小二乘/activation PCA可以不做 optimizer step，也不改原checkpoint；但它仍从数据估计新预测参数。将拟合矩阵保留到runtime、替换W1/W2、增添bias或保存因子，都是新的模型/算子状态。FLAP的新增bias、SliceGPT旋转/切片、ASVD新Linear和SVD-LLM闭式左因子，均说明“training-free”不等于“没有新权重”。

应分别记录：是否新跑dense前向；是否做SVD/闭式拟合；是否梯度测量；是否optimizer更新；是否生成/保存新因子；是否返回模型；是否新视频。仅在已授权的保存数据分析里临时拟合、只保存标量，不等于已部署新权重；它也仍是新增科学计算，不能由方法讨论自动授权。**本 Agent 此次连临时拟合也没有执行。**

## 实际检索、来源关系和图建议

2026-10-26使用web搜索并迭代四条线：FLAP→均值补偿/结构化恢复及作者prune代码；SliceGPT→RMSNorm不变性/实际rotation与slicing；ASVD→activation加权分解→SVD-LLM白化与闭式更新；SVD-Cache→当前特征投影/EMA/源码可见性。由“PCA未考虑可预测性”继续检索并亲读2011年RRR-ridge原始作者PDF §2。这条统计学引用改变了建议：用已知预测降秩目标作强对照，避免发明只改名的“函数基”。

本次论文版本明确为 FLAP v1、SliceGPT v2、ASVD v3、SVD-LLM v4、SVD-Cache v1；代码链接为此次实际读取的作者仓库main，未取得不可变commit pin，故不将其冒写成源码快照复现。未运行任何外部实现。SVD-Cache原实现和最终会议全文核验保持缺口；搜索摘要仅定位其作者仓库，方法结论依赖可读arXiv原文。与这些完整系统的关系分别是**均值补偿控制、全网络坐标重参数化组件、线性层activation-aware因子化、时间预测邻近机制**，不是整套等价。

给整理者的图建议仅是：在 P14保留后验表示边；另加从 `G18/F03` 到“指定输入函数类的系数预测”的 **CONDITIONAL/OPEN** 注释，并把表示误差、系数预测误差、训练/持出身份与 `P12` 成本分别记录。不要从 `F09/F15` 的漂亮谱直接画一条已执行的W1省算实线。此文件不修改graph、claim、索引或正式讨论状态。
