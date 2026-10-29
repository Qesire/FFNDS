# 两种完整FFN计算单位：符号偶函数组与共享二次坐标

2026-10-29 · `/root/s2_theory_design_independent` · SRC/ALG/已有OBS/OPEN。没有读取原始大张量、运行模型/GPU或拟合；不新增实验授权。两个动作分别改变实际W1与W2的计算单位，均是待反证的候选，不宣称已有实现、创新优先权或视频/速度通过。

**核心问题**：现有低秩W2保留完整W1，完整视频中的动作衰减及同h L/P/I交叉说明“输出重建好”不足以说明FFN行为保留。一个可探索的自由度是 tanh-GELU 的奇偶结构：线性部分允许跨隐藏神经元先合成为输入到输出的共同算子，偶函数部分允许有符号镜像的预激活共享。另一种独立表示把整层近似为输入共享坐标上的非线性特征，再直接映射输出。第一种单位是**有符号隐藏函数组＋线性总算子**；第二种是**输入子空间上的多项式交叉特征**。两者都必须证明其前提与总成本，不能从负cos或静态近邻直接推出冗余。

## 真实计算与信息时点

[SRC] [Wan 原BF16 FFN](../../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py:366)：attention残差后，row-LN及shift/scale得到 `x`；W1+b1给 `a=xW1+b1`，tanh近似GELU给 `h`，W2+b2得到 `Y`，随后实际gate乘法与residual BF16写回。当前M=31668=21×29×52，D=5120，N=13824；每token两次原dense GEMM名义约 `2DN=141,557,760` MAC，不包括bias、GELU、规范化及写回。原BF16无FP4 Q1/Q2。[已存capture](../../../../tools/e0_original_bf16_full_depth_collect.py:312) 的V1是调制LN后、W1之前的 `[rows,5120]`，h是GELU后 `[rows,13824]`，不是W1预激活a。

选择动作若需要x，可在原W1之前合法读取已付的当前x；若需要a、GELU输出h或真实未来运动，必须先支付其producer/观察或明确后验oracle。step和expert在调度时可得；同一step所有token共享当前参数及gate，但各自x不同。目标是同一student自身当前x上的完整FFN替代，而不是借Dense hidden修复自由轨迹。

[ALG] `f(a)=GELU_tanh(a)=a/2+ψ(a)`，`ψ(a)=a/2*tanh(√(2/π)(a+0.044715a³))` 为偶函数。在实数运算下对每行

`Y = 0.5*x*W1*W2 + (0.5*b1*W2+b2) + ψ(xW1+b1)*W2`。

这不是原生BF16位级等价：原GEMM/addmm、GELU、W2及gate/residual写回都有舍入；将它拆成两路还引入新GEMM、相加和不同舍入。旧HiF4的五层跨step创新线性／偶分量W2投影cos约 −0.64..−0.82是有界强负交叉[已有报告](../../../20261029/ASSET_FIRST_STRUCTURE_DISCOVERY_AND_METHOD_DERIVATION_20261029.md)，不能单从负cos判定二者相同、可删或动态冗余；更不能将旧HiF4结果直接搬到原BF16或把diffusion step差视为视频latent τ差。

## 候选A：镜像偶函数组＋折叠线性通路

把 `a_j` 看作输出列对应的实函数，给每组g选一个代表 `z_g=xw_g+b_g`，每列j指定离线符号 `s_j∈{−1,+1}`；候选要求在目标source/step/student状态上 `a_j≈s_j z_g`。因 ψ是偶函数，组偶输出可近似 `ψ(z_g)∑_{j∈g} W2[j,:]`，正负号不改偶输出权重。**线性半项仍以全体W1、W2的总乘积进入**，并非把负号简单视为普通GELU neuron对称：例如精确 `a_2=−a_1` 时 `f(a_1)v_1+f(−a_1)v_2 = ψ(a_1)(v_1+v_2)+a_1(v_1−v_2)/2`；v1=v2才可取消线性差，普通GELU并不满足 `f(−a)=f(a)`。

把完整线性矩阵 `K=0.5 W1 W2` 离线近似为 `A B`（A D×r、B r×D），保留 `c=0.5 b1 W2+b2`，得到 `Y_A=xAB+c+ψ(xW_group+b_group)V_group`。G个代表需D×G和G×D两个投影，线性通路D×r及r×D；名义每token `2D(G+r)`，相对原W1+W2为 `(G+r)/N`，例G=4608、r=960为0.403。这个只计算逻辑MAC，未含代表选取、离线 `W1W2` 构造/分解、bias、两路中间结果、ψ、累加、布局搬运和原生kernel利用率；K若无足够低rank，其补偿成本可能吞掉全部收益。无需先算全部a即可在线运行，但所有分组/参数及模型工作都要先在独立source上付费确定。

对当前x，令 `e_j=a_j−s_j z_g`，偶项实际误差为 `∑_j[ψ(a_j)−ψ(s_j z_g)]v_j`；各输出向量可能抵消或叠加，不能仅看各自RMS或相似度。对 `|a|,|z|≤B`，有可计算的 `L_B=sup_|t|≤B |ψ'(t)|`，进而 `||error_even||≤L_B∑|e_j| ||v_j||`；此界通常宽松，且输入外推不受限。总误差还含线性通路压缩误差与其signed cross，不能分别优化后假定联合误差相加。

**竞争和反例**：相似原W1列或权重余弦不够：即使 `||w_i−w_j||` 小，`x` 在差向量上可取很大值，或b不匹配使 `a_i`、`a_j`跨零落入不同GELU区；当前student经过attention后输入分布也可能改变。另一方面，两个a远离镜像但其W2方向相消可使输出误差小，因此静态几何失败并不必然否决输出级联合表达。预测若镜像条件在teacher及student、source-held的真实x上持续，G下降时偶项及联合gY/typed-u误差能优于相同宽度普通同号neuron合并；反预测若跨状态符号、bias或大幅值翻转频繁，低G组误差和跨分量抵消会崩溃，即使校准时平均权重邻近。

## 候选B：输入公共坐标上的对称二次特征

设 `z=x A`，A为D×k离线固定基；若所有影响当前任务的W1输入方向都近似包含在该子空间，则 `W1≈A B`，于是 `a≈zB+b1`。在受限区间上用偶函数的二次近似 `ψ(t)≈c₂t²`（tanh-GELU在0附近 `c₂=√(2/π)/2`），将每个隐藏单位展开并离线汇总：

`Y_B ≈ c + z L + φ₂(z) T`，其中 `φ₂(z)`含 `k(k+1)/2` 个不同 `z_i z_j`，L、T按B、b1、W2折叠，跨项2倍系数一并放入T；线性半项及b2也并入L和c。在线只做 `x→z`、计算对称乘积、`z→D` 与 `φ₂(z)→D`，无需完整N宽a、GELU及W2；不是只压W2。以k=64为例，名义MAC `(2k+k(k+1)/2)/(2N)=2208/27648≈0.0799` 倍原两线性，T约 `2080×5120` BF16值约21.3 MB/层。另计成对乘积、临时张量、输出相加、bias/typed写回、加载T与原权重比较、离线 `B/W2` 折叠及系统cache；大型M上的实现方式与实际延迟完全未测。

这是**条件近似**而不是PCA定理：小 `||W1−AB||`、小均方奇异值尾，不能保证重要token上的 `x(W1−AB)` 小；更不能保证GELU输出误差或下一attention/动作小。`ψ(t)`在大|t|近似|t|/2，二次在无界域增长更快，任意固定c₂不能全局逼近；bias会令即使z小也处于大|a|区。反例：校准x集中在A空间，但某source的学生x落在A的正交方向上，令权重残差与W2放大；在校准平均仍可很小。另一个反例是所有 `|a|` 很大时二次项严重过冲。可以用分段或四次特征作独立竞争臂，但四次项数 `binom(k+3,4)` 迅速超过原N，动态分段需在**不先支付完整W1**的可得信息上路由，并额外计其代价。选择BF16、FP32或其他多项式计算及输出累加顺序都是新数值recipe，必须跟原生完整FFN做typed对照。

**预测**：若source-held、step/expert及student x的有价值变化同时满足低维与受限非线性范围，k和多项式度增长时gY、u、下一attention及动作方向恢复，并可在相同算力下优于A与双侧W1/W2低秩基线；反预测是输入残差或高幅值/跨零尾部在动作源显著贡献误差，即使平均输出MSE小，时间变化和完整视频仍衰减。视为新动作假说而非已测结论。

## 符号抵消、量化和最近先例

每个误差须在同一个实际视图中报告 `||e_linear||², ||e_even||², 2⟨e_linear,e_even⟩` 和总差；对真实gate、真实r及BF16写回重新测，不能把两个百分比相加或把单个分支误差当完整风险。比较step之间差时记录各step gate、输出和cross；比较latent时间时对齐21τ，避免混淆两个时间轴。任何Q2替代下，先把新的非线性组输出重组成原hidden布局会失去收益；若不重组，则量化器实际group/scale、packed布局、融合kernel必须重新设计/核验，不能把旧HiF4 64组或其S1P2结果当原生NVFP4证据。group permutation要同步W1/W2/bias并保留实际码字和scale；跨组共用可改变动态scale，不能假设量化与聚合交换。BF16更无现成Q1/Q2可用作免费路由。

既有G14台账中 [L93 FANG](https://arxiv.org/html/2512.23014v1) 已有按上下文的neuron分组与二阶补偿，不能将普通分组、代价加权重建或近似匹配当成新颖性；其原文Eq4–7/附录Eq11已在[既有台账](../../../literature/20261028/QCFB_PRIMARY_METHOD_AUDIT.json)记录，作者代码未核。A的区别是明确偶函数±关系与独立保留/压缩奇线性总通路，但仍须与FANG式组重建及普通同号合并对照，不能声称从未提出过。 [L06 SliceGPT](https://proceedings.iclr.cc/paper_files/paper/2024/hash/316648eb8b4ffb6010f531b07848c300-Abstract-Conference.html) 用离线PCA及残差坐标变换/切片，B则希望将FFN整体变成二次交叉特征；二者都支付坐标变换，B不享有PCA尾能量保证。 [L09 Chipmunk](https://arxiv.org/abs/2506.03275) 以当前输入的blockmean→W1付费预测、同权重历史状态及稀疏W1/W2 delta执行；本候选不以跨step缓存为前提，必须同预算比较。[既有原文与作者MLP源码审计](../../../literature/20261027/G8_FOUR_METHOD_PRIMARY_GRAPH_AUDIT.md)记有代码深度；本次没有新读或运行作者代码。以上作用映射为LIT，非本地NATIVE证据。

## 最便宜的决定性判别与必要新资产

先做**ALG/SRC门**：对多个候选G、r、k及归一化成本，明确离线产物、需要全权重或sample操作、线上shape/精度及元数据；选matched MAC的普通同号聚组、独立W1/W2低秩和随机分组对照。若无廉价可行离线方式或 `G+r≥N`，A在该物理形式上直接淘汰；若 `2k+k(k+1)/2≥2N`，B的核心MAC不优于原两GEMM。二者均需额外完整数据/带宽/占用的实际效益。

随后拟注册**同输入四角原生标量判别**：先对已有两曝光pilot，在同坐标Dense与value student的V1上分别得四角的真 `a,h,Y,g,r,u`；原生完整M计算W1并采样所选行作anchor，不能用小M重算a冒称全M位相同。记录同h候选两支、分量signed cross、原始gY/真实BF16 postwrite、频带及normalization；旧保存的h无法可靠反演a（GELU在负区非单调），后验重算小M只在与fullM桥验证后使用。先用少量预先固定组/子空间、负例及随机对照检查该物理动作是否有任何可信低风险区间，再依据结果开展**预冻结来源分离**的独立source研究，不能拿两个曝光pilot作泛化。

最终缺口不以旧两图/五层封顶：为12新开发source，每源同seed、Dense及自由student全4step×40层，记录相同21τ及预冻结52个空间锚点的 `V1,a,h,Y,r,u`、实际gate/坐标/step/expert及可追溯权重、输入图、state和运行SHA；设计额外覆盖空间并行扩展，以防52锚点代表性不足。原Dense十状态与student无observer控制逐byte核验，所有flow/latent、W1 fullM及observer D2H/写入预扣闭账。仅这六种BF16 row payload为 `12×2×160×1092×(4×5120+2×13824)×2 = 403,628,359,680` bytes（约376 GiB），不含gate、metadata、临时峰值、视频、其他空间采样及独立确认；可分source/step分片并在有完整预声明候选的情况下做付费流式signed统计减少持久化，但不得以抽样行或流式统计冒充全M全空间观测。新增Dense observer重跑需既有10状态byte-neutral；新12 student轨迹及其observer control须有真实相同factor/seed RUN后才能复用。来源分为calibration开发、source-held开发和后续未曝光确认，绝不用目标视频事后选择分组/阈值。

最强可证伪比较是各动作同x、同gate、同原r的真实typed写回，以及自由4step×40层和完整81帧视频动作、背景、时序/成像、语义和真实系统成本；局部最佳与完整动作未必一致。所有新采集、离线拟合、kernel及后缀实验须单独制定producer、预算和ACTIVE登记。本文只提出问题，不启动或更改现有正常forward工作。
