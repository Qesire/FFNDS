# B1：末层 head 的最小同状态有限扰动判别

**独立候选 / ALG + SRC / 未登记、未运行。** B0 的 source01/04 × step0/3 提供四个完整 M=31668 的真实 pre-head 状态 u 和同步256行。末层归一化让 head 直读不到的方向仍可通过状态范数改变输出；同一个 base 上比较 head 可见、范数通路及一阶零方向，可先确定这个额外自由度是否在真实 BF16 算子中超过舍入噪声。这是四个嵌套 case、两个已暴露 source 的局部机制诊断，不是拟合、视频或泛化实验。

## 固定对象与方向

每个 case 按 B0 的256个原 row IDs，其他 31412 行保留原 u。固定实际 BF16 u、step 的 shift/scale、对应 expert 原 BF16 head 及原 full-M layout，原函数顺序 LN(eps=1e-6)→BF16 modulation→Default MM。不从观察结果挑选来源、行、方向、符号、幅度或成功子集。所选行是先前 B0 坐标，不是256个独立样本。

在 FP64 构造实数参考：对每个 case 令 T=W_head diag(1+scale)，P0=I−11ᵀ/D，B=T P0。每 case 对 `[1; B]` 只对该矩阵转置做一次无主元 thin QR（秩阈值为最大R对角绝对值的1e−10；秩不足的case标记INCONCLUSIVE），所有行共用其正交基；记录原矩阵形状、有效秩和正交性误差。64行的头与常数总不超过65维。对每个保存行，z=P0 u，v=z−Proj_span(1,Bᵀ) z。使用固定计算顺序以及最大绝对分量为正的符号约定，构造：

1. **visible**：B 的第0行投影到均值零空间后归一；若退化则标记INCONCLUSIVE，不更换输出行，不用当前响应选最佳行。提供直接线性通路。
2. **norm**：v/||v||；该方向均值为零且 Tδ≈0，但与 z 正相关，可单独激发范数分母的一阶通路。
3. **null**：seed42 的固定伪随机向量先投影出 span(1,Bᵀ)，再投影出 v，单位化；因此均值、head读出和 z 一阶项均近零，但有限范数变化仍可能造成二阶响应。此为后验 paid 诊断方向，不是可免费在线生成的算法。

每个方向为unit-L2；固定两个幅度 a∈{1/16,1/4} 和两个符号。形成候选 `BF16(u.float()±a*||u||₂*direction)`，名义扰动的行RMS及L2均为u相应尺度的a；这属于有限幅度恒等式检验，不称自然小扰动；逐字记录真正有效 `δ=BF16(candidate).float()−u.float()`、其 RMS、常数/head/z投影，不以名义方向假定舍入后约束仍成立。若 norm 投影退化（||v||/||z||<1e−6）、QR有效秩不稳定、u RMS为零或非有限，该行标记 INCONCLUSIVE，不能事后重新抽方向；所有256行均不合格则对应 case 不具辨别力。

每 case 增加一个常数方向 control，幅度固定为1/4行RMS、两个符号。实数 LN 对常量平移不变，native BF16 的有效扰动并不一定是常量：将 BF16 后的各通道差异作为该 control 的必要解释。它不要求假装得到绝对零响应。

## 真实写回与预算

基线完整 u 在同一进程原 M 重放 **两次**；两次 head 与 B0 head 均须逐字节一致才进入干预，否则 FAIL_IDENTITY 并停止。然后三方向×两个幅度×两个符号=12次 post-write full-M head，以及常数方向2次，共 **16次 head/case**，四 case **64次**。每次都用相同 M、原 operator/dtype/layout 和原权重；观察256行，记录其余行是否逐字节保持。每个 head 调用包含一次完整 LN、modulation和 BF16 head MM；不做 scheduler、VAE 或完整模型。head 与 unpatchify只是布局对应，主要比较原 head 坐标。

另做每 case 两次 **pre-gate** 对照，只使用固定 visible 方向和1/4行RMS、±符号。对该同步行先计算 `Δy = (1/4)*||u||₂*visible/g`，仅在 `|g|≥g_floor` 且有限的通道赋值，其余置零；冻结 `g_floor=1e−3`，记录可干预通道及实际有效方向，不将零/小 gate 反演成无限大。使用 B0 同行原 BF16 r、y 和 g，严格先 `BF16(y+Δy)`，再原 caller 的 BF16 multiply/add 顺序形成 u′；其余行不动。首先核原 `BF16(r+BF16(g*y))` 与保存 u 同步行逐字节相同，失败则只该 pre-gate 对照 FAIL_IDENTITY。这里的实际 δ 与 post-write visible 通常不同，不应把两臂响应差直接归因为 gate 几何。额外 **2次 head/case、8次总计**；全范围 **72次 full-M head**，无 FFN、QDQ、生成或训练。

分解账本：每 case 一次固定矩阵 thin QR，**4次QR**；每 case 256 行各一次 state residual 和固定随机向量的投影/归一，**1024行构造**，均离线 CPU FP64；每 case 对64×5120 head矩阵的缩放及固定 head第0行的构造一次。head矩阵秩和各方向残差用相同预注册数值容差记录，不使用SVD或拟合。每次 full-M head 前预扣、失败保留，保存 source/state/row IDs/方向规则与摘要、有效 δ、响应和输出SHA；不重跑、不自动加幅度。

## 事先判据和解释

先取两次 no-op的差作为系统噪声 N（若逐bit相同则仍记录量化楼层）；对每方向与幅度以256行整体 L2 计算奇偶响应 `O(a)=(F(+a)−F(−a))/2`、`E(a)=(F(+a)+F(−a)−2F(0))/2`。先在有效δ而非名义δ上核算连续有限公式，并计算 native−公式差的 L2；量化误差不是用连续导数替代。设置保守的判定门槛：所比较响应须超过 `max(4N, 4×该对native−连续公式L2误差)`；否则对应比较 INCONCLUSIVE。连续公式使用实际有效δ计算，不能以解析中理想方向的阈值掩盖舍入。

- **PASS_DIRECT**：两个幅度均满足门槛、visible 的奇响应为非零，且高/低幅奇响应比在[2.5,6.5]；同时有效扰动幅度比在[3,5]。这只验证该同状态直接读出的一阶可见性。可辨别但不符合范围为 FAIL_DIRECT；未满足门槛或方向退化为 INCONCLUSIVE。
- **PASS_NORM**：norm 的有效 head和均值泄漏作为实际δ线性预测的一部分量化，有效 z投影非零，奇响应满足上述门槛及[2.5,6.5]比例，并与使用实际δ的有限公式同方向（cos≥0.9）。若有效 head线性泄漏超过范数线性预测的四分之一、量化楼层主导或模型残差大则 INCONCLUSIVE；在这些前提满足时观测不符合则 FAIL_NORM。
- **PASS_SECOND_ORDER**：null的有效均值、head及径向泄漏分别以 |1ᵀδ|/(√D||δ||)、||Tδ||/(||T||F||δ||)、|zᵀδ|/(||z||||δ||) 报告；不假定BF16后仍满足理想零约束。须实际δ的线性预测范数小于有限预测范数的四分之一，两幅度偶响应超过门槛，幅度比在[8,32]（名义平方比16），高幅偶响应与公式方向cos≥0.9。BF16破坏零方向、q=Tz过小、偶响应低于噪声或公式残差同量级均为 INCONCLUSIVE；满足前提却明显不符才 FAIL_SECOND_ORDER。这些阈值是事先约定的探索容忍带，不是统计显著性或后端误差证书。

报告每 case 的有效样本数、各响应比值和连续预测误差，保留全部零/退化行；只在两个 source 内报告，不能把行或两个 step 作为独立来源推断。常数控制与 pre-gate 对照只报告实际δ和响应，不设置人为通过阈值。反例包括 B行全零、z接近head行空间、q≈0、gate为零、BF16幅度落在同一舍入格、跨case head/scale不同、M256代替真实M改变GEMM舍入；这些情形分别造成无可用方向、不可见偶响应或基线不一致，均不可解释为机制不存在。量化间断也会破坏小幅极限的幂次关系；不自动放大或修改阈值。

## 实际 B0 完成与 B1 冻结绑定审阅

B1 最终协议 SHA `1abe493a430fe592ebb2b450c0ee40a9557949df65f8c548c5cd470ebe14bb8d`；实际 B0 V3 protocol SHA `219b50ef13d20ed824d4095b749e9e52086359ec859b50bf3e33efe3686a55f8`，RUN status `COMPLETE_B_FINAL_BLOCK_CAPTURE_HISTORICAL_POSTSTATE_BYTES_MATCH`，launcher 实际 exit 0，记录四 flow、160 FFN。独立逐字节核对 B1 的11源码及16个HF资产绑定、4个 CASE 与 B0 RUN 中同 source/step 的 u/capture SHA，四条均记录历史 post FP32 原字节复现。每个实际 u stride为 `[5120,1]`、head weight stride `[1,5120]`、bias stride `[1]`；head 为 `MMWeight`，norm 为 `LNWeight`，两 dtype 均 BF16，实际 scale shape `[1,1,5120]`。这些 layout 是真实捕获，不从 CPU view 的转置布局推断。

冻结范围为72次 full-M 等价显式 Torch head/LN、4次QR、1024行方向构造及16384行FP64响应；零FFN、QDQ、训练、视频。显式 source-equivalent LN/modulation/addmm 重放不称作原类对象调用。运行前仍需实际 worker 对该最终协议的CPU preflight及独立注册；上述核验不声称 B1 干预已执行或通过。
