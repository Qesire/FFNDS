# Attention：从当前消费者推导状态与图旁路

2026-11-28。固定 current centered low / high OUT-ABS 实现，零 GPU、模型推理与生产源码变更。数学工具采用 operator-tensor-analysis 的 C03 任务读出资源证书、E1 收缩路径、E2 依赖调度与 E5 在线 softmax 状态。实际源码和冻结 CAL 文件哈希在 `SOURCE_IDENTITIES.json`；CPU 见证在 `ATTENTION_GRAPH_VERIFIED.json`。

## 0. 本轮新增了什么

旧 `round8_derivation/.../CORE_METHOD_REVIEW.md` 已指出 OUT-ABS 是共享仿射通道校正；旧 `round13_speed/core/CORE_AND_COMBINED_SPEED_CLAIM.md` 已给出恢复算子合成和 P64-first decode/tile 输出。本轮承接这两项：

1. 把 low anchor 的 dense 与 selected 两次遍历，改写为一次遍历上的两个互补 softmax 累积状态，同时产生 native、selected、rho 和 centered 历史。
2. 从 low hit 的实际消费者得到联合历史 `Z=[C,A]` 与动态读出 `G_t=[diag(σ_t);−Γ_1,t]`。精确保留 P64 后，联合压缩只作用于非 P 行，系数不变，缓存误差有精确的当前读出表达。
3. 将 high 仿射关系推进到当前输出投影 `W_o`，计算动态权重构建的真实形状费用；将 P-first 从“已有全场 selected 后只先解码 P”推进到“先计算 P 的 selected，再流式计算非 P”。

这些构造分别回答重复计算、缓存的实际信息用途和中间结果物化三个问题。没有用秩候选代替真实质量结论。

## 1. 实际状态与消费者

对一个 layer/head，省去对应下标。query 数 `N=31668`，value 通道 `d=128`，恢复斜率维数 `r=31`，`P=64`。`S_t∈R^{N×d}` 是 frozen route 上归一化的 selected attention，`B=P_v[:, :64]T∈R^{128×31}` 是冻结 CAL 投影链。

low anchor 存：

- `ρ∈R^{N×1}`，FP32，遗漏质量；
- `C∈R^{N×128}`，BF16，centered 历史；
- `A∈R^{N×31}`，BF16，anchor selected 的投影；
- route / BlockMask。

实数定义与舍入分开写为

\[
C_a=\frac{O_a-(1-\rho_a)S_a-\rho_a\mu_a}{\sigma_a},\qquad
A_a=\operatorname{bf16}((S_aP_v)T),
\]

零尺度通道的逆按实际代码置零。命中时

\[
\phi_t=\operatorname{bf16}((S_tP_v)T),\quad
X_t=[1,\phi_t-A_a],
\]
\[
b_t=(1-\rho_a)S_t+\rho_a\mu_t+C_aD_{\sigma_t}.
\]

`schur_fit` 对所有32列（包括截距）采用 `αI`，`α=Pλ=6.4`。把 `scale_r` 吸收进系数，

\[
\Gamma_t=(X_P^TX_P+\alpha I)^{-1}X_P^T(O_{t,P}-b_{t,P}),
\qquad
\widehat O_t=b_t+X_t\Gamma_t.
\]

最后 P 行覆盖原生当前全键 attention。`C` 与 `A` 的全场分别被直接读取；`A` 不是只在拟合时使用，因此原实现不能仅保留 `A_P` 后得到相同非 P 输出。high OUT-ABS 只保存 route，使用 `b_t=S_t`、`X=[1,φ_t]`。

本地实际 CAL 的 `B` 均为31列满秩：high 条件数1.06786，low 条件数1.06592。该结果核查的是已有投影链，没有测量历史 `C/A` 的联合谱。

## 2. 锚点的一次互补分区遍历

令每个 query 的支持分为 `S` 和其补集 `U`。分区统计量为

\[
Z_S=\sum_{j\in S}e^{\ell_j},\quad M_S=\sum_{j\in S}e^{\ell_j}v_j,
\quad Z_U=\sum_{j\in U}e^{\ell_j},\quad M_U=\sum_{j\in U}e^{\ell_j}v_j.
\]

则

\[
S_a=M_S/Z_S,\quad O_a=(M_S+M_U)/(Z_S+Z_U),\quad
\rho_a=Z_U/(Z_S+Z_U),
\]
\[
C_a=\rho_a\left(M_U/Z_U-\mu_a\right)D_{\sigma_a}^{-1}.
\]

因此 native、selected 与 centered 历史是同一次分区归约的不同读出。无需先获得完整输出，再重新访问 selected 支持计算第二遍 QK 和 PV。

数值稳定实现给两个分区各保留 `(m,ℓ,z)`：每个 key block 只算一次局部分数和局部加权 V，按照 frozen route 将其合并到所处分区；最后用共同最大值合并两个统计。query tile 内所有 query 使用同一个 key-block 支持，恰好吻合当前 BlockMask 路由。全场 logits 不需要物化。

当前 selected 每 query 最多31×128=3968 keys，额外一遍的上界为每层 `4 N·3968·H·128=2.57349 TFLOPs`，40层 low anchor 为 **102.93954 TFLOPs**。包含尾块时真实值略低。该构造消除的是这些重复主乘法；双归一化状态、寄存器压力、向量合并与缓存写出费用仍在。

分区公式在实数中等价；CPU 单遍合并与 dense 参考最大绝对差 `8.33e−16`，rho/centered 也在 `1e−15` 内。相对于当前 frozen BF16/FP32 路径，归约次序与中间写回改变，当前由两个后端 LSE 不一致引起的 rho reconcile 也会由统一分区统计替代。因此这里证明的是目标算子的等价，不声称生成字节已经一致或 GPU 加速已经测得。

## 3. 更简洁的联合缓存：按实际读取保存 `C` 与 `A`

将系数分成偏置 `Γ_0` 和斜率 `Γ_1`。非 P 行的历史仅通过

\[
C_aD_{\sigma_t}-A_a\Gamma_{1,t}
=Z_aG_t,
\qquad
Z_a=[C_a,A_a]\in\mathbb R^{N\times159},
\quad
G_t=\begin{bmatrix}D_{\sigma_t}\\-\Gamma_{1,t}\end{bmatrix}\in\mathbb R^{159\times128}
\]

进入当前输出；其余当前项为 `(1−ρ)S+ρμ+Γ_0+φ_tΓ_1`。

### 3.1 保留 P64 将拟合与缓存压缩解耦

保留 rho 全场、`C_P/A_P` 原值和 route。只对非 P 行编码

\[
U=Z_aE,\qquad \widetilde Z_a=UD,
\qquad E\in\mathbb R^{159\times k},\;D\in\mathbb R^{k\times159}.
\]

当前 P64 教师、P64 底座与设计都保持原值，因而在固定当前 Q/K/V 下，`Γ_t` 完全不变。非 P 行直接计算

\[
U(DG_t),
\]

无需重建两个全场缓存。局部输出变化精确为

\[
\boxed{\delta O_{t,\bar P}=(\widetilde Z_a-Z_a)G_t
=\delta C_aD_{\sigma_t}-\delta A_a\Gamma_{1,t}.}
\]

`E=R,D=Rᵀ` 为一种正交编码；一般双因子表示也满足同样误差式。编码、基矩阵、解码的量化误差可统一放入 `\widetilde Z−Z`，后续乘加舍入另计。精确 P64 既保留方法的信息接口，又使误差能被当前小读出清晰解释。CPU 见证误差式残差 `8.88e−16`。

跨 head 经输出投影的真实误差为

\[
\delta y_i=\sum_h\delta O_{hi}W_{o,h}.
\]

因此，准确的配对 CAL 目标应保留同一次 anchor→hit 的状态与读出，以及 head 间交叉项：

\[
\mathcal J(E,D)=\mathbb E_{(a,t)}\left\|
\sum_h Z_{a,h,\bar P}(I-E_hD_h)G_{t,h}W_{o,h}
\right\|_F^2.
\]

当前 `Z_a` 与 `Γ_t/σ_t` 来自同一条轨迹，通常相关，所以上式的联合期望不能自动拆成两个平均矩阵的乘积。按 head 分别优化的是分头能量之和；完整输出能量还含交叉项。使用 `H·Σ_h‖δO_hW_h‖²` 可给出简单上界。

也可以从 CAL 构造固定读取权重

\[
\mathcal O_h=\mathbb E_t[G_{t,h}W_{o,h}W_{o,h}^TG_{t,h}^T],
\]

作为便于求解的 surrogate。只有读出固定，或状态与读出满足所需联合矩可分离条件（例如相互独立）时，分头平均目标才等于

\[
\|\Sigma_Z^{1/2}(I-ED)\mathcal O^{1/2}\|_F^2,
\qquad \Sigma_Z=\mathbb E[Z_{a,\bar P}^TZ_{a,\bar P}].
\]

使用固定 `O` 的加权 PCA/SVD 是一种主动选择的近似设计，须回到真实配对 `J` 解释效果。这把“对历史做普通 PCA”提升为“按实际消费者读取的历史进行压缩”，同时保留状态与读出的相关性。当前本地只有 CAL 特征变换，没有完整 `C/A` 与未来 `Γ` 的联合数据，所以本轮已经给出构造与可检查的误差对象，未宣称某个 k 已能保持模型质量。

### 3.2 容量与算术成本

下表保持 rho FP32、非 P 编码 BF16、P64 的159维原状态精确保存，并保守计入每层每头独立的 BF16 编码基。40层仅一个 low anchor-hit 周期，route、临时 workspace 另计。

| k | 主要状态含编码基 | anchor 编码 | hit 历史读出 | 相对原历史项的周期新增算术 |
|---:|---:|---:|---:|---:|
| 64 | 6.277 GiB | 1.029 TFLOPs | 0.828 TFLOPs | 1.454 TFLOPs |
| 96 | 9.307 GiB | 1.544 TFLOPs | 1.243 TFLOPs | 2.385 TFLOPs |
| 128 | 12.336 GiB | 2.058 TFLOPs | 1.657 TFLOPs | 3.316 TFLOPs |
| 原独立 C/A | 15.195 GiB | 无联合编码 | 0.408 TFLOPs | 基准 |

hit 动态小解码器 `DG_t` 另需约0.004/0.006/0.008 TFLOPs，已计入最后一列。原命中历史项的费用是 `C diagσ−AΓ_1`，包括缩放与矩乘。

这条表示首先减少持久状态，同时增加稠密编码/解码算术。最终时间由它替代的 HBM 流量、融合方式与小矩阵运行效率决定。它与 INT8 可组合，但其数学收益来自选择任务相关状态方向，而非单纯降低每元素位宽。

## 4. 高噪声恢复吸收进当前 `W_o`

忽略特征端 BF16 舍入，high 每头有

\[
\widehat O_h=S_h(I+B\Gamma_{1,h})+1\Gamma_{0,h}.
\]

把 `W_o` 按输入 head 分块为 `W_h∈R^{128×5120}`，得到

\[
\widehat O W_o
=\sum_h S_h\underbrace{[W_h+B(\Gamma_{1,h}W_h)]}_{W_{h,\mathrm{eff}}}
+1\sum_h\Gamma_{0,h}W_h.
\]

先在 P64 拟合，再构造当前有效输出投影，即可绕过全 N 特征矩阵及逐位置恢复矩阵乘法。主 `N×5120` 到 `5120` 的 `W_o` GEMM 仍执行。

每层真实形状下：

| 对象 | GFLOPs |
|---|---:|
| 当前 `S→64→31` 特征投影 | 25.780 |
| 当前32维残差读出 | 10.377 |
| 构建动态 `W_eff` 与偏置 | 3.303 |
| 主 `W_o` GEMM | 1660.315 |

因此该实数改写每层约少 **32.854 GFLOPs** 的恢复相关主乘法，但保留1660.315 GFLOPs主投影。P64 要用原生教师乘原 `W_o`：若主矩乘跳过 P 行，它替换原本这64行计算；若仍做全N矩乘再覆盖，则另加3.355 GFLOPs。动态 BF16 `W_eff` 净载荷50 MiB，若当前 GEMM 需要权重 packing 还须计入。将固定 `P_vT` 预合成为128×31矩阵，本身已可把特征算术从25.780降至10.053 GFLOPs；这些构造的收益不能重复相加。

实际冻结特征存在 `ε=bf16(SB)−SB`，故折叠遗漏的特征舍入项恰为 `εΓ_1W_h`；最终 attention 输出 cast 再经 `W_o` 还有一项。保留原舍入时，可以融合逐 tile 的原特征和恢复 epilogue；完全吸收进有效权重则是另一数值路径。CPU 验证实数折叠差 `1.42e−14`，特征舍入缺陷公式残差 `1.61e−14`。

low 的 `(1−ρ_i)S_i` 为 query/head 相关缩放，外加独立历史项，不能整体由一个 query 共享 `W_eff` 覆盖。把多个输入字段拼接为更大 GEMM 虽可写成恒等式，却扩大乘法。low 更合适的精确执行入口是 tile 融合；联合缓存压缩则提供新的小状态读出。

## 5. P64 先行的真正依赖边界

旧方案先完成全场 selected，再只对 P64 decode。现在可以更早切开：

1. 全 K/V、route 和当前 value 矩已就绪。
2. 仅计算 P64 的 selected/full-key 教师、P64设计与底座，解出相同系数。
3. 按 query tile 计算其 selected，立即生成特征、历史读出和恢复输出。
4. tile 继续交给 `W_o` 与后继逐 token 运算。

selected(P64) 若在随后全场 attention 再算一次，额外 selected 工作上界仅 `64/31668=0.2021%`；实现也可跳过已经完成的 P 行。由于真实 attention 的 query 行互不依赖，P 拟合无需等待全部 selected 行。全 N 的设计、base、estimate 中间物化可以消除或缩为 tile，原完整历史状态仍需保存。本轮 CPU 分块读出差0，模拟最终 BF16相同；GPU融合应保留数值语义或明确它的变化。

整块跨 self-attention→cross→FFN 的 P64先行依赖由主报告另行展开；下一层 self-attention 的全 K/V 是新的全局屏障。这里不把分块输出等同于删除主要矩阵运算。

## 6. 当前状态是否已经达到无损极限？

对固定当前设计 X，令 `K=X(X_PᵀX_P+αI)⁻¹X_Pᵀ`。在 P 覆写之前，历史扰动经过 `I−KP`。其非单位特征值为

\[
\frac{\alpha}{\sigma_j(X_P)^2+\alpha}>0.
\]

所以当前 `α=6.4` 下，这个算子可逆；直接删掉“落在拟合特征空间里的历史”会留下可计算的 ridge 收缩项。截距也正则化，因此没有由无正则截距产生的额外平移规范。

P 覆写以后算子变为 `Q(I−KP)`，`Q=I−PᵀP`。固定 X 时秩是 N−P，确实有 P 维核，但该核由当前 X 决定；anchor 时未来 X 尚未知。这个无损商仅涉及约0.2%的行，且多种未来读出的联合核需要重新求交。CPU实例 N257、P64 给出覆写前秩257、覆写后秩193，与公式一致。

联合缓存的未来读出族 `G_t` 通常也需要159维的联合行空间才能对任意状态精确恢复；单次读出只有128输出并不意味着可在未来读出未知时固定压成128维。用技能 C03 的语言，固定线性编码 E 必须满足每个未来读出 `ker(Eᵀ)⊆ker(G_tᵀ)`；最小维数由所有未来读出的联合跨度决定。当前实际模型分布仍可使该任务状态高度近似可压缩，这正是第3节联合 Gramian 构造的用途。

因此，本轮找到的理论空间有清楚分工：**单遍分区归约减少确定的重复计算；P64先行减少无必要的全场物化；high W_o折叠改变收缩路径；联合历史编码用任务读出定义可控的状态近似。** 当前实现还没有把这些表示自由度全部用尽。
