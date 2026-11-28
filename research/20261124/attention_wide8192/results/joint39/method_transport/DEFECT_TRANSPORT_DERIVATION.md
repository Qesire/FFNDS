# 从历史缺陷到当前响应：值空间运输与可识别作用

2026-11-25。依据 FFNDS 研究分支 `7c8196fa5380dbe67dd9772f942553085fde55b3`。本报告推进上一轮“当前分组底座＋完整缺陷锚点＋P64 创新恢复”的具体数学机制。使用 operator-tensor-analysis 的算子作用、有限差分、群等变、数值线性代数，以及 active-abstraction 的任务充分表示与状态闭合判据。

新增执行为 NumPy/SymPy 数学计算；128维见证中的 Q/K/V 全部为明确合成张量。真实归档 P_k 仅用于选择分组方向，真实代码用于核对已有 low OUT-D 语义，真实模型尺寸用于预算。没有新增 Wan 激活、GPU 推理、视频或计时。

## 1. 缺陷是一个零质量的线性作用

固定层、head、selected 支持和分组。query轴 i、key轴 j、value通道 ν。以 S_s 和 S_s^0 分别表示完整 attention 与分组底座在原始 key 上的概率矩阵：

\[
A_s=S_sV_s,\quad G_s=S_s^0V_s,\quad
R_s=S_s-S_s^0,\quad E_s=R_sV_s.
\]

未 selected 组内每个原 key 的粗权重为 exp(q·k̄_c/√d)/Z⁰，组内 n_c 项相加就是摘要质量；selected 支持上使用真实指数权重。因此 S_s^0 非负、每行和为1，和 S_s 一样满足概率归一化。由此

\[
R_s\mathbf1=0.
\]

这个零质量条件意味着缺陷对共同 value 平移不变。其有限变化有精确分解

\[
\boxed{E_s-E_a=(R_s-R_a)V_s+R_a(V_s-V_a).}
\]

第一项是路由缺陷的变化，第二项是历史路由缺陷作用于当前 value 变化。两者可以同时存在；完整缺陷恒定只是其中一个精确作用类。

## 2. 低成本闭合：缓存标准化缺陷形状

对每个 head、每个 value 通道计算全 key 均值与标准差：

\[
\mu_s^\nu=\frac1M\sum_jv_{s,j}^\nu,\qquad
(\sigma_s^\nu)^2=\frac1M\sum_j(v_{s,j}^\nu-\mu_s^\nu)^2,
\quad D_s=\operatorname{diag}(\sigma_s).
\]

令 D_s^+ 为对正标准差取倒数、零标准差位置取0的对角伪逆，定义

\[
\widetilde V_s=(V_s-\mathbf1\mu_s)D_s^+,
\qquad C_s=E_sD_s^+=R_s\widetilde V_s.
\]

若 σ_s^ν=0，该 value 通道在 key 轴上为常数。R_s1=0 保证该通道缺陷严格为0，所以仍有

\[
E_s=C_sD_s.
\]

anchor 保存完整 C_a；当前底座为

\[
\boxed{G_s+C_aD_s.}
\]

相应 P64 目标改为标准化缺陷创新 ΔC_s=C_s−C_a。它保留了完整的逐位置、逐通道缺陷场；decode 使用当前 value 尺度。历史 D_a 只参与初始化 C_a，随后可以释放，不必增加持久字段。

### 2.1 新增可精确处理的有限变化类

若路由缺陷保持不变 R_s=R_a，且每个 value 通道发生全 key 共同的正仿射变化

\[
V_s=V_a\Gamma+\mathbf1b,
\qquad\Gamma=\operatorname{diag}(\gamma_\nu),\quad\gamma_\nu>0,
\]

则 D_s=D_aΓ，标准化 value 不变，所以 C_s=C_a。因此

\[
G_s+C_aD_s=A_s.
\]

这个精确类允许各通道幅度与均值发生有限、大幅变化。原 raw 缺陷底座 G_s+E_a 在同一情形的误差为 E_a(I−Γ)，而新的状态直接执行了对应作用。它增加了对这类值空间变化的精确响应；一般“E_s=E_a”的情形和“C_s=C_a”的情形是两种不同的作用假设，不能据此声称两种算法的全部精确输入集合存在包含关系。

路由相同还可以由有效 Q/K 的对偶变化产生。对可逆 L，令

\[
Q_s=Q_aL^{-\top},\qquad K_s=K_aL+\mathbf1\kappa.
\]

完整 logits 只增加逐 query 常数；冻结组的 K 均值作相同变换，所以完整与粗概率都保持，R_s=R_a。该结论作用于 norm/RoPE 后的有效操作数；实际 Wan 轨迹是否落入这个作用类，由真实输入决定。冻结组成员与支持是此结论的条件，当前重新分组属于另一种更新规则。

### 2.2 剩余创新的解释更加具体

对一般当前输入，标准化创新仍有精确有限分解：

\[
\boxed{
\Delta C_s=(R_s-R_a)\widetilde V_s
+R_a(\widetilde V_s-\widetilde V_a).
}
\]

因此探针承担的是路由缺陷变化与标准化 value 形状变化。共同的每通道正增益和平移已经由当前统计与底座直接处理。

若恢复器 K_s 由当前特征构成，缓存为 C_a+ε_C，最后覆盖 P，则

\[
\boxed{
\widehat A_s-A_s
=M_H(K_sP-I)(\Delta C_s-\varepsilon_C)D_s.
}
\]

标准化坐标中的未恢复变化，最终由当前 D_s 还原成原 value 单位。该式针对固定当前输入、固定特征，数值误差另列。

### 2.3 完整接口的正对角仿射等变

上一轮32列特征可以具体改写为

\[
\psi_i=[1,\ q_iP_{k,15}/\sqrt d,\
((G_i-B_i)D_s^+)P_{v,15},\ Z_i^S/Z_i^0].
\]

以标准化创新为 ridge 标签，输出修正乘 D_s。保持有效Q/K、partition/support、P位置和投影固定，在 anchor 与当前分别施加可以不同的正仿射变化：

\[
V'_a=V_a\Gamma_a+\mathbf1b_a,\qquad
V'_s=V_s\Gamma_s+\mathbf1b_s.
\]

只要各 Γ 为正对角，标准化历史状态 C_a、当前标准化标签 C_s 和当前特征均各自不变。因此同一 ridge 系数保持，最终输出满足

\[
\widehat A'_s=\widehat A_s\Gamma_s+\mathbf1b_s.
\]

这允许 anchor 与 hit 有各自的幅度和均值，并且在非零创新、非零恢复误差时仍保持上述等变关系。固定正则不会破坏这个性质，因为求解坐标与标签都不变。当前均值由正权底座 G_s 携带；零质量缺陷只需要尺度 decode，不给 C_a 额外加value均值。

这个性质针对固定模型 value 通道坐标中的正对角作用，不能视作任意改动完整Wan内部状态后的模型不变性。Q/K视为给定有效操作数；decode恢复当前应有的幅度和均值，后续W_O与FFN接收相应变化。一般线性混合需要下节的更宽表示。数学定义在 σ=0 时使用明确零分支（正增益保持这个分支）；实现中的 scale floor 若被激活，其影响属于额外数值项，需按实际阈值记录。

### 2.4 尺度很小时的数学界与数值边界

每行 R_i 是两个概率行之差，故 ||R_i||₂≤√2。标准化 value 的每个非零通道具有 key 轴范数 √M。因此

\[
|C_i^\nu|\le\sqrt{2M}.
\]

小 σ 本身不会令精确标准化缺陷无界；σ=0 的缺陷列严格为0。这个界描述数学对象，不是BF16误差保证。

真实 anchor 输出可能已是 BF16。若 A_a、G_a 带误差 δA_a、δG_a，则初始化标准化缓存的误差包含 (δA_a−δG_a)D_a^+；之后由当前 D_s decode。共同 value 均值很大、缺陷很小时，A_a−G_a 的相消和原输出舍入需要单独计入，理论零质量性质不能恢复已经丢失的浮点信息。生产实现应使用已有原生输出与明确的 FP32 差分/统计路径，不能把另算一次 centered-V full attention当作免费构建。

## 3. 一般值空间运输：只识别未来需要的作用

设当前 value 可表示为

\[
V_s=V_aT+\mathbf1b+r.
\]

由零质量性直接得到

\[
\boxed{
E_s=E_aT+(R_s-R_a)V_s+R_ar.
}
\]

这给出一个更宽候选：先让完整历史缺陷执行当前 value 的通道变换 T，再由真实教师恢复未覆盖项。它允许当前底座 G 继续处理已经直接计算的交互。

### 3.1 当前组均值提供独立于原生 P64 的观测

使用固定组，令 w_c=n_c/M。构造

\[
X_c=\sqrt{w_c}(\bar V_{a,c}-\mu_a),\qquad
Y_{s,c}=\sqrt{w_c}(\bar V_{s,c}-\mu_s).
\]

精确仿射演化时 Y_s=XT。当前 Y_s 来自粗底座本来就在计算的 value 组均值。它不需要当前原生全行 A_s，也不消耗额外 full-key P 查询。

anchor 预计算 L_X=X^+ 与 P_0=I−X^+X；hit 定义

\[
\widehat T=P_0+L_XY_s.
\]

P_0 为未观测方向采用恒等运输，避免任意把它们清零。满列秩时 P_0=0，只需保存 L_X。

### 3.2 目标作用可识别性弱于整个 T 的可识别性

希望从 Y=XT 唯一确定的是 E_aT。其必要且充分条件为

\[
\boxed{\ker X\subseteq\ker E_a.}
\]

等价地，row(E_a)⊂row(X)，或 E_a=E_aX^+X。

证明：若 X(T_1−T_2)=0，每一输出列的差都属于 ker X；条件保证 E_a(T_1−T_2)=0。反之，若存在 z∈ker X 且 E_az≠0，取 T_1−T_2=z uᵀ，就得到相同 Y 与不同目标作用。

因此 X 满列秩128是一个充分条件；即使 X 秩亏，目标作用仍可能完全可辨识。此时

\[
E_aT=E_aX^+Y.
\]

其观测扰动放大由 ||E_aX^+|| 控制，通常比粗略的 ||E_a||/σ_min(X) 更精确。这是面向实际读出的稳定性，所需的是完整历史缺陷怎样读取那些不稳定方向。

### 3.3 近似仿射情形的精确误差归属

记当前组观测剩余为 ρ=Y_s−XT，则

\[
\widehat T-T=P_0(I-T)+X^+\rho.
\]

运输作用的误差为

\[
\boxed{
E_a(\widehat T-T)
=E_aP_0(I-T)+E_aX^+\rho.
}
\]

前项对应观测未覆盖的历史作用，后项对应当前组均值中的模型剩余。再结合 E_s 的有限分解，就能把运输后创新归到未观测作用、组均值剩余、路由缺陷变化与组内 value 剩余四个具体来源。随后仍可用同一 P64 剩余恢复接口。

## 4. 与当前 low OUT-D 的源码区别

已核对 `attention_online_structured.py`：

- rho_a=1−Z_selected/Z_full，是逐query遗漏概率质量。
- u0=A_a−(1−rho_a)B_a，保存为BF16。
- hit base=(1−rho_a)B_s+u0。
- scale_r 来自固定 CAL 拟合文件，用于标签与残差的单位缩放。

因此当前 rho_a 是 attention 质量，scale_r 是固定校准尺度；它们都不是当前/anchor value 标准差之间的运输。新 C_a 的语义是“零质量缺陷的标准化形状”，decode 使用当前 value 尺度。其精确作用类与参数作用位置都已明确。

标准化、仿射运输本身属于已有数学操作；本轮贡献应表述为对当前残差缓存组织的具体推进，并与最接近的原始方法比较。不能仅凭新记号给原有 residual caching 或 P64 ridge 重新申报新颖性。

## 5. 真实尺寸的构建、读取与状态成本

N=31668，H=40，d=128，40层，四步有80个anchor站点、80个hit站点；两组方案每head最多496个组槽。

**逐通道标准化版本：** 完整 C_a 取代 E_a，持久状态净增0。D_a初始化后可释放；即使保留全部40层FP32尺度，也仅0.78125MiB。每层hit增加162,140,160个decode乘法，能否与已有读出融合取决于实现。当前value均值/方差需真实归约，归约可与构建组均值共享读取；本轮没有测量此融合。

**完整 T 版本：**

| 项 | 静态结果 |
|---|---:|
| 每层由当前组均值构造 T | 325,058,560 MAC |
| 每层 E_a T | 20,753,940,480 MAC |
| 两者相对 full attention QK/PV 成对 MAC | 0.205262% |
| 四步80个hit合计 | 3.37264 TFLOP |
| L_X FP32，40层持久 | 0.378418 GiB |
| P_0 FP32，40层持久，秩亏完成时 | 0.097656 GiB |
| E_a＋组标签＋L_X | 12.505990 GiB |
| 再加P_0 | 12.603647 GiB |
| 当前层 T FP32 工作区 | 2.5 MiB |

anchor 另有3200次496×128的小分解和伪逆构建。满列秩使用显式thin-Q的Householder QR加三角求解，其主要算术估算约0.12108TFLOP/四步；秩揭示SVD/QR的实际执行与稳定性另按所选算法计入。已知满秩时无需构建/保存P_0；不能将小系统构建称作免费。

运输需要读取完整 E_a：80个hit为24.16077GiB的BF16净读取；若将 E_aT 全场物化为FP32，则另有48.32153GiB输出写入。分块融合可改变工作区与流量，不能直接由MAC比例推导延迟或新峰值。

## 6. 本轮实际核验结果

`verify_defect_transport.py` 在96 query、768 key、128维的明确合成数据上执行，使用归档P_k第一列分组，共192个非空组：

| 核验 | 结果 |
|---|---:|
| R1的最大绝对值 | 3.17e−16 |
| 原坐标有限路由/value分解残差 | 5.33e−16 |
| 标准化路由/形状分解残差 | 2.97e−16 |
| Q/K对偶变换＋key平移后的概率变化 | 2.29e−16 |
| 0.2–5倍各通道正增益＋平移，标准化底座误差 | 6.22e−15 |
| 同例raw缺陷底座Fro误差 | 6.64958 |
| anchor/hit采用不同正仿射变换，cache差 | 9.44e−16 |
| 同例当前feature差、ridge系数差 | 1.55e−15、4.91e−16 |
| 同例完整候选输出等变残差 | 4.00e−15 |
| X实际合成秩、条件数 | 128、9.18851 |
| 组均值恢复全T后的输出误差 | 5.77e−15 |
| 秩亏但目标作用可辨识的作用误差 | 3.55e−15 |
| 同例T参数仍有的Fro差异 | 12.20656 |
| 两个观测相同但目标作用不同的例子 | 观测差0，目标差6.40312 |
| 常量value通道的精确缺陷 | 0 |
| SymPy有限仿射运输恒等式 | 通过 |

所有小误差是公式与构造核验，合成条件数也不代表真实模型条件数。真实Wan的分组均值谱、标准化缺陷变化及成片收益需要相应激活或运行证据。

## 复算与来源

```bash
OPENBLAS_NUM_THREADS=1 python verify_defect_transport.py --basis /path/to/P_k.npy
```

省略 `--basis` 时使用固定种子的合成方向，仍可复算数学关系；使用实际归档方向时，脚本将其SHA256记录在JSON中。

- `verify_defect_transport.py`：完整核验程序。
- `defect_transport_verified.json`：实际执行结果、形状预算、真实源码与P_k指纹。
- [上一轮当前交互构造](https://github.com/Qesire/FFNDS/blob/7c8196fa5380dbe67dd9772f942553085fde55b3/research/20261124/attention_wide8192/results/joint39/TWO_GROUP_CURRENT_INTERACTION.md)。
- [实际在线attention](https://github.com/Qesire/FFNDS/blob/7c8196fa5380dbe67dd9772f942553085fde55b3/research/20261124/attention_wide8192/results/joint39/attention_online_structured.py)。
