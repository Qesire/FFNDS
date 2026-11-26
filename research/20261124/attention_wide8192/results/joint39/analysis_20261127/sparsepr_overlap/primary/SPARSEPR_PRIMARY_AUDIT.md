# SparsePR v1/v2：恢复算子与相似性边界

核验日期：2026-11-27。阅读两版 §4.2、附录 A–C；v2 另核查 D.3–D.6。官方 GitHub contents API 本轮仍只返回 README.md（Git blob `d4c7b77a639c9542bf43392b56ce17636831ae69`），因此本报告区分论文定义与未发布的实际作者内核。

## 1. 论文实际定义

令 S 为 renormalized sparse attention，A 为完整输出，R=A−S。两版均使用 S 的全输出通道作为回归输入，P64 实测 R_P。query-response 坐标只参与分组、probe选择，不进入回归。采用 probe 覆盖权重、逐通道加权标准化、残差中心化与 ridge；bias 单独加回、不惩罚，最后 probe 行覆写 A_P。

- **v1（2026-08-19）**：§4.2 Eq.(8–10)，λ=0.1；将预测的中心化残差右乘由实测 R_P 的加权 SVD 得到的 rank16 输出投影。
- **v2（2026-09-27）**：§4.2 Eq.(7–8)、A.3 Eq.(A.11–19)，λ=1；默认无输出截断、无残差 SVD、无 norm cap。中心化预测自然属于 probe 残差行空间。

v2 C.1 复用旧 centroid/factor 作为迭代初值；没有全位置历史输出残差。应称“当前调用的输出恢复，无完整历史响应状态”，避免称完全无状态。

## 2. 已有消融的范围

v2 D.3 改 probe 数、输出截断rank、λ；D.4 比较均值、scalar、diagonal、full affine；D.5 用当前 S 的 SVD 做全标签 oracle 表示诊断；D.6 比较在相同 probe 上训练的32宽 GELU MLP。本次未在这些部分发现离线 CAL 输入投影、跨步残差状态或历史运输消融。

因此，将当前方法的 CAL 输入投影、惩罚 bias、不同 λ、固定 probe 与作者的加权标准化实现区分，是准确的实现辨析。它们仍属于 affine(S) 恢复家族中的不同限制与估计器，单靠这些差别不足以论证新的恢复原理。v1 的 rank16 是输出方向截断，应与输入维度压缩分别比较。

## 3. 从 v2 算子推出的条件等变性

以下为本次推导，不是论文声称的定理。固定完整和 sparse 权重、probe P、覆盖权重 W；对非退化 probe 特征方差取 ε=0；保持同一 λ>0；使用 v2 无输出截断恢复。令 Γ 为正对角阵，V'=VΓ+1c^T，则

\[
S'=S\Gamma+\mathbf1c^\top,\quad R'=R\Gamma,
\quad\mu_S'=\mu_S\Gamma+c^\top,\quad D_S'=D_S\Gamma.
\]

故标准化特征 \(\bar S'=\bar S\)，中心化目标 \(\bar R_P'=\bar R_P\Gamma\)，相同 ridge 求解满足 \(B_\lambda'=B_\lambda\Gamma\)，从而

\[
\widehat A'=\widehat A\Gamma+\mathbf1c^\top.
\]

这表示估计器的等变性；其已有误差也按 Γ 变化，并不说明该估计器在任意稀疏残差上精确。

保留 ε>0 时，设原始逐通道方差为 σ_j²，则标准化特征改为 \(\bar S'=\bar S T\)，其中

\[
T_{jj}=\frac{\gamma_j\sqrt{\sigma_j^2+\epsilon}}{\sqrt{\gamma_j^2\sigma_j^2+\epsilon}}.
\]

令 \(H=\bar S_P^TW\bar S_P\)、\(C=\bar S_P^TW\bar R_P\)，则变换后的中心化预测可写为

\[
\bar S\,(H+\lambda T^{-2})^{-1}C\Gamma.
\]

ε可忽略时 T≈I；纯平移时 T=I，条件平移等变性仍精确。固定 λ=1 本身不破坏 ε=0 时的性质。v1 的各向异性输出 SVD 截断一般不与 Γ 可交换，所以这一定理明确绑定 v2。

实际路由 B.3 的 value-error score 依赖 V，值变换可以改变支持；全系统等变性需要另核对路由，因此不能从上述固定支持推导直接外推。

## 4. 对本项目判断的影响

P64 原生输出、ridge、残差覆写，以及 v1 的输出子空间都已有明确先例。对 Chipmunk 的“静态残差在尺度变化下留下误差”构造，不能直接用来区分 SparsePR；v2 的标准化当步回归同样具有上述条件尺度适应性。更有区分力的是：全位置历史保存了哪些当前 sparse-output/probe 恢复器不能重新推断的方向，以及运输后是否以更少成本保持这些方向。

## 原始来源

- [SparsePR v1](https://arxiv.org/html/2608.18484v1)：§4.2、A.3、C.2；当前probe残差拟合、rank16、λ=0.1。
- [SparsePR v2](https://arxiv.org/html/2608.18484v2)：§4.2、A.3、C.1–2、D.3–6；无输出截断、λ=1及消融。
- [arXiv版本记录](https://arxiv.org/abs/2608.18484)：v1 2026-08-19，v2 2026-09-27。
- [官方仓库](https://github.com/PardisTaghavi/SparsePR)：本轮 API 仅 README；代码待发布。
