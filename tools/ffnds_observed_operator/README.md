# 观测条件 FFN 算子

`core.py` 实现两种有限割线构造的 FP64 参考计算。原生 BF16 生成入口为 `wan_full80_job.py` 和 `wan_scope6_job.py`，逐位置教师入口为 `student_query.py` 和 `student_query_scope6.py`。相应的独立复审、冻结范围和执行身份收录于本轮结果。[实测、源码身份与回执](../../reports/20261118/observed_secant_ffn_operator/README.md) · [完整设计](../../research/20261118/OBSERVED_SECANT_OPERATOR_DESIGN.md) · [实验方案](../../research/20261118/OBSERVED_SECANT_EXPERIMENT_PLAN.md) · [独立审查](../../research/20261118/OBSERVED_SECANT_INDEPENDENT_REVIEW.md)

1. 固定廉价端点特征、当前完整 FFN probe 残差、在线全输出读出；
2. 少量隐藏系数模式、当前 probe 辨识共享系数、编译当前满秩矩阵。

参考核心只依赖 NumPy，其单元测试使用小型合成矩阵验证 FP64 代数。校准、原生绑定和视频作业使用各自审定的模型与运行环境。

## 全 80 读出的成本与修正方向

[完整成本与关系保持分析](../../research/20261118/READOUT_COST_COHERENCE_REASSESSMENT.md)区分主要 MAC、完整 observer 权重访问、研究记录 I/O 与真实部署性能；并给出保持 anchor 和全输出读出的关系加权割线拟合。当前实现未切换到该新目标；固定特征/参数和已发布视频保留原身份。

## 结果解释与模式基底审查

0392 全 80 在线读出已产生相对静态控制的完整视频保留，接近仍有全画幅损坏的 Global-P64。详见[结果与算子分析](../../research/20261118/OBSERVED_SECANT_RESULT_ANALYSIS.md)。

当前已运行的 `mode_local._structural_fit` 在 source 拟合与 POD 中使用 GELU 后增量 ΔH，`mode_templates` 和 `compile_matrix` 使用前激活增量 ΔZ；这与设计中基底生产也应使用 ΔZ 的约定不同。后续 theta/prior/lambda 确实按部署模板拟合，所以当前视频对应合法但选基目标不同的算子。继续模式路线前应在一致的 ΔZ 作用下重建基底，不重用旧参数宣称语义已修复；本次分析没有改写原实验代码或参数。

## 运行

从仓库根目录：

```bash
python -m unittest discover -s tools/ffnds_observed_operator/tests -v
```

## 张量约定

| 对象 | 形状 | 含义 |
|---|---|---|
| `U` | `[N,d]` | norm/AdaLN 后、W1 前的 token 行 |
| `W1`, `W2` | `[m,d]`, `[d,m]` | 原生 FFN 权重 |
| `Phi`, `X` | `[N,r]` | 端点特征、两端特征差 |
| `V` | `[r,d]` | 当前有符号全输出读出 |
| `modes` | `[q,m]` | 原生隐藏系数模式 |
| `templates` | `[q,p,d]` | 各模式在 probe 输入位移上的作用 |
| `probe_rows` | `[p]` | 当前数组行号；与全局 token ID 区分 |

全部预测为 **pre-gate** FFN 输出。共享模式的拟合可以使用当前 gate 度量；返回值不乘 gate。
原生调用方在真实残差边界应用一次 gate，并绑定 expert、层、token 顺序和缓存窗口。

## 在线读出

Anchor 保存 `C0 = F0 - U0 @ A` 与 `Phi0`。Hit 计算
`base = U1 @ A + C0`、`X = Phi1 - Phi0`，仅给 observer 传入选中的 `p` 个当前 token：

```python
anchor = make_observed_anchor(U0, F0, A, Phi0)
prediction, receipt = observed_hit(
    U1, Phi1, anchor, probe_ffn,
    p=64, ridge=0.1, cal_floor=0.01,
    prior=V_static, cal_gram=Gamma_cal,
    policy="fixed", token_ids=global_token_ids,
    fixed_ids=frozen_probe_ids,
)
```

示例 `ridge=0.1` 只是调用演示；实际冻结值由研究方案规定，不能从该示例推断已经拟合。
`prior=V_static` 使用同特征的静态完整输出读出；`prior=None` 对应零先验消融。
`adaptation=False` 通过同一 `C0/Phi0` 路径返回静态先验预测，且不调用 observer。
其数学对照是 `base + X @ V_static`；移植时不把它换成采用另一种 BF16 残差缓存的旧端点程序。

求解目标为：

\[
\frac1p\|X_PV-R_P\|_F^2
+\lambda\operatorname{tr}\left[(V-V_0)^\top
\left(\frac{X^\top X}{N}+\tau^2\Gamma_{\rm CAL}\right)(V-V_0)\right].
\]

`cal_floor` 是 \(\tau\)，默认 0.01。生产规格使用正的 \(\lambda\) 与 CAL 尺度正则底；
实际 \(\tau\) 按设计文档中的一次 CAL 数值差异估计与 0.01 取最大值。
`cal_floor=0` 保留此前纯当前预测度量的代数极限，不能据此声称近零位移噪声会衰减。

`Gamma_cal` 是已冻结的 SPD 特征度量；省略时为 CAL 标准化坐标中的单位矩阵。
先用它白化，再对当前特征做 SVD；默认保留当前 RMS 奇异值大于
`max(1e-12, 1e-10 * largest)` 的方向。在该支撑上用 Cholesky 解正则系统。
无正则模式仅用于代数极限；probe 无法辨识有效支撑时显式拒绝无正则求解。
对可逆换基 `X -> X @ T`，同时变换 `Gamma_cal -> T.T @ Gamma_cal @ T`
和 `V0 -> solve(T,V0)`，预测保持不变（数值误差内）。

`paired_features` 实现 `[Z_K, GELU_tanh(Z_K)-beta*Z_K] / CAL_scale`。
研究主配置为同一组 16 个原生通道、总计 32 列；函数允许小规模测试尺寸。
`beta`、通道列表、两块特征的 CAL 差分尺度均由离线准备提供，本核心不拟合这些资产。

`select_probes` 有两种明确策略：

- `fixed` 使用给定全局 ID，或对排序后的全局 ID 使用固定 seed42 排列。后者不宣称与历史 P64 的真实 64 行相同。
- `geometry` 使用当前白化特征的贪心 log-det 信息增益，信息矩阵从单位矩阵开始；近似平分时选择最小全局 ID。每轮使用秩一更新，复杂度约 \(O(pNr)\)。

观测策略不读取真实 residual。观测行不会以真实 FFN 输出替换预测，全部行统一使用拟合的算子。
几何选点只用于当前已表示特征的可辨识性，不能保证覆盖特征之外的稀有响应。
诊断中的 `*_rms` 是 token 向量的均方根，即 Frobenius 范数除以 token 数的平方根；
没有再除以输出通道数 `d` 的平方根，不能直接当作逐元素 RMS 或 MSE。

## 共享满秩模式

\[
B_j=W_1^\top\operatorname{diag}(a^{(j)})W_2^\top,\qquad
\widehat F_1=F_0+\Delta U\left(A_0+\sum_j\theta_jB_j\right).
\]

`native_mode_templates` 直接计算
`(delta_U_probe @ W1.T) * modes[j] @ W2.T`，返回 `[q,p,d]`。
`fit_mode_coefficients` 在当前 gate 度量下求同一组共享系数；`cal_metric`
为冻结的 CAL 模式作用 SPD 度量。省略时对应已经归一化的模式坐标。
`compile_native_modes` 先组合隐藏系数，再进行一个 `[d,m] @ [m,d]` 收缩。
整个流程不物化 `[q,d,d]` 矩阵组，也不物化隐藏对角矩阵。
模式的 `adaptation=False` 返回使用相同 `U0/F0` 缓存和编译顺序的
`A0 + compile_native_modes(theta_static, ...)` 静态对照，且不执行 probe。
主比较显式传入同一组模式在 CAL 上拟合的 `prior=theta_static`；省略 prior 只表示零先验消融。

```python
anchor = make_mode_anchor(U0, F0, A0)
prediction, receipt = mode_hit(
    U1, anchor, probe_ffn, W1, W2, modes, current_gate, probe_rows,
    ridge=mode_lambda, prior=theta_static, cal_metric=mode_cal_gram,
)
```

## 原生移植边界

参考核心将输入显式转换为 FP64，因此它不能验证 BF16 的 GEMM 累加、GELU 舍入或缓存误差。
原生规格拟保存 BF16 的 `C0`、FP32 的 `Phi0`；相对 BF16 `Nd` 状态，额外 `Nr` 特征状态为
\(2r/d\)，当前 \(r=32,d=5120\) 时是 **1.25%**。不要继续沿用双 BF16 缓存的 0.625% 数字。
共享模式保存 BF16 的 `U0,F0`，其状态是 `2Nd`。完整成本还包含真实 probe FFN、求解、布局与权重读取。

在模型移植中，anchor 与 hit 必须使用绑定后的同一特征数值程序，正确应用当前 gate，并在窗口结束后丢弃缓存。
同输入时的数学端点恒等式与 BF16 逐位相等分别验收。主实验范围、BF16 动作顺序、离线训练和视频预算由研究设计文档规定。
