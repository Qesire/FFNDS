# D176 的代码语义与信息流审查

审查对象：Qesire/FFNDS@0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c 的 D176 实现、导出收据和逐站点结果，以及前一轮封存的 joint39 执行与原 attention/FFN 读出。此审查只读代码并重算已发布统计；未执行新的真实模型或学校资产回放。

## 1. 本轮实际检验了什么

D176 检验一个具体的函数族：在固定底座 flow 上，拼接常数、跨头均值31维、cheap FFN 的 q16，以及两种固定 P64 核列，随后只用 P64 的64维终端 flow 拟合一次统一岭回归。

它保留了若干原方法的可计算坐标，也明确改变了恢复对象、算子约束、监督位置和正则几何。该实验为“哪些坐标能解释终端误差”提供证据；将其结果解释为原 FFNDS 恢复机制整体扩展的成败，会越过实际实验的对象。

## 2. 同样 P64，对应不同监督信息

| 路径 | 实际 P64 标签 | 跨行恢复的主要结构 |
|---|---|---|
| 原 attention OUT-ABS | 每头 P×128 的原生 attention 输出；40头 | 每头32维特征分别恢复128维输出，随后使用真实 W_O 与后续模块 |
| 原 attention FIN-C0 | 同上 | 每头矩特征响应加当前 value 方向的乘性质量校正 |
| 原 wide FFN | 原生 attention 输入上的 P×5120 原生 FFN 输出 | q16 对固定输出方向的 gain，以及当前激活割线核和既有 inverse metric |
| cheap→D176 flow | P×64 的终端 flow | 统一176列线性读出，无逐头输出约束 |
| B11_fixed→D176 residual | 同一 P×64 终端标签减覆盖前 B11_fixed | 原结构留在底座中，再拟合同一批 P 的剩余误差 |

执行代码 attention_joint39.py 在 native attention 分支上计算原生 FFN P 标签，分别供旧 FFN 拟合；attention_panel_fit.py 的 readout 则以原生逐头输出为监督。D176 的 solve 只把 flow_B00 作为 truth。

P 数目相同意味着完整查询位置相同，并不意味着恢复时保留了同样的监督分解。P×64 flow 是上述中间观测经过模型图后的任务输出。它有助于直接针对最终作用，但不自动保留逐头或 FFN 输出的结构约束。反过来，64维任务目标也不要求完整恢复5120维中间态；判断条件应是现有观测能否区分会改变目标的剩余变化。

用允许的缺陷空间 V、观察算子 O、目标算子 T 表达，精确恢复要求：

\[
\ker(O|_V)\subseteq\ker(T|_V).
\]

这个条件针对结构约束后的缺陷空间，而非用参数总数直接判定方法成败。

## 3. D176 的 FFN 核坐标来自 cheap_U

attention_joint39_export_ffn.py 明确调用：

```python
_, q, x = ffn_operands(data["cheap_U"], cached, prepared)
```

并以旧 joint_phi 的 q16 逐位匹配作为真实性检查。该 x 对 cheap 和 B11_fixed 两种底座完全共用。

原 joint39 的独立 FFN 预测则用 candidate attention 路径的 a_op=(base,q,x)；其拟合 probe 坐标取自 native attention 路径 n_op。cheap_U、candidate U、native U 是三个有明确语义的状态。因此，“加入原生 FFN 核”准确指复用了原 kernel 坐标构造与锚点规则；此次坐标是在 cheap 路径上计算，未恢复独立 FFN 分支使用的全部核几何。

这不是导出错误：导出按照预定 cheap 字典完成，q16、真实入块状态和四步背景轨迹均有匹配收据。它限定了本轮的解释范围。

## 4. 核列线性读出与原核算法有不同几何

令 cheap 输入的 FFN 核为 K_F(i,p)=x_i x_p^T/2560，P-RMS 对角缩放矩阵为 S_F。D176 将

\[
Z_F(i,:)=K_F(i,P)S_F^{-1}
\]

作为64列特征，再统一做 ridge。因此这部分诱导的 query kernel 是

\[
\widetilde K_F(i,j)
=K_F(i,P)S_F^{-2}K_F(P,j).
\]

原 FFN 使用直接 K_F(i,P)，在 P 上解 K_F(P,P)+M，同时保留 q16 对已知输出方向 v 的 gain 约束；不是上述核列再次平方的统一 ridge。原实现的形式为

\[
\widehat F_i=b_i+(q_i\odot g)V
+\tfrac12 K_F(i,P)C,
\]

其 g 与 C 来自有 inverse metric 的联合拟合。D176 用相同核坐标作为字典，保留了信息来源的一部分，却未保持同一恢复算子或相同函数先验。

简单未缩放、单核情形下，直接 K ridge 的探针谱滤波是 s/(s+\mu)，核列再做 ridge 的谱滤波是 s^2/(s^2+\alpha)。P-RMS 后需对实际矩阵求谱，不能直接套这条标量式；它只说明两种过程不能视为相同算法。

## 5. FIN-C0 的当前 value 乘性结构没有完整进入 D176 cheap 分支

原 readout 每头具有

\[
\widehat O_{hi}=O_{hi}^{base}+X_{hi}B_h
-(X_{hi}a_h)c_{hi},
\quad c_{hi}=O^{selected}_{hi}-v_h^*,
\]

其中 X_h 有32列（常数+31坐标），B_h 的输出宽128，a_h 控制沿当前 value 方向 c_hi 的校正。a_h 与 B_h 通过 Schur 结构拟合。

D176 的头对比坐标仅由 X_h 去跨头均值得到，再与固定探针做内积；没有单独保留向量 c_hi 或 (X_hi a_h)c_hi 的输出相关作用。一个标量头对比核对所有 flow 通道共享 query 几何，不能自动等同于原逐头、逐输出方向的乘性算子。

在 B11_fixed 底座里，旧 FIN-C0 的作用已存在。因此该底座加零校正包含独立路径本身，H 全标签最优误差必不高于底座；实际 P64 读出是否达到这个嵌套中的有用位置是另一问题。

## 6. 需要补上的 P64 可辨识子空间诊断

对一次固定设计 D∈R^(512×d)，实际预测在 H 上为：

\[
\widehat r_H
=D_H D_P^T(D_P D_P^T+64\lambda I)^{-1}r_P.
\]

当前报告的 H 全标签 OLS 下限允许任意系数 C，搜索 col(D_H)。实际 P64 ridge 的系数限定于 col(D_P^T)，故只搜索 col(D_H D_P^T)，维数最多64。

已发布 D176_REPLAY.json 的8个站点全部记录：

- phi48 的 P64_rank=48；
- head112、ffn112、D176 的 P64_rank=64。

所以 D176 的系数零空间维数至少176−64=112。该计数不意味着这112个方向都在 H 上有不同作用，或都被真实误差需要；H_rank、相应谱和目标对齐信息目前未发布。已有 D176 H 全标签下限0.300×/0.382×属于更大的函数族，不能直接当作同一 P64 读出的可达值。

Φ48 则可作更强的判断：8站点的 D_P 都满列秩48，所以 col(D_H D_P^T)=col(D_H)，在精确算术下 E_Pspan=E_full。Φ48 原有的下限到实际预测差距可定位到已有函数空间内的系数恢复／正则问题；扩大到 D176 后，多出一个必须单独检验的函数空间限制。这是根据已发布 rank 和线性代数推出的结果，不需要假造尚未保存的 H 谱。

适当的三层诊断是：

\[
E_{full}=\min_C\|r_H-D_HC\|_F^2,
\]

\[
E_{Pspan}=\min_A\|r_H-D_HD_P^TA\|_F^2,
\]

\[
E_{actual}=\|r_H-D_HD_P^T(D_PD_P^T+64\lambda I)^{-1}r_P\|_F^2.
\]

在同一 FP64 点误差度量、秩阈值明确且列空间嵌套的条件下，有精确正交能量分解：

\[
E_{actual}=E_{full}
+\|\Pi_{full}r_H-\Pi_{Pspan}r_H\|_F^2
+\|\Pi_{Pspan}r_H-\widehat r_H\|_F^2.
\]

最后一项共同包含 P 标签信息、正则化与估计所得系数的差距，不能单独命名为“采样方差”。FP32 输出取整应另列差额，不能把上述精确等式不加修改地套到低精度输出。

D176 相对于更小字典的 full-H 函数族嵌套，但新增列使诱导核求和，P-representer span 一般会旋转，不能断言 E_Pspan 随字典扩大单调降低。

这里采用 operator-tensor-analysis 的 A11 采样插值思想：先检验目标子空间在采样限制下的可见性与稳定性，再解释探针数和恢复成本。CPU 诊断程序 diagnose_joint39_observability.py 已提供；它复用原 D176 producer 的真实资产加载、指纹、归一化与构造，只增加上述指标。

## 7. 在同一批 P 上追加残差校正没有新增精确观测

在冻结特征、线性读出的简化条件下，若第一阶段为 S、第二阶段为 R，写 r_P=Y_P-b_P，则：

\[
\widehat Y_H=b_H+S_{HP}r_P
+R_{HP}(I-S_{PP})r_P.
\]

第二阶段使用第一阶段训练后的 probe residual；它重新组合已有观测，没有增加新的独立 target 约束。raw flow_B11_fixed 避免了“P 已覆盖为教师导致零残差”的实现错误，但这些 P 剩余误差仍受原读出拟合影响。

实际 attention→FFN→head 有非线性，不能把上述 S 公式直接当作全图精确恒等式。它给出应检验的解释：追加校正的 probe 残差可能比 H 残差更受已有拟合塑形。当前公布收据不足以将这一因素单独归因为误差增大的原因。

## 8. 回到最初问题的建设性判断

实测表明，原 attention 读出利用的逐头分解、当前 value 方向及 FFN 的真实非线性输入响应，在自然复合图中产生有用作用；H 全标签 D176 下限下降也表明头对比与 FFN 坐标携带互补的终端误差信号。

可扩展的对象因此是“便宜但保留作用结构的当前计算，加锚点与少量完整查询所约束的未计算部分”。它在不同模块需要不同的可见缺陷空间。把最后输出统一成64维 flow 为评估联合效应创造了便利，但这一坐标压缩并不会自动替代原生模块约束。

最有区分力的下一项低成本证据是一次既有资产的可辨识性诊断，定位 H 下限改善到底有多少位于 P64 实际可用的函数族。如果 E_Pspan 仍高于独立底座，继续在同一字典/同一探针上调 λ 无法达到目标；如果它已明显低于底座，才有具体证据表明问题集中于可用系数的恢复。这个检验用于决定原理与观察的匹配，而不自动启动新的局部优化路线。

## 执行记录与资产边界

本地存在代码、JSON收据和原 CAL 投影矩阵；没有学校 joint39 向量或导出的 FFN kernel safetensors，且本地无 torch/safetensors。因此未生成真实 D176 新谱或 E_Pspan 数字。

新增诊断程序已运行一个明确标注的合成代数检查：两探针可观察两个方向，H有四个函数方向，full-H 下限0而 P-span 下限25；可见子空间降到两维后，两种下限相等。这核对了上述数学分解与可识别性区别，不构成模型实验结果。结果保存在 observability_algebra_check.json。

原始来源：attention_joint39_d176.py、attention_joint39_export_ffn.py、attention_joint39_head_contrast.py、D176_REPLAY.json（上述仓库 head）；attention_joint39.py（封存执行代码）；attention_panel_fit.py 和 exact_port.py（原读出实现）。

