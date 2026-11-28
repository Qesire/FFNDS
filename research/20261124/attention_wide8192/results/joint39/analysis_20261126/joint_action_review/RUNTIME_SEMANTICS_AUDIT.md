# 逐阶段联合回放：真实计算图与数值语义审计

证据提交 `6baa2a44ffaac5cd90d9a4f4df81809a651a784d`，学校 A100 作业 `201276`。本次工作读取归档源码和原始收据，执行 CPU 复算；新增模型运行、视频解码和性能测量均为 0。使用更新后的 operator-tensor-analysis 中的轴语义、有限差分、真实 dtype 边界和完整调用周期判据。

## 1. 本次新增了什么证据

两条固定轨迹是原生 self-attention 加 wide8192 FFN：0392、1013。每条读取两个命中步的第 0、13、26、39 层，共 16 个嵌套站点；独立来源仍为 2。每站 H448 × 5120 留出场上保存 10 个阶段的 raw/transport 2×2 Gram，并在 3 个 FFN/出口阶段增加 Dense 背景，形成 3×3 Gram。

已独立核对原收据未压缩 SHA256 为 `73f5d6f446f2a14d307170df3851ad186857849956f0b9c778a9da6cad1daf66`，执行主文件 SHA256 与收据一致；方法 SHA256 绑定上一轮冻结运输实现。160 个 2×2 与 48 个 3×3 Gram 都符合对称、半正定及公共子块一致性。不同形状 GPU Gram 归约存在浮点末位差，公共子块按 1e−10 相对容差核对，未将它们误写为逐位相同。

两条真实参考轨迹均有精确轨迹见证。16 站点 FFN P64 输入与捕获的现场输入相对差均为 0；原生回放尾部对现场块出口的最大相对 L2 为 5.962273007753538e−6。回放代码的阈值 0.025 是断言门槛，实际测得误差远小于门槛，应报告实际值。

## 2. 阶段的真实边界

固定本次块输入 X、条件、历史 FFN 锚点及共享 P64 拟合参数 θ。设 `a_m` 为原生、raw 或 transport 的注意力多头响应。

| 归档字段 | 实际读取的对象 | 误差比较的含义 |
|---|---|---|
| `attention_head` | BF16 注意力响应拼头 | 原生 attention 的同位置输出误差 |
| `output_projection` | `self_attn_o.apply(a_m)` | 实际输出投影后的有限差 |
| `attention_gate_response` | `projected * gate_a` | 本步门控后、加入 X 前的响应 |
| `self_residual` | 真实 `infer_cross_attn` 返回的 X 更新 | `X + gate_a * projected` 的实际数值写回 |
| `cross_response` | 依当前 self residual 重算的 cross attention 输出 | 含 norm3、Q 投影、固定 context K/V 的响应变化 |
| `cross_residual` | self residual 再加 cross response | 进入 FFN 归一化前的真实状态 |
| `ffn_input` | norm2 与当前 scale/shift 调制后输入 | FFN 实际读取的输入差 |
| `ffn_response` | 同一 θ 下的 wide8192 预测 | 门控前的 FFN 响应变化 |
| `ffn_gate_response` | `ffn_response * gate_f` | FFN 门控后、末尾残差前的响应 |
| `block_exit` | 原生 `post_process` 写回 | 完整当前块尾部的输出变化 |

`cross_residual` 和 `self_residual` 是不同张量。报告主表省略了后者，但原始 Gram 保留了它。cross attention 的 Q 随分支改变，context K/V 固定；回放通过实际 cross 算子执行这一变化。

### 残差加法的数值作用

[原生 transformer 固定源码](https://github.com/ModelTC/LightX2V/blob/ed146db64d0b6130c669285fc1aa0b280a6a4823/lightx2v/models/networks/wan/infer/transformer_infer.py)的当前 same-dtype 路径为 `x.add_(y_out * gate_msa.squeeze())`。诊断保存的是同一 projected 值乘 `gate_a.reshape(-1)`；当前全局 5120 通道 gate 下，squeeze 与 reshape 的广播相同。这条诊断回放直接调用 eager 的原生 cross 函数，乘积在两处都是独立 torch 乘法，因此这里的附加余项可定位于加 X 的 BF16 写回。若改变 dtype 或编译/融合配置，需重新检查此边界。

对实数算术，同一个 X 加到各分支上不会改变两分支相对原生的误差。当前 BF16 路径的实际加法含舍入，不能用这个实数消去式越过真实边界。令 Q_B 为实际写回，则

\[
e_m^{self}=Q_B(X+g_m)-Q_B(X+g_n)=e_m^{gate}+\rho_m-\rho_n.
\]

因此在相同原生 BF16 路径中，self residual 的 Gram 可以与 gate response 的 Gram 不同。0392 low39 的能量比从 gate 的 0.9838143 到 self residual 的 0.9623532，随后 cross residual 为 0.9623565。这里已经可以准确定位：第一次比值降至 1 以下发生在 self gate，之后进一步的变化发生在残差数值写回；cross response 没有造成这次主要排序变化。该处 cross response 的 raw/transport 能量仅为约 0.0004092/0.0003842，而 self residual 为 115.8097/111.4498。

这一定位基于当前实际图；不能把舍入变化解释成新的实数非线性恢复机制。归档没有每阶段误差向量或舍入余项向量，尚不能把后续残差合成的能量差逐项归因给 FFN 与残差之间的真实内积。

## 3. 共享探针确实闭合到同一 FFN 拟合

注意力比较用全查询 native attention 中对应的 P64 响应覆写三个分支。短查询 SDPA 教师另作数值见证。于是三个分支在 P 位置的注意力输出相同；当前尾部在固定 context 下按 query 行独立执行，FFN 输入也逐位相同。

新运行器在 live wide8192 调用期间捕获原生 FFN 的 64 行输入和输出，恢复现场函数后才执行诊断回放。θ 由捕获的这一次 P64 标签、同一缓存和相同 FFN 权重计算一次，并用于三个回放分支。H448 Dense 标签在拟合完成后才生成，未进入 θ。

这解决了原生 FFN P64 教师重复计算可能带来的细微差异。该证据适用于受控的 full-query attention P 教师；实际在线运输使用 short-query attention 教师时的 θ 变化仍属于另一条数值路径。收据保留两者误差，不能把受控闭合自动推广为所有在线调用逐位一致。

## 4. Dense 背景的定义与中间排序翻转

在三个保存 Dense Gram 的阶段，统一定义

\[
z_m=T^{wide}_\theta(a_m)-T^{wide}_\theta(a_n),\qquad
b=T^{wide}_\theta(a_n)-T^{Dense}(a_n).
\]

这里的 Dense FFN 使用 **原生 attention 在当前同一块输入上生成的 FFN 输入**，只在 H448 额外计算完整 FFN。因此 `b+z_m` 是该站点同输入的 Dense 目标误差；它不是独立原生 Dense 整条去噪轨迹的误差。两类目标各自清楚定义后，这是一项有效且有辨识力的局部比较。

\[
\Delta E_{Dense}=\|z_t\|^2-\|z_r\|^2+2\langle b,z_t-z_r\rangle.
\]

新 3×3 Gram 直接提供上述背景交叉项，以下翻转都可以由实测内积验证：

| 站点/读出 | fixed wide 比值 | Dense 目标比值 | fixed wide 能量差 | 背景交叉差 |
|---|---:|---:|---:|---:|
| 0392 low39，FFN gate response | 0.9128368 | 1.0077989 | −2.5813933 | +3.7731799 |
| 0392 low39，block exit | 0.9843578 | 0.9897708 | −1.3781550 | −1.0197323 |
| 1013 high0，FFN gate response | 1.0878228 | 0.9948808 | +2.0745977 | −2.7755337 |
| 1013 low39，FFN response | 0.9929906 | 1.0002896 | −1.5566523 | +3.1417758 |

所以“16 站点 fixed wide 与 Dense 的改善方向一致”必须限定为 **block exit**。同一个物理站点的中间读出已有实测目标翻转。这进一步支持以完整后继读出确定所需恢复信息：FFN 响应、FFN 门控响应和最终残差状态分别对应不同目标。

另一项可重复现象是 high0：两个来源的纯 FFN 响应比为 0.8275904、0.8387079，经 FFN gate 后变成 1.1681775、1.0878228，块出口又为 0.6713709、0.6957317。这个序列不能概括为“后续各阶段都继续改善”。完整块出口在两个来源上都受益，是该比较直接支持的肯定判断。

## 5. 路由、形状和尺度分解测到了哪个算子

对每个 head，R_t=S_t−S_t^0 的输出轴为 query，归约轴为 key；value 的最后一轴为通道。S_t^0 同时包含选中真实 key 与未选块的两组当前均值，按统一分母归一化，因此 R_t 1=0。支持集合和组成员由 anchor 固定，Q/K 数值和当前均值可以改变。

在全 keys、同一 512 行上直接计算 R_a Ṽ_a、R_a Ṽ_s、R_s Ṽ_s，给出

\[
U=(R_s-R_a)\widetilde V_sD_s,\qquad
V=R_a(\widetilde V_s-\widetilde V_a)D_s,\qquad
W=R_a\widetilde V_a(D_s-D_a).
\]

活动通道上 raw 创新为 U+V+W，transport 创新为 U+V。当前常量通道按实现掩码令相应创新为 0。对固定当前特征与 P64，剩余算子

\[
L_H=M_H\left[I-\Psi(\Psi_P^\top\Psi_P+6.4I)^{-1}\Psi_P^\top P\right]
\]

沿 query 轴作用；D_s 沿 value 通道轴作用。因而可在解码后分别恢复三项并用同一 Gram 计算平方范数及全部交叉项。尺度的净影响是

\[
\|L_HW\|^2+2\langle L_H(U+V),L_HW\rangle,
\]

而不是尺度项自己的能量。正或负都具有明确解释。路由与形状分解选择了“先在 anchor 路由下改变 V，再改变路由”的指定路径；换分解顺序会重新分配交互项，整体创新保持不变。

该路由作用独立计算使用 FP32 全 key softmax、FP32 value contraction，并关闭 TF32；生产路径使用 selected FlexAttention 的 BF16 输出/LSE，再以 FP32 合并。全 key 分解与运行时物理缺陷最大相对差约 2.04%，所以它是经过数值对照的机制见证。归档的 BF16 实际拟合结果仍单独报告。`independent_queries=512` 是代码字段名，表示执行了 512 个 query 行；统计独立样本数仍为两个来源。

## 6. 完整性与可推断范围

Gram 由先升格 FP32 再作差的路径输出构成，内积用 FP64 累加。因此 `z_t−z_r` 的能量、与 z_r 的内积、与 Dense 背景的内积都可从同阶段 Gram 精确复算到归约精度。160 个 2×2 Gram 不是 160 个独立统计样本。

阶段之间只保存各自 raw/transport Gram。没有保存 `cross_residual` 误差与 `ffn_gate_response` 误差之间的跨阶段内积。即使三个范数都已知，由于实际末尾 BF16 相加含舍入，也不能通过 `E_exit−E_residual−E_gate` 将结果命名为两分支的真实交叉项。若以后确需分解 FFN 内部抵消，应直接在同一现场计算该跨阶段 Gram，并另记实际加法舍入差；当前轮已足以回答阶段排序和 Dense 背景作用，无需把缺失项当成新实验前置条件。

## 7. 成本范围

本次共额外执行 16×448=7,168 行 Dense FFN，复用现场 native FFN P64 标签。注意力短查询计数为每来源 512 行，两来源 1,024 行；全查询原生 attention 仍执行以保持固定轨迹。每来源只对 4 层建立 attention 诊断状态，8 anchor、8 hit，峰值 4 槽；实际 wide8192 仍贯穿全部 40 层。

路由见证额外保存 anchor K/V，并对 512 行进行全 key 交叉收缩；这些开销只服务机制测量。本次运行时间和峰值不能作为生产运输方法对冻结 C 的性能结果。现有全字段运输的持久状态、anchor 构建与一次 hit 的生命周期，仍以既有生产实现和匹配计时条件评估。

## 8. 可复算交付

- `verify_runtime_semantics.py`：独立检查收据 SHA、主文件 SHA、来源/站点/行数、全部 Gram 基本性质、live 探针闭合及关键阶段边界。
- `runtime_semantics_verified.json`：上述检查及逐站 stage 比值与 gate/residual Gram 差。

复算命令：

```sh
OPENBLAS_NUM_THREADS=1 python verify_runtime_semantics.py --source ../joint_action
```

当前 CPU 复算全部通过。源码审计入口为归档 `attention_joint_action.py`、`run_attention_defect_transport.py` 和冻结 `method_transport/attention_defect_transport.py`；已有原生 transformer 源码快照用于核查真实 residual 算术边界，动态绑定代码在运行时检查其 SHA 对应实验 base。该快照本轮未重新取得学校完整导入依赖包，因此没有把单个主文件指纹扩写为所有运行依赖逐字节的新验证。
