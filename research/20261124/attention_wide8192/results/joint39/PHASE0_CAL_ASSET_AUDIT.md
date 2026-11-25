# phase0 专属 CAL：现有资产与 gate 审计

审计对象：FFNDS `9abe82a6e5131d86f4f4ac5f5c7ba30312da2bff` 的 C0 首步 OUT-ABS，以及已冻结的 CAL3 面板和原生运行代码。此轮只查阅代码与回执，未运行模型、未生成新轨迹，也未重算大面板。

## 结论与可执行范围

**现有面板足够构造 step0 的 `out_current` basis，并做固定原 λ=0.1、固定原目标尺度的离线 OUT-ABS 对照。** 这一操作直接使用双端点张量中的 anchor 端，构造过程不需要 self-attention gate。

**按原有下游损失合同重新选择 λ，还需要可验证的 step0 self-attention gate 和 step0 native 数值噪声基线。** 当前已公开 light schema 与 `Assets.prepare()` 只显式提供 `self_gate_s`，对应 hit 端。已保存的 time embedding 模型权重、层 modulation、实际 scheduler 时间及运行配置在数学与代码依赖上足够重建 step0 gate，计算不依赖图像或 prompt，也不需要重跑完整视频。不过，本次未持有这些权重的数值内容，未实算该 gate；不能把“可重建”写成“已核对 gate 数值”。

还要进行一项很小但必要的实现区分：当前 `OnlineStructured.approximate()` 把所有 OUT-ABS 都映射到 `self.fits['high']`。若希望只更改 step0，应给 step0 单独选择校准对象；直接替换 `self_high_transform` 会同时改变 step1。

## 1. 哪些现有资产直接可用

`attention_panel_fit.py::Assets.prepare()` 从每个 light safetensors 读取 `o_native[0:2]`、`o_selected[0:2]`，其两个端点按 anchor/hit 排列。self/high 的 anchor 是 step0，hit 是 step1。张量轴为 `[endpoint, query=512, head=40, dv=128]`，P64/H448 和全部索引已有冻结资产。

| 用途 | 既有字段或资产 | 审计判断 |
|---|---|---|
| step0 当前 selected 输出特征 | `o_selected[0] @ P_v[:, :64]` | 足够构造当前输出的 raw64 特征 |
| step0 full/selected 标签差 | `o_native[0] - o_selected[0]` | 足够进行 P64 ridge 拟合与 H448 head-space 评价 |
| 固定来源与层范围 | CAL=`0032/0304/0456`，Dense/wide8192，层=`0/13/26/39` | 每个来源 8 个 self/high 面板，总共 24 个 CAL 面板；与原校准的加权规则一致 |
| 输出投影 | sidecar 的 `projection_binding`；`Assets.weight()` 校验并加载 `weight_in_out` | 同一 expert/layer 的 step0/1 共用，可复用已绑定权重 |
| step1 self gate | `self_gate_s` | 原 hit 端下游评分所需字段 |
| step0 self gate | 已公开 schema 无 `self_gate_a`；`Assets.prepare()` 未读取它 | 当前材料未形成可直接使用的 anchor gate 资产证明 |
| step0 原始 native 噪声计算输入 | 既有 raw endpoint 命名 `endpoints/self/s00_bLL.safetensors`；`q_effective/k_effective/v_effective/o_native` | `native_noise()` 已具备同类读取机制，但当前入口使用 `hit_step`；需要切到 `anchor_step` 并生成 step0 评分回执 |

`CAPTURE_AUDIT.json` 记录 12 条背景轨迹、192 个配对 light 面板、384 次 native endpoint 核对，说明两端点属于已经完成并核对的捕获范围。这里没有把早期 FFN 面板当成注意力面板，也没有把 raw 路径的存在证明升级为已逐文件检查当前主机内容；现有完整 raw 数据在拟合主机的可用性应按已绑定索引解析。

## 2. 固定 λ 的 phase0 basis 对照为何无需 gate

对每个已有 CAL self/high 站点取 anchor 端，令

\[
z_{h i}=o^{S}_{0,h i}P_v^{(64)}\in\mathbb R^{64}.
\]

按原 `sufficient_stats()` 和 `transform_cal()` 的规则，先对每个站点的 head/query 行计算二阶矩，再在同一来源的八个站点内平均，最后对三个 CAL 来源等权平均。由该 Gram 的对角线得到 RMS，归一化 Gram 的前 31 个特征向量形成 `out_current_basis`；加常数列后总设计宽度为 32。

原 OUT-ABS 的逐头读出为

\[
\widehat o_0=o_0^S+
s_R X_0(X_{0,P}^{\mathsf T}X_{0,P}+64\lambda I)^{-1}
X_{0,P}^{\mathsf T}\frac{o_{0,P}-o^S_{0,P}}{s_R}.
\]

因此，在精确代数中，标量 `s_R` 相消，`scale_m` 与 gate 均未进入 OUT-ABS 的系数拟合。为使对照尽量单一，直接保留原 `s_R`、λ=0.1、P64、BF16 映射规则、32 列设计 与全部路由设置，只换 step0 所使用的 `out_current_basis` 即可。有限精度下仍应保持原代码的 cast 顺序；这里没有宣称不同尺度选择会得到逐比特相同结果。

构造 basis 不需要新的 full-key teacher；P64/H448 的离线评价也已经有 anchor 标签。`Assets.prepare()` 目前把 `out_current` 固定成 `(None, o_s)`，实施时要明确提取 `o_selected[0]`，而不是只改变量名称后仍读 hit 端。

这一对照回答的是“在既有原生背景 step0 面板上，换到匹配 step0 的表示是否改善可恢复性”。C0 自身经过前层替换后的 Q/K/V 分布仍可能继续变化；已有共享输入面板的改善需要和闭环视频结果分开表述。

## 3. 为什么早期 `gate1` 不能填补 anchor self gate

本地 `evidence/capture_current_panels.py` 的模块说明是“Capture shared high-noise native FFN panels”。它在 `infer_ffn` 内插入：

```python
capture_endpoint(self, phase, norm2_out, y, c_gate_msa)
```

只有 `step == 1` 时保存 `gate1`。所以该字段同时具备两个明确的身份：**FFN 分支的 `c_gate_msa`，step1。** 它既不是注意力分支的 `gate_msa`，也不是 step0 的 gate。该脚本/索引的 `U0/U1/Y0/Y1` 等 FFN 端点没有为 step0 self gate 提供直接观测。

同样，已有 Q/K/V、selected output、native output 本身并不一般地唯一确定 gate。原生 `infer_self_attn()` 完成 attention 和输出投影后，`infer_cross_attn()` 才用 `gate_msa` 将该分支加入 residual；门控不在上述注意力输出标签之内。若额外保存了加门控前后的 residual，才可考虑从残差增量恢复门控，但当前公开的 light 合同没有这一组配对 residual 字段。

## 4. 从实际 time path 重建 gate 的依赖是确定的

已绑定的 `joint_graph/native_transformer_infer.py` SHA 与 base identity 一致：

`5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225`。

它的 `pre_process()` 明确从 `modulation.tensor + embed0` 沿 6 个分量切分，顺序是 self shift、self scale、**self gate**、FFN shift、FFN scale、**FFN gate**。简记为

\[
g^{\mathrm{self}}_{0,\ell}
=\operatorname{chunk}_3\!\left(M_\ell+E(t_0)\right).
\]

这里第三分量用从 1 开始的编号，即 Python 的索引 2。`joint_graph/pre_infer.py` 给出 time path：

`sinusoidal_embedding_1d(t)` → `time_embedding_0` → SiLU → `time_embedding_2` → SiLU → `time_projection_1` → unflatten `(6,5120)`。

当前蒸馏合同是 `wan2.2_moe_distill`，单步全局时间，不进入 vanilla `wan2.2` 的 token-mask timestep 分支。因此在相同模型、step、层和配置下，gate 对所有来源和 token 广播；C/C0 的不同 residual 轨迹不会改变该 gate。phase0 面板所需的是四个层的 5120 维向量，FP32 净载荷仅 80 KiB；若一次导出 40 层，净载荷为 800 KiB。

重建时应使用实际运行的 scheduler 时间与原生 dtype/后端。合同中的 `denoising_step_list=[1000,750,500,250]` 是选点索引；代码还施加 `sample_shift=5`。step0 的实际时间为 1000，核对 step1 时应读取 scheduler 的实际值而不能直接把列表中的 750 当成 time-embedding 输入。

可行的最小补齐是：用已校验 high checkpoint 的 `time_embedding_0/2`、`time_projection_1` 与相应层 modulation，在绑定配置和原生 dtype 路径下计算 step0/step1 gate；将重建的 step1 gate 与已有 `self_gate_s` 核对，然后存 step0 gate 及输入权重/配置/时间/实现哈希。此项只涉及小 time 子图，不依赖 full attention、FFN 或视频生成。若 runtime 仍驻留，也可从同一 `pre_process()` 导出该向量。

从 `g1 + E(t0) - E(t1)` 在实数代数上能够消去层 modulation，但已有 gate 的舍入信息不足以保证这种差分做法恢复原生 bit pattern。直接走 `M_l + E(t0)` 的原生路径更适合作为 actual gate 的证明。

## 5. 若重新选择 λ，确切还要补什么

原 `select_lambda()` 采用 CAL 留一来源：训练两个来源的变换，在留出来源上用 P64 拟合、H448 计算实际输出投影和 self gate 后的点误差，再相对 selected-only 误差归一化。主机所需 step0 损失为

\[
\frac{\|\bigl(\widehat o_{0,H}-o_{0,H}\bigr)W_O
\operatorname{Diag}(g^{\mathrm{self}}_{0,\ell})\|_F^2}
{\|\bigl(o^{S}_{0,H}-o_{0,H}\bigr)W_O
\operatorname{Diag}(g^{\mathrm{self}}_{0,\ell})\|_F^2}.
\]

分子和分母都有同一 gate，并不表示它一定可以约去：gate 是通道向量，两个误差方向一般不同，改变 gate 会改变通道加权和 λ 排名。

当前缺口可精确表述为两份小范围评分依赖：

1. **可验证的 `self_gate_a` 或等价 step0 gate 文件/重建回执。** 旧 `self_gate_s` 和 FFN 的 `gate1` 不能替代。
2. **step0 的 native-vs-recomputed noise 回执。** `select_lambda()` 会跳过 selected-only 点误差不高于 native noise 的站点；当前 `Assets.native_noise()` 使用 `hit_step`。需将同一已有 raw Q/K/V/o_native 路径切换为 `anchor_step`，结合 step0 gate 计算相应 H448 baseline。原 step1 noise 值不足以保持原评分合同。

这些补齐不要求另采 CAL 视频。W_O、P/H 索引、三来源身份、anchor 标签、selected 特征与原有 raw endpoint 的读取结构已有。若主机的原始数据已清理，则先核对 raw 索引和 tensor header；当前仓库未发布全部原始 endpoint 清单和大张量，本审计没有确认其现时驻留状态。

固定原 λ 的 basis 对照可以先利用已知完整的 anchor 输出资产开展。重新调 λ 则在上述 step0 gate 与 noise 评分依赖补齐后，仍只使用 CAL3 选择参数；DEV0392/1013 保持结果检查角色。

## 证据路径

最新 head 下实际读取：

- `research/20261124/attention_wide8192/results/attention_panel_fit.py`：`Assets.prepare/weight/native_noise`、`sufficient_stats/transform_cal/readout/downstream_point/select_lambda`。本地副本 `round6_schedule/phase0_sources/attention_panel_fit.py` 与先前已下载实现一致。
- `research/20261124/attention_wide8192/results/joint39/attention_online_structured.py`：`approximate()` 共用 high fit，P64 原生标签及写回路径。
- `research/20261124/attention_wide8192/results/CAPTURE_AUDIT.json`、`EVIDENCE.json`：完成的两端点数量、light 资产规模与绑定哈希。
- `research/20261124/attention_wide8192/wide8192_binding.json`：FFN anchor/hit 调度与运行主机的模型权重绑定要求。
- `upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json`：蒸馏 model_cls、scheduler、时间选点、CFG 状态与 high/low checkpoint 身份。只引用其运行结构，不用旧几何字段覆盖本次实测 832×464。

原始方案/既有源码证据：

- `attention_experiment_plan_20261123/schemas/tensor_contract.json`：端点轴及仅显式列出的 `self_gate_s`；该原始合同的 fast512 背景字段已由当前 wide8192 实现和回执更新。
- `evidence/capture_current_panels.py`、`evidence/capture_index.json`：早期 FFN 捕获内容与 `gate1` 身份。
- `joint_graph/native_transformer_infer.py`、`evidence/base_identity.json`：实际 transformer SHA、self/FFN gate 分量和作用位置。
- `joint_graph/pre_infer.py`、`joint_graph/native_step_scheduler.py`：time 子图与实际时间生成；pre_infer 为已有源码快照，尚需在导出 gate 的目标主机记录其当前实现 SHA。

源文件指纹见 [phase0_cal_source_manifest.json](phase0_cal_source_manifest.json)，其中工作副本路径记录审查来源，不要求仓库具有同名本地目录。此文是资产可执行性与实现语义审计，不是 phase0 专属 CAL 的已执行结果。
