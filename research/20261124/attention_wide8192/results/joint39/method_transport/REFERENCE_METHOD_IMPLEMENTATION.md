# attention 候选的完整数学参考实现

2026-11-25。基于 FFNDS `7c8196fa5380dbe67dd9772f942553085fde55b3` 的上轮候选继续推进。使用 operator-tensor-analysis、active-abstraction 和 research-implementation；本轮交付为 NumPy CPU 数学参考、固定构造上的数值核验和实际资产可用性清点。

## 1. 已形成的可执行对象

本实现包含原始缺陷缓存，以及按当前每通道 value 尺度运输的缺陷缓存。主候选采用后者：当前 K/V 两组底座负责当下交互，完整的标准化缺陷保留历史，P64 拟合标准化创新。

对每层、每头，输入轴定义为：

- Q：`[H,I,Dk]`，实际 norm/RoPE 后的 query；
- K：`[H,J,Dk]`，实际 norm/RoPE 后的 key；
- V：`[H,J,Dv]`，当前 value；
- selected：每个 head、query block 的非空 key-block 支持；
- P：64个互不重复的 query 行，当前教师读取这些 query 及全部 K/V。

生产形状为 Dk=Dv=128，H=40，I=J=31668；参考实现支持其他有效形状。参考数值采用 float64，生产 BF16 缓存、FlashAttention 后端和舍入次序留待实际绑定。

### 当前底座

每个原 key block 以固定归档 P_k 第一列对中心化 K 做正负二分。anchor 冻结组成员与 selected 支持，hit 使用当前 K/V 重新计算各组均值。selected blocks 的逐 key 响应精确，其对应摘要被排除。遗漏组以

\[
\bar z_{ic}=n_c e^{q_i^\top\bar k_c/\sqrt d}
\]

形成正权补偿：

\[
G_i=\frac{Z_i^SB_i+\sum_c\bar z_{ic}\bar v_c}
{Z_i^S+\sum_c\bar z_{ic}}.
\]

代码先得到 selected 的输出 B 与 LSE，再将遗漏组分块在线 log-sum-exp 合并。空组跳过，末块按有效 key 数计数；不会物化全部 `I×J` 或 `I×全部coarse组` 矩阵。

### 标准化的历史状态

当前每头、每 value 通道定义

\[
\sigma_{s,\nu}^2=\frac1J\sum_j(V_{s,j\nu}-\bar V_{s,\nu})^2,
\qquad D_s=\operatorname{diag}(\sigma_s).
\]

anchor 保存完整字段

\[
C_a=(A_a-G_a)D_a^+.
\]

当前底座为 `G_s + C_a D_s`，P64 监督标准化创新

\[
T_P=(A_{s,P}-G_{s,P})D_s^+-C_{a,P}.
\]

无需保留 anchor 的 σ 场。新增 per-channel 统计只在当前调用中使用。

若 anchor 某个 V 通道在所有 keys 上恒定，该列 A_a=G_a，因此缓存 C_a 的该列定义为0。若当前某列恒定，则当前 A_s=G_s；该列直接使用 G_s，跳过历史与校正。全部列恒定时连原生 P 教师也无需调用。非零尺度使用 float64 下限 `1e-12` 明确数值行为；下述尺度等变定理限定为下限未触发的通道。

### 当前32列特征与解算

主候选将归一化放在 value 投影之前：

\[
\Psi_i=
\left[
1,\ q_iP_{k,15}/\sqrt d,\
\big((G_i-B_i)D_s^+\big)P_{v,15},\ Z_i^S/Z_i^0
\right].
\]

固定 P64、每样本 λ=0.1，求

\[
W=(\Psi_P^\top\Psi_P+6.4I)^{-1}\Psi_P^\top T_P,
\qquad
\widehat A_s=G_s+(C_a+\Psi W)D_s.
\]

标签在原生覆盖前形成，最终将 P 行覆盖为原生教师。零方差列的 W 固定为0。无锚调用取 C_a=0，目标相应为当前标准化缺陷。

`raw` 模式保留上轮原始 E_a，并用每头全局 RMS 归一化当前特征与标签；它作为固定的机制对照。`scalar_normalized` 保留兼容接口，未进行参数扫描。主核验比较 `raw` 与 `channel_normalized` 两种语义。

## 2. 新坐标得到的精确作用类

### 2.1 跨步正对角 value 演化

在 Q/K、支持与分区保持不变时，若

\[
V_s=V_a\operatorname{diag}(g)+\mathbf1 b,\qquad g_\nu>0,
\]

则 A/G 都按相同仿射关系变化，且 σ_s=σ_a⊙g。因此标准化缺陷 C_s=C_a，当前底座 `G_s+C_aD_s` 已经精确；P 拟合只接收到舍入量级的创新。

这里允许每个 head、每个通道有不同正增益。负增益携带符号翻转，不属于仅用标准差运输的这一精确类。

### 2.2 包含非零创新与 ridge 的完整方法等变

对 anchor 与 hit 的 V 分别施加各自的正对角仿射变换，各次调用对应的 Q/K 保持原值：

\[
V'_r=V_r\operatorname{diag}(g_r)+\mathbf1 b_r,
\qquad r\in\{a,s\}.
\]

允许 g_a≠g_s、b_a≠b_s；Q/K 本身也可以在 anchor 与 hit 之间发生一般有限变化。则 C_a 不变，当前标准化 response 特征不变，T_P 不变，带固定 λ 的 W 不变。最终输出满足

\[
\widehat A'_s=\widehat A_s\operatorname{diag}(g_s)+\mathbf1 b_s.
\]

这条性质覆盖一般非零创新与两时刻独立的 value 坐标变化。把 `(G−B)` 先按通道标准化、再乘 P_v，正是保持该关系的顺序。仅对投影后结果除一个全局 RMS，只保留更窄的标量尺度关系。

这描述 attention 方法对 value 作用的协变关系；最终仍向真实输出投影与后续模块写回当前单位的 attention 响应。

### 2.3 缓存误差的实际读出

若标准化缓存保存为 C_a+ε，令 ΔC=C_s−C_a，固定当前特征与尺度，则最终 P 覆盖后的误差为

\[
\widehat A_s-A_s
=M_H(K_sP-I)(\Delta C-\varepsilon)D_s.
\]

代码也执行了该式的固定输入数值核验。生产缓存采用 BF16 时，应在这些标准化坐标中记录 ε，并通过当前 D_s 和真实后续模块读出其影响。

### 2.4 固定同一特征，隔离历史更新

使用同一标准化state、当前特征、P教师、ridge和P覆盖，把可选`transported_anchor_defect`设为原始E_a，可得到仅历史项不同的对照。令K_s为相同当前特征的恢复算子，则

\[
\widehat A_{\rm transport}-\widehat A_{\rm raw,same\ feature}
=M_H(I-K_sP)(C_aD_s-E_a).
\]

程序分别在正对角演化和一般有限Q/K/V变化上核验该式；当前特征与教师逐位相同。它把运输作用与特征坐标变化分开，适用于后续真实回放的同单位比较。

## 3. 本轮执行结果

命令：

```bash
OPENBLAS_NUM_THREADS=1 python verify_reference_attention.py --replay-zip ../../replay_evidence.zip
```

也支持 `--basis-dir /path/to/plan_assets/bases`。脚本与模块放同一目录即可。测试使用真实归档 P_k/P_v 字节；激活均明确合成。主体形状为 H=2、N=190、d=128、P64、block64，包含空组与62-key末块。

| 有区分力的核验 | 本轮结果 |
|---|---:|
| anchor 输入回返，raw / channel normalized | 最大误差均2.78e−17 |
| 正对角 value 演化，raw 未拟合底座的留出L2 | 4.00535 |
| 同一构造，raw P64拟合后的留出L2 | 3.97488 |
| 同一构造，channel normalized最终输出最大误差 | 1.11e−15 |
| 一般非零创新下，完整方法正对角仿射等变最大误差 | 1.44e−15 |
| anchor/hit使用不同正对角增益与shift，完整方法等变最大误差 | 2.66e−15 |
| 组内K相同、V任意，当前完整候选最大误差 | 5.83e−16 |
| ±K/V的selected输出碰撞，两组底座最大误差 | 6.94e−18 |
| 原始缓存有限误差算子恒等式最大差 | 5.55e−17 |
| 标准化缓存有限误差算子恒等式最大差 | 8.33e−17 |
| 同特征/教师的历史运输输出差分恒等式 | 最大差2.36e−16 |
| 同特征正对角演化，未运输/运输后留出L2 | 3.97776 / 2.38e−14 |
| 同特征一般有限变化，未运输/运输后留出L2 | 7.33626 / 7.37903 |
| P缓存保留原值时，当前系数逐位相同 | 通过 |
| current全部V恒定时的原生教师调用数 | 0 |

正对角演化的固定增益范围为0.34505–3.31718。表中 raw / normalized 的差异是精确作用类的构造性辨识，不是 Wan 误差或成片提升。

同特征对照进一步隔离了历史运输：正对角演化中，其精确性在完整拟合后保留；一般随机有限变化中误差增加约0.58%，给出了作用边界。真实比较应使用J_raw=E_s−E_a与J_transport=(C_s−C_a)D_s这两个相同输出单位的对象，并追踪真实后续模块作用；仅比较标准化和原始范数不能判断改善。

两时刻独立坐标变换的进一步核验中，anchor/hit 的增益差L2为12.03376、shift差L2为9.32281。当前特征最大差6.11e−16，标准化创新最大差3.64e−15，拟合系数最大差4.13e−16，最终等变误差2.66e−15。该例的Q/K在anchor与hit之间本身也发生了变化，因而核验覆盖非零routing创新。

一般当前Q/K/V变化下，重新分组本会改变104个标签；实际 hit 始终保留 anchor 的 labels 和 selected 支持。该例实际 P 创新范数5.23407，最终 P 覆盖误差严格为0。教师调用记录确认仅接收64行 query及全部190 keys，而非所有当前query。

另外验证了 partial constant channels、constant anchor→variable hit、无锚调用，以及朴素 exp 会溢出的 logit 范围内在线归约保持有限。

## 4. 实现和状态接口

`reference_attention.py` 公开以下入口：

| 入口 | 含义 |
|---|---|
| `group_labels` | 生成当前/anchor正负分组 |
| `summarize` | 从冻结labels与当前K/V取得counts及组均值 |
| `grouped_attention` | 稳定合并selected精确响应和遗漏组摘要 |
| `build_anchor` | 构建原始/标准化完整缺陷，可直接接收已有原生anchor输出 |
| `recover_hit` | 当前特征、P教师、标签、ridge与最终P覆盖 |
| `defect_projection` | 可选的anchor字典接口，不自动加入持久状态 |

最小状态包含一个完整 field、uint8 分组标签及已有 selected 支持。参考实现以 bool mask 表示 support 以便阅读；生产绑定应复用已有 selected block 索引格式。`BaseResult.groups` 暴露当前组统计供其他推导使用，但主候选不额外缓存 anchor group means。

`recover_hit` 的可选 `transported_anchor_defect` 参数接收已还原到当前输出单位的全位置字段，供未来明确指定的其他运输规则使用。本轮默认规则是 per-channel std，没有额外拟合128×128运输矩阵。

## 5. 真实资产清点与当前执行范围

实际读取的 `replay_evidence.zip` SHA256：

`01152bd9b72afd9346cecc9546d97c3b4e5b170a8b94d5f51d496015b815256a`。

归档共有16个文件，包含两个128×128投影、四组CAL变换、索引、见证和metrics。CAL主要投影是64×31，并非模型Q/K/V激活。归档中没有当前构造所需的真实完整 anchor/hit Q/K/V。

原 `CAPTURE_AUDIT.json` 记录学校曾生成192个轻面板、合计10,147,109,376字节的light safetensors；当前可读取归档只提供其汇总证据。这一记录支持历史采集已存在，不能替代本候选需要的实际输入字节。

当前完成状态是：**方法语义已闭合，CPU数学参考已执行，运行结果可复算。** 实际模型回放需要匹配的有效Q/K/V、固定selected支持、P索引和原生anchor响应，或另行定义并核验的充分统计包。生产q64 FlashAttention绑定、BF16缓存路径、成片与GPU性能本轮没有执行。

文件：`reference_attention.py`、`verify_reference_attention.py`、`reference_attention_verified.json`。
