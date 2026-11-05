# 分组输入统计与响应对照：测量 schema 草案

状态：**方案，未采集、未冻结、未运行**。参考 `G19_input_relations/measurement_plan.json`、`odd/NEW_RESPONSE_STRUCTURE_NOTES.md`、`operator/OPERATOR_INPUT_RELATIONS_NOTES.md`；这些笔记的相关性为探索性，不是分组规则、因果结论或新盲测证据。先登记候选、来源划分、组边界、阈值、权重和主终点，再执行；已有 HELD 已被查看，须另取新来源/seed确认，历史 exposure UNVERIFIED。

## 标识、组与采样

主键 `(source_cluster_id, source_id, seed, trajectory_model_id, eval_weights_id, layer, expert, endpoint_or_step_pair_id, panel_id, mask_id, group_kind, group_id)`；并记录两端实际 timestep、capture/weight SHA、条件/投影版本 SHA、运行与数值精度。`trajectory_model_id` 表示生成输入的模型，`eval_weights_id` 表示计算响应的权重；Oq 用 Dq 轨迹不能标成 Oq 自由轨迹。只有 trajectory_model 与 eval_weights 身份一致、且实际捕获该轨迹的 Y 时才标记该臂 native 端点；新 E1 若实际生成并捕获 Oq 自身轨迹，Oq 同样可有 native 端点，交叉臂不得冒称自身 native。`metric_gate_id` 必须绑定实际同一 state 的 gate、来源/端点及 SHA；跨权重评估保持同一状态的 metric gate，禁止混用不同轨迹 gate。静态单端统计使用明确 `endpoint_id`，不可将两端或两专家误合并。`mask_id` 是预登记的 token 筛选/有效性集合及版本，**不能**由 HELD 响应或误差选择；`group_id` 必须关联 group manifest 的 `group_kind`、边界、坐标系和版本，避免不同轴上 ID 碰撞。

| 组种类 | 定义与用途 | 不可混淆 |
|---|---|---|
| token/时间空间 | 固定 latent 帧索引 `t`、空间 `(y,x)`、tubelet 的16固定空间位置，或 CAL 冻结位置/径向分箱；token_id 与 `t,y,x` 保留映射 | latent 时间是帧索引，不是去噪 step；tubelet 与 uniform 为不同抽样面板，不直接拼接 |
| hidden channel | 冻结的原 W1 行索引连续块或 CAL 冻结分组、channel manifest SHA；同一组跨数据分割保持不变 | hidden 通道分组不是输出维度组，也不是 token 组；若权重/专家变化须保持原始编号及映射 |
| 输入分布分箱 | 用 CAL 固定的 `m` 有符号区间 × `|d|` 分位/阈值箱，及 CAL 固定 q 径向箱；同一阈值用于 DEV/HELD | 每源重分箱或重白化会掩盖来源偏移；预留 under/overflow、零和非有限标志 |

来源是主要分组/报告单位，其独立性未确立；层、通道、token 和重复面板不是独立来源。新实验优先每来源-seed冻结同一2048 token mask供全部去噪步骤、层及权重臂使用，并按latent时间分层记录包含概率；跨步必须验证ID完全一致。**仅旧数据**的2048 uniform同源同step跨层对齐、跨step需按token ID求交（通常约132而非2048）；旧336 tubelet同源跨step/layer同坐标，跨源仅在显式检查ID后才配对。记录端点原总数、交集数、未匹配数、mask 前后数及 `(sum_w,sum_w2)`；uniform 与 tubelet 权重、抽样单位分别登记，未知总体包含概率时标记 `DESCRIPTIVE_UNWEIGHTED`，不伪造无偏总体估计。

## 分组字段与何时可得

每个 token 先按同一确定性 mask/空间组赋组，再分别聚合输入和响应；不能把跨通道聚合后的均值当成逐通道统计。两个端点为 `z0,z1`，`Δz=z1−z0`；在相同评估权重下 `a_i=W1 z_i+b1`，`m=(a0+a1)/2,d=a1−a0, Δh=φ(a1)−φ(a0)`。**分开记录**实际运行的 `a_native,h_native` 与由已存权重/输入进行 F64 连续函数重算的 `a_smoothlift,h_smoothlift`，各自标 dtype、舍入位置、端点、数据来源与 SHA；直接计算两者的组内差及能量。若实际 native 中间量未采集，标 `NOT_CAPTURED`，不得将 smoothlift 冒充 native。

| 域 | 组内预定字段（sum 为带样本权重的可合并量；均值均由分母导出） | 可用时点 |
|---|---|---|
| z：幅度/均值 | endpoint 的 `sum_wz,sum_w_norm2,sum_w_norm4`，`Δz` 的 `sum_w_delta,sum_w_delta_norm2`，两端内积；RMS/能量/均值位移，保留有效行数 | 单端在 W1 前；Δz 需可用且 ID 对齐的两端，不等于仅凭单端统计得到的值 |
| z：谱与径向 | CAL 冻结中心、PCA 基/白化尺度及 SHA；每源 `sum_wq,sum_w_qqT` 或登记的对角/迹与投影能量，`sum_w||q||², sum_w||q||⁴`、固定边界径向直方图 | CAL 构造后冻结；新来源不得自行重拟合白化；属于输入测量，投影/汇总成本应计入 |
| a：每通道 | `A_j=Σw d_j²,B_j=Σw d_jΔh_j,C_j=Σw Δh_j²`；`sum_w a0_j,a1_j,m_j,d_j,m_j²,d_j²`、符号/零计数 | W1 后；需要激活计算的 B/C 与误差不是可免费的 pre-W1 路由信号 |
| a：联合尾 | `sign(m)` × CAL 冻结 `m` 和 `|d|` 联合箱：计数/权重、`Σwd²,Σwd⁴`、真实有限弦 GL2 hidden SSE、`ΣwΔh²`；可附固定阈值超尾和排序 sketch | a 后；误差仅在目标/近似计算完成后；联合尾不能由边际分位数相乘替代 |
| 同组响应 | hidden `Σw||Δh||²,Σw||e_h||²`；全输出 `Σw||W2Δh||²,Σw||W2e_h||²`；midgate 与当前 g1 的 target/SSE、signed cross、closure；仅在 trajectory_model==eval_weights 且 Y 实际捕获时单列该臂 native `Y0,Y1,Yhat1` 和 g1 风险 | 输出投影/gate/native endpoint 已得后，属于事后验证；交叉臂不冒称自身 native endpoint；绑定同 state 的 `metric_gate_id`，明确 F64 lift 与真实 BF16 算术边界 |

alpha 仅由 CAL 逐 channel 的 `B_j/A_j` 得出并 SHA 冻结；`A_CAL,j=0` 时 alpha=0 并标记不可识别，HELD 非零输入/目标能量和全部误差照计。同组误差以逐样本直接残差平方累加作为主指标，`C−2αB+α²A` 作为矩式交叉核验；记录 F64 消减量级及负残差原值/数值状态。仅按 scale-relative 容差处理微小负舍入，严重负值失败；**正残差也可能由大数相减而静默消去**，故验证阶段必须独立用直接逐样本残差平方累加、稳定中心矩或 QR 计算，并与 A/B/C 推导结果按来源/通道块/组比较，保存绝对与尺度相对差、消减条件指标及失败状态；不能仅检查负值或将零当无损。这不是区间证书。

**输出交叉项合同：** 对互斥 hidden 组 `e_h=Σ_g e_g`，输出完整风险是 `Σ_g ||W2 e_g||² + 2Σ_{g<h}〈W2 e_g,W2 e_h〉`，且 gate 版本须先逐行乘对应 gate 再算。故每组投影的平方和不是全输出 SSE：同时直接保存全向量 SSE、组单独贡献及 `cross_remainder=full−Σgroup_diagonal`（有符号）；如需可归因具体组对，另存受预算约束的组间 Gram/交叉矩阵或流式探针，并明确是否精确。不同分组族可以重叠，不能将不同族的 q_g 或贡献一起求和；仅在同一互斥且覆盖完整的分组族内进行闭合核验。不能把 hidden A/B/C 外推成完整 W2/g1 风险，不能将中点 gate 外移到期望而忽略条件依赖。

## 存储、合并和空覆盖

1. **冻结注册层**：`design.json` 登记所有 source/split、模型及原始数据 SHA、分组/mask/projection/bin 定义、采样规则、时点和估算上限；组规则只由 CAL 或无响应的预定规则确定。CAL、DEV、HELD 的来源图像簇不可交叉。
2. **逐 source × site × panel 小记录**：`site_manifest.jsonl` 一行一个主键域，含端点匹配/coverage、权重、各统计文件路径/SHA、版本、分片序号、状态及错误；按原始 source 保留，先不跨源汇总。
3. **压缩数值层**：每 source/site/panel 的列式数组/NPZ 分开保存：`input_z`（均值/投影及固定直方图）、`input_a_channel`（长度 hidden 的 A/B/C 与覆盖）、`joint_bins`（二维有限箱）、`response_group`（hidden、全输出、g1 同组原始和交叉项）；需要位置诊断时保存固定面板的匿名 token ID/索引及所需标量，不存逐 token×hidden 全矩阵。逐通道按块流式计算；把每 token 的完整输出误差与相同 mask 的输入汇总关联，避免笛卡尔积。
4. **后处理汇总层**：存每来源、预设层/步/专家/组的原始 `sum/count` 和分母、来源级风险与描述分布；明确同时报告比值中位数与总和之比。跨源比较按预先指定的来源权重，绝不把重复 token 当作新来源。

分片只能在相同主键域、权重含义、组边界/投影版本与非重叠 token ID 验证后，以 F64 稳定累积合并；量化分位数用固定 bins（可加记录算法/参数的可合并 sketch，不能平均各分片分位数）。`coverage=0` 时 sums=0、count=0，均值/比例/分位数为 null 并标记 `EMPTY`；目标能量为零时 SSE 原值保留而相对风险为 null；有无效值、重复 ID 或不一致版本须显式失败/记录，不以零填补。SHA 绑定逐文件与最终索引；保留异常与部分回执，缺组不能作完整结果。

## 最小输出行结构

```text
site_manifest: primary_key..., endpoint0/1, timestep0/1, input_sha0/1,
  group_manifest_sha, projection_sha, mask_sha, metric_gate_id, sample_design, sample_weight_definition,
  n0, n1, matched, unmatched0, unmatched1, mask_count, sum_w, sum_w2,
  input_z_uri/sha, input_a_channel_uri/sha, joint_bins_uri/sha,
  response_group_uri/sha, execution_phase, status, numerical_status

group_record: primary_key..., group_kind, channel_or_token_axis, group_definition_sha,
  count, sum_w, sum_w2, input_stats{z,a,m,d}, response_stats{hidden,output,midgate,g1,native},
  target_energy, error_sse, signed_cross, full_output_sse, cross_remainder,
  zero_denominator, zero_cal_channel_count, timing_availability, dtype, status
```

存储用细组预设上限和分层稀疏条目，优先 160 site × 来源 × 两 panel 的紧凑通道/直方图统计；不创建全 token×全 hidden×所有组的副本。最终可执行预算须按选定的 channel 组数、箱数、dtypes 和每 source 文件数计算并另经复审，不以本草案隐含授权新运行。

## 与主方案 §6 的实现复核

§6 已明确 3 段 latent 时间 × 4×4 空间块、逐帧边际、13824 通道及 32 连续块、CAL 冻结机制箱、完整输出 cross remainder 与按来源对照，足以作为实现合同的科学骨架。为防存储爆炸，P64 全 64×64 协方差限主域，细时空组只存均值/对角/迹；组对交叉限预设 8 粗组或逐组 q_g，避免全 token×hidden×时空组笛卡尔表。分片/实际字节上限和失败门仍须执行方案冻结，不能仅凭组数保证磁盘预算。需以同一 mask 与 token ID 显式连接输入和响应、按 expert/权重及 native/smoothlift 身份分开，不能把旧交集或异来源归一化坐标当作同一对象。新实验同一个 P64 应由同 expert-layer CAL 两端合并固定，两端和各新来源共用；旧 step 独立基只复现旧结果。
