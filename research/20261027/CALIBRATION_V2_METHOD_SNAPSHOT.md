# CALIBRATION_V2 · FFNDS 的分目标校准、数据量决定与状态匹配

**METHOD_SNAPSHOT · 2026-10-27 · 研究方法冻结前的可实现设计（DESIGN / STATIC_ONLY）**

## 1. 目标函数及数据对象

研究对象：Wan2.2 I2V A14B 的四步 D-Q（high/high/low/low）共 40 Block，实际调用为：
`norm2/AdaLN → Q1 → W1 → GELU(tanh) → HiF4 Q2(64→8→4) → W2 → gate(e5) → residual → consumers → 4-step scheduler → video`。

**校准是对特定 estimand 的估计，并不只有一个统一样本量**。分为：

| 过程 | 统计量 / 科学对象 | 真实数据独立单位 | 必须核对的目标 |
|---|---|---|---|
| **C-support** | 每组 gated signed contribution \(v_g(x)=e_5(x)\odot W_{2,G_g}a_{G_g}(x)\) 及组联合结构 | image/source cluster；时步、block、token为嵌套观测 | 支持排序的源级稳定性、组间交叉项、部署相同 QDQ 的 heldout signed output |
| **gain/bias** | 原生压缩输出到 dense 输出的中心化增益和偏置 | source cluster | heldout signed residual（逐输出、逐step）及数值实施一致性 |
| **R256** | 真实 \(\mathrm{V1}_{Q1}\to \mathrm{gate}\times \mathrm{W2}\) 残差条件映射，rank256、正则化系数冻结 | source cluster，token采样为重复观测 | 协方差谱/有效维度、训练与来源留出风险、native BF16/QDQ 风险差 |
| **state** | \(\mathbb P(X_D)\) 和 \(\mathbb P(X_S)\) 下同输入差异；Student 自回馈 | source×seed 的真实自由轨迹，source为确认统计簇 | teacher/student 同坐标、相同QDQ、40×4后续流与 motion |
| **terminal** | 动作/镜头运动、主体背景一致性、画质及端到端成本 | 独立 image/source cluster（seed重复） | 实际完整视频、连续 motion、官方逐维VBench、latency |

令第 \(s\) 个source内采样 \(m\) 行、共 \(S\) 个独立来源。在理想二层随机效应假设下，均值估计方差近似
\[
\mathrm{Var}(\hat\mu)=\sigma_{\mathrm{source}}^2/S+\sigma_{\mathrm{within}}^2/(Sm).
\]
固定 \(B=Sm=6144\) 并增加 \(S\)，只会降低第一项；若同时更改图像类别/源状态，因果解释需要显式配对和独立因素设计。64/128/256/512行及12/24/48/96来源均是**候选实验坐标，不是理论正确校准量**。

### 支持分解：何以不能只追求 group-energy 拟合

原有对角排序估计 \(C_g=\mathbb E\|v_g\|^2\)。对删除组集合 \(D\) 的实际线性输出：
\[
\mathbb E\|\sum_{g\in D}v_g\|^2
=\sum_{g\in D} C_g + 2\sum_{g<h;\ g,h\in D}\mathbb E\langle v_g,v_h\rangle.
\]
含义：校准样本再多，只估对角项无法确定带符号的组间抵消/协同。新实验提供同budget `diagonal_energy` / `joint_signed_group` / `native_gate_output` 三种证据，不能预先将新评分器等同较好视频结果。

### R256 条件维度

把每 source 的原生输入视为 \(X_s\in\mathbb R^{m\times5120}\)。原 CAL24 每位置 6144 行，仅使名义 \(N/d=1.2\)；由于行间相关，真正影响低秩回归的是条件化谱、来源变异和实际有效维度
\[
d_{\rm eff}(\lambda)=\operatorname{tr}\{C(C+\lambda I)^{-1}\}.
\]
本式要求 **与实现同一中心化/尺度/正则化约定** 的 \(C\)。单源covariance eigenvalue不能直接平均成全局covariance eigenvalue；应从同一截面的足量来源实际联合输入估计，或明确标记 `OPEN_RRR_COVARIANCE`。

### Student 状态目标

\[
R_D=\mathbb E_{x\sim P_D}\|f_S(x)-f_D(x)\|^2,\qquad
R_S=\mathbb E_{x\sim P_S}\|f_S(x)-f_D(x)\|^2.
\]
当模型在全部 FFN 层同时受干预时，\(P_S\) 与 \(P_D\) 的差异可随四步去噪累积。Student state 的同输入拟合还需要与**真实free-rollout终端质量**分开评价，避免 teacher-forced 的局部改进被解释为 motion 保留。

## 2. 选择校准数目的可证伪程序

**第一阶段：来源方差和支持稳定性（已保存 CAL24 静态资产可优先复用）**：
- 对每个 step×block×state 重建来源级 `group_energy[216]`，一个image cluster内多prompt先平均；
- 以完整来源簇为单位计算 leave-one-source-out TopK Jaccard、边界分数 margin、源间方差；如果真实组间 signed Gram 可得，再计算交叉项；报告具体位置/噪声分支和源级尾部；
- 位点样本 \(S<2\) 或统计量不存在时记录 `INCONCLUSIVE / NOT_MEASURED`，提供需要的 producer，而不是以0代替。

**第二阶段：固定总行数，改变来源数**：

| arm | Sources × rows/位置 | 场景配额（subject/background/camera） | 目标 |
|---|---:|---:|---|
| C12_R512 | 12×512 | 6/6/0 | 稳定性小样本参照 |
| C24_R256 | 24×256 | 12/12/0 | 原UB1名义预算参照；v2新坐标新拟合要另运行 |
| C48_R128 | 48×128 | 24/24/0 | 独立来源效应 |
| C24_R128 | 24×128 | 12/12/0 | 同源降低行数 |
| C24_R512 | 24×512 | 12/12/0 | 同源增加行数 |
| C48_R128_MOTION | 48×128 | 20/20/8 | 独立的**运动域扩展**效应，不参与仅来源数的等价比较 |

场景分类配额是此轮可解释对照的起点，摄像机类别在独立的 domain arm 中加入；其真实比例应在正式外部目标分布已定义后重新冻结。source 选择首先按真实 `image_id` 组成独立簇，按类在**同一冻结source bank与seed**下置换簇并抽取嵌套前缀，然后在每簇内选择一条明确的image-prompt条件；不能先抽重复source再把撞到同图像解释为来源不足。若增加来源或修改类别，形成新cohort版本并记录实际被选的source/prompt，跨版本嵌套关系须从保存的cohort名单实证核对。标准VBench完整Image Suite保留全部所需image-prompt条件，条件数量与独立image cluster数量各自报告。采样行使用同一 source×step×block 的随机前缀。存量 CAL24 的 NumPy 选择与 v2 的 Python PRNG 不属于相同“精确行配对”，真正共同对照须都从 v2 采集，或严格重放历史坐标。

**第三阶段：相同来源与预算的机制竞争**：
1. 支持评分差异（diagonal、signed joint 等），每条运行都重建原生 Q2 与W2；
2. 相同支持、相同训练预算下比较 `teacher_input` 与 `student_input`，只改变目标状态；
3. 对rank 256 检查正则化谱、训练-留出误差及子空间稳定；对增益/偏置记录有符号输出来防止只降低平均 SSE；
4. 原生 40×4 同输入检查与 Student 自由轨迹及视频指标构成新的 deployment evidence。

**停止/扩容规则**：事前冻结 metric direction、来源级CI/尾部验收和 cost budget；先找统计稳定点，再在与目标一致的 holdout 上评估 native signed output 和全视频。当扩大独立来源提供清晰边际收益时，追加来源优先；当固定来源上增加 token 显著改善 RRR 泛化时，追加行优先。若改动准则是根据已看见的目标结果提出，则新建 `POST_HOC_EXPLORATORY` method_version，再在未暴露确认来源上检验。

## 3. 运行合同与明确状态

- **REGISTERED_DESIGN**：设计存在；无实际操作数。
- **NATIVE_CAPTURE**：source、step、block、expert、state、原生张量边界、坐标及dtype对应真实有效文件。
- **SOURCE_DIAGNOSTIC**：统计量由真实source-native operands计算，已报告来源数量和稳定性。
- **NATIVE_FITTED**：cohort-aware fit + 同精度 QDQ 数值验收和真实source holdout。
- **VIDEO_VERIFIED**：真实配对生成、完整的质量/动态/成本证据和未见确认来源。
- **SCIENTIFIC_CLAIM**：明确 claim、比较前提、独立实验单位与估计不确定性。

旧 CAL12/CAL24/UB1 的原始运行收据仍属旧方法事实；不改变、重命名或隐式继承为新方法的 `NATIVE_FITTED`。

## 4. 实现映射、资产及检查

- Frozen source bank → `tools/ffnds_experiment_plans.py`；
- 实际 geometry + row draws → `ffnds_native_geometry.py` 和 `calibration-plan --source-state teacher_input/student_input`；
- 本轮方法注册和 source-level 支持分析 → `tools/ffnds_calibration_design.py coverage/diagnose`；
- 原生 Q1/Q2/W2/gate operands header 与抽样坐标 → `ffnds_validate_calibration_operands.py`；
- 独立 video + motion → `ffnds_run_registered_video.py`、`ffnds_stage_registered_i2v.py`；
- source-cluster 统计 → `ffnds_cluster_metric_report.py`。

实验 intake 即建立 `REUSABLE_ASSET_PLAN`，包括：source bank / exposure、geometry、selected row IDs、teacher/student states、group 统计与cross terms、native BF16/QDQ operands、fit policy、完整 video / motion 指标、源级 paired effects。每种记录 producer、语义、状态、坐标、units、format/locator、可重用条件和失效条件；本次设计与CPU测试的产物只标 `STATIC_ONLY`。

## 5. 关键既有证据与成熟实现

- 项目：`config/calibration24_ffn_operands.json` 记录24×256和来源类别；`tools/fit_step_budget.py` 的循环固定24×256；`reports/worker_20261020/calibration_row_coordinate_exposure_audit.json` 确认原生同坐标采样；`UB1_SOURCE_GENERALIZATION_AUDIT` 记录外部来源影像质量尾部及状态拟合预算差异。
- SparseGPT：原始论文通过数量消融、下游困惑度和随机种子波动来选择校准量；其128条语言序列不能移植为视频样本数。https://arxiv.org/abs/2301.00774
- Q-Diffusion：覆盖真实去噪时间并通过时间/数量消融决定样本点。https://openaccess.thecvf.com/content/ICCV2023/html/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.html
- APQ-DM：使用时步分布和信息量作为校准数据选择因素。https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Towards_Accurate_Post-training_Quantization_for_Diffusion_Models_CVPR_2024_paper.html
- S²Q-VDiT：长视频token校准方差和Hessian-aware显著数据选择提示按信息质量而非只增量。https://proceedings.neurips.cc/paper_files/paper/2025/hash/31ed129feae64a7e44a15b148c15558d-Abstract-Conference.html
- ICLR 2025 Beware of Calibration Data、NAACL 2025 Self-calibration：数据分布接近程度和来源对量化/剪枝有强烈影响，适合作为来源敏感性对照，而非数目抄写。https://proceedings.iclr.cc/paper_files/paper/2025/hash/2ede933e10afa991a10b6f36b6522129-Abstract-Conference.html ; https://aclanthology.org/2025.naacl-long.509/

## 6. REUSABLE_ASSET_PLAN 与 IMPLEMENTATION_RECORD（本轮）

| asset_id / 格式 | producer 与真实来源依赖 | 复用合同、失效条件与当前状态 |
|---|---|---|
| `calibration_v2.source_bank` / JSON | `ffnds_experiment_plans.py source-bank`，绑定数据集revision、图像簇、prompt、曝光历史 | 仅按完整role与scene-stratum复用；重新分类/识别近重复后更新version；本轮 **NOT_PRODUCED** |
| `calibration_v2.geometry` / JSON | `ffnds_native_geometry.py` 从原生full160 capture读取step/block grid及population | 绑定同一I2V预处理、分辨率、native位置与sample语义；本轮 **NOT_PRODUCED** |
| `calibration_v2.draws` / JSON | `calibration-plan --source-state`，来源ordinal、分层嵌套随机过程与真实row IDs | teacher/student共享坐标；旧CAL24 NumPy与v2 Python PRNG坐标方案不同，必须标别；本轮 **NOT_PRODUCED** |
| `calibration_v2.native_operands` / BF16 safetensors+index | 原生模型采集者；V1/V2/Y、b2、gate、QDQ与实际row匹配验证 | 依赖实际tensor_boundary、shape、dtype、完整160位置；新拟合后继续保留可重用原生资产；本轮 **NOT_PRODUCED** |
| `calibration_v2.source_stats` / JSON | 从原生 V2、W2(QDQ)与gate计算每source/prompt每step×block×state唯一的 `group_energy[216]`，可选完整 signed Gram | 所有值带source/image/step/block/state；同模型/同gate/同QDQ才复用；本轮 **OPEN_RUNTIME_BINDING** |
| `calibration_v2.diagnostic` / JSON | `ffnds_calibration_design.py diagnose` 对真实来源输入的统计处理 | source簇LOO、bootstrap、signed残差；仅代表统计稳定性，非视频确认；本轮 **NOT_PRODUCED** |
| `calibration_v2.fit` / native policy + numerical receipts | 新cohort-aware支持/gain-bias/R256拟合器，真实训练-留出与QDQ部署 | 原拟合 `fit_step_budget.py` 的冻结24×256策略为独立历史基线；新fit尚待binding；本轮 **OPEN_RUNTIME_BINDING** |
| `calibration_v2.video_metrics` / MP4、metric JSON、配对run收据 | 真正 Dense/student 完整生成、官方指标、连续运动/相机动作 | 真实无噪声复用或重命名不增加样本；预注册独立确认集；本轮 **NOT_PRODUCED** |

上述表是**预期资产及复用边界**。每次真实运行需将生产条件、scope、locator、format、shape/units、method revision、determinism、校验命令、validity及失效条件写入独立的 `REUSABLE_ASSET_RECORD`；没有实际产物不填虚拟路径。

**IMPLEMENTATION_RECORD（CPU / STATIC_ONLY）**：`ffnds_calibration_design.py` 注册检查、按独立image cluster分层配额、逐step 54/108/108/162支持稳定性、可选216×216有符号Gram、teacher/student同坐标行计划和VBench自定义参考图像映射均已提交。新方法的功能性验收只能沿真实 `source→native tensor→operator→full video→metrics` 运行路径进行；本轮仓库CI覆盖的是CPU合同和数学合成数据，并不代表GPU环境验收。

**首次真实运行前 ENVIRONMENT_CONTRACT**：冻结 `research/ACTIVE_RESEARCH.json` 下的实验注册范围、模型与native runtime版本、校准source/image/prompt数据版本、每步expert、V1/V2/Y BF16实张量与QDQ权重、GPU设备/显存、磁盘配额、随机种子及状态、异常后新attempt路径、真实视频解码及VBench原版依赖。目标统计含义与实际native取值若不一致，应记录 `METHOD_SPEC_GAP` 或 `OPEN_BINDING`，用最小source/step/block probe确认后再扩大到全160。

### 本轮证据状态

- `METHOD_SNAPSHOT`：**DESIGN**（科学对象、估计量、竞争假说、消融、原生数值绑定）。
- `IMPLEMENTATION_RECORD`：**CPU_CONTRACT_PASSED**（[2026-10-27 GitHub Actions](https://github.com/Qesire/FFNDS/actions/runs/40512223468)，合成边界测试，不代表完整GPU）。
- `OPEN_BINDING`：实际24来源的 `group_energy[216]` 和 signed Gram 原生采集者；cohort-aware支持/gain-bias/R256拟合及消费者级的数值验收；运动完整视频与独立确认集。
- **研究结论**：历史CAL24是可追溯基线预算，尚无新方法质量、运行成本和最优校准来源数的实测证据。



## 项目专属校准研究操作

遇到校准量、校准数据或剪枝/量化模型状态问题，加载 [FFNDS calibration METHOD_SNAPSHOT](CALIBRATION_V2_METHOD_SNAPSHOT.md)。先提炼当前已有数据揭示的结构：源间方差、signed group crossing、rank-256协方差有效维度或Student分布转移；将每项信息对应到独立的科学estimand，再用来源级稳定性、同预算对照、真实原生算子/视频结果分别判别。借鉴SparseGPT的scale ablation、Q-Diffusion的时间覆盖、S²Q-VDiT的显著数据选择，通过实际资产可得性确定最小区分性研究。用真实producer和可复用数据使下一步可执行；以`NOT_MEASURED`记录尚需的校准资产，继续积极推进理论与静态证据。

本段保留项目提交 `ec949769` 的领域指导；通用 AutoResearcher 文件继续与 ACTIVE_SOURCE_MANIFEST 的上游 blob 保持一致。
