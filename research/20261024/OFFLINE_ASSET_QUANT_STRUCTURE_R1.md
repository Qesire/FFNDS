# FFNDS · 已保存资产离线检验 R1（2026-10-24）

## 任务边界与输入

用户在原先暂停所有实验后明确要求“利用资产开始离线实验”。本轮仅恢复**已有来源报告的离线重计算及有限码字数学穷举**；未恢复 GPU、模型推理、生成或自动作业。权限收据：[OFFLINE_ONLY_RESUME_SCOPE_R1.json](OFFLINE_ONLY_RESUME_SCOPE_R1.json)。此前的 [EXPERIMENT_PAUSE.json](EXPERIMENT_PAUSE.json) 保留历史原样。

- 真实资产 1：[OP2D_native_state_description_RUN.json](../../reports/worker_20261020/OP2D_native_state_description_RUN.json)：24 个原来源、Step 0–3、Block 20，C24/UB1 轨迹的原生 V1 与 TeacherY 连续张量差异汇总；原配对通过 24 个前缀 bitwise controls。
- 真实资产 2：[C1_boundary_target_audit.json](C1_boundary_target_audit.json)：6 个原来源、72 个已采集 FFN 位置、每位置 31,668 行的两个真实边界标量 SSE。
- 已核验结构来源：[OP2A_complete_dataset_integrity_RUN.json](../../reports/worker_20261020/OP2A_complete_dataset_integrity_RUN.json)：3,840 份、40,345,743,360 B 的历史原始张量校验记录；文件本体不在 Git 仓库中。
- 已核验量化结构：[R12_HiF4_group_separability.json](../20261022/R12_HiF4_group_separability.json)：8 份原先的逐组/逐行分离检查全通过，组内置换有 401 个 BF16 不一致元素。**这是已有证据复用，并非本轮重复执行 HiF4 CUDA。**

可机读的本轮结果：**[OFFLINE_SAVED_ASSET_QUANT_STRUCTURE_R1.json](OFFLINE_SAVED_ASSET_QUANT_STRUCTURE_R1.json)**、**[OFFLINE_FP4_ORBIT_FULL_W1_COST_R1.json](OFFLINE_FP4_ORBIT_FULL_W1_COST_R1.json)**。

## E1：已保存的 24 来源原生状态漂移，离线重新聚合

以下均为*相同来源的两条不同轨迹在 Block 20 的 V1 相似度*，不是同一步不同 token 的 Q2 量化组码字重复率。

| Step | 24 来源宏平均 V1 cosine | 宏平均 V1 relative RMSE | TeacherY cosine | UB1 / C24 自身状态门控 SSE 总和比 |
|---:|---:|---:|---:|---:|
| 0 | 0.577669 | 0.917300 | 0.528582 | 1.732122 |
| 1 | 0.804275 | 0.603007 | 0.769754 | 0.974490 |
| 2 | 0.726036 | 0.720027 | 0.716242 | 0.981780 |
| 3 | 0.589228 | 0.904989 | 0.599382 | 0.467053 |

Step 1 的 V1 cosine 高于 Step 0：**23/24** 来源；高于 Step 3：**24/24**。宏平均 Step 1−Step 0 = **0.22660528**，24 次 leave-one-source-out 宏均值范围 **[0.22223410, 0.23853281]**。

**机理指向：** 同一个原来源的轨迹差异会随着去噪步明显变化。跨步缓存不能由“已有量化分组”本身论证；后续实际组共享应首先固定 step/expert/prefix。自状态 UB1/C24 SSE 比用的是不同轨迹的各自 Teacher，不能转译为同一前缀的优劣或视频接受率。

## E2：已保存的六来源下一层误差，目标改变导致分组优先级改变

72 个位置、每位置 31,668 行的两个事后 full-row oracle top-25% 选择，在真实记录中重新求和：

- 平均 Spearman = **0.87563494**；均值 top-25% 重叠 = **0.78297580**。
- 按当前 FFN 写回 SSE 选择 top-25%，捕获总下一层 norm SSE 比例 **0.35132682**。
- 按下一层 norm SSE 选择 top-25%，捕获同一 norm SSE 比例 **0.36711267**。
- 总量加权差异 **0.01578586**，约 **1.58 个百分点**。按 Step 的差异分别约 **2.20、1.12、1.09、1.16 个百分点**。
- 72/72 个位置上，按照同一下一层 norm 标签排序的 oracle 捕获量不低于按照不同写回标签排序；这条**单调性来自 oracle 选择定义，本身不是新的经验发现**。值得解释的是变化幅度和按步骤异质性。
- 独立来源数只有 6，不能把 72 个位置当成 72 个独立场景。

**机理指向：** 若后续量化 cell/组共享需要非严格等价的 fallback，则必须与 downstream-conditioned 可观测风险对比；本次 C1 数据只给出该问题必要性，不包含任何新算子的独立结果。

## E3：E2M1 FP4 码字尺度轨道穷举

对 15 个不同数值的 E2M1 字母表（±0 合并），把数值乘以 2 后得到整数集合 `{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}`。将元组按整体非零实数倍取等价类，对实际枚举的每一个元组计算绝对 gcd 并把首个非零分量符号归一化。

| 子块宽度 k | 穷举元组 | 穷举获得的尺度轨道数 | 128 token 同一位置子块中被强制重复的最少次数 |
|---:|---:|---:|---:|
| 1 | 15 | 2 | 126 |
| 2 | 225 | 41 | 87 |
| 3 | 3,375 | 914 | 0 |
| 4 | 50,625 | 17,105 | 0 |

穷举轨道数与独立评议的容斥闭式 `1+(15^k−11^k−7^k+5^k)/2` **四个 k 全部逐项相等**。它验证了形式等价类容量，*不测量真实 Wan Q1 原生量化码分布*，也不保证量化额外因子可直接被 E8M0 NV/Tensor Core 原生吞入。

## E4：把轨道复用放回完整 W1 成本核算

对 Wan FFN `D=5120,H=13824`、一次 `T=128` token tile：

- dense W1 理想 MAC 数 = `T·H·D=9,059,696,640`。
- 若切成 `k=2` 的码字子块，`D/k=2560` 份，每个 token 每个输出仍需搜集相应部分和，合计 **4,529,848,320** 项查表/重缩放/累加贡献。
- 对 k=4、16、32，该数量仍分别为 **2.265B / 0.566B / 0.283B**。
- k=2 的 41 类在 128 输出通道且 4-byte 表项时仅需 20,992 B 的原型表，但这个空间便利**不免除所有 token×输出×输入子块的归约工作**。
- 所谓 18.12 GB（k=2）是每贡献取 4-byte 值的**逻辑读请求量**，不是 GPU DRAM 实际搬运量，也不是带宽基准。真实 kernel 可能利用缓存、SIMT、warp/tensorcore 重构，故不能从此计算数推出加速/减速实测值。

**决策：** k=2 的必然重复是统计恒真式，但天真逐子块查表不是值得直接落地的 W1 替代。若继续此路线，应首先提出能**消除或融合数十亿次部分和 gather/accumulate**的物理算子；若没有，则优先讨论更宽 k 的实际重复性，或把 FP4 量化产生的信息用于 W1 前证书/原生 group packing。

## 数据可得性与后续离线检验

现有 Git 不包含 OP2 `.safetensors`、C1 `.npz` 原始字节或大型 model/cache；运行环境也未挂载这些对象，因此本轮对 quantizer 原始 codes/scales 的**实测**无法进行。原始操作数还需检查字段，不能将 V1/TeacherY 的连续张量相似度等同于 Q1/Q2 编码重复。具体缺失的是：(1) 选定来源和步骤的逐元素 Q1 输入/实际码和 effective scale；(2) GELU 后 Q2 完整 block/global scale 与码字；(3) post-Q2→pre-W2 的逐 token 有效 operand（含 gain/BF16）；(4) 对应 BF16 实际 W1/W2 权重与布局。

再利用工作节点已有资产时，只需取一个 source/step/block 的少量原始位置并沿用原始 SHA 校验，优先回答：

1. 按原物理 token tile 报告 MXFP4/NVFP4/HiF4 相同子块的真实字母表熵与重复率（格式适配/weight 路径固定）。
2. 记录局部与全局 scale leader，区分 Oracle 完全相等、由便宜输入信息能证相等、完整 W1 后才能知道相等。
3. 用 Q2 真实 post-gain W2 operand 比较完整组复用，而非只比较原始 Q2 码字；注明以哪个版本的 Dense/QDQ 为数值参照。

本轮**不自动启动这些依赖缺失字节的下一步**。此文档不把缺失操作数替换为 toy 分布。

## 可复现入口

在包含此仓库的本地环境运行（无 GPU、无需下载 HF）：

```bash
python tools/offline_saved_asset_quant_probe.py --repo-root . --out /tmp/ffnds_offline_r1_replay.json
```

输出重新计算 OP2D 24 来源、C1 72 位置汇总及 E2M1 码字 k=1..4 的精确轨道数。R1 原始 JSON 还包含每个来源/Step 的分层统计、SHA 绑定、资产可用性及 E4 成本枚举。
