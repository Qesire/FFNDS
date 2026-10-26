# E0 paired trace versus historical panel metrics — independent domain audit

日期：2026-10-26。结论：**旧 8.54% 与新 full-M teacher-operator 约 20%–28% 的数值不能直接构成部署退化或全 M 分布失配的定量对照。** 门控、来源与汇总、token 支持、计算对象均有变化。旧 block0 的 0.85% 也不是新首 site 的同条件基线。本文保存此前已完成的 JSON 与源码审查，不运行模型、张量计算或测试。

## 实读证据与身份

| 对象 | SHA-256 |
| --- | --- |
| v3 source0 `dc_source00/RUN.json` | `4b51d835b2f493a06dbac1d1b100c66d380b3ea128fa645591cec6531c49fc98` |
| v3 source0 `dc_source00/TRACE.json` | `26789cad90a54019df901838d6354fad40b229d12b652b9afc5ec4bd180c22c8` |
| 旧 `all_block_reconstruction_attempt1/RUN.json` | `27889f48aa2ce19ead820b09b69bc8f52b0b70a51cbd44fec950b82100b74b6b` |
| source0 v3 protocol SHA | `3027bc17588efdcdab32aacc5084cca6705b58d76f75529be43edeeda1b7bcc5` |

前三项均在 HF derived cache；新结果相对位置为 `e0_paired_full_depth_20261026/v3_attempt1/dc_source00/`，旧结果为 `e0_function_basis_20261026/all_block_reconstruction_attempt1/RUN.json`。已读取公开汇总 `research/20261026/E0_ALL_BLOCK_RECONSTRUCTION.{md,json}`、`E0_40_BLOCK_DISTRIBUTION_AND_CUMULATIVE_NEXT_20261026.md`，以及 `tools/run_e0_all_block_reconstruction.py`、`build_e0_reconstruction_bank.py`、`e0_paired_ffn_observer.py`、`e0_full_depth_trace_contract.py` 的实际计算。

source0 新 RUN 状态为 `PASS_NATIVE_AND_CORE_METRICS_NOT_QUALITY`，160 个 site、4 个 step，实际 FFN panel 为 `[31668,5120]`、scheduler 状态为 `[16,21,58,104]`；`quality_accepted=false`。source12 后续 PASS 是 root 告知，本文件没有对 source12 JSON 独立计算，也不合并其数据。

## 四处不可混用的口径

1. **gate 与分母。** 旧全层指标为 `||g ⊙ (prediction−Y)||² / ||g ⊙ Y||²`，源码实际乘保存的 gate。新 TRACE.FFN 为 ungated FFN 输出的能量比。TRACE.native_writeback 又是实际 residual/gate 写回，参考能量包含 residual；它也不等于旧 gated FFN 能量比。没有统一 gate/分母前不能把两列百分数直接相减或相除。
2. **来源与汇总。** 8.54% 是 source0/12 × 4 step × 40 block 的 320 个 case 比值等权平均；source0 旧均值单独为 **9.8414479%**。新给出的 28.28/26.21/21.29/20.25% 则是单 source、每 step 40 层的中位数。case mean、层中位数与能量 pooled ratio 是三个不同统计量。
3. **token 支持。** 旧误差评价是保存的 uniform_disjoint32（另有 center128），并非全 M；旧 full-M 指部分斜率统计，不能移用成旧误差覆盖。新 FFN 四臂能量真实归约全部 31,668 行，另有九个 raster 分区；它给出新部署对象的完整覆盖，但没有单独隔离“扩展支持”造成的差值。
4. **函数与部署对象。** 旧离线重构直接使用保存的 native `q2_post` 系数与 native `ffn_output` 标签，以 BF16 原权重升 FP64 做 affine/ridge/投影。它不是未经量化的 GELU hidden 重构。新部署用 BF16 pack、真实 partial W1/GELU/Q2、BF16 新读出与折叠仿射矩阵；选宽 GEMM、Q2、系数/折叠舍入等差异与支持变化同时存在。

source0 的旧与新数值只能并列说明范围，不能解释为已测净退化：

| step | 旧 gated uniform32 case mean | 新 ungated full-M teacher-op 层中位数 | 新 student-op / 同一 teacher 能量层中位数 |
| --- | ---: | ---: | ---: |
| 0 | 14.5126% | 28.2782% | 13.6106% |
| 1 | 10.0565% | 26.2124% | 9.7400% |
| 2 | 7.8578% | 21.2917% | 7.8159% |
| 3 | 6.9390% | 20.2485% | 7.8645% |

旧 **0.8469391%** 是 block0 两 source × 四 step 共八 case 等权平均。真正 source0/step0/block0 的旧 affine54 uniform_disjoint 比值为 **1.0127095%**，center128 为 **1.0649056%**，仍是 gated 保存数据诊断。新首 site 的 **2.5248876%** 是 ungated full-M native 误差；因此“0.85%→2.525%，退化近三倍”不是同条件比较。

## native-versus-continuous 的实际边界

新 source0 teacher 32 行 native-continuous 相对 SSE 的跨 160 site 中位数为 **0.2487064%**。observer 的 continuous 函数将**同一部署 BF16 pack**升 FP32，重新执行 GELU/linear，并移除 Q2 与中间 BF16 cast；分母是这些 32 行的 continuous candidate 能量，而非 dense FFN 能量。它不是旧 FP64 saved-q2_post 重构，也不包含原拟合参数被 BF16 pack 化之前的误差。

因此，小的 0.249% 不能证明旧拟合与部署 bank 在 full-M 上等价，也不能证明部署数值是失败主因或可忽略。它只约束同 BF16 系数、明确 32 行上 native candidate 与该 FP32 重算的差。没有同时固定支持、gate/分母与参考函数的桥接对照，H-NUM 与覆盖/函数容量解释仍混杂。

## 可成立的新认识与最强反解释

新首 site 的 teacher/student post-Q1 输入差为零、prefix response 为零，teacher-op 与 student-op SSE 相等。因此该处实际 same-input approximation mismatch 的存在不依赖 student prefix 漂移。其余 teacher-op 也在真实 teacher 输入上测量，不能将它们本身归因于 student 前缀。

相反，student-op / teacher-energy 较小不等于“student 轨迹更准确”或“漂移自愈”：这是改变输入后、仍除以 teacher 能量的绝对误差比较。输入/函数能量收缩或非线性工作区改变都可使该误差变小；当前四臂字段没有完整保存 dense-student 输出自身能量，不能直接计算其同分母相对风险。更小的局部误差可以与更大的 prefix/最终状态偏离共存。

最强反解释首先是**测量域不同**：gate 的坐标重加权、source/统计汇总与支持改变，可能解释相当部分旧新表面差距；另有 BF16 pack 折叠与选宽 native 路径的真实差异尚未在同一域中单独识别。当前 JSON 支持完整新对象的描述及有符号几何分解，不支持把旧新差距单因归为 full-M 分布、部署数值、固定支持失效或下游传播。最小判别条件是同 site、同 rows、同 gate/分母，分别列旧 saved-q2 重构、部署 BF16 pack 的 FP32重算与 native 输出；这是未执行的比较条件，不是新增作业。

图锚点为 F03/F04/F09、F12/G20、F13/G21 与 P22/P24；覆盖/能量值是已保存 OBS，错误来源的边仍为 conditional，跨层/最终视频因果归因仍为 posthoc/OPEN。本轮按任务只审本地 JSON 与源计算，没有新增外部论文/新颖性主张，也没有追加模型授权或质量接受。
