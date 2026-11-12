# G20 R0 资产与接口盘点（只读）

2026-11-12；首批 R0+R1 的 R0 交付。依据 FFNDS main@ab41d738 的 G20_NEXT_EXPERIMENT_ROUTE.md 第 2 节。
盘点方式：登录节点只读查询 + 计算节点 srun 验证（verify_r0.py，CPU 分配）；未做任何生成、拟合或视频。
R1 冻结 scope：G20_R1_SAME_INPUT_SCREEN_20261111，scope SHA 4bbf8496484f80506bf59565e2b9c1556e19a908de0c8736643126724cf41625。

## 1. 可复用 / 缺字段表

| 资产 | 位置（cluster） | 身份绑定 | 状态 | 备注 |
| --- | --- | --- | --- | --- |
| CAL4 capture（chip10_02/04/07/09，seed42） | work/g19_native_capture10_20261103/<source>/output_<source>/capture/ | sidecar SHA + payload file_sha256（加载时双重验证）；checkpoint_sha256；grid[21,29,52] | 可复用 | 4 源×4 步×40 层=640；均匀 2048 行面板 + 336 tubelet 分管；U=z（pre-W1）、w2_output_before_gate、gate(1,1,5120) |
| DEV5 capture（chip10_01/03/05/06/08，seed42） | 同上 | 同上 | 可复用（已暴露开发源） | 800 个；来源外开发评估，不是盲测确认 |
| checkpoint high | weights/Wan2.2-I2V-A14B-LightX2V-Distill-4step/high_noise_model/…safetensors | SHA 39d3af74e46e59e989db5cc36f401e666f3ae77c7648eb55d3aff9fee47716fa | 可复用 | BF16；R1 加载时再哈希 |
| checkpoint low | …/low_noise_model/…safetensors | SHA a3b8680affe2ab2e2121210a83ffa7d351f77da18191e57bb359d7839beba8a2 | 可复用 | 同上 |
| Q1-r2 部署 BF16 矩阵（80 站点） | work/g19_strengthen_20261107/fit_{high,low}_r2/output/fit/layer_NN/Q1.npy | q1_matrix_manifest.json SHA d127f0941fa018cfded78aa7476d96abcee141fcc13a3260af91c7060f57410f；每站点 SHA+provenance（fit scope/receipt SHA） | 可复用 | BF16_RAW_UINT16 (5120,5120)；recipe F64 W1^T diag(alpha) W2^T → F32 → BF16 RNE；不重拟合 |
| ICC svd_r512 因子（80 站点） | work/ffn_baselines_20261107/icc_all80_{high,low}_r1/output/ | icc_svd_manifest.json SHA 6d0a576481f8d36c367171bd47f2b81029d5470bff06a3584740f550047c876e；每站点 SHA | 可复用（已真实执行） | fc1(13824,512)+(512,5120)、fc2(5120,512)+(512,13824) BF16；冻结配置 svd_r512 ffnds_two_residual |
| ICC cdsvd_r512 因子（80 站点） | 同上 | icc_cdsvd_r512_manifest.json（ref_remote 存档） | 可复用 | 本轮不评分（冻结配置仅 svd）；保留备查 |
| ICC 实现与收据 | ref_remote/icc_ffn.py、icc_runner_all80.py、icc_RUN.json、icc_EXIT.json | 代码 SHA 见原 scope | 可复用 | 两残差语义已逐行核对一致 |
| capture 生产代码 native_tap.py（9 源） | work/g19_native_capture10_20261103/<source>/native_tap.py | 9 份逐 bit 相同 SHA 67a269a0487df5714932fa8b86a07c7b1ad3fa60f4073133f57253564d15f034 | 可复用 | 已列入 scope external_source_sha256 |
| Chipmunk 作者块均值实现 | ref_remote/chip_faithful_core.py（MIT 上游保留）；compiled author selector .so | 代码存在 | 代码可复用；数据缺字段 | voxel(4,6,8) 序、block_mean_valid(block=128，尾 52 行)、topk_indices(0.7,256,0.05) |
| Chipmunk full-M proxy 分组统计（CAL 源） | 不存在 | — | **MISSING_ASSET** | 缺 full_M_U_anchor、full_M_U_hit、voxel_block_means、author_topk_mask、valid_tail_rows；2048 行散点不能替代块均值；R1 不打分，延后 R2 共享采集 |
| Chipmunk global-P64 变体 | ref_remote/chipmunk_README.md、chip_scope.json | 独立 scope（CHIPMUNK_FFNDS_GLOBAL_P64_VIDEO） | 存在但非臂 6 | 与作者块均值不同架构；不得混称为作者忠实臂 |
| 原 40 步 CAL（g19_original40_anchor10_20261104） | work/g19_original40_anchor10_20261104 | 64 行、40 步同 ID、双 CFG | 存在；本轮不用 | 权重×间隔支线（方案第 5 节）另审；02/09 E1a 另列 |
| O-CAL 02/04 | 旧 O 资产 | source/prompt/preprocess/noise 交叉核验未完成 | 存在；本轮不用 | 不得假定与 D 同名源已匹配 |
| G19 视频/指标归档 | 旧 14 视频范围与八轴指标 | 原身份 | 只读复用 | 本轮 0 新生成 |
| Attention 强方法兼容实现 | 本批未核验 | — | 未实现/未核验 | 按方案前移到 R2 系统背景决策；不假定任意 Attention 方法可用 |

## 2. 计算节点实测（verify_r0.py，srun CPU 分配）

- sidecar 模式：含 file_sha256、checkpoint_sha256、grid、image/prompt_sha256、expert、step、layer、seed、sampling_panels 等；抽样 payload SHA 与 sidecar 一致（True）。
- gate 身份：high/0,4,19,34 step1 与 low/4,19,34 step3 共 7 站点，9 来源 gate 逐 bit 相同（全部 true）。gate 为逐通道 (1,1,5120)，支持共享 Gram 精确特征解。
- token 交集（high/4，steps0∩1，精确 ID 计数）：

| source | chip10_02 | chip10_04 | chip10_07 | chip10_09 | chip10_01 | chip10_03 | chip10_05 | chip10_06 | chip10_08 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 交集行数 | 126 | 137 | 129 | 125 | 136 | 122 | 114 | 135 | 134 |

接近条件期望 2048²/31668=132.45（独立均匀抽样假设下的数学估算）。全部 80 站点×9 来源的实际交集由 R1 job 的 matched_counts 逐站点记录（读 ID，不估算）。

## 3. 运行环境

- 分区 P107-A100，account competition，qos qos_p107-a100；A100-SXM4-80GB×8/节点（anode16–23）。
- python /public/home/scc/pb23061276/miniconda3/envs/hif4/bin/python3.11；torch 2.10.0+cu128。
- 全部计算在 Slurm 计算节点；登录节点仅只读查询与传输。runtime_guard 拒绝 login 主机并要求真实 RUNNING 分配。

## 4. 缺字段汇总与 R2 采集需求

R1 唯一缺数据臂为 Chipmunk 作者块均值（臂 6）。R2 共享采集需补：六个（或全 160）expert-layer 调用的 full-M U_anchor/U_hit、真实 voxel 块均值、作者 topk mask、有效 52 尾行；以及方案第 3 节列出的端点/scheduler/RNG/条件 payload。缺端点与 R2 合并采集，不为 R1/R2 各采一遍。