# A5+HWG 匹配边界补采：主 Agent 自审

作者 `/root`；2026-10-30。状态 `AUTHOR_SELF_AUDIT / INDEPENDENT_REVIEW_REQUESTED`。该文件不代表独立批准或启动。

M0 真实文件检查发现：HiF4 CAL24 的 1,920 个同专家跨步 block 对，匹配 token 数为 0–8，平均 2.0354，完整原始 M16/M32/M64 tile 均为零。8 源 BF16 随机行亦不配对；两源 BF16 spatial52 虽有 1,092 个匹配坐标，却不包含完整物理 M16，且数值族与 HiF4 不同。因此旧数据不能回答全层增量支持规则性，必须补采。逐项地图见 `research/20261030/A5_HWG_M0_ASSET_COVERAGE_20261030.csv`。M0 只检查 JSON SHA、坐标、payload 长度及 header，未读取模型数值。

原 A5 runtime 的 `beta_row * group_scale64 * ternary` 和原 HWG gate×W2 列范数源码已找到。参数 producer `hif4_scale_sign_proxy.qdq_emulation` 和 compact 参数尚未找到；不得声称原 A5 复现。新数值方案明确命名 `A5_INSPIRED_MEANABS64_ROW_LS_V1`，独立标记其参数推导。源恢复/新适配是不同身份。

## 本次有限范围：只采集，不做选择/拟合/干预

4 个历史暴露开发来源 ord898、96、1040、979，seed42。每来源一个独立进程，在同 checkpoint、HiF4 fake-QDQ+BF16、同硬件及 scheduler recipe 下先 plain Dense，再被动 observer Dense，共8条完整81帧视频。四步 high/high/low/low，全部40块，160位置。无训练、rank分解、稀疏选择、oracle、官方评分或自由学生候选。本次准备的是下一 E0 的实际可重放资产，不能作为 E0 方法胜负结果。

每位置保存真实 post-Q1 `V1[31668,5120] BF16` 全量；另保存 21 latent τ 上每τ一个真实对齐 M64 raster tile（共1344行），坐标在四步/40块相同，包含原生 U/Z/H、Q1/Q2输入输出、Y、gate、residual/writeback及b2。后续从 fullM V1 与原有效W1/W2、bias、smooth2/native Q2重建 fullM FFN，必须首先与本次采样的实际 Z/H/V2/Y 核验；重放是后续另审科学范围。本次不保存 fullM preQ1 rawx，不能据此重建其完整 Q1 动态尺度；Q1已支付且实际输出V1保存，未来W1/W2候选均计完整Q1价格。

需要区分的机制：H1 增量支持比当前贡献更集中；H2 元素集中但原K64/tile并集饱和；H3 真正空间共享优于shuffle。此次采集提供三者共同缺失的匹配坐标、fullM输入、两专家和全40层信息；不提前选择有利层/源。

## 数值与中性

使用现存完整 HiF4 cache：56,223,524,243字节，SHA `bfb400f94ce42760070bab5916721fd942b0864907bb4e4e5207c008dc4c3df3`；M0已在实际缓存文件上重新全量读取核验。原权重 high SHA `39d3af74e46e59e989db5cc36f401e666f3ae77c7648eb55d3aff9fee47716fa`、low SHA `a3b8680affe2ab2e2121210a83ffa7d351f77da18191e57bb359d7839beba8a2`。真实 `input.float()/smooth.float() -> BF16 -> native QDQ(force_fp32) -> BF16 F.linear`，GELU tanh、bias一次、current gate及postwrite不改。不得用 `Q2(h1-h0)` 代替 `Q2(h1/s2)-Q2(h0/s2)`。

每来源 plain/observer 四个最终 step latent 文件和完整视频必须 SHA 相同；原生 runtime 每视频1600 linear及1600 activationQDQ、cache miss0；observer必须160位置齐全，Q1/Q2/有效W2和原生postwrite断言通过。任何一项失败即当前job EXIT非零，不自动重跑、不丢失败。

## 代价、资源与失效处理

会话 GPU 并发≤4，每进程1张已核空闲 A100，每节点最多2个本任务完整模型，实际内存不足则串行。4 CPU线程/进程，GPU reserved≤76GiB，host available≥128GiB，峰RSS≤320GiB，每进程新HF输出≤128GiB，总≤512GiB，wall≤14400秒/outer14520秒，单次尝试。所有模型、tensor和视频放HF缓存。公共记录不含节点名/IP/UUID；私有profile和job保存实际映射。

选择/代理/真实skip计时、static校准及 E0/E1/E2 均不借用本次采集许可。下一步对实际新资产重新绑定和独立评审。当前设计的反例风险：全M V1仅保证可重放postQ1之后的FFN；若原生GEMM复现因shape/dtype/调用顺序导致差异，后续需以真实采样边界校验定位而非把它称neutral。固定坐标的稀疏raw边界也不能被当作全M rawx。

## 实施冻结要求

实现中的 owner/observer、unittest、protocol、私有job spec和reviewer最终recheck均待完成；独立review不能只批准本概念文件。最终关键源码和资产SHA、精确argv、supervisor退出及ACTIVE登记必须齐全后才能启动。主Agent逐项响应独立反例；reviewer需核对最终不可变版本。
