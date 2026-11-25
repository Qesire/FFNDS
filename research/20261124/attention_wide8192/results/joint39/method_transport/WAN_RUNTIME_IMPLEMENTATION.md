# 标准化缺陷运输的 Wan 运行绑定

方法依据 `0496ef979a4add0c50a50351a37ced61fd0faf79`。运行入口复用冻结的 BF16 wide8192 FFN 和原生 P64。第 0/2 步保留原生 self-attention，同时记录同层 `C_a=(A_a−G_a)D_a⁺`、分组标签和选中支持。第 1/3 步用当前 K/V 更新组统计，求标准化创新，再用当前通道尺度解码并覆盖 P64。原生锚点、命中和缓存生命周期按四步 HHLL 调度。

## 实现

- [Torch 算子](attention_defect_transport.py)复用已核验的 FlexAttention 选中块输出和 LSE。遗漏组用计数加权。按 2,048 query 行和全部 496 个组槽做规则批量 GEMM，再排除已精算的块和空组。用共享指数基准合并两个正质量；中间只保存一个 query 分块的粗分数。
- 组统计、通道方差、合并和 32 列 ridge 用 FP32。完整锚点缺陷字段用 BF16，标签用 uint8。正尺度地板 `1e−12`，当前常量通道旁路历史和校正，全部通道恒定时跳过 P 教师。求解的 α 是 `64×0.1`。
- [Wan 入口](run_attention_defect_transport.py)可在当前实验 runner 的独立进程中使用。`bind(experiment, method)` 作用于真实 norm/RoPE 后的 self-attention；FFN 走冻结 wide8192。每条轨迹核验 160 个站点、80 个锚点、80 个命中、5,120 个精确 query 行，最多 40 个活跃槽。
- 固定轨迹比较用相同当前底座、特征、全查询 P 教师和固定正则，在 H448 比较原始历史项与 FP32 标准化运输。实际 BF16 缓存另列。第 39 层使用实际输出投影、gate、cross-attention、AdaLN 和 wide8192 FFN，检查共用 P 后的 FFN 输入逐位一致。

## 已验证的数值范围

[CUDA 交叉核验](runtime_oracle_verified.json)用归档 P_k/P_v 和明确合成的 H2×N190×D128 操作数，与本目录 NumPy 参考对照。Torch FP32 最大绝对误差为 `3.061e−7`，BF16 为 `1.850e−3`。两种精度均检查了 P 行逐位覆盖、空组、62 个有效键的尾块，以及全常量 V 跳过原生 P 教师。验收脚本为 [verify_attention_defect_transport.py](verify_attention_defect_transport.py)。

## 学校作业

学校 A100 作业 `198062` 已完成。0392 固定轨迹逐位复现归档，八个注意力站点和两处第 39 层真实 wide8192 尾部通过核验。0392 与 1013 的 81 帧完整视频已生成并校验。[实测报告](WAN_RUNTIME_RESULT_198062.md) · [原始收据](wan_runtime_receipt_198062.json) · [数值复算](wan_runtime_summary_198062.json)

执行目录沿用冻结 wide8192 release 的 `generate.py` 和配置、旧 `attention_online_structured.py`、第 39 层辅助代码 `attention_joint39.py` 与原 CAL 资产。将 Torch 算子和 Wan 入口放入这个目录，使用原拟合目录在 `PYTHONPATH` 上执行 `python -B run_attention_defect_transport.py FIT_DIR ONLINE_INPUTS.json`。后者绑定方案提交、历史 `ONLINE_RESULT.json` 路径及其 SHA-256。`attention_P_k.npy` 使用原已封存的投影字节。
