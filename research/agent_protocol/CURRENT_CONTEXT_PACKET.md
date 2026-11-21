# 逐位置锚点联合读出 · 接手说明

截至 2026-11-21，当前成片基线为 **`fast512`，每次近似调用使用固定 64 个真实观测位置**。A100 上复用末端缓冲已获单站提速，尚待整片验证。当前 64 点、固定 128 点和轮换布局的证据都保存在 [PR #61 的固定版本](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/当前方法.md)。按主题查找原表、审片文件和历史方法，见[研究导航](../RESEARCH_NAVIGATION.md)。

## 一层在四步中实际做什么

Wan2.2 图生视频依次使用高、高、低、低噪声专家，每个专家 40 层。固定专家和层，$p$ 表示潜视频中的时空位置，$F$ 表示门控前的原生 FFN。第 0、2 个去噪步在所有位置计算 $F(u_{0,p})$，保存 $u_{0,p}$、原生输出和 2,560 个选中隐通道的预激活。第 1、3 步在同一专家、同一层、同一位置估计 $T_p=F(u_{1,p})-F(u_{0,p})$。邻边 $p,q$ 上的 $T_q-T_p$ 描述两处之间的相对变化。高低专家分时复用 40 层的显存槽。[生成实现](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchored_scale/exact_port.py)

命中步用锚点输出、输入变化的折叠线性项和选中通道的实际激活变化形成基线 $B_p$。在 64 个位置执行完整原生 FFN，以观测减去各自基线为标签，同时拟合探针点和同帧邻边。联合读出写回 $B_p+Q_p\operatorname{diag}(g)V+\tfrac12K_{pP}C$。第一项保住位置自身的起点，第二项校正 16 个共享输出方向，第三项通过激活变化的相似度传播探针未解释的全输出残差。观测位置直接写回本步实测值，随后施加当前门控和外层残差。64 是探针数，80 是联合读出的列数，512 是执行分块的行数。[算子与数值路径](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/保留锚点方法总结.md)

**校准与可见信息。** 离线校准来源为 0032、0304、0456，固定参数按专家和层准备。24 个成片来源与这些校准 ID 分离，0800、1013 等经过反复开发观察。在线 64 个位置的当前原生响应计入推理成本。Dense 是配对度量和跟踪参照，首帧、原提示、噪声种子、去噪步序及轨迹必须能对应到该视频的生成记录。[校准与来源核对](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/provenance.json)

## 已知结论与边界

- **整片基线。** 24 个来源，seed 42，832×464、81 帧、16 fps。原版锚点和 `fast512` 的各步预测、潜变量、解码 RGB 在 24/24 源均逐字节一致。相对原生 Dense，单 A100 热墙钟平均 40.059→39.470 秒，21/24 源更快。锚槽 30.20 GiB，候选峰值 47.57 GiB。每源每臂正式计时一次。原批已看视频的运动幅度获人眼认可。[整片等价与计时](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/计时与等价性.md) · [VBench 与人工判读](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/vbench/VBench完整评测.md)
- **局部能力。** 64→96→128 在第 19 层的 11 个开发来源、两专家共 22 组上，留出位置的点边误差、门控邻边角、差模正交余能均逐档下降。固定 128 在 0800、1013 上平均每片多 0.256 秒，两片的运动和纹理变化有升有降。1013 的右手指区域在两种固定预算下都未被直接观测。保持 64 点并跨层轮换虽扩大覆盖，双手共同可追踪位置上的轨迹差仍可能变大。[预算证据与局部方向统计](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/当前方法.md)
- **数值一致性与成本。** A100 单站试验将高噪声第 19 层归档的 512 个真实位置铺至 31,668 行，并在原坐标还原对应行。复用末端 FP32 缓冲后，162,140,160 个 BF16 输出元素逐位一致，单次命中耗时中位数 29.521→28.818 ms，30/30 对更快。无损编码逐位置锚点可将槽预算折算至 23.51 GiB，RTX 4060 Laptop 上写槽加命中变慢；整片峰值仍待测。两项是独立候选。[A100 原始记录](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchor_next_perf/A100_RESULT.json) · [编码组件记录](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchor_next_memory/GPU.json)
- **已知反例。** 去掉逐位置锚点的版本曾在 0392、0800、1013 均明显模糊，局部指标有利也会在整片失效。在 0951 左摇扰动中，四个新种子 43–46 均经人眼确认出现此前的主体不稳定；已看其他视频未见此问题。按审片决定保留案例，暂缓专项排查。新一轮 0800、1013 的量化结果已记录，终局视觉判断仍由用户给出。[三片盲审](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_zero_fused_DEV3/人工判读.md) · [0951 跨种子复核](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_0951_seeds/跨种子人工复核.md)

## 从没有 GPU 的新检出开始

下列命令在主仓库根目录运行，假设 `origin` 指向 Qesire/FFNDS。它们取回证据对象并读取当时的文件，当前检出的分支和工作文件保持原样。

```sh
git fetch origin 2c4aa753761194994ee25ea8caba76a29445912d
git show "2c4aa753761194994ee25ea8caba76a29445912d:reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/计时汇总.json"
git show "2c4aa753761194994ee25ea8caba76a29445912d:reports/20261121/anchored_video_experiments/work/anchor_next_perf/A100_RESULT.json"
```

第一份 JSON 检查 `backend=fast512`、`verified_trajectories=24` 和 `overall.faster=21`。第二份检查 `site=high19`、`nfe=0`、`changed=0` 和 `elements=162140160`。两份记录各自支持整片和单站结论。

**原生出片的前提。** 历史 `generate.py` 从封装好的学校 Slurm 作业目录读取 `base.json`、`cohort.json`、`reviewed.json` 及参考运行器，校验配置、图像和提示的哈希。还需要匹配的外部 Wan 模型与蒸馏权重、原始输入图、80 站 CAL 参数包及兼容的 LightX2V/CUDA 环境。先核对实际可用资产和绑定，再决定是否重建同环境。仓库中的运行记录、部署源码、旧环境清单和文件指纹各有用途，单独取出该脚本不能复现成片。[原运行入口](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchored_scale/generate.py) · [运行回执](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/RESULT.json)

## 下一步如何判定

1. **工程。** 固定 64 个坐标，仅切换末端缓冲复用。在同卡同输入、同噪声与同调度下先比对四步预测、四步潜变量和解码 RGB；逐位相符后交错重复计时。分别报告热整片墙钟（含解码、写出）、四步 GPU 事件、峰值显存和首次准备成本。整片等价与速度均有证据后再考虑更新默认实现。[待整片验证的补丁](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchor_next_perf/reuse_base.patch)
2. **能力。** 保留原 64 个位置，按相同的额外原生 FFN 行数，对比全局新增点与由首帧指定的动作区域邻点对。两臂使用同图、同提示、同种子。布局只读生成时可见的信息，Dense 跟踪只作事后评分。报同一可见支撑上的方向投影、角差、正交和共同／差分误差、跟踪覆盖、运动补偿纹理变化及代价。像素或局部 MSE 只作辅助。成片用同步盲审核查动作、轮廓与纹理。[已有成片对照](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/budget128/逐片方向与区域实测.md)

早期 scope、历史批准和曾经的队列保留其原始归属。这次交接更新知识，没有提交新的生成作业。[旧版入口与归档目录](../../archive/20261121/handoff/README.md)
