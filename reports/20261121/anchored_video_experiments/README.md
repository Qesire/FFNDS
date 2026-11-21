# FFNDS 保留锚点读出、24 源配对实验与 0951 镜头干预

本批研究从三条开发视频上的 FFN 联合读出开始，比较保留逐位置锚点与去掉跨步状态的构造，随后在 24 个来源上验证保留锚点版的优化实现。最后用同一张泰迪熊图检验镜头改变时的运动与纹理失稳。所有成片使用 Wan2.2 四步 HHLL、seed 42、832×464、81 帧。不同来源、同图镜头变体、同一视频的逐帧统计分别计数。

| 实验 | 得到的结果 | 记录 |
|---|---|---|
| 三源原生生成与离线折叠 | 0392、0800、1013 均有三源实测。取消逐位置锚点后的融合版在三条人审片中损失清晰度，0800 和 1013 出现接近平均色的格子。0392 的 Dense 重影也与观看者的场景判断冲突。 | [三源联合读出](outputs/FFNDS_joint_DEV3/MEASUREMENTS.md) · [盲审揭盲与纹理](outputs/FFNDS_zero_fused_DEV3/人工判读.md) · [缓存与离线编译](outputs/FFNDS_cache_principles/缓存与离线编译原则.md) |
| 保留锚点版的等价优化 | 24/24 条逐步预测、潜变量和解码 RGB 与旧锚点实现逐位相同。单 A100 上平均墙钟为 Dense 40.059 秒，锚点 39.470 秒，合计耗时之比 1.0149。每个来源各一次，计入解码和写出。 | [计时与等价性](outputs/FFNDS_anchor_scale24/计时与等价性.md) · [逐源计时](outputs/FFNDS_anchor_scale24/逐源速度与等价性.csv) · [方向测量](outputs/FFNDS_anchor_scale24/实验结果.md) |
| VBench-I2V | 24 个原提示和同一张 0951 的七种补充镜头指令。按已列出的官方维度和权重计算，本批综合分锚点 87.898、Dense 88.187。镜头指令两者均命中 5/7。用户审片认为运动幅度基本无法区分。 | [九维与补充诊断、适用范围](outputs/FFNDS_anchor_scale24/vbench/VBench完整评测.md) · [逐来源及视频指纹](outputs/FFNDS_anchor_scale24/vbench/逐源配对.csv) |
| 0951 同图分步对照 | 左摇时，Dense 和旧锚点背景邻帧位移分别为 2.476、2.473 像素；运动补偿后熊身纹理相关为 0.548、0.141。只在低噪声步近似时，用户确认动作正确、无可见抖动，主要损失在纹理。只在高噪声步近似时，熊身暖色面积由 Dense 的 14.85% 降至 7.97%。 | [原提示与左摇四路析因](outputs/FFNDS_0951_mechanism/0951失稳机制与最优性复核.md) · [八条件原数](outputs/FFNDS_0951_mechanism/motion/镜头条件汇总.json) |
| 折叠与替换层数 | 高步全折叠在左摇动作上与旧版同样差，保留一半高步原生层减轻了抖动，动作仍然不同。末步全折叠与 low_only 的目视表现基本一致。高步全原生且仅低步 20 层近似，使锚槽由 30.20 降至 15.10 GiB，整片峰值由 47.57 降至 26.52 GiB。单次四步 GPU 计时较 Dense 缩短 0.417 秒，尚无该版稳定整片提速的配对重复证据。 | [分臂结果与审片](outputs/FFNDS_0951_improvement/0951改进实测.md) · [独立进程计时](outputs/FFNDS_0951_improvement/驻留与复算.md) |

## 0951 的提示与校准归属

原提示为 `A teddy bear is climbing over a wooden fence`。七条镜头变体由同一文本追加逗号和镜头短语，和主体描述一起送入正向文本条件。例如左摇追加 `, camera pans left`。输入图、初始噪声及采样种子相同。原提示产生轻微背景运动，明确固定镜头的版本背景位移更小。在同一张图上，原提示与固定镜头的锚点纹理相关降幅为 0.078、0.094，六种运动镜头的降幅为 0.243–0.416。[生成条件](outputs/FFNDS_anchor_scale24/camera7/COHORT.json) · [分条件原数](outputs/FFNDS_0951_mechanism/motion/镜头条件汇总.json)

80 个正式校准站点复核后均只用了 0032、0304、0456，24 个出片来源与三者的记录 ID 没有交集。0951 的 64 行 FFN 响应是在自身生成轨迹上按预先固定的位置取得。另一次同图跨提示诊断有意使用 0951 原提示下的真实剩余拟合，再到同图左摇回放，约 4% 的局部误差改善属于这次有教师真值的诊断，其权重未用于成片。0951 最初来自 21 条扩展样本，后续选层方案参考过该图的人审反馈，应以新来源检验泛化。[校准核验](provenance.json) · [选样规则](work/anchored_scale/cohort_provenance.json) · [同图回放](outputs/FFNDS_0951_mechanism/0951真实站点最优性.md)

## 复查材料

各批次生成和审片视频的分组下载见[本次视频快照](https://github.com/Qesire/FFNDS/releases/tag/ffnds-results-20261121)。

归档保留原报告、逐来源数表、指标定义、成功与失败对照、代表性视频原片和分析实现。文本里的集群绝对路径已改为相对标记。历史报告中仅在工作站保有的材料以“本地归档”标注。全部生成和审片 MP4 的文件名、尺寸和原始 SHA-256 见 [视频索引](video_inventory.csv)。发布文件的 SHA-256 见 [校验清单](SHA256SUMS)。同一输入图派生的镜头条件与同一视频的各帧均不构成独立图像来源。

- [0951 左摇 Dense](outputs/FFNDS_0951_improvement/compact_dense/vbi_0951_pan_left.mp4) · [旧双步锚点](outputs/FFNDS_0951_improvement/old/vbi_0951_pan_left.mp4) · [仅低步近似](outputs/FFNDS_0951_improvement/low_only/vbi_0951_pan_left.mp4) · [低步 20 层近似](outputs/FFNDS_0951_improvement/compact_low20/vbi_0951_pan_left.mp4)
- [高步折叠](outputs/FFNDS_0951_improvement/fold_high/vbi_0951_pan_left.mp4) · [高步半数原生](outputs/FFNDS_0951_improvement/h20/vbi_0951_pan_left.mp4) · [仅末步折叠](outputs/FFNDS_0951_improvement/fold_low_native_high/vbi_0951_pan_left.mp4)
- [原提示低步 20 层近似](outputs/FFNDS_0951_improvement/compact_low20/vbi_0951.mp4) · [校准折叠的数学与编译](outputs/FFNDS_0951_improvement/折叠编译核对.md) · [参数预算与跨步跨层的前期研究](../../20261120/observer_budget_cross_step_depth/README.md)
