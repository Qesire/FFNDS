# E0 · 四个项目新来源与两组交叉 seed 的原生结果

**相同图片和提示词下，seed可以改变候选作用与既有状态响应的有符号关系。** 镜头平移来源425在seed42的后三步cross全负，在seed54全正；两次的局部误差计数却完全相同。镜头缩放来源437又复现了“候选本步部分抵消状态传播，但完整轨迹仍偏离”的现象。这把条件/seed异质性与一个固定误差排序分开，为寻找真正可共享的FFN函数和必要计算提供了反例。

本轮2026-10-27完成。四个原始官方输入在[执行前审计](../method_discussion/discoveries/E0/independent-phase2-source-history-and-official-inputs-20261026_v11.md)中与38份冻结项目输入清单排除身份重叠。其含义是**项目此前固定cohort未用过**，不涉及Wan预训练是否见过这些照片。原Affine54 bank、参数、全部原生数学路径保持不变；没有按新结果拟合或挑选新候选。

## EVIDENCE · 四个身份，六条嵌套轨迹

| 原metadata行 / seed | 场景指令 | 局部FFN更小 / 等 / 更大 | 后三步dense有限能量增益 | 后三步cross符号 | 终端latent SSE |
|---|---|---:|---|---|---:|
| 908 / 42 | 小猴主体 | 154 / 1 / 5 | 4.948, 4.376, 7.050 | +, +, + | 198,484.10 |
| 908 / 54 | 同图片/提示词 | 155 / 1 / 4 | 5.296, 4.494, 7.265 | +, +, + | 202,343.32 |
| 376 / 42 | 城市/倒影背景 | 159 / 1 / 0 | 5.212, 3.998, 6.909 | +, +, + | 233,060.24 |
| 425 / 42 | 镜头向左平移 | 158 / 1 / 1 | 6.596, 4.678, 7.313 | −, −, − | 322,025.73 |
| 425 / 54 | 同图片/平移指令 | 158 / 1 / 1 | 5.458, 3.856, 6.698 | +, +, + | 185,184.71 |
| 437 / 42 | 镜头推近 | 158 / 1 / 1 | 6.519, 4.937, 7.395 | −, −, − | 552,300.27 |

942/960处Student-prefix raw绝对operator误差小于Teacher-prefix，6处相等、12处更大；实际native writeback则为943小、6等、11大。两种边界的计数不同：来源908两seed各少一个增大位置，来源376多一个增大位置，说明gate/残差/BF16写回后的比较需要单列。18个非零位移上的完整dense有限能量增益均>1。结合[六个已暴露来源](E0_SIX_SOURCE_DIRECTIONAL_RESULTS.md)，当前共10个source身份、12条source×seed轨迹、1,920个嵌套site；不能将site、帧或配对视频当成独立source样本。

令 `P=T_D(zS)−T_D(zD)`、`N=T_S(zS)−T_D(zS)`，均含实际native BF16 step_post。cross为`2<P,N>`，候选本步相对dense的净能量变化是`||N||²+cross`。来源437的step2/3净变化分别为 **−4,083.3982 / −5,682.6629**：在这些具体Student状态上，候选比同状态dense单步更接近该步teacher。其实际next SSE及整条轨迹仍显著偏离；这个条件反事实没有证明候选完整视频质量更好。负cross与负净变化亦不是同一判断。

来源425的seed对照在图片、提示词、camera指令、候选bank和实现均相同时发生cross变号，因此该符号不能仅用subject/background/camera内容标签解释。它不揭示某个具体FFN单元控制相机运动，仍允许随机初始状态、上游attention及后续非线性运输等解释。

## 视频功能保留各轴

| 来源 / seed | RGB RMSE均值 | 时序RGB delta RMSE | 光流幅值比¹ | 边缘能量比¹ | 相邻RGB变化比¹ |
|---|---:|---:|---:|---:|---:|
| 908 / 42 | .08909 | .03758 | .47491 | .93516 | .68006 |
| 908 / 54 | .08868 | .04435 | .33809 | .94193 | .57044 |
| 376 / 42 | .09419 | .05062 | .62390 | 1.76792 | 1.36115 |
| 425 / 42 | .14790 | .04915 | .51875 | .98878 | .83408 |
| 425 / 54 | .11825 | .03388 | 1.29540 | .92559 | 1.71234 |
| 437 / 42 | .18969 | .07093 | .34458 | 1.27597 | .72054 |

¹ candidate/dense各自均值的比值。JSON同时保留逐帧/transition比值的均值、亮度、配对光流差、motion-compensated luma及16-transition窗口。Farneback为max-side320的uint8灰度诊断；纹理和亮度变化会影响估计。camera prompt身份不等于相机指令已经被视频正确执行，后者尚无CoTracker/官方camera结果。

来源908两seed的RGB RMSE非常接近，光流幅值比却从约.475变为.338；来源425则跨越1。这些轴的分歧提示：固定候选的局部相似、状态距离、运动和细节分别约束解释。[机制优先指引](E0_FRONTIER_MECHANISM_FIRST_20261026.md)仍要求先发现计算/函数单元及其条件性，随后用这些观测检验作用；本轮没有选出新算子、频带校正策略或部署算法。

## EXECUTION · 固定范围完成，按用户要求暂停

全部六对一次完成，零失败/重试。单张空闲A100串行六个新进程；12条完整四步latent轨迹/视频、960paired site、24额外完整dense flow、12VAE decode。12份逐视频报告覆盖 **972帧、960transition**，每条均通过core bundle及严格独立native inventory/model身份门禁。新来源的首条dense控制使用同一冻结controller、初始步中性控制和此前四来源byte-identity集成证据；没有新来源同source未插桩历史视频逐字节等同的结论。

实际计费：32,640次native QDQ，34,560次native GEMM（含affine），5,760次FP32数值诊断GEMM；采样物理峰值76,847MiB；HF输出 **1,110,811,963bytes**，小于1.5GiB上限。观察时间含两轨、额外whole-flow、局部shadow及解码，不能当候选单次部署时间。零新拟合、训练、权重分解或官方质量模型调用。

**官方VBench-I2V/VBench2及人工完整视频审阅NOT_RUN，quality_accepted=false。** 原评分环境v1–v3准备失败及原因保留为独立工程事实，未把环境检查当作质量。v4与anchor-neighbor新范围均未登记/执行，其准备草稿仅留本地。

用户本轮最后指令为“跑完这个上传后暂停”。本scope已结束；执行端确认串行父/模型进程均退出，所占GPU回到15MiB空闲、无本项目计算PID。新实验入口关闭，后续研究执行等待用户重新指示。

- [完整汇总JSON](E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.json)：绑定每对实际RUN/TRACE/bands/媒体指标SHA，保留全部step、layer及source/seed功能向量。
- [冻结协议](../../config/e0_fresh_official_directional_native_20261026_v1.json)，SHA `a0d666545296c1a2de4260fd6b3ceed9bd4239b12b689ac1d5ed94a9c0f2d813`；[执行前独立审阅](../method_discussion/discoveries/E0/independent-fresh-official-frozen-native-review-20261027_v13.md)。
- [收尾汇总器](../../tools/summarize_e0_fresh_directional.py)仅读取JSON及文件byte-SHA，0新模型/张量数值/解码/指标执行。

图锚点F04/F06/F08/F09/F12/F13/P23/P24/G21：有限原生响应为NATIVE-OBS，正交频带为posthoc；源/seed条件性→功能单元、可获得信息→真实省算、局部表示→完整视频保真均保持OPEN。当前结果没有改变E0研究阶段或正式质量状态。

[最终独立结果审阅](../method_discussion/discoveries/E0/independent-fresh-four-source-six-pair-results-audit-20261027_v15.md)复核六份原生RUN/strict gate、300组标量统计及24个opaque状态文件byte-SHA，无新增张量数值或模型执行；审阅SHA `edf0c5664a63478fdd09f5aca85c34b9d769d9b5d2c7a902833a27586a3283df`。
