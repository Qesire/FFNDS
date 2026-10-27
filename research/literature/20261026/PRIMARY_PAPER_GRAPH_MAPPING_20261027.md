# G7 · 原论文方法与冻结Wan计算图的逐边同步（2026-10-27复审合并）

本页是G7 K01–K12的**现行索引和计算时点映射**。详细原始公式见 [DIFS/FIS-DiT v1一手审计](PRIMARY_METHOD_EQUATION_AUDIT_20261027.md)，其余计算图及强比较见[FFN现行综合](../../20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[主图G0–G12](../../../docs/98_端到端计算图与证据注解_20261025.md)。论文的训练/实验和本项目 Wan2.2 I2V 四步HiF4 QDQ 是两种独立证据身份（LIT vs SRC/OBS/NATIVE）。

## L34 Dynamic-in-Few-Step · K01–K03

**一手方法已经升级**：arXiv 2607.06631v1 §3.2–5与附录A原PDF已核对，而非仅alphaXiv二手细节。论文以**离线训练**得到step-specific binary block/head/FFN gates，使用STE、按W1权重L2重排FFN隐藏列、每**8个**通道为group，逐step逆向蒸馏D-DMD/GAN＋output rollout，最终union模型和每step index table供compact真实参数计算。原文实验包括Wan2.1-14B **T2V四步**，论文24%单step FLOPs和1.2×总wall的口径含非FFN算子及部署，不能直接称原Wan2.2 I2V的稀疏效果。

**边**：G02（原Wan high/high/low/low只是两套独立expert选择） ⇢ K01（论文训练gate/结构） → K02（新compact W1/W2及模型step参数） → K03（终端rollout训练）。K02与原F04/F09是**模型替代的LIT比较边**，需要重新处理HiF4 Q2 64→8→4尺度闭包和video function，非原native变更。

## L35 FIS-DiT · K04–K06

**一手方法已经升级**：arXiv 2605.11869v1 §3.2–4.1原PDF直接确认block/frame 的anchor gather→仅对anchor frames计算block→**两相邻anchor线性插值恢复其它frame**，边界帧保留，anchor相位在连续sparse blocks中轮换；对head/tail或末尾denoise steps保持完整计算。论文的Wan2.2-14B为 **T2V 4-step**，HunyuanVideo1.5为**I2V 8-step**，非原Wan2.2 I2V。

**边**：G04/G17/G21（本模型block/attention/下游状态） ⇢ K04（frame×block×step anchor选择） → K05（减少输入M行及其对应attention/FFN执行） → K06（跨帧插值、更新完整latent-frame状态） ⇢ G21（下一block）。插值不等价于跳过的原block真实输出；评估对应frame的context、RoPE、gate与后续flow。

## L37 Mobile Video DiT · K07–K09

[原始工作](https://arxiv.org/abs/2508.09448) 的训练后head/FFN width结构和紧凑GEMM计划已经在G7源图中登记：K07选择训练后保留通道→K08将W1中间维度与W2输入缩窄、部署真实compact dense weight→K09对feature一致性/KD及内存tile作优化。**区别**是静态重训后的物理维度缩窄，而非冻结Wan后验TopK；原HiF4 Q2 group的排列、码字尺度以及flow视频保真需要独立重建。这一项由[先前源码/论文审计](ADJACENT_SOURCE_METHOD_REVIEW_20261026.md)决定版本，未产生当前Wan新原生结果。

## L50 NerVE · K10–K12

[论文](https://arxiv.org/abs/2603.06922)与[作者 metrics.py](https://github.com/nerve-eigenspectrum/NerVE/blob/main/nerve/metrics.py)（原文件blob记录于[前序源审计](ADJACENT_SOURCE_METHOD_REVIEW_20261026.md)）：K10 pre-GELU hidden sample covariance→K11 post-GELU eigenvalue histogram→K12谱熵SE、参与率PR、EEE和JS。代码使用torch.linalg.eigvalsh，只读**特征值**，并未输出隐藏函数方向、可在线分派的通道掩码或真实GEMM跳算。O(H²) covariance/计算谱的成本和作者的离线描述证据应保持LIT而非本Wan优化结论。

## 统一的跨论文验证轴

| 分支 | 改变真正的执行单位 | 数据何时可用 | 迁移至冻结Wan的关键证明 |
|---|---|---|---|
| DIFS/K01–03 | step-specific训练后compact block/head/FFN | 离线训练的gate/权重，在当前step读取 | 当前双expert和Q2尺度能否与新训练架构一致；视频质量与模型成本 |
| FIS/K04–06 | latent frame×layer计算行M与插值 | 本次step/layer的真实frame index和条件 | 被插值的frame和后续attention/flow功能保真 |
| Mobile Video DiT/K07–09 | 训练后FFN H宽度+head | 离线模型裁剪和蒸馏 | 原冻结函数能否训练出新窄模型，native GEMM能省多少 |
| NerVE/K10–12 | 激活谱离线统计，不改变执行 | 已有完整pre/post-GELU激活 | 特征谱是否提示功能表示，不得视为在线免费selector |

论文细节和原源版本在各自审计中更新；本页不维护另一个“已读论文总数”。[机器图](../../agent_protocol/GRAPH_AGENT_BINDINGS.json)是稳定节点ID来源，[Crosswalk](../../agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)维护G7与G11/G12/G10的机制比较边。原实验仍暂停。
