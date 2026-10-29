# FFNDS · 2026-10-29 用户纠偏：以 Wan 原生、可运行、免大训练的稀疏基线为准

## 最新用户实际目标 / 生效规则

**只从已经整理的论文列表和相关原生作者代码中寻找真正能在 Wan 上复现的省算方法；直接用 FFNDS 的 Wan2.2 I2V A14B 四步、两独立专家及全部160 FFN位置，以固定 Dense 同来源/初始噪声/seed 为对照取得视频行为和真实执行成本，再由成功或失败的相反证据反推新机制。**

不再把 TEAL 原 Llama 生成、tokenizer/权重、LLM PPL 或 Chipmunk 原 FLUX 图片生成当成 Wan 基线的入口门槛。上述过去真实已闭合工作仅作历史技术资产，不再耗用本轮 P0 资源。原作者环境验证只有在它**消除当前 Wan 执行的具体技术阻碍**时才申请有限scope、独立审查、真实运行。相反，保留已真正跑出的原 Wan 视频、Chipmunk A100 单组和 TEAL Wan 全160成绩；不重写失败或成功收据。

## 最新进度快照（GitHub main 56b708c；所有数值仅关联各自证据）

- **TEAL post-GELU / 原BF16 Wan**：四来源、全160位置的 V3 20条视频（8中性/12研究）真实EXIT0；四来源 RAFT、八轴、24页全81帧视觉均真实EXIT0。25%门限来源等权RAFT比1.170、50%比1.376，后者source 0.423–2.231；979 threshold50出现多彩碎片，较高全帧流可能来自新生物体而非主体动作。完整Dense W2仍执行，**不是物理FFN加速**。详见 [实际视频](TEAL_WAN_ALL160_VIDEO_V3_NATIVE_RESULTS_20261029.md)、[实际质量](TEAL_WAN_ALL160_FOUR_SOURCE_QUALITY_RESULTS_20261029.md)。
- **Chipmunk Wan单组付费proxy移植**：source898,high.block20,step0→1, 128非raster行，付费均值代理实际K=5120/13824，门后相对同组Dense L2=0.279554；等K三个随机0.309026–0.320663，固定0.321217，算子同mask参考约0.00158，真实EXIT0。其作者启发式安全选择不是作者Hopper kernel，也非全160或自由视频。详见 [实测](CHIPMUNK_PAID_PROXY_EQUAL_K_RESULTS_20261029.md)。
- **无已证实达标的方法**：本仓库此时没有在匹配Wan2.2-I2V-4step、全160真实自由视频、来源级主体/事件质量和物理Kernel/系统时间同时通过的FFN-only无大训练正基线。

## 对已有论文索引的源级筛选（PRIMARY_LITERATURE_INDEX / G14 PAPER_METHOD_FOCUS_GRAPH）

| 本仓索引 | 论文及其真实Wan关系 | 训练/算法成本 | 可执行代码与主要差异 | 基线位置 |
| --- | --- | --- | --- | --- |
| **L35 FIS-DiT** | 论文明确在**Wan2.2-14B四步T2V**做frame-interleaved sparsity，训练自由，Few-step同构最接近 | 无大型训练；frame subset跨层interleave | 官方Tencent/KsanaDiT支持Wan2.2-I2V，但仓库README明确**FIS核心代码尚在整理/待开源**；论文结果T2V，不是我方I2V；稀疏单位为跨帧Block，不是FFN列稀疏 | **P0源方法/系统运动正向参照候选**，先确认公开代码身份和可实现机制，不冒充已release |
| **L74 SharQ** | 论文明确在**Wan2.2-T2V-A14B**报告与SageAttention结合最高1.58×加速 | **免训练/免校准**；FP4 N:M sparse backbone + dense FP4残差 | actypedef/SharQ开源模型支持列表为Llama/Qwen/Mixtral，缺已核实的Wan原生集成；需在RTX PRO6000 SM120复现两物理路径、动态scale、FP4权重布局、masked FFN真实W1/W2成本，不能把SageAttention组合收益称FFN-only | **P0 Wan FFN稀疏+量化候选**，作者source/kernel移植+I2V双专家测试；单独比较原生dense NVFP4 |
| **L09 Chipmunk** | 作者在**Wan2.1**例使用动态Attention/step-cache，MLP默认关闭，原版Wan FFN未挂SparseDiffMlp | 无训练、动态selected-column delta+付费W1 proxy | 本仓A100单组已验证作者启发式选择价值与计算正确性，但重排分组、Hopper实现、4步high/low每专家两次和全160皆未成立 | **P1 FFN局部机制备选**，如多来源proxy保留运动/成本才升级 |
| **L17 ProfilingDiT** | 论文在Wan2.1视频报告foreground/background block-aware caching | 免训练；step/cache | 未核实Wan2.2 I2V四步实装；完整block/cache系统非FFN-only；较多步有效性不可直接移植 | P1跨步/场景机制对照 |
| **L14 TeaCache** | 视频DiT 训练自由缓存，有Wan相关代码家族 | 无训练，step-level | 四步双专家时间窗口过窄，缓存命中与跨专家失效需真实核 | P1轻量系统对照，不作为FFN正基线 |
| **L20 BWCache** | 论文视频block caching | 免训练 | 作者hsc113/BWCache README明确Wan2.1支持仍 TODO，当前公开例OpenSora；**不能标作者原生Wan复现** | P2研究参考 |
| **L19 DiffSparse** | Wan2.1-1.3B 25flow-step T2V曾实测 | 作者需要离线学习/STE/蒸馏和多GPU训练 | 不是所需“无大训练”基线，且非Wan2.2四步 | P2方法/论文结果对照，不投入训练复现 |
| **L34/L36** | Wan2.1 few-step/视频动态路由 | 需要distill/router训练 | 与当前不大规模训练的门槛冲突 | 不入本次执行队列 |

**直接可运行的Wan2.2 I2V系统补充对照（非原先L编号，需作为新paper关联资产记录）：MagCache**，官方NeurIPS 2025仓库 [Zehong-Ma/MagCache](https://github.com/Zehong-Ma/MagCache/tree/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4Wan2.2)，直接提供 `i2v-A14B` 调用及 `magcache_generate.py`，作者README附Wan2.2 I2V示例。**它跳过整个Transformer Blocks，不是FFN-only**，可提供真正原生质量/运动/系统计算参照，严禁与FFN-only耗时公平性混称。原仓40步硬编码mag_ratios经nearest_interpolation到四步、`model.__class__.forward`全类monkeypatch、`cnt%2`分CFG而非专家、`split_step`和双expert切换等必须先独立源码复审；4step skip count和cache可用性以实际观测为准。不能把原官方40步1.5–2×当四步保证。

### 用户真正要求的一个合格基线

1. **相同原模型**：Wan2.2 I2V A14B同原BF16或另列原生NVFP4物理路径；四步high/high/low/low；相同配图、prompt、seed、初始noise、backend、解码与两个expert身份。作者在Wan2.2 T2V上的报告只能作为转移先例，I2V仍须另测。
2. **真实省算层**：分别报FFN-only/kernel与端到端系统全成本。FIS/MagCache是block/frame/system，SharQ涉及W1/W2及FP4，TEAL postGELU QDQ保留Dense W2，Chipmunk delta涉及proxy/cache和两核。各自实际算子边界不能混合比倍率。
3. **质量/机制**：至少开发源级多独立来源、正式未暴露确认；Dense配对；81帧完整播放/对象轨迹/动态变化/语义/主体/背景/平滑/成像/RAFT/VBench；必须标异常生成、运动增加并非一定同动作。事件局部和全帧分别观察。
4. **每个具体实验先审**：作者自审与独立异Agent反例审查，修订recheck、独立冻结ACTIVE scope与资源、执行前校验本体/数据/代码SHA，实际RUN/EXIT确认结果。source级独立统计，不用160 sites当160个source。
5. **分支优先**：A. **可直接上机的Wan2.2 I2V方法**（MagCache原生代码，系统级reference）；B. **论文真正在少步Wan2.2上成立的frame sparse**（L35 FIS，核代码可得性是入口障碍）；C. **Wan2.2原论文有视频数据、公开原生稀疏FP4内核的FFN扩展**（L74 SharQ）；D. 已有TEAL-Wan和Chipmunk单组只作为可解释的FFN机制对照、按证据决定再执行。上述是**优先级/复现资格筛选**，不是已经上GPU申请运行授权；三个路线需要分scope真实独立复审。

## 原始定位和当前源码身份

- G14: [PAPER_METHOD_FOCUS_GRAPH.json](../agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json) 对 L09/L14/L17/L19/L20/L34/L35/L36/L74 的图定位；[PRIMARY_LITERATURE_INDEX.json](../literature/20261026/PRIMARY_LITERATURE_INDEX.json) 的`model/training`字段分辨真Wan/大训练。
- FIS论文：https://arxiv.org/abs/2605.11869；KsanaDiT https://github.com/Tencent/KsanaDiT/tree/2d7f227d14be2d868a6ac845ecc558f70de368de （论文代码release仍列in progress，框架I2V并非FIS算法证明）。
- SharQ论文：https://arxiv.org/abs/2606.26587；代码 https://github.com/actypedef/SharQ/tree/3cc584fe48e32e59c8b4e37323470dcdd606bb11 （作者README仅列LLM模型；Wan论文组合加速不代表FFN-only）。
- MagCache论文/官方代码：https://github.com/Zehong-Ma/MagCache/tree/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4Wan2.2 （有i2v-A14B示例，默认40步ratio与四步转移尚待严审）。
- Chipmunk: https://github.com/sandyresearch/chipmunk/tree/177fdefc4803e621fd8a02aad491812e9d1ecc94/examples/wan （MLP OFF、Attention/step cache ON）。
- BWCache: https://github.com/hsc113/BWCache （官方README Wan support TODO）。

**本文件不发起实验，也不改变已完成RUN、EXIT和历史文档：没有本轮新GPU运行与Agent实际独立复审。下一真实Agent应先按本文提交匹配的Wan执行scope，再获review许可。**
