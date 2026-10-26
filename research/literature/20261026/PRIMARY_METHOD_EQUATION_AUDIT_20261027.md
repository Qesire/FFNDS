# FFNDS · 原论文 v1 方法公式与 Wan 计算图核查（2026-10-27）

## 版本与证据等级

已直接核验两份原始 PDF：[Dynamic-in-Few-Step v1](https://arxiv.org/pdf/2607.06631v1) 19页 §3.2–5、附录A，以及 [FIS-DiT v1](https://arxiv.org/pdf/2605.11869v1) 13页 §3.2–4.1。FIS-DiT 另有2026年9月的v2修订，本轮没有核对v2差异。两者论文实验不等于FFNDS的Wan2.2 I2V、high/high/low/low、HiF4 QDQ真实运行。

## Dynamic-in-Few-Step：每步训练得到的实际模型

**§3.2 Eq.(2)**：对每个step t_i单独配置block、attention-head、FFN-hidden的二进制gates，输出抽象为 x_out=g_t⊙F(x_in)，gate形状分别是 [T,Nblock]、[T,Nhead]、[T,Nchannel]。

**§4.1 Eq.(3)**：门控使用STE surrogate g_tilde=g_t+m_t−stopgrad(m_t)，前向硬二值、反向学习mask。block直接按m>0；head按Dhead维均值判断；FFN先分chunk共享mask。**附录A明确C=8**，并先按照pretrained原W1对应权重的L2大小重排hidden通道，然后每8相邻坐标编组。这与当前Wan HiF4的64→8→4 Q2 scale树不是同一分组规则，不能免费交换排列和量化。

**Eq.(4–5)**：各step局部可学习gate与其它step已经冻结的gate共同进入平均保留率稀疏约束；优化D-DMD及GAN增强的训练目标与L_sparse。不是按后验local SSE自动删除原Wan通道。

**§4.2 Eq.(6)**：先t4，再t3→t2→t1，逐步把较早去噪步加入训练，对新步施加稀疏项；后续共同训练。Output Rollout由中间step继续用student直到最终输出，后续step stopgradient，终端rollout用来训练fake-score网络。

**§4.3**：部署使用训练后所有step曾被激活参数的并集theta_union，每step的index table I_t选取紧凑参数再执行；不是每次重新从所有神经元计算后验mask。

**§5.1**：论文实际主模型Wan2.1-14B T2V四步，小模型Wan2.1-1.3B用于大量消融。训练在Koala-36M采样的5万视频上进行，16 Kunlun P800 XPU，约2000训练step。原作者报告四步基础上24% per-step FLOPs降低与约1.2x额外wall gain，不能归功于单独FFN或迁移至当前Wan2.2 I2V。

**G7映射 K01–K03**：G02 step→K01离线训练门控→K02 step子模型和真实compact W1/W2→K03终端rollout训练。外接F04/F09仅表示改变H和GEMM形状的研究分支，非当前原生执行。

## FIS-DiT：block-frame重算与插值恢复

**§3.2**：从各latent frame同位置token的L2特征幅值图出发，计算相邻帧变化及跨prompt/block稳定度；middle blocks更稳定，head/tail blocks变化更明显。此统计不是感知视频帧误差。

**§3.3.1 Eq.(7–10)**：给block l和latent frame的anchor集合A_l，先X_anc=Gather(X_l,A_l)，仅对anchor调用本block module得到Y_anc=Phi_l(X_anc)，然后X_(l+1)=R(Y_anc,A_l,F)。默认R为两相邻anchor间的线性插值；首尾帧始终作为anchor且直接复制。这不是跳过帧残差完全保持旧值，也不是与完整模块等价。

**§3.3.2 Eq.(11–12)**：anchor比例kappa，对应stride n=ceil(1/kappa)，层相位r_l=(l−l_0) mod n；A_l由模n选出的帧加{0,F−1}构成。跨n个连续稀疏块，非边界帧各获得一次anchor计算。

**§3.3.3 Eq.(13–15)**：只对稳定middle-block集合M，以及非最后T_tail个step启用稀疏。G(l,t)=1[l∈M]·1[t≤T−T_tail]。G=1时执行Gather→Phi→R，其余走完整Phi(X_l)。

**§3.2 Figure3**：原文v1报告Wan2.2被插值帧的特征幅值图相对误差平均约9.8%，HunyuanVideo1.5约2%；不是RGB RMSE，不等于无误差。**§4.1明确Wan2.2-14B是4step T2V**；HunyuanVideo1.5为8step I2V。它不能冒充Wan2.2 I2V实测。

**G7映射 K04–K06**：G04–G06/G17→K04跨block和step确定A_l→K05仅anchor module计算→K06线性插值为完整frame状态→G21下一层。形状变化是M行和attention时空上下文，不是H通道；原本所有frame还承担QKV/position与residual依赖，替换合法性必须在真实源码中核查。

## 重要学术区别

Dynamic-in-Few-Step通过大规模训练得到**step-specific结构**，FIS-DiT则在固定模型上通过**训练自由的layer/frame重排与特征插值**省算。前者的FFN宽度C=8与本项目量化组64的相互作用属于一个未解数值问题；后者的特征插值误差会跨后续blocks传播，不能拿VBench整体变化推定每个latent frame作用弱。图谱边均标为外部论文LIT，尚无本Wan4step I2V新实验或可接受的新算法。
