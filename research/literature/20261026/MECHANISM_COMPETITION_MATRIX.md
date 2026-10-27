# FFNDS · 现行研究机制与强竞争论文矩阵（G7–G12）

> 2026-10-27综合版本，覆盖79项论文索引。[完整研究报告](../../20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)详细陈述数学、原生证据和物理预算；[Crosswalk](../../agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)保存机器可检索的M01–M12机制组/源节点/论文ID/成本。下表是**同构性与区分性**导航，而不是方法优先序或GPU任务单。

| 研究机制 | 最强公开先例（LID / 图节点） | 何时可观察、何处产生真实费用 | 反向预测与当前Wan待验证边 |
|---|---|---|---|
| **冻结FFN有符号函数可预测** | Geva L01/superposition L02；NerVE L50/K10–K12；Prox L69/Q19–20 | preW1 q能读、weight静态统计能离线预处理；postQ2 signed函数是后验标签 | 可能只有W1/GELU后才分辨，或shared Q2 group/跨行相消使单通道不足；验证原gate×W2和整视频 |
| **FP4有符号直方图与严格区间** | XNOR-Net L77/I13、量化乘积理论 L78/I14；经典重排定理（G12 I04） | 当前激活先Q1才能生成signed histogram；每行每token分组查询/partial-pair读价必须计算 | 单行紧界可能宽，所有行共享输入导致joint vector界不紧；与低位dense相比每组检查可能不划算 |
| **付费INT4 proxy/MLP预测** | Prox L69/Q19–20；Chipmunk L09/K25；Spark L70/Q21–22 | 代理投影和mask不免费；Prox对LLM SwiGLU的两次proxy分摊到三次selected GEMV | 原Wan是W1/GELU/Q2/W2，不是SwiGLU 3投影；比较不支付proxy的码字证书及实际Q2 group |
| **编码域LUT/bitset/Psumbook** | BiQGEMM L75；LUT-GEMM L76/I11–12；CodeGEMM L73/Q27–29 | 码book与current x预计算partial sums、lookup、scale和cache | 两份官方代码只在single-vector fast，多token dequant+cuBLAS/matmul；要有真正M×K原生code kernel |
| **稀疏硬件与quant layout** | RT-Lynx L08/K13–15；SpQt L71/Q23–24；MACKO L79/I15–16 | 实时activation N:M pack、离线zigzag/静态weight sparse index与kernel访问 | tokenwise后验稀疏≠原M16×K64空tile；GEMV优化不能自动迁移大量视频token |
| **FP4 sparse＋dense residue** | SharQ L74/Q30–33，NVFP4官方代码 + T2V A14B4卡原论文 | 在线按相邻pair幅值4:8取稀疏FP4、构造R=x−DQ(Qsparse)，**两个GEMM都执行** | 不是免费省W1/W2或已适配原HiF4；1.58×需要SageAttention，与完整native NVFP4 dense比较wall |
| **训练后条件专家/硬树函数** | Dense2MoE L10/K16–19；FFF L53/K32–35；Spark L70 | 训练shared/normal、路由、专家负载或硬树path的序列决策 | 冻结原Wan神经元天然可删与训练新函数是两个命题；对照相同训练预算和视频功能 |
| **少步蒸馏/训练紧凑宽度** | DIFS L34/K01–03；Mobile Video DiT L37/K07–09；SQ-DM L72/Q25–26 | training使step/hidden结构变化，部署压缩参数，或SiLU→ReLU重训严格零 | 原Wan两expert各两步，HiF4 group64 vs训练channel-chunk8；SQ-DM是U-Net定制加速器 |
| **按latent frame/layer分配M行** | FIS-DiT L35/K04–06；SCD L60/S13–14 | Gather+Block+Interpolation、causal encoder每帧一次和renderer每步 | FIS的Wan2.2结果为T2V且插值不等价；SCD训练新模型，原I2V frame时间依赖另证 |
| **历史主/残差及FFN增量** | SVD-Cache L39/K20–23、Chipmunk L09/K24–27、L²P L52/K28–31 | 历史缓存、全功能刷新、预测参数/FFT、same-expert合法状态 | 50step FLUX历史难直接用于4step high/low，student前缀偏移可能导致背景光照变化 |
| **真实因果流式与低延迟系统** | CausVid L55；LongLive2 L61/S15–16；EdenLive L66/S23–24 | 改训causal AR、few-step、自回归KV/shot sink、StreamVAE、低位GEMM和SLO队列 | clip平均FPS ≠ TTFF/p99/响应；Wan2.2-TI2V-5B与原I2V-A14B-D-Q不同 |
| **视频完整功能及物理验收** | OBS-Diff L11、VBench L29/VMBench L30；本仓库P23–P25 | 全40×4、独立seed/source、gate/BF16写回和当前硬件完整kernel | 局部SSE更小仍可能末端亮度/运动/小物体失真；作者外部指标不能升级本仓库正式质量 |

## 统一反查流程

先到[Graph Bindings](../../agent_protocol/GRAPH_AGENT_BINDINGS.json)找具体 F/G/P/K/S/Q/I 节点；[Crosswalk](../../agent_protocol/MECHANISM_GRAPH_CROSSWALK.json) 的M01–M12对每族给出**源节点→最早信息时点→真实计算开销→强比较论文→最小反例**；再读[精确原始论文方法/源码](../20261027/)及[实验谱系](../../EXPERIMENT_FINDINGS_ATLAS.md)。每一条声称性能有效的路径都需绑定同format、同模型/任务、同质量和完整生成/实时系统的物理比较器。

**证据身份必须可定位**：原Wan本地观察不支持外部论文新模型已在本repo运行；作者论文摘要只证明作者报告的结果；原文公式与原生源码检查只提升它们所覆盖的部分。用户暂停的模型/GPU试验不因本矩阵自动恢复。
