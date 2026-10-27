# FFNDS · 每次生成必有指标；全40层/四步必须评估真实误差传播

**适用**：2026-10-26之后所有新视频生成任务（Dense、候选、对照、失败视频、调参、定性演示）。已有冻结协议和原始视频RUN字节不修改；后验补评必须使用有原始视频SHA的单独版本。**用户当前暂停实时生成**，本文件只针对四步Wan整段I2V。

**2026-10-26方向协议补充**：[扩来源＋有符号评价设计](../20261026/E0_MULTISOURCE_DIRECTIONAL_EVALUATION_20261026.md)。候选发现的科学评价优先保留源级多维功能/运动/光照质量，以及原生完整状态P/N与互斥频带signed cross；pooled SSE只是诊断分量。来源校准暴露程度与真实held-out严格区分，两个source的320sites不得当320独立视频。此补充不改既有视频RUN或本文件的完整逐帧必备契约。

## A. 视频是一个“结果包”，不只是MP4

每个独立 source×seed×prompt×method×hardware×模型版本的视频须有以下相互绑定的产物，不能仅给总体均值：

1. **来源与执行**：实际source/image/prompt（可用哈希以保护敏感输入）、seed、expert/scheduler、git/source/config/pack SHA、完整视频文件SHA、帧数、FPS、逐帧解码验证、运行状态与失败原因；dense/候选成对保证同输入/同编码尺寸/帧率。
2. **物理工作与时延**：cold/warm分开、完整generate wall、所有160个 FFN 边界及step/block拆分、W1/W2/GEMM/QDQ计数或矩阵形状、权重转移、pack构建/拟合/预载/保存时长、峰值实际显存、GPU隔离；明确“FFN边界”是否含norm/attn residual，不能把单块/重复行latency当整网。
3. **每帧空间、细节、颜色**：同Dense逐帧RGB MAE/RMSE、编码BT.709亮度漂移/误差、按frame分布及p95/max；边缘/高频损失与结构度量；光照与纹理分开，不让一种明暗偏移吞并其他故障。用LPIPS/DISTS等感知模型时固定版本/归一化，并另列，不冒充无模型RMSE。
4. **每帧时序、运动、背景**：相邻帧差与Dense配对差、各自的相邻帧变化幅度（运动冻结与额外抖动可能符号相反）、光流方向/幅度和质量、运动补偿下重建差；用Dense运动/有效mask定义静态区域背景亮度闪烁。低运动mask尚不等于语义背景；需要明确相机移动、曝光与遮挡。按连续窗口和source子集输出 p50/p95/max，而非只报全视频平均。
5. **专门视频质量/主观审查**：官方VBench-I2V或在协议中冻结的相应维度，至少subject/background consistency、temporal flickering、motion smoothness、dynamic degree、imaging/aesthetic quality；可按任务增加 VBench-2.0 intrinsic、光流/感知度量。配对人工审阅需要实际完整视频证据、评审人与具体观察及帧号。**没有运行就标 NOT_RUN，不能补零或宣称完整画质通过**。
6. **聚合/统计/验收**：每一条视频单独结果、source/seed/运动场景分层、按来源聚合（不是把160处block×steps当独立生成样本）；性能与质量独立验收。失败案例完整保留。不得把保存操作数的 local-SSE 当作视频质量。任何指标门槛须在看结果之前冻结，质量缺失则只能归档/科学探索，不能晋级算法。

**已有工具**：`python tools/ffnds_video_metrics.py evaluate --candidate path/to/approx.mp4 --reference path/to/dense.mp4 --identity-json run_identity.json --out path/to/metric_report.json`；`python tools/ffnds_video_metrics.py audit --report ...`。这会真正解码**全部帧**，保存SHA、空间RGB/luma、参考光流/候选光流、邻帧运动、局部锐度proxy与低参考运动区亮度差及p50/p95/max。依赖NumPy/OpenCV；仅CPU读视频，不运行模型。计算的是自主命名的工程诊断量，**不冒称官方VBench/LPIPS、因果背景mask或人类评价**。原视频不存在时没有办法仅凭JSON重算未保存的特征；显式登记 MISSING_ARTIFACT。

每条视频的 `run_identity.json` 最少形态：

~~~json
{
  "identity": {
    "source_id": "image-sha256-or-frozen-id",
    "seed": 42,
    "expected_frames": 81,
    "input_prompt_digest": "prompt-image-condition-sha",
    "model_digest": "actual-model-checkpoint-sha",
    "protocol_sha256": "frozen-protocol-sha",
    "run_sha256": "completed-run-receipt-sha"
  },
  "reference_identity": {
    "source_id": "image-sha256-or-frozen-id",
    "seed": 42,
    "input_prompt_digest": "prompt-image-condition-sha",
    "run_sha256": "matched-dense-RUN-json-sha"
  },
  "cost": {
    "generation_wall_seconds": 0.0,
    "ffn_boundary_seconds": 0.0,
    "setup_seconds": 0.0,
    "peak_gpu_memory_mib": 0
  }
}
~~~

数值为字段形态示范，**不是一条实际测量**。真实运行必须从计时和资源收据填写。paired候选除两个视频必须实际匹配，还要绑定参考 Dense 的 source、seed、输入图像+prompt+conditioning digest 和原dense RUN SHA；不允许拿不同prompt/seed的视频凑“质量差异”。无Dense配对的参考视频也做全帧自身时序/纹理统计，之后与候选组合时生成paired报告。

多视频完整性入口：`python tools/ffnds_video_metric_gate.py --bundle path/to/video_bundle.json`，要求同一视频文件有唯一的逐帧报告、原始 RUN 收据以及全部文件SHA；有Dense配对则额外核对Dense视频与Dense RUN的文件SHA。最小bundle形态（SHA值必须由实际文件计算，下面只是字段名）：

~~~json
{
  "schema": "ffnds.video.bundle.v1",
  "expected_video_count": 1,
  "videos": [
    {
      "source_id": "image-sha-or-id", "seed": 42, "variant": "affine54",
      "video_path": "generated.mp4", "video_sha256": "sha-of-generated",
      "run_receipt_path": "RUN.json", "run_sha256": "sha-of-RUN",
      "metric_report_path": "metrics.json", "metric_report_sha256": "sha-of-metrics",
      "reference_video_path": "dense.mp4", "reference_video_sha256": "sha-of-dense",
      "reference_run_receipt_path": "DENSE_RUN.json",
      "reference_run_sha256": "sha-of-dense-RUN"
    }
  ]
}
~~~

每条视频须在运行前冻结预期帧数（例如本轮Wan为81帧）、输入image/prompt/condition的digest，并保留全部frame索引的实际测量值，不仅是全段平均。实际解码未达到预期帧数则拒绝完成报告。官方指标和人工结果必须独立保留工具/模型版本和实际评价凭据；核心指标报告允许尚未获质量验收，但任何缺文件/缺真实帧都会使证据bundle检查失败。

分析器写出的 `external_quality` 默认明确为 `NOT_RUN`，`quality_accepted=false`；必须从真实官方输出和实际视频人工审查另行提供具有版本、来源/运行签名的证据才能变更。验收模式 `audit --require-acceptance` 遇到缺失必须失败；可产生**有未决项的客观指标包**，但不能宣传“完整视频质量合格”。核心指标或逐帧覆盖不齐时连客观指标包也不得标为完成。

**已有六视频当前证据**：`research/20261026/E0_RECONSTRUCTED_VIDEOS.json`保存原始两来源×81帧、RGB/encoded-luma与成本/SHA；[独立后验补评](../20261026/E0_EXISTING_VIDEO_QUALITY_BACKFILL.md)已完成486帧/480帧间核心工程指标，包含修正后的Farneback运动和运动补偿亮度proxy。原RUN/旧报告保持不可变；正式VBench-I2V/VBench2、语义背景与人工全片仍按实际验收状态标记。

参考官方指标定义：[VBench CVPR2024原仓库](https://github.com/Vchitect/VBench) 及 [VBench-2.0](https://github.com/Vchitect/VBench/tree/master/VBench-2.0)。官方报告需要原工具、权重、版本、数据范围；本repo廉价proxy绝不是“VBench时序闪烁”的实现。

## A5+HWG历史观察后的指标修正：时空双轴与小物体功能

[历史A5+HWG背景闪烁/小物体模糊机制研究](../20261027/A5_HWG_SPATIOTEMPORAL_FUNCTIONS_FLICKER_AND_SMALL_OBJECTS_20261027.md)说明，粗略主体结构、空间高频细节和跨帧背景稳定性可以分离；本节细化原有A.3–A.5定义，**保持现有结果包和当前已实现CLI的字段/验证标准，不把尚未实现的ROI指标写成已执行的自动化功能**。

- **空间细节**：对比对象的小尺度纹理/边缘/轮廓与身份保持；仅Laplacian高不能证明更锐利，更不能证明小物体保留。检测/人工ROI存在且有效时分别输出小目标面积、结构/局部锐度、中心漂移和可见帧比例，缺少可靠ROI标\`NOT_RUN\`，别用任意低纹理区代替小物体。
- **时序背景**：将静态镜头与相机/对象运动分开；记录同源Dense与候选的背景匹配区域。先估计并冻结运动warp、可见性/遮挡置信度及背景语义/低运动代理身份，再报告对齐后的luma和颜色变化、光度残差和高时间频率分量。**整帧相邻RGB MAE不能区分真实动作与闪烁**；低运动区域不自动等于背景。
- **运动本身**：保留连续RAFT/Farneback光流幅度、方向/ROI、五个连续时间窗口及二值Dynamic Degree。静止视频可能具有很高Motion Smoothness且过原二值阈值，不能因“动作平滑”而推断动作遵循prompt；同样背景静止描述中减少运动可能正确。
- **空间×时间双频**：背景照明闪白可以是空间低频×时间高频；细小物体持续模糊可以是空间高频×时间低频；闪动纹理可两轴均高。这三者需要独立描述。任何latent时间FFT须注明21个latent-time槽不等于81帧RGB，去噪step也不等于视频时间。
- **mask机制归因**：mask时序异动、阈值margin、当前W2/gate有符号贡献、Q2 scale副作用与Attention/后缀传播均只能按实际同身份和合法graph/稀疏插入点观察后报告；动态mask的Jaccard/xor率及背景闪烁相关性本身不足以证明因果，需有同prefix的固定/稳健mask对照。论文的VBench Temporal Flickering通常以静态prompt视频作官方指标，此处运动补偿背景ROI是另一个**自定义工程指标**，两者严禁同名。
- **资产与身份**：A5+HWG历史视觉描述尚不是本仓库新得到的逐帧ROI数值；同方法族旧TopK pilot与固定HWG/τ版须根据manifest区分。旧质量文件和旧RUN原样保留，只有真实找到原MP4与mask载荷才开展新的再评价（需符合实际ACTIVE授权）。

**可交付状态**：现有\`ffnds_video_metrics.py\`仍只实现它已经声明的工程代理；本节增设的是研究规范与未来**可选**深入评议字段，非静态研究阶段新增必填/质量gate。这些指标只有在需要声明对应行为保真时，才作为事先冻结的匹配实验重点。

## UB1实测新增的动作保真测量：二值动态与连续运动分开

[UB1运动衰减、光流可观测性和因果研究](../20261027/UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md)已从**既有四条视频的原版RAFT原始数组**计算出两组连续均值和五窗口比例：犀牛source882 UB1/Dense≈38.13%（38/40对较低），住宅source80≈9.19%（40/40对较低），而四条原版Dynamic Degree在10.875px、至少10对超阈的规则下全为0。原formal20历史Motion Smoothness几乎不变，这**不是矛盾**：平滑与幅值不相同。**这不是全20来源结果，也不是UB1主体真实速度/动作正确性的因果结论。**

今后当研究或对外声称**运动保真/动作保留**时，需要明确分层附加：
1. 官方 **Motion Smoothness**、官方 **Dynamic Degree binary**、原函数中间\`top5% RAFT-pair score\`的**连续诊断值**分开列出（后者不冒充另一个官方质量分数）。原方法16fps隔帧41帧/40对、原尺寸、checkpoint/hash、阈值、所测对象需登记，个别source原型不能代表正式cohort。
2. 视频背景/相机/主体ROI按可信分割/运动warp分离；先以跟踪或标注直接确认轮廓位移、关键点或动作相位，再论证“实际生成运动衰减”。同时报告锐度/模糊、遮挡/匹配置信度，避免**blur让RAFT幅值降低**或Dense纹理噪声抬高flow的测量混淆。
3. **动作语义**与光流分开：\`rhino grazing\`带运动语义但还需检查实际嘴/头/腿连续过程，静态houses不要求相机平移；不能简单以越接近Dense光流绝对值作为始终正确的生成目标。
4. 可先按已注册来源身份报告全视频40-pair中多少对低于Dense、5个连续窗口的时间稳定性和源级主观观察；40 pair是同一个source内部相关时间点，不是40个独立视频。真正的确认要预冻结来源分层、包含被筛出的静态/相机类、未暴露source/seed及可复核的视频SHA。
5. 若进一步宣称底层FFN承担了运动功能，要在合法Q1/Q2/gate的同前缀native后缀上做**时间维函数响应**与显式干预；光流或local SSE相关性不直接认定是哪组神经元。

**实现/验收关系**：\`ffnds_video_metrics.py\`和当前完整结果bundle仍以实有字段为准；这些是**运动保真主张的专项研究评议字段**，未实现的主观ROI/RAFT追踪一律\`NOT_RUN\`，不扩写旧RUN或伪称现有通用指标工具已具备全部功能。新的GPU/模型/视频任务必须遵循[ACTIVE](../ACTIVE_RESEARCH.json)登记的执行范围，不由本段自动启动。


## B. 40层同时替换：首先量化沿真实Student前缀发展的函数差，不只用Dense输入

当前 `a54` 对原模型 `f_{k,l}` 拟合了固定 `\hat f_{k,l}`，其中每个去噪step `k∈{0,1,2,3}`、block `l∈{0,...,39}`；最终原生运行确实替换了**全部160位**。但以前“native验证”只在block0/30进行一次有限partial same-input，而全40层8.54%是**teacher输入的抽样函数近似**。block30不是最后一层39。两者都不能当成全网误差归因。

### B1. 逐块四种真实对象

对相同step、同一expert与相同用户条件，分别记录原模型前缀在该FFN算子入口产生的 `x^D_{k,l}` 与完整近似模型产生的 `x^S_{k,l}`。在真实 Q1/postQ1（不是自由选的未量化 hidden）边界计算：

```text
f_D(x_D)             teacher-input dense
f_S(x_D)             teacher-input approx, native same input
f_D(x_S)             student-input dense, native same input
f_S(x_S)             actual student-input approx
```

令 `e_op=f_S(x_S)-f_D(x_S)`（同Student输入上的算法重构误差），`e_shift=f_D(x_S)-f_D(x_D)`（前缀导致的函数响应），则**向量级精确**：

`f_S(x_S)-f_D(x_D)=e_op+e_shift`。

在原生FFN输出/门控残差writeback/下一个block输入分别保存分量范数、signed内积、cosine、相关协方差/极值、来源/步/层/空间行分层；考虑BF16量化时，应在每个实际边界重新测量**有限输出差**，不能将浮点分量恒等式声称为逐位writeback恒等式。带gate校准残差与多路norm/attention的前缀必须分开。

**关键区分**：`f_S(x_D)-f_D(x_D)`是既有teacher-forced的静态局部误差；`e_op`是在**真实student受扰输入**上的同输入误差。这能真正观察distribution shift。另需对`f_S`的数学连续预测与真正HiF4/QDQ/BF16运行作同输入对照，先识别部署数值差而不是把它错归因为跨层传播。

### B2. 全层传播与四步去噪

- **全面遍历而非挑两层**：每个执行source/seed，全部40 blocks×4 steps=160 sites。每个site至少记录该处teacher/student前缀输入配对的错误分解与成本、门控残差误差、block输出误差；每步结束对head flow和scheduler latent保存全向量差的范数/内积/sha。全M=31668中可以使用预冻结/分层采样进行**向量内容保存**，但源/步/层的标量聚合应全覆盖，不可用sample面板冒充整个M。
- **累积不是简单SSE相加**：以离线/本次运行可用的网络状态标记 `H^D_{k,l},H^S_{k,l}`，计算各block后 `H^S−H^D`；分辨误差递增、收缩、交叉抵消。必须记录每层输入、输出误差方向与范数和层末gate/bias写回；不能用“较晚层更大”就假定误差线性积累。
- **跨step**：保持实际Wan调度器的 `z_{k+1}=z_k−h_kv_k(z_k)+\rho_k`。保留每个step的 `delta=z_S−z_D`、`d=v_S(z_S)−v_D(z_D)`及 `\langle delta,d\rangle`；这决定下一步是校正还是漂移。图像视频的第40帧**不是**去噪step2。
- **后缀敏感性**：完整 teacher/student 双轨迹本身不隔离单层因果贡献。只有在少量、分层冻结的**混合原生前缀/后缀干预**中，才能测一个site的有限目标贡献；不必运行160条完整视频。先扫160处轻量迹线，再跨depth、expert、step、来源固定包括首中末、峰值层和低误差层的多层扰动组，保留选择规则/额外成本。用新的held来源防止事后挑选偏差。
- **视频消费者**：把每个step末 `delta`、flow变化与终端画质各维度相关起来；空间上分析背景、运动物体、低纹理区与边缘；失败类型可截然不同（当前熊运动抑制 vs 建筑场景额外变化），需要跨source/semantic strata检验。相关性不是因果归因。

### B3. 机制淘汰准则（互相竞争但不硬指定算法）

| 假说 | 支持它的判别结果 | 使其降级的结果 |
| --- | --- | --- |
| **数值部署偏差主导** | 同输入native vs连续误差在多个层显著，且与放大/质量异常吻合 | 全层同输入native一致，主要偏离出现在前缀与下游 |
| **受扰前缀失配主导** | `e_op(x_S)`显著恶化于teacher输入的近似误差，且和中后层误差变化相关 | 同输入op误差各层稳定，而少量方向造成大下游变化 |
| **下游有符号方向主导** | 局部SSE近似，原生后缀/flow对不同方向放大不同且跨层出现相干或抵消 | 层/step下游响应主要与局部范数排序一致、无显著交互 |
| **去噪轨迹目标偏差** | 部分step`||delta_next||`增长与`\langle delta,d\rangle`符合；单纯改善当前flow并不能修复latent | 错误主要出现在step内，step间delta由其它分量解释 |
| **特定算子容量不足** | 即使消除可测部署数值/输入偏移，固定54组在多层/多source上仍显著损伤下游功能 | 新机制同MAC恢复功能，说明不是必要的54组物理极限 |

以上证据阈值要由独立Agent在看下一新实验数据前冻结；将已淘汰的**具体比较器**和开放函数族区别对待。当前whole94/affine54失败不授权新训练/新视频。

## C. 任务准入与禁止假进展

从现在起新video protocol必须写`metric_report_required=true`、绑定`metric_contract_version`、匹配参考来源、每条视频分组/文件SHA、成本分母、官方质量评测实现或其不可执行原因。未收到report/未审查则保持`COMPLETED_EXECUTION_QUALITY_PENDING`，不称结果合格。新multi-layer方法必须按160-site维度呈现逐层过程，**单层teacher forced只可作底层算术/局部构造证据，不能作为全网络结论**。

Agent通过[更新的G5d/P23/P24计算图](../../docs/98_端到端计算图与证据注解_20261025.md)定位已测原生site与下游消费者；区分v3的真实四臂、gate写回和四步flow/latent记录，与未来有限反事实/受控后缀的OPEN因果边。

**最新执行核验**：[v3两来源320site结果](../20261026/E0_PAIRED_FULL_DEPTH_RESULTS.md)已保存每来源160处真实配对与四步signed flow/latent，四条视频、16个latent与历史指纹一致，核心全帧/身份检查通过；v2后处理失败的未保存site独立记为0。318/320处Student-prefix raw绝对误差变小、latent差同时增大。新实验如对`v_D(z_S)`作完整dense有限差分须另行授权；官方VBench与人工视频验收保留NOT_RUN。
