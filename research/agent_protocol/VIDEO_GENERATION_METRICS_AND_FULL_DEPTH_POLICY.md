# FFNDS · 每次生成必有指标；全40层/四步必须评估真实误差传播

**适用**：2026-10-26之后所有新视频生成任务（Dense、候选、对照、失败视频、调参、定性演示）。已有冻结协议和原始视频RUN字节不修改；后验补评必须使用有原始视频SHA的单独版本。**用户当前暂停实时生成**，本文件只针对四步Wan整段I2V。

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

每条视频应保留全部frame索引的实际测量值，不仅是全段平均。官方指标和人工结果必须独立保留工具/模型版本和实际评价凭据；核心指标报告允许尚未获质量验收，但任何缺文件/缺真实帧都会使证据bundle检查失败。

分析器写出的 `external_quality` 默认明确为 `NOT_RUN`，`quality_accepted=false`；必须从真实官方输出和实际视频人工审查另行提供具有版本、来源/运行签名的证据才能变更。验收模式 `audit --require-acceptance` 遇到缺失必须失败；可产生**有未决项的客观指标包**，但不能宣传“完整视频质量合格”。核心指标或逐帧覆盖不齐时连客观指标包也不得标为完成。

**对已完成六视频**：`research/20261026/E0_RECONSTRUCTED_VIDEOS.json`已经有两来源81帧RGB、encoded-luma、相邻帧差、运行成本/SHA，属于**legacy partial-quality coverage**。缺真正的参考光流、语义分区/运动补偿闪烁、感知距离、正式VBench-I2V和完整人工看视频；不能把缺项补成PASS、也不重写原RUN。可在原始MP4可访问时以本工具**新增一份**后验报告，并保留与旧指标公式/视频SHA的关系。

参考官方指标定义：[VBench CVPR2024原仓库](https://github.com/Vchitect/VBench) 及 [VBench-2.0](https://github.com/Vchitect/VBench/tree/master/VBench-2.0)。官方报告需要原工具、权重、版本、数据范围；本repo廉价proxy绝不是“VBench时序闪烁”的实现。

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

Agent仍使用 [Graph-Agent实际计算图](../../docs/98_端到端计算图与证据注解_20261025.md) 的F03–F13/G20–G21/P22/D09，并支持新的 `L00...` 全深度双轨迹迹线/消费者节点；图增量是待执行的诊断设计而不是凭空完成的测量。

下一项有效交付是**静态可验证的跟踪/质量协议与工具**，之后才是新授权的全160位实际配对记录；不能宣称当前已有这些新迹线、指标或改进视频。
