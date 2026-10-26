# E0 · 全深度双轨误差账本：初始静态设计、后续原生实现

**版本说明：本文件记录先前静态设计阶段。其后原生采集器已在[paired-native实现](E0_PAIRED_NATIVE_IMPLEMENTATION.md)中完成软件接线，新严格证据层见[paired-v2门禁](E0_PAIRED_NATIVE_STRICT_EVIDENCE.md)，另获固定v2范围授权；实际native测量/运行状态必须读取[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)与原生RUN。静态设计时记录的“尚未实现/未授权/零GPU”均仅表示本阶段的历史截面，不代表当前状态。**

初始阶段交付两个静态软件工具：[有符号向量账本](../../tools/e0_full_depth_trace_contract.py)与[原生接口检查器](../../tools/check_e0_full_depth_interface.py)。**这两个工具本身**是校验/归约层，并非原生采集程序；真正的四臂Wan采集程序已在后续[paired-native实现](E0_PAIRED_NATIVE_IMPLEMENTATION.md)接线。本阶段当时新GPU调用/生成/真实site均为0，属于明确的历史计数；当前执行与质量须读取[活动状态](../ACTIVE_RESEARCH.json)和真实RUN。后续[v3两来源320site实际结果](E0_PAIRED_FULL_DEPTH_RESULTS.md)已取得P23/P24的NATIVE-OBS证据；其全模型因果分解仍OPEN，G6实时场景暂停。

## 新结果与有符号分解

已有六视频补评显示[来源间相反的时序异常](E0_EXISTING_VIDEO_QUALITY_BACKFILL.md)。因此，下一轮不能只测局部相对SSE或统一寻找“冻结”阈值。对实际两轨输入的四个native FFN输出定义：

```text
a = fD(U_D), b = fS(U_D), c = fD(U_S), d = fS(U_S)
teacher_operator = b-a
student_operator = d-c
prefix_response = c-a
total = d-a = student_operator + prefix_response
||total||² = ||student_operator||² + ||prefix_response||²
            + 2 <student_operator, prefix_response>
```

`U`是调制后、Smooth1/Q1之前的实际caller输入；账本的`teacher_q1/student_q1`则明确要求真实post-Q1矩阵。实际producer应证明两者的边界关系，不能把U直接重命名q1。包含W1/GELU/Q2/W2的四函数要求相同真实输入，并对每个site另附绑定系数来源和明确row_ids的native-versus-continuous检查。连续诊断仍可为预冻结采样，不能把它的误差泛化到full-M。

账本沿M轴分块转FP64归约整个矩阵，保存能量、交叉内积、cosine和重组闭合；不保存160个完整激活。分块归约有额外计算/同步成本，原生采集时应另行计费。软件反例中两个分量各能量6，却因内积−6完全抵消；简单相加SSE会误判。

门控写回需要另一套四臂实际张量：`W(rD,a,gD), W(rD,b,gD), W(rS,c,gS), W(rS,d,gS)`，其中W包含原生乘法、加法与dtype cast。其prefix项同时含residual基底和gate变化，不能称为纯FFN输入响应，也不能用`gate*输出误差`替代原生写回。两个实际block输出差单列。这些是双轨相关性观察；某一层对终端视频的因果贡献仍需另冻有限hybrid干预。

## 调度器边界纠正

此前“外层latent一直FP32”的叙述遗漏继承的`BaseScheduler.step_pre`。独立源码审阅确认：固定BF16、SENSITIVE_LAYER_DTYPE=None且keep_latents_dtype_in_scheduler=false条件下，`prepare`的FP32 latent在每步入口可能被转为BF16；原`step_post`先做FP32两次更新，再回转当前latent dtype。这是条件静态推断，旧回执把latent转FP32后才哈希，不能恢复转换前真实dtype。

因此工具强制区分：

1. `step_pre_geometry`：实际入口前后四份状态，记录两轨各自cast误差、轨间差变化及有符号交叉项。
2. `scheduler_geometry`：before固定为**after_step_pre**，after为**after_step_post**，绑定实际sigma和sigma_next。令h=sigma−sigma_next，理想式`delta_next=delta-h*d_flow`只作为解释坐标，另外保存原生更新次序及出口cast造成的defect。
3. 每步记录六个状态dtype边界哈希；完成时检查前一步post与下一步pre前状态连续。完整latent shape另行声明；非synthetic记录必须为C,F,H,W四维，不能把局部采样面板标为full state。

相同flow差范数可以纠正或扩大旧latent差，符号取决于`<delta,d_flow>`。软件夹具也显示入口cast可吞掉原先非零差异。两者均为代数/软件验证，不是Wan已测传播结果。

## 原生接口和资源设计

[独立源审阅](../method_discussion/discoveries/E0/independent-full-depth-native-interface-20261026.md)给出精确方法名、23份实际源码SHA、offload时序和互相竞争的实现。设计优先保留单pipeline原生block-offload：在同一compute stream和当前GPU buffer中依次完成两轨与shadow，再swap一次。scheduler需通过原setter绑定pre/transformer/post，两个轨的hidden和可变state独立；初始candidate的单pending槽不能直接视为可重入四臂dispatcher；后续已单独实现四臂调度。

原non-FFN驻留31.25GiB、affine80pack约9.18GiB及当前原FFN矩阵约0.264GiB，已知payload合计约40.70GiB。它不是显存峰值上界，未含双buffer、pre/head、attention/QDQ临时、VAE、observer及allocator。现有AllWeightResidency在shadow dense访问后会永久留下另约21.09GiB原FFN，不能直接复用；所需bounded策略在本静态设计阶段尚未实现；后续接线见paired-native实现文档。双GPU替代方案还要支付full-M跨轨传输，不能靠标量norm重建有符号内积。

[静态设计](../../config/e0_full_depth_trace_design_20261026.json)绑定25份源码，检查7个关键API和NoCaching/SDPA/block-offload/无CFG/无seq有效配置需求；拒绝配置漂移、错误接口与无界FFN驻留声明。通过只能证明已绑定文件与接口要求相符，不能证明接线中性、资源可行或真实配置已运行。

```bash
python3 tools/check_e0_full_depth_interface.py
.venv/bin/python -m unittest discover -s tests -p test_e0_full_depth_trace_contract.py -q
.venv/bin/python -m unittest discover -s tests -p test_e0_full_depth_interface.py -q
```

10个账本测试覆盖抵消/放大、4D状态cast、采样冒充、完整160site缺口、非有限数、dtype及跨步连续性；4个静态接口测试覆盖实际配置模式、源码漂移/API变更和执行权限误升格。全部fixture保持SYNTHETIC标签，无模型加载。[原生接口的后续独立复核](../method_discussion/discoveries/E0/independent-full-depth-native-interface-20261026_review.md)单列剩余缺口。

## 下一步边界

以上属于静态设计阶段曾经开放的实现任务；其后双scheduler、四臂dispatcher、bounded FFN生命周期及严格身份检查已另行实现并有软件验证，paired-v2有界授权已登记。真实native物理峰值与320处配对已由[v3独立回执](E0_PAIRED_FULL_DEPTH_RESULTS.md)测得并经核验；当前开放部分为额外完整dense-on-Student-state函数与有限后缀因果机制。没有把模板当授权，没有选择新算子，也没有把测试夹具当原生数据。未来每条新视频仍需全81帧SHA绑定评估，官方/人工NOT_RUN时不得验收质量。
