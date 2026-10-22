# R13：已完成空间误差拆解，继续研究平均非线性响应

当前入口：[CURRENT_DECISION](../CURRENT_DECISION.json)、[路线表](../20261021/method_triage.json)、[说明与公式](../../docs/71_空间误差的可共享部分与双点非线性构造.md)、[机制差异记录](R13_mechanism_record.json)。最新原生实验仍是[已拒绝D0](HANDOFF_R12_D0.md)，24次已完成，不重跑、不触发D1/预测器/rollout。**本轮是保存数据、源代码、CPU与纯随机张量GPU成本研究，模型资格仍为空；不是用户暂停研究。**

## 可以依赖的新结果

- 固定24个开发暴露来源、四步block20的96份OP2原生文件，1,008,643,584字节及张量身份通过CPU核验。坐标来自真实`[21,29,52]`行号，不是256采样offset。没有一份完整四行stripe被保存。
- 完整输出邻居余弦0.525–0.875，但UB1尚未解释误差仅0.093–0.162。直接复制这部分剩余误差，损失变成UB1的1.675–1.814倍。该数据不能区分量化噪声与内容非线性，勿编造因果归因。
- step1/2等主MAC的193对不重叠真实邻居，完整anchor复制0/193胜过UB1。允许任意理想共同输出的连续下界，也只有7/193可能胜过UB1；逐对理想选择最多改善0.236315%。这是本样本和共同输出动作族的局部界，不是所有空间FFN或R12最终输出splice的界。
- 复用原R9两个专家仿射矩阵，96次CPU线性计算，无重拟合。逐token保留L再复制单anchor余项仍差；但理想共同**均值**余项的参考误差在step2是UB1的0.6682，原偶数未拟合开发来源为0.8101。step1/3理想参考也差，不能一律推广。参考量来自稀疏行对的设计权重，不是已执行完整stripe或新来源质量。
- 原HiF4 Python参考函数的固定反例：逐通道边缘分布完全相同，仅改变通道联合关系，平均post-Q2有63/64通道不同。因此纯逐通道高斯矩公式不足。保留联合关系的双点草图在两个合成例中比单点好，但单次草图方差大；不是Wan结果。

## 当前值得深入的构造

保留每个token自己的冻结全通道仿射L。真实四行stripe内，从当前post-Q1输入形成均值mu和共享符号的中心化contrast c。单点计算mu；双点计算mu±c，经原W1/tanh-GELU/group64Q2之后平均激活，只做一次W2；减掉mean(L)再给各行加回自己的L。

固定3958/7917组做双点，其余单点，计入额外W1后主投影MAC=49.767729%。全部双点会到56.0185%，不合格。全行Q1、更多Q2行、pool/sketch/排序、BF16骨干驻留、激活平均、广播和完整写回都还需真实计费。主MAC不能替代完整FFN1.5×。

未来可用W1前contrast范数分配额外点，但真实增益可预测性未知。必须与完全相同总点数、在训练来源选出的最强**固定**空间放置比较；单有输入相关输出或固定位置双点不是动态计算证据。需要同时比较UB1及相同骨干的单点响应。原R9 hidden贪心、TC1支持迁移、R20跨步缓存、D0 raw宽度拒绝均保持原范围。

当前缺少真实完整四行V1操作数和新算子的原生结果。已准备[可复算坐标草案](R13_complete_stripe_operand_DRAFT.json)：四个已暴露来源各64组真实horizontal4，共256行/来源；没有采集任何新操作数。下一模型检查需要独立的冻结登记和用户执行授权；不得把保存的零散四行拼成空间stripe，不把理想Rbar当可读取标签，不为低集合平均误差免费运行16符号。可以继续源/CPU工作和实现准备。直接GPU实现的成本预检失败后，先检查下述三方向重构的后端，不立即采集Wan操作数。

## 复算与证据

不需要私有大张量即可复算公共记录、恒等式和预算：

```bash
python3 tools/check_research_routes.py
python3 tools/check_r13_artifacts.py
python3 -m unittest discover -s tests -p 'test_spatial*.py' -v
python3 tools/analyze_spatial_sharing_limits.py --output /tmp/r13_sharing_limits_review.json
```

保存值分析的原始标量记录在忽略目录`runs/r13_saved_spatial_audit`，SHA见公开结果；两份既有R9矩阵镜像在节点HF派生缓存，总105,099,920字节，未新增权重拟合，未把权重放入Git。[收据](R13_original_affine_asset_receipt.json)。重执行96份CPU分析需要原OP2和R9资产，工具拒绝覆盖首次结果。合成源/参考检查还需要固定的原HiF4纯Python源码，不导入CUDA扩展。

公开入口包括：[空间数据](R13_spatial_residual_coherence.json)、[同预算轴](R13_saved_compression_axis_value.json)、[仿射余项](R13_anchored_affine_remainder.json)、[有限下界](R13_spatial_sharing_limits.json)、[合成联合矩判别](R13_pooled_nonlinear_moments.json)、[7项原文检索及访问层级](method_search_R13.json)。完整数值/来源身份由`check_r13_artifacts.py`核验；CI新增同一离线检查与14项空间/数学测试。

[实际验证记录](continuation_checks_R13.json)：最终82项相关测试通过；不含`.runtime`/`runs`的干净索引导出中，6个证据CLI、同82项测试及合成smoke通过。公开索引扫描通过，未加入私有节点名或机器绝对路径；D0与本轮未注册种子的模型选择均返回预期退出码2。

## 合成GPU成本与新的线性改写

[固定随机张量A100结果](R13_synthetic_spatial_response_cost.json)：dense55.315ms，半hidden＋R256代理33.741ms，固定/动态双点40.768/40.816ms，动态仅1.355×。8项原QDQ逐位控制先通过；峰值分配8.230GiB，零Wan权重/训练/视频。直接实现不放大，不能把它的49.77%MAC当1.5×。

利用四个中心化输入只有三个独立方向，`L_i−meanL=A(X_i−mu)`可改为3份Haar对比投影再逆变换，取消冗余均值投影和中间量，保持双点数不变，主MAC成为45.1381%。[24份保存输入的CPU算术控制](R13_three_direction_contrast.json)中，改写差异仅为UB1 gated误差的0.02364%，通过预定1%精度检查；这些任意四行只是算术fixture，不是空间stripe，TeacherY均值只作测试加数。新改写的GPU速度、原生数值和质量未测。下一步先核对投影/重建融合的内存计划及独立固定合成成本，再考虑单独登记Wan实验，不扫q/seed/邻居池。
