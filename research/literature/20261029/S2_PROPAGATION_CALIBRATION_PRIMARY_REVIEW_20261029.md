# S2：把局部重建、传播敏感度与闭环校准分开

2026-10-29，ROOT实际在线检索及原文核对。搜索覆盖 diffusion accumulation/student trajectory calibration 与低秩误差传播，并从PulseQuant追到AccuQuant。下面是LIT/SRC及推导；本地640条完整位置结果、干预或新方法表现仍各以真实RUN证明。原文快照和作者代码SHA见[台账](S2_PROPAGATION_CALIBRATION_SOURCE_LEDGER_20261029.json)，源码只读取，未执行。

**C14 · [PulseQuant v1](https://arxiv.org/html/2609.33384v1)**。读§4、§5.1、附录B.1，公式4–8。一次block–step量化脉冲后沿Dense继续，以`g²=(E_H+eps)/(E_0+eps)`衡量传播；风险权重用于离线半径选择，再用激活方向指导合法相邻码字修改。作者Wan2.2启动脚本实为40步T2V/CFG4和3/shift12；本地是四步I2V/无CFG双调用，不能照搬其40步插值或H=4。固定提交的profile构造、trace分析、风险与code-refine局部源码已读；不是完整部署核验。脚本的旧参数名不直接证明执行了对应损失：实际_finalize使用环境中的temporal_delta和固定trust=-1，须沿实际分支理解。其Dense后缀pulse与本项目学生后缀恢复是不同处理，最小判别应比较同方向在两种后缀上的响应。

**C15 · [AccuQuant v1](https://arxiv.org/html/2510.20348v1)**。读§3.2–3.3、公式5–18及组长/梯度消融。将连续M个采样更新的末状态配对，而非逐步同输入误差；以省略模型状态Jacobian、保留scheduler标量的梯度近似降低存储。作者方法不是无拟合的后验统计。特别地，其`δ_t=epsilon_Q(x_S)-epsilon_D(x_D)`已经含输入状态差，不能把它当本项目同h局部L。DDIM系数和近似也不能直接替代四步Wan实际更新；本地应先用真实后缀方向响应检验近似是否成立。作者执行代码未核查。

**C16 · [LEMS/KFAC-SVD v2](https://arxiv.org/html/2610.32878v2)**。读§4.1–4.2、公式8–15。双侧token矩以Kronecker近似构造曲率；rank代理为谱尾形状×单点任务损失标尺×深度传播偏置，再用参数预算ILP分配rank。此为LLM任务与代用目标，非视频因果或实测时延优化；代码未核。它提供“谱只确定层内形状、跨层价值另校准”的可比较对照。token外积可提高矩阵的代数秩，不把同一图内token变成独立来源。

由这三个竞争机制得到两个不同的下一动作。第一，在真实学生自身h上恢复一次原W2，并沿下一attention、flow、下一step观测，区分局部误差幅度和后缀传播方向；Dense继续与Student继续分开。第二，若后缀传播确实主导，再比较同数据/同计算预算的状态感知或短后缀目标与原局部目标，显式保留拟合、observer和真实rank成本。二者都以新12开发图的Dense动态分层和source-held观测扩展，不能把两张pilot的P较大直接认作反馈不稳定或给所有早层加rank。

对应[独立正常后缀因果设计](../../method_discussion/discoveries/E0/s2-normal-forward-causal-factorial-independent-20261029.md)。这里新增的是可反证的目标和信息时点对照；没有借作者成绩宣布本地BF16低秩方法质量或速度通过。G14的C14/C15/C16分别落到采样状态、传播、量化/重建与宽度分配的既有锚点，映射为LIT。
