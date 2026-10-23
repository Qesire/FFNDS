# PRD successor: residual-complete action value

本轮输入快照：`7c7019ec66835e740b3573d5572a8da59f0efa69`。该输入快照中的C1已完成、C2已准备；C2的当前独立授权与执行状态见[现行C2交接](HANDOFF_C2N0.md)。完整研究见[docs/88](../../docs/88_AutoResearcher再探索_残差双路径价值与取样修订_20261023.md)。

1. C1两个whole64交换在72整位置上均输UB1，停止该动作集合的router/q扩展。原R16/R17按用户指示退出活动研究；作为历史匹配对照保留。
2. 重分析C2：捕获率+0.978586个百分点，6/6来源汇总为正；15/72位置为负，12处在step2/3。保存数据CPU结果不是新视频。
3. 新机制：下一norm抹去的径向误差仍经残差直通传播。有限合成反例已反转排名；主方向改为完整block/step的条件净动作价值，在线只用pre-W1特征。
4. 保留[C2原配对视频草案](C2N0_native_smoke_DRAFT.json)及状态，不在已暴露数据上继续改评分。B方向准备完整因果边界的标签与低成本读出；跨位置动态预算作为后继种子，与最佳位置固定策略比较。
5. 采用[新取样设计](PRD_sampling_revision.json)：场景簇隔离、来源/seed分层、四步四深度带轮换、完整tile概率取样、镜头图像交叉、最终样本量由来源级配对方差与精度确定。示例12来源/96上下文是覆盖设计。

复现：`python tools/analyze_prd_successor.py --check`，`python -m unittest discover -s tests -p test_prd_successor.py -v`。表示账本和research_state v2.1均按新版schema验证。新增模型、GPU、官方评分调用均为0。原实验记录与C2代码/草案字节保持不变。

当前C2固定范围已经另获用户授权。PRD重分析所述代码/草案不变属于上述输入快照；执行前的FP32统计精度修正及原未执行v0记录见[C2精度修正](C2N0_prelaunch_precision_fix.json)。六视频与原对照保持原范围，不据15个负位置改评分。
