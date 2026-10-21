# 历史交接 — R8F2已完成并拒绝

当前入口已迁至CURRENT_DECISION、method_triage、docs/61和HANDOFF_R10。本文件保留完成边界，不再是执行队列。

固定64组Hessian编码只执行两个矩阵，48项旧UB1与8项完整教师原生逐位控制通过；组RTN逐位复现原全矩阵RTN。训练仿射误差降低约77%，但完整HiF4误差/UB1在step2为0.10298–0.12405、step3为0.24134–0.27171，八项均超过固定0.1门槛，仅3/8优于RTN。因此固定编码修复关闭，不搜索阻尼/actorder/组大小，不启动压缩。

R8F2_transfer_audit和R10_saved_failure_geometry只读保存数据，定位到训练行张成空间外误差增长；未运行完整总体采集。完整结果见R8F2_native_hessian_results和docs/60。

范围更正：原R2规定学习型BF16仿射骨干和原生非线性余项；R8F1/F2固定0.5+额外HiF4失败不关闭原设计。R8D1/R9A1均为未运行草案。没有新的候选质量、速度或50%压缩通过。
