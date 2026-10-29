# 协调者自审：付费均值代理与等K更新

协调者 `/root`，实现者 `/root/sm120_full_quality_build_terra`，独立复审者 `/root/teal_probe_preexec_reviewer`，理论提出者 `/root/s2_theory_design_independent`。独立初审REVISE、修订和最终确切身份APPROVE已发生；本页确认有限注册责任，不代替审阅。

固定此前真实source898/high block20/step0→1/非raster128行组，比较作者启发的安全代理、三个同实际K随机集、前K固定集、全/空共七臂。H1预测合法的当前组均值pre-GELU信号优于随机与固定列；H2预测相反token变化在均值中抵消、GELU及W2符号交叉使其不优。当前完整Dense只在所有mask冻结后用于评估，不参与选集。

原作proxy使用 `fc1(mean(x)) - cached_mean(mid)`，包括付费W1+b1和历史pre-GELU缓存。前1024分数阈值、0.05随机接受和256补齐采用已明确的Python固定seed和安全升序padding适配，不声称作者CURAND/atomic/Hopper忠实复现。原作者源码和已闭合算子RUN/EXIT均绑定，未修改原两核。真实分数和mask保存并由闭合核验重放。

接受初审意见：强制终止时未持久化hash读取的保守上界为264 MiB；另以实际完成状态记录准确逻辑读取，物理读UNKNOWN。每臂包含独立同mask数值参考、当前gate和一次b2；数值界提前冻结。代理、选择、拷贝、缓存、kernel、部分回散和参考成本分别保存；一次含编译/账本的wall不外推系统速度或视频质量。

最终协议SHA `0092f4e7f3fb246931f2535cf2b7bcb325ffaf9896bdefe841ac690628d98eb1`、指纹 `c93c6d9a4f8563d8818d64f172d0280835cd388359ea9ead0f6af6ba287aa6ce`、私有spec `f8a0bb9f0d8d13d827dc7e7b807e5c24e0ef361933d1a45c58cbd8fd925c893c`，由真正独立reviewer复核。执行前将核对GPU0/CPU0–3、source/env/worker/spec及fresh output、确认全会话不超过4卡；该轻量单函数范围单卡24GiB reserved、host RSS64GiB边界检查、900/1020秒、HF512MiB、无训练/新视频，失败保留且无自动重试。
