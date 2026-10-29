# 对独立初审 REVISE 的作者逐项回应

初审确切对象为 reviewer 记录的旧 draft `da8fd...`；真实结论 REVISE。以下修订后请求同一独立 reviewer 对最终协议和私有 spec 复核；没有执行或 ACTIVE 注册。

1. **硬中断的 hash 账**：接受 8MiB 不确定上界不足的指正。按既有 8MiB 读取块、每最多256MiB持久化，改记 `logical_bytes_read` 为最后持久化的已完成逻辑读取下界，并在未完成时给 `durable_unrecorded_read_bytes_max=264MiB`（256MiB 间隔加一个8MiB在途读取）；hash 完成后该上界为0。正常异常 finally 会保存更多进度，但 SIGKILL 无需依赖 finally，也不会把逻辑读取声称为物理磁盘字节。完整 payload/checkpoint SHA 和读取前后 stable stat 保留。合成测试覆盖255MiB附近潜在未持久化区间，以及文件身份改变时已经预扣、inflight 和读取进度留存。沿用已闭合旧producer只是绑定其历史证据，**没有修改**旧账。

2. **完整对象与最终启动身份**：在实际GPU0空闲前不启动。最终静态配置 `config/e0_chipmunk_paid_proxy_equal_k_20261029_pending_review_v1.json` SHA `0092f4e7f3fb246931f2535cf2b7bcb325ffaf9896bdefe841ac690628d98eb1`，fingerprint `c93c6d9a4f8563d8818d64f172d0280835cd388359ea9ead0f6af6ba287aa6ce`；私有 durable spec `.runtime/chipmunk_paid_proxy_equal_k_20261029/GPU_JOB.json` SHA `f8a0bb9f0d8d13d827dc7e7b807e5c24e0ef361933d1a45c58cbd8fd925c893c`。spec固定同一私有profile、既有GPU0 UUID合作锁、CPU0–3、项目 `.venv`、output/RUN 和1020s outer timeout；worker900s，scope GPU1、全局上限4由协调者负责，绝不继承旧2卡范围。代码、旧producer/kernel/tests/config、作者MLP和两份selector源码、parent payload元数据/GEN/RUN/EXIT及先前实际成功RUN/EXIT均绑定；`.venv` 中 torch2.10.0+cu128 / triton3.6.0 / safetensors0.8.0由RECORD和选定入口文件字节绑定，而非所有二进制的逐字节证明。supervisor 静态 validate PASS；无ACTIVE gate实际拒绝。启动时身份和资源仍须复核，host/output限制是边界及高水位检查，不是连续硬限制；无自动重试。

3. **选择与成本闭合**：保留实际 K、阈值、全H `>=` 截断ties数、random附加数、padding数、全部mask唯一索引及SHA；保存每个 BF16 proxy分数并在CPU closed verifier重放同固定seed的安全selector、三随机等K对照和固定集，核K/索引/SHA。非作者CURAND和atomic，padding真实进入更新。K0/H 保留并标注不辨识。计时分报初始Dense、每个payload load、checkpoint tensor load、旧mean、新mean、完整proxy W1+bias、分数复制与selector、所选历史均值更新、随机对照生成、每臂缓存/索引/两核/回散和独立参考；编译的单独纯耗时UNKNOWN，包含于冷单次核wall，不称热的端到端加速。closed 再检验七臂、空控制、全选中性、同mask参考和动态kernel入口数，物理launch/HBM UNKNOWN。合成测试增加 ties、样本边界、可变K0/H、未定义零分母/空集参考、重复/越界、硬中断账与未选坐标反例；10项测试通过，未执行数值GPU。

两项独立反例继续保留：token变化正负抵消可能低估逐token变化；两个W2有符号贡献相互抵消时更新其中一项可反而增大误差。任何结果仅是这个非raster历史source单组描述，不是作者完整Hopper选择器、自由student、Wan全M或视频质量证据。

修订 producer SHA `56f23845bbfd31f58a29de3cda5032c225b61850ae360194785b928852626701`、selector SHA `fa436ecc150bd287cc4c81d090f5b27c7d6caa4822001c8b26f13af5856855e1`、tests SHA `afe4588aa0a28fb4012d7f98a01793db492079f4d3d7ee58243b13ba74cfcc1e`。请独立 recheck 此确切 protocol/source/环境/spec。
