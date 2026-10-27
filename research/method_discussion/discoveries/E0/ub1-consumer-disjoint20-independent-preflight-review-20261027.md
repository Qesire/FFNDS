# UB1 consumer disjoint20 原生驱动：独立冻结前审阅

**结论（STATIC/CPU-MOCK）**：交审版本的配对数据、原生噪声与失败账本、全160位置检查和单管线两臂驻留设计彼此一致；未发现需要改代码的确定阻断。当前协议仍是 `assets_finalized=false` 草案，不能以本次审阅本身启动。协调者需冻结最终协议、登记其精确 SHA 与40个 run_id，并在实际目标环境完成资产/版本/资源只读 preflight；实际原生生成和质量仍待对应结果。

审阅版本：driver SHA256 `8a5fa265fe585e944bbace3b7f351a43225dd8b1adf2489eaac1be14abf40748`；tests SHA256 `aa457d8170598ae960c38aaf97196b1dbe8b32e911916346fe49346864d04ae2`；未finalize协议 SHA256 `737e8df8fd6d514825460eba7dae37c1c92bf291b97a331400268bb2f58c5852`。41项仓库源码绑定逐项匹配。独立执行9项 CPU 软件测试全部通过；包含真 safetensor 原dtype及正/负零字节差拒绝、首次调度更新前噪声失配中止、失败前预扣不退还、CPU原字典异常恢复、完整10视频mock预算、原生计数/160位置和 staging.collect 接口。它们不代替真正 GPU 或视频运行。

每part固定5个来源顺序执行 fresh Dense→candidate，四part总20 source×两臂=40个真实注册run；每part10视频、40个post-step latent、10个实际初始状态、40次完整flow，按任务前预扣，失败保留部分记录并禁止自动重试。每part 768 MiB、总3 GiB上限和8 CPU/单卡限制已写进固定配置。训练24与formal20的现有实际图像 SHA 零交集由绑定表和恢复记录校验；历史 disjoint20 已用于方法选择/评测，角色为 DEVELOPMENT，不是未经触碰的确认集。图像 SHA 不自动证明不存在近重复语义来源。

调度器 `prepare_latents` 创建原生初始状态，`step_post` 首次更新之前包装器按原dtype保存，并在 candidate 首次更新前与 fresh Dense 文件逐字节比较；源码和单测支持此配对入口，实际实现仍以每次生成的文件和receipt为证。Dense/候选持有同一CPU records原对象，candidate浅引用筛出非FFN记录，按臂预加载、退出释放且核验无残留后再切换；测试覆盖异常时原引用恢复。registered receipt 复用新 cohort/stage 格式，实际 pipeline 的正确性需由每条完成的运行记录及收集检查证实。

先前两来源 neutral 的已完成精确复现通过源码绑定、四latent与视频SHA/计数校验；它们是此前控制，此scope不新增neutral。原版上游runner并未实现新candidate，因此本驱动借用审计与staging合同并走项目已有candidate运行器；不能把该兼容性称成上游原版candidate执行。原生视频质量、独立来源确认、完整官方九维和真实性能仍各自待证。
