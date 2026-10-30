# 来源1040全 M E0 V1 独立复核

评议者 `/root/sync_abba_actual_review_terra`；作者 `/root`。本精确范围为历史暴露校准源1040的80个同 expert 层对、M=31668、K=5120、58臂付费 teacher-state 比较；它不产生新视频，不包含 E1/E2，也不将 Dense FP32 masked W2 说成实际 native skip。A5-inspired适配不冒充缺失参数 producer 的原A5。来源1040用于CAL1拟合，它的比较结果是校准源描述，不能充当盲确认。

已核查父采集和独立 VERIFY 的实际 EXIT0、342文件/160 site及原 manifest 对160份metadata的精确引用；已核查 CAL1_V3 实际 EXIT0、80对完成、开发 tensor 读取0、其RUN/EXIT与支持artifact的 SHA 和schema绑定。当前比较必须先验证已冻结 CAL1 artifact 才可读取本来源 tensor；未来开发源须独立审核自己的父链和scope。

重新核对本协议全部1035源码大小与SHA、140项namespace、58臂、资源/调用上限、seed、bias checkpoint与原模型view的绑定、完整私有 argv/worker/profile 和fresh输出。cache SHA 读取56,223,524,243字节、两个原checkpoint SHA读取合计57,155,831,760字节；再加采集张量及其他预扣读取须受256GiB总显式读取上限约束。实际原生导入来源在首次 replay 前及结束时验证，V1/Z/H/Q2输入/V2/Y 每状态1344采样行逐字节检查；较大M的算法差异可能导致失败，不得热修阈值。

预算单GPU、4 CPU、76GiB reserved、96GiB RSS、host 128GiB余量及64GiB forecast、单host至多两个模型、全会话最多4GPU、HF新增32GiB、6小时加120秒外层；实际共享设备、锁和host剩余量在启动时重新检查，失败不自动重试。57个mask候选W2及full-delta W2均为已收费FP32 Dense诊断，其event区间和host wall不构成kernel活跃时间或系统速度证据；物理occupancy按还原原坐标统计。实测峰值/耗时与完整逐bit通过都仍待本任务实际运行。

已独立复跑native replay、discovery contract和CPU summary 13项软件fixture通过；不证明真实资产或数值结果。未来三源汇总、898/979开发比较各需实际前置闭合及精确独立review；来源96实际失败不纳入数值分母。

**APPROVE**：本来源1040 V1 可进入登记。冻结提交 `e6fa1b9eb3424d3d8f3cc8ce6d874731c4b89df5` 中协议blob与精确私有job绑定SHA一致；登记和启动前仍须核验真实HEAD、ACTIVE和实时资源。
