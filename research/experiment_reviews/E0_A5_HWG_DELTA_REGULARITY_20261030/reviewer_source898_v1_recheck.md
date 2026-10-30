# 来源898全 M E0 V1 独立复核

评议者 `/root/sync_abba_actual_review_terra`；作者 `/root`。范围为历史暴露开发来源898、80个同expert层对、M31668、K5120及58臂付费teacher-state比较，不是盲确认、native skip或新视频质量实验。原A5参数producer缺失，新adapter不称原方法复现。来源96的实际失败单列；1040 V1的10对失败数据不得混入1040 V2。

本来源producer和独立VERIFY的实际RUN/EXIT、protocol/job/RUN SHA绑定、实际EXIT0和无timeout已复核；VERIFY记录342个文件和160匹配位置，producer绑定的manifest逐site与输入metadata引用一致。CAL1来源1040实际EXIT0、80对、开发tensor读取0、support artifact及schema SHA均已复核；在本任务读取开发tensor前仍须实际验证该冻结artifact。

逐键对比已审1040范围，科学输入/方案、58臂、资源和调用上限一致，差异限于开发来源、scope、worker/CPU和新输出路径。1035项源码的实际大小和SHA逐项核对；完整私有job的argv已依worker配置重构，精确CPU、设备索引/UUID、worker profile/asset locator、超时和fresh输出均核对。实际私有主机和设备标识不公开。

每源单GPU/4CPU、76GiB累计reserved、96GiB RSS、128GiB host可用余量、64GiB/model forecast、单host最多2模型、session最多4GPU、256GiB显式输入读、32GiB HF新输出、6小时加120秒外层，不自动重试；实时UUID/锁/foreign-compute及host预留仍须在启动时核查。每个原生状态的1344实际采样行须对V1/Z/H/Q2输入/V2/Y逐字节一致；若失败应保留实际失败及费用，不热改或放宽门限。全M数据是离线付费Dense诊断，软件测试不证明真实峰值、时间或视频效果。

**APPROVE**：精确来源898 V1范围可进入登记。实际冻结提交 `c9c5fae734db7e46eab7482d6cb98eb83672b769` 中对应协议 blob 与精确job绑定SHA一致；启动前仍须核验实时HEAD、ACTIVE及资源。
