# 三来源 E0 收尾汇总 V1 独立复核

评议者 `/root/sync_abba_actual_review_terra`；作者 `/root`。本范围为0 GPU、单CPU读取已退出且SHA冻结的三份E0结果元数据，并独立顺序核验240个mask/proxy文件的整文件及typed raw SHA；不运行模型、CUDA或新tensor数值实验。每个来源先对80个site等权，再对三个来源等权；1040为CAL1校准来源，898/979为历史暴露开发来源，均不是盲确认，层对和token不冒充独立source。

已逐一复核三份父E0的真实RUN/EXIT、实际EXIT0、无timeout及protocol/job/RUN SHA，校验相同cache/recipe/seed/CAL1绑定及各80个唯一双expert全层site。另只读复核240份结果元数据及其SHA：每site包含精确58臂、两状态六边界实际采样逐byte检查通过；逐份核调用记录SHA、所有类别完整计数和ordinal无重复。240个mask文件按元数据声明共54,704,331,360字节，小于96GiB读取上限；本独立评审未读取这些raw payload，实际whole/raw SHA须由本次获准汇总独立完成。

来源96实际采集EXIT1、零phase，保持未测不纳入数值平均；1040 V1因foreign compute失败的10对不混入V2完整80对。两项独立resource预读helper实际均EXIT1，记录`read exceeded predebit`及inflight UNKNOWN；helper返回tensor payload字节0不等于本次失败metadata真实读取数或内核物理I/O为0。两个helper未改变父E0科学代码或数值输入，三个最终父结果均实际EXIT0；本汇总只读其已关闭的固定RUN，仍逐文件核SHA。

冻结protocol、完整私有CPU1/空CUDA job、locator/profile/fresh输出、1039项源码大小及SHA、全部预算和共同静态支持已核。资源为RSS≤512MiB、显式读≤96GiB、新HF≤256MiB、输出单文件≤128MiB、7200秒（外层7320秒）、失败不自动重试；实际内存及时间待本次运行。事件区间与host wall不充当kernel active time或公平系统加速；没有E1 native skip或E2候选视频质量结论。

**APPROVE**：精确 CORE3 汇总可进入登记。冻结提交 `c04ae4a26f0c3529bf8a9b09f013ac0d6fdab22a` 的protocol blob、作业SHA与fingerprint一致。登记/启动前仍须核真实HEAD、ACTIVE和资源；本复核未启动实际汇总。
