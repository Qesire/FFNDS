# 来源898 E0 资源预读 V2 独立复核

评议者 `/root/sync_abba_actual_review_terra`；作者 `/root`。本范围仅为运行中的来源898 E0 提供有界 CPU 文件预读建议，不运行模型、数值候选或CUDA，不向父进程发信号。对真实 tensor payload 不执行 read/mmap；内核因 `POSIX_FADV_WILLNEED` 产生的实际物理 I/O 明确为 UNKNOWN。父任务原完整file SHA、raw tensor SHA和原生逐bit门保持不变。

已核该来源原采集及独立VERIFY实际 EXIT0、160site原manifest，并核父 E0 protocol/SHA、当前RUN身份与新scope绑定。精确新protocol/job/fingerprint、worker/单CPU/空CUDA、私有locator/profile、fresh输出，以及所有继承和新增源码的大小与SHA均复核；不公开私有主机/设备标识。

单scope为0GPU、1CPU、RSS及显式metadata读各≤256MiB、新HF≤4MiB、wall≤7200秒（外层7320秒），文件预扣≤96GiB、至多25000次4MiB建议，最多提前两个pair；每次hint前检查父EXIT与128GiB可用内存。每文件预扣日志在hint前fsync，打开后先核regular和inode等身份，O_NOFOLLOW，事后再核；遇到父退出正常结束，低内存的partial不重试。进程自身返回payload字节为0与内核物理I/O未知是不同口径。四项软件fixture独立通过，含中途EXIT/内存下降、打开inode竞态与有限range。

**APPROVE**：精确资源预读scope可进入登记；提交 `ce2eb638ecf170a6ce0b708420154920b4b87ff4` 的对应协议blob、私有job及review fingerprint已核对。实际启动时仍应检查HEAD、ACTIVE、父进程与宿主资源；不把提示或并发时长差当成可归因系统加速。
