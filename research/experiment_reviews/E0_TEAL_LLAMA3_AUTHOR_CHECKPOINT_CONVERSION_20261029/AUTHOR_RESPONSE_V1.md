# 对独立初审 V1 的回应

初审为 **REVISE**。作者接受并修订：

1. **ACCEPT — 全 worker 超限失败账**：转换与验证各在执行前 `O_EXCL` 创建并 fsync 固定 65,536 字节 `EMERGENCY_FAILURE.reserve`，计入各自总预算。普通 RUN 的 atomic persist 达到软上限时拒绝新增写入；finally 不能正常持久化时对该已分配文件原位 `pwrite` 固定长度、fsync，记录失败身份、各 predebit/completion、hash进度、目录用量与监测结果。即使空间软预算已超仍无新增长。若底层 NFS 已不可写或 SIGKILL，中断写仍可能丢失，真实 supervisor EXIT 和目录事后核查作为独立证据，不声称该备用记录绝对成功。超额检查/备用写保持尺寸、锁冲突和已预扣未完成数据由合成测试覆盖。
2. **ACCEPT — 实际 staging 路径的反例**：提取生产用 `stage_pinned`，两 shard 微型合成测试通过同样的 O_NOFOLLOW+flock、`/proc/self/fd` symlink、实际原作者 converter、tokenizer copy、Q/K非平凡排列/QKV拼接及 BF16 验证；同测试另验证另一个文件描述符持排他锁时拒绝、固定 fd 被改时拒绝、路径替换时保留原打开对象。输入转前后整文件 SHA 及 stat 仍只约束非对抗、合作写者；独立 scope 启动前必须确认 closed 私有 cache 没有任何写者，并保留 uncooperative 短暂修改再恢复不可排除的范围，不将 advisory flock 说成硬只读。
3. 父资产获取与独立验证各自实际 EXIT0 及 RUN 都被绑定，两个转换/验证 scope 分别审核和授权；完整 dtype/key/shape 检查及 output 读取只在实际新 scope 获准时执行。原作者转换全量成功、LLM inference/quality均未被预先声称。
