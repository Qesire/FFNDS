# 作者回应质量初审：仍待独立复核

1. 各消费者新增 `e0_sharq_wan_metric_read_budget.HashReads`，每次原base SHA真实文件读取先在持久journal预扣已知字节和路径SHA、成功后记完成；失败保留inflight UNKNOWN、未调用不计为完成。每phase独立16GiB读取上限、journal 16MiB；scope另明确普通配置、child元数据和未被SHA入口观测的模型内部读取为UNKNOWN。视觉`inputs`全部文件、三个视频第二次、JPEG及manifest等所有base SHA逐次入账。实际视频数、独立state、代码/源文件绑定来自父闭合后协议，不预写未来字节。
2. 原八轴/连续RAFT preflight、GPU执行的全部base SHA以及snapshot helper的独立base SHA都接入上述journal，包含all_bindings逐文件初次哈希、verify_lineage三个视频、构建expected的三个视频、八轴I2V image两次hash以及模型loader审计中base.bound和source snapshot真实复验；多次读取逐次预扣，未缓存未宣称一次。文件大小只在操作前stat获得，长度增大时硬中止。已用小文件真实成功、中断恢复、超限拒绝测试。
3. 新 `e0_sharq_wan_metric_verify.py` 为每个quality/raft/visual单独独立ACTIVE CPU VERIFY scope及exact private supervisor spec，先验父评分进程实际EXIT0，再对旧完整核验函数实施同样的逐文件SHA预扣；核验本身保存RUN、真实supervisor EXIT0后才可调用只读 `inspect` 判闭合。直接调用旧的SharQ wrapper verify-closed会拒绝，没有沿用旧TEAL注册授权。CPU verifier上限1线程/1800s/16GiB摘要/32MiB新输出、单attempt；对原作者metric的真实调用和计算账不更改。

已完成隔离CLI与16项合成测试；父GPU与父独立VERIFY尚未实际闭合，inputs SHA/预算明细和scope spec尚不可冻结。请先核源码修订，再在真实父闭合后执行exact recheck。持续保留两个source×三个臂、八轴与独立连续RAFT重复调用，16fps stride2为0→2,…,78→80共40对/视频，不是80个相邻原帧对。

追加静态复核：worker顶层journal与四个supervisor/child日志、receipt单列64MiB额外软预算，每个SHA事件采样；journal自身16MiB，原results预算不变，监测不是连续峰值或硬quota。冻结时 `known_hash_work` 根据真实父绑定列出去重输入的确切文件数/字节和必然重复视频/I2V图像的哈希调用次数/字节；source snapshot、其余模型source的SHA按真实journal记录。CPU输入证明与三种scope的已成功journal在独立verifier中重新验证；child小metadata与非包装alias/直接native读仍UNKNOWN。
