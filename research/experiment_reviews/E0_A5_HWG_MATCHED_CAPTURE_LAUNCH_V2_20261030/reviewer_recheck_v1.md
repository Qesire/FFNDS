# 独立最终复核：source96 启动失败后的新有限范围

评议者 `/root/sync_abba_actual_review_terra`；作者 `/root`。**APPROVE 对精确 PART1_V2 采集与其条件 VERIFY 范围登记**，前提仍是实际资源/身份复核，以及producer真实EXIT0后才启动VERIFY。本review不修改ACTIVE、不启动模型、不声称数据或质量结果。

原 PART1_V1 supervisor实际退出收据 SHA256 `ca08577fc65217ad0c035e82510ecde0d5d53fcbb283e2d49970acf566ff1657` 已按协议绑定读取：`error_type=JSONDecodeError`、`child_pid=null`、`process_exited=false`、`child_returncode=null`、无自动重试。实际收据支持“未启动子进程”；不将旧范围称为EXIT0。当前登记记录保留其 `FAILED_SUPERVISOR_BOOTSTRAP_NO_CHILD_NO_MODEL_PRESERVED`，对应VERIFY为 `NOT_RUN_PARENT_START_FAILED`。作者说明的ACTIVE非原子写入竞争与该失败类型一致；我未将其作为实验测量或恢复旧范围。

新协议与已独立复核的 PART1_V1 对象逐键比较，除scope ID、fresh HF输出目录及实际no-child失败绑定外完全一致；source96、seed42、HiF4 fake-QDQ数值路径、21τ匹配M64、两条Dense视频、40×4位点、producer/VERIFY资源和失败不重试均未扩张。实际1018源码/扩展的size/SHA全部核对，capture owner/prepare/tests相对前次受审源码提交 `f1d7cdf25213b3ca06fbb894133dee4fd7d5a515` 无更改。按实际私有profile重构RUN与VERIFY全部argv、输出和超时并逐对象比较，两个job SHA与协议fingerprint记录于机器收据；公共文档不暴露worker身份。

原前次独立审查的科学反例、十状态/视频neutral、原生QDQ/全M V1边界、逐字段预算和独立文件验证要求继续适用。两视频的媒体解码及原生加载器内部物理读取量仍标UNKNOWN，不混入明确计量的SHA读取；当前是暴露开发源的被动采集，不能当E0/E1/E2方法、速度或质量证据。此次未重跑GPU/CPU科学实验，21项既有软件fixture是前次f1d7代码验证；producer、单独条件VERIFY及其实际RUN/EXIT仍须真实完成。登记更新须以完整原子替换实现，旧失败及另外三个已启动scope保持原身份。
