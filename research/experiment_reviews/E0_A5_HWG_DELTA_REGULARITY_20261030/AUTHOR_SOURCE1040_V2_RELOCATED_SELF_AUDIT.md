# AUTHOR_SELF_AUDIT · 来源1040资源迁移 V2

实际作者 `/root`。用户仍要求完成核心比较后收尾；本次提出新的有限scope，不自动重启旧job，也不更改原生数值门。

旧SOURCE1040_V1实际EXIT1，GPU监控发现foreign compute，已完成10/80对；最终RUN SHA `b553588d2f9d60c117a11e7eb447ad66e6452c276b076f55908080ec8b150784`，EXIT SHA `28a1e287bba1f9e4deb014f775d88d0824daea5ae983a8bc1a5098a083fa6df3`。ACTIVE已关闭并释放额度。部分结果及全部费用保留，不进入新80对的平均。

新scope `E0_A5_HWG_FULLM_DISCOVERY_20261030_SOURCE1040_V2_RELOCATED`：protocol SHA `0bf693680daa186c48c584c370a65032d5502eb497f15cd2a17d9071256e5ca7`，job SHA `e19a8c367f5d2e759221c13f700223d7647931574aa8070dd980ae287adea276`。新私有设备及CPU8–11已M0核查：A100 80GB PCIe、4卡当时均空闲、host可用约497GiB、CPU包含所分配4核；启动时仍用原UUID锁/即时空闲/host reserve/foreign process检查，M0快照不能代替运行时检查。

CAL1、采集及独立VERIFY链、source1040校准身份、seed42、全80对×M31668×58臂、K5120和原数值入口全部不变。既有1035源码逐项SHA重核不变，只添加一个metadata-only迁移preparer绑定；该工具仅产生新协议/私有job，不批准或启动实验。新输出为独立attempt2_relocated，旧路径不覆盖。

预算维持单GPU/4CPU、reserved≤76GiB、RSS≤96GiB、显式读取≤256GiB、新HF≤32GiB、6h+120s外层、每host≤2模型和128GiB reserve、会话≤4卡，失败不自动重试。80对从头完整重做；六边界逐bit检查和全部paid Dense诊断/调用账本不变。不训练、不生成候选视频、不做E1/E2、不把离线付费运算称为native省算。

请真实不同评议者独立核对本冻结commit、精确job/协议/worker、旧失败闭合、原1035源码不变及新preparer实际argv、fresh output；本文件不是批准。
