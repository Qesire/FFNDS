# 严格复现：审计、设计与本轮检查材料

当前实施要求以[完整复现与重做](../STRICT_REPRODUCTION_AND_REDO_20261116.md)为准；本目录不建立第二进度队列。实际状态继续写在现有G20资产/判断台账。

| 文件 | 内容与证据范围 |
| --- | --- |
| [CHIPMUNK_UPSTREAM_AUDIT.md](CHIPMUNK_UPSTREAM_AUDIT.md) | 固定作者commit的Wan调用链、官方Ampere/Triton、MLP机制与必要接口适配；未运行GPU |
| [UPSTREAM_WAN_EXECUTION_SOURCES.json](UPSTREAM_WAN_EXECUTION_SOURCES.json) | 39份原文的固定URL/Git blob/SHA256，branch/release/issue来源；作者代码不重复vendor |
| [ICC_STRICT_REFERENCE.md](ICC_STRICT_REFERENCE.md) | 四状态、SVD/CA/CD、四边界统计、精度与缓存实施规格 |
| [CONTROLLED_REPRODUCTION_PLAN.md](CONTROLLED_REPRODUCTION_PLAN.md) | 单一数据/候选/消融/生成预算；897次D4为所有阶段提案上界，不是已执行数量 |
| [COHORT_PROPOSAL.json](COHORT_PROPOSAL.json) | CAL11/22/64/256、DEV11、EVAL22元数据候选；图像实体/近重复簇和历史曝光尚待绑定 |
| [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md) | 真实独立审阅、初稿修订与最终回核；设计意见不是运行验收或ACTIVE |
| [audit_chipmunk_counter.py](audit_chipmunk_counter.py) | 对固定作者LayerCounter的无torch/CUDA接口轨迹检查 |
| [UPSTREAM_COUNTER_TRACE.json](UPSTREAM_COUNTER_TRACE.json) | 实际纯Python检查输出，包含调用坐标错配，不包含模型结果 |
| [COUNTER_SOURCE_IDENTITY.md](COUNTER_SOURCE_IDENTITY.md) | 初稿旧工作副本末尾多换行的SHA差异、修订与复跑记录 |
| [PUBLICATION_CHECKS.json](PUBLICATION_CHECKS.json) | 本次文档/元数据完整性检查；不表示原方法已复现或全局研究图已验证 |

复跑计数检查须从固定上游commit取`src/chipmunk/util/layer_counter.py`原文字节，再执行：

```bash
python audit_chipmunk_counter.py --source /path/to/chipmunk/src/chipmunk/util/layer_counter.py --output counter_trace.json
```

脚本严格核原文SHA256；若编辑器改变末尾换行，应恢复原字节，不放宽校验。当前模型/GPU新增运行0；输入实体、端口producer、编译环境、资源与实际运行范围均未据本轮文档自动得到验证。
