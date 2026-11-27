# 来源身份与核查范围

| 来源 | 固定身份 | 本轮核查范围 | 用途 |
|---|---|---|---|
| FFNDS 新结果 | `a1eb64d63910b4d90b94ea4c641ef461c83d0d92`，job `206224` | 正式计时与冻结方法绑定；质量结论由同轮专项审计交叉核验 | 当前实际竞争点 |
| [SparsePR 论文](https://arxiv.org/html/2608.18484v2) | arXiv `2608.18484v2` | §4、§5.2、附录 A、C；在线重新打开全文 | 原理、H100/BF16、N、执行对比例、estimated 全生成时间 |
| [SparsePR 作者仓库](https://github.com/PardisTaghavi/SparsePR) | `main@e028c78e6b40a29cb9683a9f90f7d7fa422b2b0e` | 本轮实时 branch 与 root contents API；README 867 bytes，blob `d4c7b77a639c9542bf43392b56ce17636831ae69` | 确认代码发布状态；排除搜索摘要冲突 |
| [Chipmunk 论文](https://arxiv.org/html/2506.03275v1) | arXiv `2506.03275v1` | §3.3、§4.1、Table 4、B.4、D.1 | 长步端到端分母、视频 MLP 关闭、offload |
| [Chipmunk 作者源码](https://github.com/sandyresearch/chipmunk/tree/b5e8d1c3da6bddc7ca9007c48e91b82087c11490) | `master@b5e8d1c3da6bddc7ca9007c48e91b82087c11490` | 本轮实时验证 master 未变；复用 round14 精确 git-blob 快照，重新读取 attention、MLP、storage 和 Triton 配置 | 主要缓存字段、dense warmup、A100 适配路径、索引/offload |
| [ICC 作者源码](https://github.com/ccccczzy/icc/tree/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4) | `dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4` | 复用 round14 的线性分解、FFN、cache 固定源码审核 | 四字段布局代数；不声称此为最新 HEAD |

源码逐文件 blob 和 SHA256：[Chipmunk 源码快照身份](CHIPMUNK_SOURCE_SNAPSHOT.json)；ICC 等逐文件 blob：[ICC 等源码快照身份](ICC_AND_NEIGHBORS_SOURCE_SNAPSHOT.json)。本轮检索 ref 与固定版本保存在同目录 `SOURCE_IDENTITIES.json`。

两项证据处理：SparsePR 搜索摘要曾显示 src/config，但实时 API 只有 README，未采用摘要内容；Chipmunk 的旧 main 分支与默认 master 分叉，本轮使用默认 master。
