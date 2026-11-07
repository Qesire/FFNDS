# G19 本轮轻量证据与源码

[完整方法、推导和结果](../../../research/20261107/G19_METHOD_DERIVATION_AND_RESULTS_20261107.md) · [新对话接续](../../../research/20261107/G19_NEXT_CHAT_20261107.md)

本目录保存完成实验的机器可读结果、来源级统计、身份索引和实际运行源码。大型权重、A数组、RAFT场、原始MP4及完整日志保留在原位置，未重复上传。原始视频已下载本机E盘；本仓库的路径索引不是媒体文件本身。

## 结果与身份

- [quality_per_video.json](quality_per_video.json)：六臂×八维×七来源×两seed，共672个原始分数。
- [quality_summary.json](quality_summary.json)：每来源均值、源等权总均值、配对差值和改善来源数；包含原输入文件哈希。
- [motion_summary.json](motion_summary.json)：r1/r2逐来源、逐视频的全场β、模长比、余弦及正交残差；40个两帧间隔的top-5%时间序列与原始场索引。完整场及CoTracker坐标/共同可见性详细记录见接续文档中的本机`motion_and_tracks.json`与服务器索引。
- [engineering_summary.json](engineering_summary.json)：两个执行对照的每次耗时、输出相等性、setup、内存和逻辑GEMM计数。两个节点只作各自内部配对，不能横向拼接时间。
- [fit_summary.json](fit_summary.json)：80处逐层旧/新选择、真实方程残差及目标变化。
- [receipt_index.json](receipt_index.json)：8个实际成功作业的scope/EXIT SHA、job ID及原文件位置。
- [INDEPENDENT_RESULT_REVIEW.json](INDEPENDENT_RESULT_REVIEW.json)：实际不同agent的最终数值复核；不等同于独立盲测或综合质量验收。
- [Q1矩阵清单](Q1_matrix_manifest.json) / [Q2矩阵清单](Q2_matrix_manifest.json)：80张BF16矩阵的形状、路径、哈希与构造身份；不含矩阵payload。
- [local_video_index.json](local_video_index.json)：28段精修版原始MP4的本机路径及哈希。
- [evidence](evidence)：本轮8个冻结scope和真实独立review。历史scope中的授权状态保留原样；实际完成状态以receipt_index为准，不能重复提交。

## 实际源码快照

- [fit_core.py](source/fit_high_r2/fit_core.py)：输出感知目标、隐式正规方程、预条件CG和秩1校正。
- [runner_fit.py](source/fit_high_r2/runner_fit.py)：真实配对行、旧/新BF16目标选择、FP64折叠与回放。
- [direct.py](source/q1_r2/direct.py)、[fused_post.py](source/q1_r2/fused_post.py)、[selective.py](source/q1_r2/selective.py)：默认全融合执行、BF16舍入与命中预取省略。
- [packed.py](source/packed_r1/packed.py)、[runner_panel.py](source/packed_r1/runner_panel.py)：W1/A宽GEMM及有限配对计时。
- [省W1预取实现](source/skip_w1_r1/selective.py)、[对应runner](source/skip_w1_r1/runner_panel.py)：首buffer与后续buffer的W1/W2独立有效性标签。
- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)：37个源码文件均为实际scope绑定的逐字节副本，保留SHA。目录内`.gitattributes`禁止源码/冻结证据的换行转换。

源码供复核实际计算，不是脱离模型与数据即可运行的发布包。完整外部模型、LightX2V、Triton、指标环境、原输入图片、捕获数组和scope输入仍是运行依赖；没有自动启动脚本。未来运行需另建版本化的有限范围，并核对当前用户指令、资源和实际资产。历史成功收据不意味着本机被清理过的中间数组仍然存在。

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)记录本次发布文件的SHA与字节数。归档没有上传任何模型权重、视频或完整光流数组。
