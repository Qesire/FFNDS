# G20 R1 同输入筛查 · 证据与源码（job 156476）

[科学结果与决策](../../../research/20261112/G20_R1_SAME_INPUT_SCREEN_RESULTS_20261112.md) · [实验路线](../../../research/20261111/G20_NEXT_EXPERIMENT_ROUTE_AND_DECISION_TESTS_20261111.md) · [复审规则](../../../research/agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)

本目录保存 scope `G20_R1_SAME_INPUT_SCREEN_20261111`（SHA256 `4bbf8496484f80506bf59565e2b9c1556e19a908de0c8736643126724cf41625`）的机器可读结果、治理记录与冻结源码快照。实际运行：Slurm job 156476，anode16，A100-SXM4-80GB×1，EXIT 0，1059.6 s。大型 ridgeD BF16 矩阵（6×52,428,928 字节）与原始捕获保留在集群原位置，未重复上传；其 SHA256 见 [evidence/NPY_SHA256.txt](evidence/NPY_SHA256.txt)。

## 结果与身份

- [SUMMARY.json](SUMMARY.json)：80 站点廉价景观 + 6 昂贵站点三臂读出、omega/秩/残差诊断、触发判定、自检与 Chipmunk MISSING_ASSET 记录；SHA `46ba72860d19fa453bdda5d2eaedad0ad90ddbbb6a598484694d52f69962dd5d` 与 EXIT 绑定。
- [output/](output)：87 个逐站点 JSON（80 cheap + 6 expensive + SUMMARY）。
- [R0_ASSET_INVENTORY.md](R0_ASSET_INVENTORY.md)：R0 只读资产核查（可复用/缺字段表、gate 7 站点×9 来源逐位相等核验、跨步交集计数、环境）。
- [receipt_index.json](receipt_index.json)：job 156476 的 scope/EXIT SHA、资源、时间与原文件位置。
- [evidence/](evidence)：scope、独立审查（APPROVE/recheck APPROVE，真实不同 reviewer）、作者逐项回应、作者自审、阶段级 ACTIVE 执行注册（扁平结构，区别于 research/ACTIVE_RESEARCH.json 仓库注册）、RUN/EXIT/SHELL_EXIT、SUBMITTED/DISPATCH_INTENT/LOCAL_SUBMIT_INTENT、slurm out/err、NPY SHA 清单。
- [evidence/frozen_inputs/](evidence/frozen_inputs)：scope 冻结的三个输入清单（assets_g20、q1_matrix_manifest、icc_svd_manifest），逐字节 SHA 与 scope.frozen_inputs_sha256 一致；不含张量 payload。

## 冻结源码快照

[source/](source) 内 9 个文件为 scope.code_sha256 绑定的逐字节副本（复制后重新核 SHA）：guarded_runner.py（执行守卫）、runner_screen.py（两遍筛查）、fit_core.py（原生 FFN/BF16 回放/共享 Gram ridge/ICC 两残差）、selfcheck.py（运行内数值自检）、selector.py、runtime_guard.py（参考实现原样保留，含未启用的旧接口以保持来源身份）、job.sh、dispatch_once.py、build_assets.py。[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) 记录形状与哈希。`.gitattributes` 禁止 output/evidence/source 的换行转换。

源码供复核实际计算，不是脱离模型与数据即可运行的发布包；完整外部模型、捕获数组、LightX2V 与 torch 环境仍是运行依赖。未来运行需另建版本化有限范围并重新独立复审；历史成功收据不意味着集群中间数组仍然存在。

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) 记录本次发布全部文件的 SHA 与字节数。归档未上传任何模型权重、视频、光流或 ridgeD 矩阵 payload。