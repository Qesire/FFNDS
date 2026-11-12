# G20 R1b 数值复核 · 证据与源码（job 157314）

[科学结果与裁决](../../../research/20261112/G20_R1B_NUMERICAL_RECHECK_RESULTS_20261112.md) · [R1 结果（已版本化更正）](../../../research/20261112/G20_R1_SAME_INPUT_SCREEN_RESULTS_20261112.md) · [实验路线](../../../research/20261111/G20_NEXT_EXPERIMENT_ROUTE_AND_DECISION_TESTS_20261111.md) · [复审规则](../../../research/agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)

本目录保存 scope `G20_R1B_NUMERICAL_RECHECK_20261112`（SHA256 `e03d1748ef988181666a5350b40053d0eb73f0605fadf22a70c00aba8a783561`）的机器可读结果、治理记录与冻结源码快照。实际运行：Slurm job 157314，anode16，A100-SXM4-80GB×1，EXIT 0，2097.9 s，峰值 RSS 1.59 GB。任务为对已发布 R1 六站点昂贵拟合的**纯数值复核**：稳定 F64 样本空间精确解重解同一注册目标（两 λ），逐字重算原 FP32 解做 SHA 身份门，ICC 两残差 F64 分解；0 次新生成/视频/基准。裁决：**H_overfit 成立（6/6 站点），H_artifact 被否定**——认证精确解仍使 DEV5 退化 +15%–28%，原 D 子空间外能量仅 0.42%–0.86%。24 张 BF16 矩阵（各 52,428,928 字节，~1.2 GiB）与原始捕获保留集群原位置，未上传；SHA256 见 [evidence/NPY_SHA256.txt](evidence/NPY_SHA256.txt)（计算节点上哈希）。

## 结果与身份

- [output/SUMMARY.json](output/SUMMARY.json)：六站点裁决表、身份门、证书、ICC 份额与预登记规则读出；SHA `ef35bcd695578cee0343165daacb02b03e026969d6d80856395f55eef40dea9a` 与 EXIT 绑定。
- [output/](output)：6 个逐站点 JSON（expensive_recheck_{high,low}_{04,19,34}），含 10 臂逐源 SSE、omega_stable/omega_r1_reference、ICC 分解、身份比较、矩阵 SHA、峰值显存。
- [evidence/](evidence)：scope、独立审查（首轮 REQUEST_CHANGES → 修订 → recheck APPROVE/APPROVE，真实不同 reviewer）、作者逐项回应、作者自审（含修订历史）、阶段级 ACTIVE 执行注册（扁平结构，区别于 research/ACTIVE_RESEARCH.json 仓库注册）、RUN/EXIT/SHELL_EXIT、DISPATCH_INTENT/SUBMITTED、slurm out/err、NPY SHA 清单。
- [evidence/frozen_inputs/](evidence/frozen_inputs)：scope 冻结的 9 个输入（assets_g20、q1_matrix_manifest、icc_svd_manifest 三清单 + 6 个 R1 参考收据 reference_r1/），逐字节 SHA 与 scope.frozen_inputs_sha256 一致；6 个 R1 参考收据与已发布 [R1 output](../g20_r1_screen/output) 对应文件逐字节相同。不含张量 payload。

## 冻结源码快照

[source/](source) 内 8 个文件为 scope.code_sha256 绑定的逐字节副本（复制后重新核 SHA，全部一致）：guarded_runner.py（执行守卫）、runner_recheck.py（身份门+稳定重解+ICC 分解）、fit_core_v2.py（F64 加权薄 SVD 精确解 + F64 eigh 交叉验证 + calgram 基）、selfcheck_v2.py（修订后病态测试：F64 构造、FP32 舍入副本喂原路径）、selector.py、runtime_guard.py、job.sh、dispatch_once.py。[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) 记录字节数与哈希。本目录 `.gitattributes` 禁止 output/evidence/source 的换行转换。

源码供复核实际计算，不是脱离模型与数据即可运行的发布包；完整外部模型、捕获数组、LightX2V 与 torch 环境仍是运行依赖。未来运行需另建版本化有限范围并重新独立复审；历史成功收据不意味着集群中间数组仍然存在。

## 治理要点

首轮独立审查 REQUEST_CHANGES（1 阻塞：selfcheck 病态测试 FP32 生成的 X 数值秩 12≠数学秩 6，会在任何站点前误触发秩断言 abort）→ 作者按 reviewer 首选方案修订并恢复 checkpoint SHA 守卫、对齐 omega 基判据 → reviewer 用纯 Node F64 独立复算同一自检电池（8 种子）→ recheck APPROVE/APPROVE → 新指纹重新冻结（旧指纹 `c7ffb052…` 未复用）→ 计算节点 preflight + 发射前逐文件 SHA 复核 → 单次提交。运行内自检 emitted dict 与 reviewer 独立复算一致（stable_D_rank=6，probe 3.89e-9，cross 5.49e-9，oosFull 1.32e-14，原 FP32 泄漏 0.9797）。

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) 记录本次发布全部文件的 SHA 与字节数。归档未上传任何模型权重、视频、光流或矩阵 payload。