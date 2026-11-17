# 阶段3执行接口与未闭合绑定

状态：代码候选，STATIC_ONLY；非实验运行记录。禁止将这份文档作为 ACTIVE 或独立复审通过的证据。

`stage3_assemble_inputs.py` 校验旧 V2 bank manifest、逐文件 sidecar/SHA 与新两源 capture manifest、四步 neutral 和逐文件 SHA，产生 `input_pairs`。`--mode dev` 固定两新源和九旧源、66对；`--mode cal` 固定原VBench CAL11、全部80站点，共880对。新源使用当前 capture 的坐标，不假设与旧坐标相同。参数 `--bindings`、`--bindings-sha256`、`--output` 均为必填；dev 还需 `--capture-manifest`、`--capture-manifest-sha256`、`--capture-summary`、`--capture-summary-sha256`。装配器只在全部校验成功后排他创建输出。

`icc_stage3_cli.py SCOPE_PATH SCOPE_SHA256` 不调度作业；须在审查通过、匹配 ACTIVE 且经 runtime_guard 证实的 A100 计算分配内调用。每次 scope 单独固定 `operation`：

- `icc_factor_build`：`mode=six_site_three_family` 限36次矩阵分解；后续另一个 `mode=selected_remaining_74` 绑定冻结 selection，限148次。每种CA/CD scales须来自真实 CAL11、四边界与经核验的统计产物。
- `icc_grid`：绑定完整66对DEV输入、18个六站点因子对、两checkpoint和Stage1 live-weight manifest，限制792格与≤7GiB临时缓存。每个canonical native P512端点只算一次；选族前保留198组预测h/y、66对native h/y，另保存约定的4个z0。
- `icc_decomposition`：独立scope绑定 grid `selection.json` 与 `cache_manifest.json` 的SHA，固定 norm floor；66格/132次B2 F64 action。
- `icc_numeric`：固定两个neutral源、两expert、layer19，保持BF16 native输入/anchor/因子值，仅将4S/2R缓存与增量算术提升FP32，TF32开关各一，共16格；明确记录设置，是否不同内核仍待实测。
- `fixed_comparators`：绑定共用的native缓存、合法Q1全80参数manifest、原B_CHIP_P64_INPUTS manifest；读取新两源full-U和历史9源64行，输出Q1/P64各66格。不会重复canonical Dense端点。
- `q1_fit80`：单独scope绑定CAL11 880对历史V2输入、原V4 full-M panel-Y、V4/BWS A0及PCG代码/参数、当前checkpoint。仅身份完全相同且已收敛的已有产物可抵扣；最多80次拟合。F32导出重载，在线明确cast BF16。

各 scope 还需绑定所属源码及其依赖SHA、Python/Torch、`runtime_guard.py`、`review_path`、`active_path`/`active_key`、输出路径、资源上限、checkpoint及数据源；scope和独立评审必须满足仓库执行状态机。分解与比较操作的 `output` 应与其具体 `decomposition_output`/`comparator_output` 同属已审查输出根。运行前要核查方法依赖来源和这些输出目录尚不存在。

**身份缺口**：所有必要 CAL11 CA/CD统计尚需完成并通过oracle；Q1 原producer全80合法参数尚未闭合，V4 full-M训练Y与BW local512 own-W训练Y必须分开；B_CHIP_P64_INPUTS清单列全DEV11×六站点；协调代理报告9源108份文件的字节数和SHA已核验，仍须将该实际收据与评分scope绑定；作者Chip true-M 24格、Q1/GP64的全M执行仍需独立实现/复审；16格精度诊断已有候选代码，尚未运行或复审。以上代码没有通过模型或GPU运行验收。


