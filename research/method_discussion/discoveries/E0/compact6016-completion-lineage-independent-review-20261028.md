# A0 full80 completion：独立来源与部署链审阅

2026-10-28；审阅者 `completion_lineage_review`。**SRC＋合成CPU结论：本版本的40旧学生复用、120新学生分片、160学生混合来源assembler与loader链自洽，未发现阻断项。** 本审阅没有使用GPU、导入原模型、读取真实模型tensor、拟合新数据或重哈希40个大pack。配置仍为 `CPU_PREPARATION_DRAFT_NOT_REGISTERED_NOT_RUN`、`assets_finalized=false`；源码审阅不改变运行授权。

| 审阅对象 | 实际文件SHA256 | bytes |
| --- | --- | ---: |
| `tools/e0_compact6016_full80_completion.py` | `9b9de00fd43417136b542b5e4b0307c56b4b0b9129894368b852a5eee1cdc7ce` | 65095 |
| `tools/compact6016_completion_runtime.py` | `52b940823723c31c3c4d4506e0484eb362cafa7e9f0d6a834799ec92953fe102` | 5035 |
| `tests/test_e0_compact6016_full80_completion.py` | `639ffa59fc476edecbefd31662000ce7e126a5b83459a5ec493b0c4d2bd14709` | 27934 |
| `config/e0_compact6016_full80_completion_20261028_v1.json` | `fcd74e0a5a202c4d0281668a8a2ef4ed977e980cc747a0479e3f41c91d2663d0` | 158216 |

以上是实际工作区文件身份；当时Git HEAD为 `8d4d55517eac8add58db963ab56b7e89d450d830`，不能以HEAD代替未提交文件SHA。补充只读审阅了原pilot的绑定/dispatch/native-library核验、`audit_compact6016_partial_bank.py`、原full80 producer和 `compact6016_native_runtime.py` 的部署pack校验；未修改这些文件。

**来源和producer。** `validate_partial_reuse` 按path、size、SHA绑定原冻结协议，并要求training、initializations、source_split、sources、inputs、asset_view、random_supports、state_distribution、support/diagnostic合同与原实验相同。`run` 在第一项FFN计算前验证冻结repo_sources、原资产view/CPU producer/独立bias验证、15个原生shard和720个操作数的SHA及五tensor BF16 header。原资产producer的外层wrapper错误仍明确保留为 `UNRECOVERED_WRAPPER_ERROR_PRESERVED`；独立metadata/bias验证与每part实际大文件SHA/header门承担各自证据，不把该历史producer改写为恢复的exit0。

`expand_cases` 绑定24份原INPUT报告，核image SHA/source group/seed、160个step×block位置、expert/grid及所选256个真实唯一row坐标；每part生成24来源×15函数×2 step＝720个case。支持选择只读取train0–15的teacher V2，每函数32次source×step贡献；heldout16–23只进initial/final诊断。学生训练从原post-Q1 V1经过自身W1/GELU/Smooth2/native Q2；训练入口再次拒绝heldout和错expert step。所有24来源保持历史暴露身份，step和row未计为额外独立source。原随机支持沿全80函数顺序生成后取剩余函数，保证复用与新训练的seed42身份连续。

**40旧学生的真实小文件核验。** 从HF根 `/home/xiang/.cache/huggingface` 只读audit协议、MANIFEST、RUN、LAUNCH_EXIT及其绑定的原父RUN/RESULT/SUPPORT小JSON，并对pack做stat。`validate_partial_reuse` 实际返回40个cell、20个paired函数，producer_part计数12/12/16，且整个调用未导入Torch。audit的四个绑定均与配置匹配：MANIFEST `2b977dd3f681423b99abe8da3f1380699ff243acb19e4488f886a7f6f76642e5`，RUN `cee2fd7f206dcc6f378e04bd28ba53f69bc5bc11c4e435990f45295a9c64b033`，LAUNCH_EXIT `cda945a3801201f52ddbea91bd6bec290308c9571e490b7c182d6d9922928b66`，audit协议 `9a6df833a1237cfaf2ac43989b3b4236937e508ce00000c1790f2e032e008459`。RUN记录已审计5,047,017,920 bytes，LAUNCH_EXIT的RUN_sha256精确绑定该RUN且记录process_exited=true/returncode=0；该字节数来自已有真实audit，本审阅未重复哈希这些pack。

三个原父RUN实际均为 `FAILED_NO_RETRY`，各有12/12/16个完整cell。父RUN的functions列表另含已开始但未完成的函数，共7/7/9项；复用选择的是完整cell及其完整RESULT/export，而非函数列表长度或部分训练状态。代码要求准确的20函数×2init集合，逐cell验证256-update完成标签、8 epoch、48×2相位诊断、source均值、GPU gate/gradient证据、support和12字段export头；原父exit1、失败状态保持独立身份。复用路径不加载旧pack参数或optimizer。

**四新part到部署。** 其余60个函数固定分为4×15，每part两种初始化30个cell，共120个新run ID。`assemble_manifest` 要求四份按part0–3排列的完成RUN、精确协议SHA、预算预扣/实计、复用链、15函数分区及30个cell，并逐part要求真实process_exited/returncode0和launcher的精确RUN_sha256。最后按完整80函数×2init集合核160个唯一cell；重复/半init无法补足集合。返回的 `old_scope_status=FAILED_NO_RETRY_PRESERVED` 没有改变父scope结果。

completion `load_bank` 先重组并逐字段比较冻结manifest，再选择同init的80函数。读取时重新核完整export文件SHA和八个部署tensor的shape/dtype/逐tensor SHA，核实际原生cache权重/scale身份，并复用原部署pack校验核原bias、Smooth、sorted whole64 groups/channels。默认safetensors loader要求原12字段pack，但只读取八个部署字段；training_LoRA字段不进入部署。原Runtime/Residency类直接复用，无全局producer ID改写。metadata assembler的pack检查有意只做stat；完整文件hash与tensor装载属于loader的另一个验收门。

**实际CPU验证。** 使用项目 `.venv/bin/python`，completion测试21项exit0，既有native_runtime小尺寸CPU测试10项exit0。另做8组临时合成反例：对变更后的RUN同步重绑launcher RUN_sha256，对39-cell audit同步重绑manifest→RUN→exit，确认observed Adam=0、FAILED状态、错scope、重复/半init、新cell占用旧函数、错15函数分区、旧39cell及改source seed各自在预期语义门拒绝。两项loader检查分别确认改manifest被重组比较拒绝，以及同尺寸folded字节损坏被真实export文件SHA门拒绝；后一项只用小asset-view metadata shim定位该门，未调用tensor loader。所有合成测试结束时CUDA_initialized=false。

**非阻断的测试改进建议。** 当前正式测试中，改new RUN的observed或cells、以及删audit MANIFEST一个cell的负例，未同步相关launcher RUN_sha256，新增的wrong-RUN门会先拒绝这些输入。因此这些正式负例目前不能单独证明其名称所指的count/partition门。上述临时重绑反例已独立确认生产门正确；建议将同样的重绑动作保存在正式测试中，另保留专门的wrong-RUN负例。该建议已即时通知协调者。

下一有效证据是协调者完成最终文件冻结、原生运行登记和真实四part执行，再以其完成RUN及实际exit0组装资产。这里的SRC/CPU支持不代表120新学生已训练、160包已真实装载、视频质量通过或系统提速。M-B B0/B1的科研优先级保持原有身份。

**正式测试修正附注（同日）。** 经协调者委派，上述测试辨识度建议已落实，仅修改正式测试与本附注。`tests/test_e0_compact6016_full80_completion.py` 最终SHA256为 `17f788e7e6a1bcc26cf6833762a97ca567eae9c33ee5b8a1676adcb99fe4b1e3`，31,272 bytes；表中旧测试SHA保留初审版本身份。39-cell负例同步重绑MANIFEST→RUN→launcher，new RUN语义负例同步重绑launcher RUN_sha256；断言核对预期拒绝消息，计数和scope用例另直接核对退出收据绑定。新增正式重复/半初始化、错误15函数分区、FAILED/外来scope/错误part用例，并使原跨part用例实际覆盖旧函数与另一新part函数。专门wrong-RUN负例继续单独保留。最终24项CPU测试全部exit0（3.156秒），CUDA_initialized=false；生产代码、config及ACTIVE均未由本审阅者修改。
