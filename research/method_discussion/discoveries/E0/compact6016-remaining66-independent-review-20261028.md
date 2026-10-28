# A0 remaining66：独立动态预算与混合来源审阅

2026-10-28；审阅者 `completion_lineage_review`。**SRC＋合成CPU结论：审阅版本的94旧cell复用、66新cell分配、动态预算及160集合assembler符合当前合同。初审发现的driver缺少CPU preflight强制门已由协调者修复并复核；本版本未发现阻断训练与metadata assembly的问题。** 本审阅未使用GPU、导入Torch或真实模型、装载tensor、拟合数据或重哈希大pack；未修改生产代码、config、ACTIVE、图或queue。

| 审阅对象 | 实际SHA256 | bytes |
| --- | --- | ---: |
| `tools/e0_compact6016_remaining66.py` | `779ee3bff085a710c32fde4c7017202dd99582bceab664c4b8b00b1b12ead2d5` | 63604 |
| `tools/compact6016_remaining_contract.py` | `8cf5eef390d47437f8fd7a1e8f42d0a73a68e28d224399162e972438d35f542e` | 5942 |
| `tests/test_e0_compact6016_remaining66.py` | `0be515f998f97036e22e2803efe09f0d15d6309f9b1814a8bb869483b38b9856` | 5415 |
| `tests/test_compact6016_remaining_contract.py` | `d64ccd10fd033f097dc20b2339e9c7e21a9028b8738323545aec048cc6d97c47` | 4168 |
| `config/e0_compact6016_remaining66_20261028_draft.json` | `f052bd227e35fd6c6afe874b3d318ce3c001ff261bd5ce8b70b89d83d11aa266` | 150524 |
| `tools/audit_compact6016_completion_partial.py` | `01f1b99084cf65d2a2b2f9abe1c0642d113bec055f26680b8639228bc9a3367e` | 12544 |

配置在本次读取时仍为 `DRAFT_NOT_AUTHORIZED`、`assets_finalized=false`、新audit四项绑定 `NOT_BOUND`。实际来源链检查只在内存副本填入真实小文件binding，未改变草案。最终冻结的配置SHA和真实worker preflight/执行收据由协调者另行登记。

**原算法与缺失初始化。** 新driver从 `selected = groups if init ...` 到每函数释放原权重的完整训练body，与冻结completion代码逐字相同；全80 expert/block顺序的seed42随机支持生成函数亦逐字相同。仍从原native QDQ权重切片与rank32 seed42因素开始，8 epoch×16训练source×2 step＝256 Adam更新；bias/Smooth不训练，student自身W1/GELU/Q2，不读取部分optimizer或训练参数。支持选择仍用原train16的teacher V2、每函数32次贡献/864 FP32 BMM；heldout8仅作已暴露诊断。

contract将原40和新父54组成94个准确key；新父54为14/14/13/13个cell。low block16/31只有energy94完成，`initializations_for`仅返回其random94，support预扣使用该函数真正可授权的首个缺失run ID；这两个energy94不会重训。其余32函数各两初始化，共34函数/66新cell。新ID和复用集合无交集，组合覆盖完整80函数×2init。

| part | 函数 | 新cell | Adam更新 | support FP32 BMM | 原操作数case |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 9 | 18 | 4608 | 7776 | 432 |
| 1 | 9 | 17 | 4352 | 7776 | 432 |
| 2 | 8 | 15 | 3840 | 6912 | 384 |
| 3 | 8 | 16 | 4096 | 6912 | 384 |
| 合计 | 34 | 66 | 16896 | 29376 | 1632 |

**动态账本。** remaining driver无遗留的全局 `PART_LIMITS` 引用；contract以原每cell系数和每function支持系数推导不同part预算。Ledger预扣/observe、part完成比较、adapter/backend Q2增量、assembler预算检查均读取 `part_limits(part)`。总计23,364次student forward/native Q2、16,962次backward（含每cell一次GPU gate）、17,028次merge、6,336个诊断source-step、1,088次support source-step；full model、teacher FFN、Q1、video和official quality仍为0。compute 3600秒、每part3GiB输出上限、CPU2/GPU1及全局GPU4属于合同，实际占用需由新运行测得。训练和诊断均维持24图像来源的16/8身份，step/rows是源内嵌套观测。

**94旧cell的真实小文件链。** audit协议SHA `e39141917dbd4ccb82c9ac107497fb1395ac42e016a5af94de9e29907b59574a`；真实HF audit RUN SHA `a881c7b41b84ad816e7838d228550017237eb0e03a8e9259768db1da5f4b0b6f`、LAUNCH_EXIT SHA `c2f105909ed431508f0b4263e6f2656e8ff1f5c68ea8b7cf61b4235e7f25056a`、MANIFEST SHA `6153cae853fc3f9147c3bfb8a68df17f10bcc5b9adcb0294d1508d3d6adc2c7e`。逐小文件SHA核验后，RUN为 `COMPLETE_PARTIAL_ASSET_AUDIT`，已hash 6,813,474,624 bytes、54cells/66remaining；launcher为process_exited=true/returncode0且精确绑定该RUN。这个字节数来自已执行audit，本审阅未重复hash大pack。

四个completion父RUN实际均为 `FAILED_NO_RETRY`、`BrokenPipeError`，cell数14/14/13/13。其原launcher returncode仍是 `UNRECOVERED`；termination_observation只提供后来两worker/四part无同名及assigned GPU进程的只读观察，不能升级为原exit0。新audit及remaining validator明确保留该边界。新的小文件来源链验证实际返回94个唯一cell（原full80 40、completion 54），未导入Torch，耗时约0.708秒；原40的独立audit/原父exit1链继续由冻结completion validator复核。

**新退出与assembler。** 每新part必须完成全部本part cell和function，预算预扣/实计精确相等，复用链为94且partial parameters/optimizer均false。四个新RUN必须具有本scope ID、精确训练协议SHA、`COMPLETE_REMAINING66_PART`；每份新launcher还须本part/协议、process_exited=true/returncode0和精确RUN_sha256。最终检查160个cell的长度与完整key集合，防止重复/半init靠计数通过；manifest持续标明旧scope FAILED及原returncode UNRECOVERED。metadata assembler对大pack只stat并复核已审计小文件；实际部署仍需文件和tensor bytes门。

**CPU preflight修复及范围。** 初稿只有内部资产preflight报告，没有强制消费真实worker CPU入口收据；协调者补入 `validate_cpu_preflight`。现代码在repo_sources绑定后、94-cell复核和Torch导入前，读取ACTIVE本scope本part唯一的HF path/bytes/SHA绑定；要求精确协议/part/PASS状态、source绑定数量、94reuse、9或8shard、432或384操作数、无Torch/CUDA、所有CUDA隐藏及native_model_calls=0。initial authorize核冻结design，逐操作guard继续核当前ACTIVE scope、协议、part run IDs和预算；省去逐操作重复读取同一159KB冻结设计不改变训练语义。

使用项目 `.venv/bin/python` 跑contract4＋driver5＋audit4＝13项CPU测试，全部exit0（0.044秒），结束时Torch未导入。另以真实已核94-cell metadata、合成66-cell小文件构造完整assembler正例，得到160个唯一key；8组负例拒绝wrong-RUN SHA、未真实退出、重绑后的Adam=0、FAILED、错function分区、重复/半init、尝试重训low16 energy、缺part。变更RUN的语义反例同步重绑launcher，确认命中对应语义门；所有pack均为小metadata fixture，没有tensor payload。

**部署后续边界。** 当前既有 `compact6016_completion_runtime.load_bank` 硬调用旧completion assembler，不能直接加载remaining66协议，虽然两者metadata使用相同mixed_lineage schema。该问题已即时告知协调者；协调者将本scope限定为训练＋metadata assembly，并安排独立remaining loader，因此本审阅不把160-cell metadata称为已可部署bank。durable worker启动、独立stdout和新worker实际退出收据属于本轮真实执行的另一个系统验收；本审阅未运行native任务。下一实际证据是最终冻结、四份正确CPU preflight及durable运行的真实退出，随后才能组装160-cell资产。
