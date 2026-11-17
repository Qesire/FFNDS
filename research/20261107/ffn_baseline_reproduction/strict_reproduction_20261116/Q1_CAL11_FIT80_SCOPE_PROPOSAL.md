# Q1 CAL11 80站点拟合：待独立复核的执行范围

状态：REVIEW_REQUESTED。以下是具体候选，未登记为ACTIVE，未执行拟合；实际冻结以独立复核后的scope与guard为准。

目标是在V4同一训练合同下生成当前Wan高低expert共80处Q1矩阵：原VBench CAL11的seed42、各站点512个固定panel，来自V2 full-M native W2 pre-gate输出的Y端点，以源等权构造增量X/Y及当前hit gate。Q0的alpha与A0由同一CAL11重新计算，不使用BWS mixed-CAL3参数或BW的local512重算Y。固定调用已核对V4的 `q0_prior`/`fit_q1`等价实现：FP32 addmm+tanh-GELU、F64矩构造A0、lambda=1e-3、PCG最多240迭代、真相对残差有限且处于[0,1e-5]；FP32公式导出A、逐值重载。宿主线上将A转BF16，采用已有Q1 cache算术合同。

输入需冻结：

1. `stage3_assemble_inputs.py --mode cal` 从已验证V2 `B_BINDINGS.json`/BANK_MANIFEST及各文件/sidecar生成的**880个anchor/hit pair**清单及SHA；每个source、expert、layer各一，包含token索引、seed、原Y面板、gate、图文、来源簇和checkpoint身份。CAL来源固定 `vbi_0032,0304,0456,0648,0895,0937,0966,1005,1039,1082,1094`（排序后）。
2. 两个当前H/L蒸馏BF16 safetensors完整路径和SHA；11源原V2 checkpoint身份必须一致。`target_identity=V4_FULLM_Y_PRE_GATE_PANEL512`、`calibration_manifest_sha256`、原V4源码/方法身份和历史目标语义。V4与BW的目标构造不同，不将BW六点当作V4自动抵扣。
3. `icc_stage3_cli.py`、`q1_cal11_reproduction.py`、`q1_cal11_fit80.py`、`icc_stage3_assets.py`、`icc_stage3_grid.py`、`icc_local_replay.py`及其本次实际导入的固定BWS源码完整路径/SHA，包括 `b_common_replay.py`、`q1_fit_core.py`、`ridge_fit_core.py`、`icc_ffn.py`、`icc_decompose.py`、`chipmunk_fullm_proxy_math.py`、`chipmunk_wan_packed_pregate.py`；记录原V4 `v4_repr.py`/`v4_runner.py` 的源码身份及静态等价核验。
4. 与REVIEW和ACTIVE精确相符的scope SHA；指定 `operation=q1_fit80`、`max_new_fits=80`、`ridge=0.001`、`maxiter=240`、`tolerance=0.00001`、`remaining_time_multiplier=1.25`、无条件随机搜索/调参、`compatible_existing=[]`（仅实际证明兼容后可增加）。
5. 当前hif4 Python可执行文件/SHA与torch版本、runtime_guard路径/SHA、A100 80GB真实分配；TF32关闭，BF16 reduced-precision reduction与V2固定设置相同。预建议单任务1 GPU、4 CPU、host上限160GiB、GPU上限78GiB、wall 10800秒、soft 10500秒、可用盘至少20GiB，输出上限16GiB；按实际调度与集群政策复核。

执行顺序固定 high00、high01、low00、low01，再执行其余76站点。前4个是正式拟合、各自完整保存F32 A和alpha及校验收据；完成后记录单站点耗时、进程和GPU峰值，并按 `max(前四耗时) × 76 × 1.25` 与共享soft deadline余量比较，满足时同一作业自动继续，不重算这4点；不足则中止并保留文件/日志，另经审核将其作为精确兼容项接续。单站点失败不自动重试。每个拟合前后有 `FIT_EVENTS.jsonl`，完整80点均收敛并身份通过后才产生 `Q1_MANIFEST.json`。

运行时间目前不能可靠预估。可核历史BW不同工作量：6站点×两权重的Q1+ICC+评分总460.86秒；V4的6站点×两个行数、每组六模型总1122.25秒。这些不是Q1-only 80点的计时；前4点真实观测包含实际checkpoint校验、I/O、PCG和序列化，供上述执行中资源决策，仍不保证剩余站点相同。主Q1矩阵payload约80×5120×5120×4 bytes≈7.8125GiB，另有alpha、元数据和日志；实际文件大小/峰值由运行记录给出。

调度入口为单独校验scope/review/ACTIVE和实际GPU分配的 `icc_stage3_cli.py SCOPE_PATH SCOPE_SHA256`，本文件只提出候选，不提供或替代已批准的REVIEW、ACTIVE、作业ID或结果。
