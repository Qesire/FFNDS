# S2 V3 续算 runner 实现者自审

2026-10-29 · `/root/s2_theory_design_independent` · **IMPLEMENTER_SELF_REVIEW，不是独立评议、支持票、运行授权或真实数值证据**。本文件只记实现者检查，主Agent正在独立审阅和决定是否冻结。

自审快照 SHA256：V3 `tools/e0_s2_spatial_prefix_replay_batched_v3.py` `adb7a1a0e684efc1b086b0b43879c6c6ade880527c026839f0c3c109e2e81c40`；软件控制 `tests/test_e0_s2_spatial_prefix_replay_batched_v3.py` `c7772044062538d4599f5bac0d5340a42fbf520880629278b0dbb4530f5b60e2`；batched helper `tools/raw_bf16_spatial_prefix_torch_batched.py` `e0bc5c201073388839ee5c0f328e98b8ecb2138c6355c58221b98d1d9847cc87`。依赖既有V1 `tools/e0_s2_spatial_prefix_replay.py` `ee80084aeebbec75804ba2782d8c5b537447da9444475671de210408f33d54bb` 与V2 wrapper `tools/e0_s2_spatial_prefix_replay_v2.py` `b24d80a2576309d3c4a6496026be363e49476c669edab813d84ec11b8c4d48dd`；V3 draft对parent repo源码绑定逐项验证，不能静默换掉历史数值代码。以上为审阅时快照，并非冻结状态；任何后续编辑须重新绑定和审阅。

## 实现与计数

- 父V2需实际EXIT、RUN/EXIT/positions/operations hash绑定、`completed_positions`非零且小于160并能被4整除、按block→source→step→family顺序每position恰两record，predebit、observed和journal均等于严格整block预算；父input_index直接从已绑定的V2协议复用，不在新协议重复写index。当前自审版只检查父为失败且非零退出，**尚未要求特定错误文本、timeout=false和GPU资源violations=[]**；主Agent已说明将强化这些迁移边界后再考虑冻结。
- 先冻结并执行parent prefix索引0、1、completed−1三个controls，逐字段对比同position两family的旧NumPy结果；字段/整数/null/bool严格一致，实数 `atol=1e-9, rtol=1e-7`，复相位圆周绝对差`1e-8`。低参考能量 `<=1e-20`路径单列，**任何容差或null不符仍失败**，不改变EPS。control记录/对照单写 `CONTROLS.jsonl`，只有三position均过后才计算remaining。
- V3保留原M1092原W2及两次BF16 factor GEMM、原bias、相同state/gate/r、typed postwrite、rowLN、native full-M anchor与prefix input定义；统计通过CPU Torch从CPU numpy数组构建共享视图、batched FP64归约与一次materialize。按每position预算3原W2 GEMM、8因子GEMM、7 Y D2H、10 gY、10 rowLN、6 view、6 bridge、32 FFT（每family的gY view+bridge各4输入×2时间视图共16）。FFT数由冻结helper控制流确定，成功返回后整体记observed；这不是逐个FFT硬件事件计数。
- 剩余及control分阶段按block复用对应W2与两因子，原始读取和H2D按涉及的唯一block计费；新输出余量和三个control不覆盖父record。只有结束后精确账本/controls均通过，按严格唯一顺序合并320条至 `MERGED_POSITIONS.jsonl`；最终仍需实际RUN与外层EXIT receipt及hash才算完成。实际性能与显存未测。

## 合成验证与限制

实际命令：`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 timeout 60s .venv/bin/python -m unittest tests.test_e0_s2_spatial_prefix_replay_batched_v3 tests.test_raw_bf16_spatial_prefix_torch_batched`，**14项通过**。测试覆盖父退出/账本闭合合成反例、顺序与位置唯一、预算上限/predebit失败保留、control比较的null/bool/近阈值及phase wrap、同原NumPy的CPU适配、逐bin、τ0和小差分桥。测试没有调用真实parent artifacts、GPU、真实h或整个runner，因此不证明生产写回、控制通过、完成时长、内存峰值、监控或launcher收据。

明确风险：

1. V2若在最后完整block后开始下一block的任何预算动作，或进程不是正常完成异常清理、RUN缺少完整journal绑定，严格父prefix校验拒绝继承；不能将失败或未对齐部分自动抹去。迁移在旧`authorize`下一完整block边界拒绝的方案需由主Agent实际收据核对，不能预设其已发生。
2. 低能量及相消相位接近阈值时Torch/NumPy归约顺序可能使某些值、null身份或精度在容差外；这会停在controls，不得声称已消除数值风险或放松判定。
3. `Session.materialize`单次批量输出与CPU结果对照来自合成测试。V3累计7个Y、10 gY/rowLN及helper临时张量的真实并存峰值可能不同于单pair合成基准；GPU/host/IO/wall必须在未来独立登记范围实测。
4. 新实现需主Agent逐项审阅CPU preflight/worker launcher、异常落盘/外层EXIT、源closure、控制先行、数值recipe、会话并发≤2及长时资源预算。源码的业务`COMPLETE`不替代外层actual EXIT与完整产物验收；不存在自动重跑、在线速度或视频主张。
