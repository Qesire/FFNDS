# SharQ Wan 两来源质量闭环：作者自审，待独立评议

问题：相同原BF16权重的Dense32与SharQ全160真实视频，相对同source/seed原BF16视频，各维质量和运动有何变化？竞争预测为量化误差是否通过多步动力学累积，以及两分支误差较小是否足以保留运动/外观。两来源1040/979均历史暴露开发来源，各一seed，来源等权，非heldout；三臂共6个真实完整视频。首BF16、Dense32、SharQ各评分一次，第二BF16是父范围neutral，不重复作为独立样本。

每来源三个独立有限scope：原函数八轴（8实际维度调用、24个视频维结果，动态维3×40 RAFT forward×20迭代）、连续原RAFT（再做3×40 forward×20迭代，重复成本单列）、81帧contact sheet（243帧、6页JPEG）。八轴原函数含两个I2V维，均为custom单seed诊断，不伪称官方五采样或添加第九轴。连续RAFT为全帧高运动区域统计，可受相机/背景影响；contact sheet仅供后续观察，不是人工完整播放审查。各scope复用此前真实闭合TEAL原环境、原源码/资产和操作账，新的adapter仅改变parent lineage、真实视频身份及输出ID。

父依赖是各自V6 GPU实际RUN/EXIT0以及独立VERIFY实际RUN/EXIT0；严格要求十状态/视频neutral、same-source/noise及原四臂全160闭合。所有新的metric draft要等真实parent及verifier EXIT0后才绑定视频SHA；不存在占位hash。一个source先闭合不代表另一source可评分。每scope fresh attempt、源码与权重固定、原环境禁下载、实际CPU byte preflight EXIT0后GPU、1 GPU、会话全节点不超过4卡、不自动重试。视觉独立CPU scope，原有可验证81帧和JPEG输出。结束须实际RUN/EXIT和closed校验；机器身份仅私有spec。

代码：`tools/e0_sharq_wan_quality_lineage.py` 的隔离子进程只做小证据身份与stat、不是自称大视频字节复验；每消费者在本scope独立付费哈希所绑定视频和状态。`tools/e0_sharq_wan_metric_adapter.py` 复用 `e0_teal_wan_full_quality_v3.py` 与 `e0_teal_wan_motion_metrics_v3.py` 原数值路径和预算；启动入口分别 `e0_sharq_wan_full_quality.py` 与 `e0_sharq_wan_continuous_raft.py`。视觉入口 `e0_sharq_wan_visual_cpu.py` 是逐行适配既有已闭合CPU producer。父输入未闭合，不能生成最终协议、spec或声称运行结果。

请独立审查：原metric脚本实际调用与权重/加载审计、子进程/真实父 verifier EXIT身份，重复RAFT成本和decode预算，源码闭包与状态绑定、ACTIVE/预检/worker/输出和失败账、三个选定arm与等权聚合、非官方单seed边界。必要时给可触发误标或启动失败的反例。最终SHA和launcher须父实际闭合后再作exact审查；当前仅静态设计与合成测试。
