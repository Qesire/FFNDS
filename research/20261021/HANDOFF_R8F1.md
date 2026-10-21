# 已完成交接 — R8F1既有仿射骨干精度前提

CURRENT_DECISION→method_triage→docs/59。R7D1已完成24次原生生成，R7D2已完成18次并拒绝固定双增益配方；两者不重复。旧R4与所有失败继续关闭。

R7D2实际flow训练改善0.0357%，代理预测的2447.64 SSE只兑现44.20；开发留出flow-fit下降0.0770%，local-fit下降0.1655%，前者不胜后者。全4flow仅改善0.0489%，没有候选视频或质量通过。

R8F1是已登记R2 all-channel affine backbone的full-support控制，非新方法、非GELU半线性剪枝。固定step2/3 block20，OP2输入00/06/12/18。先48项UB1原生主输出/R逐位回放、8项全教师重建，然后比较BF16、混合HiF4、完整HiF4三种全支持表示。BF16误差/UB1≤.01、完整HiF4≤.1需全部8项通过；否则关闭这一个固定表示实例，不扫支持/阈值。

工具probe_native_affine_fold.py、native_affine_fold.py，协议R8F1_native_affine_protocol.json。只用一张空闲工作卡，原始全M/stride与采样坐标，精确source view；无训练/视频。完整支持主MAC1.185185×dense，绝非50%候选。矩阵留HF缓存，保留80GiB节点余量和本轮2GiB写入额。

结果更新CURRENT_DECISION、triage、README、docs/59和本交接；记录全部失败及误差归因，仍须真正新来源、完整时间与三项质量联合验收。按大结果正常快进推送main。

R8F1已完成，BF16通过、HiF4八项均拒绝；当前迁移HANDOFF_R8F2.md，未启动压缩。
