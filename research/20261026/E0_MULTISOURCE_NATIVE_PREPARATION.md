# 四个新增暴露来源：原生方向与视频功能复验

**研究对象是局部 FFN 误差收缩与完整状态偏移扩大的共存，以及这种关系在不同来源上的有符号方向。** 竞争解释是此前两个来源的状态响应具有偶然性，或者同样的全域误差掩盖了互相抵消的频带与不同视频功能。按最新[多来源方案](E0_MULTISOURCE_DIRECTIONAL_EVALUATION_20261026.md)，本轮以来源级指标向量检验这些预测。

## DISCOVERY

固定原 bank，新增 CAL24 来源4/8/16/20，seed46/50/58/62。每来源保留全部40块×4步、teacher/student输入上的四臂 FFN，及实际 gate/writeback。每步另求完整 `T_D(z_S)`，分解 `P=T_D(z_S)-T_D(z_D)`、`N=T_S(z_S)-T_D(z_S)`、`delta_next=P+N`，保留有符号交叉项。四个固定互斥子空间分别为全轴DC、时间DC空间低频除DC、时间非DC空间低频、其余空间高频；它们描述 latent 坐标，不对应语义主体/背景标签。

预登记预测是：(1) 局部误差多数收缩与非零状态对 dense 有限映射能量增益>1在这些新增来源共存；(2) 同一频带的交叉项可能跨来源变号，或与全域符号相反。所有反例按来源/step完整保留，160site是同一轨迹内重复测量。预测和定量输出见[本轮冻结协议](../../config/e0_multisource_directional_native_20261026_v1.json)。

## EVIDENCE

原始六来源回执、实际图片、提示词、seed、四步 dense latent 与MP4身份已经在配置执行节点完成只读SHA预检。当前80个参数包保持不变；当前 dense 轨迹与视频必须逐字节匹配历史 passive 输出。新的 candidate 在这些来源上没有历史输出，采集结果不会冒称为旧 candidate 重现。

每条新视频单独计算81帧/80transition核心指标，以当前 byte-verified dense 和不可变 `NATIVE_RUN` 为权威 reference，通过核心 bundle 与独立原生 inventory gate。原VBench-I2V/VBench2、LPIPS/DISTS及人工审阅在本范围中为NOT_RUN，quality_accepted=false。核心指标描述帧、时序、运动、亮度、细节，后续比较保留各轴和来源差异。

独立[来源审查](../method_discussion/discoveries/E0/independent-multisource-exposure-audit-20261026_v10.md)确认全部CAL24已有历史暴露；其中4/8/16/20直接用于bank校准。本轮提供暴露来源稳健性证据。独立未暴露来源和交叉seed属于后续另行冻结的范围。

## EXECUTION

依据用户临时持续授权，序列化新固定范围：单张空闲A100、四个全新进程、八条完整视频、640新paired site、16次额外完整dense flow、8次VAE decode；输出≤1GiB，零拟合/训练/分解/官方评分。任何失败保留扣账并停止该scope；旧v2/v3及两来源whole-flow协议和代码保持原字节。实际启动、完成和预算由ACTIVE及HF的STUDY/RUN记录，不由此准备文档推断。

新增源码分离于冻结driver。七项CPU测试覆盖互斥维数、signed cancellation、neutral control、大能量近零cross容差、失败扣账、固定顺序和损坏前缀。自然尺度 `2||P||||N||`控制cross舍入容差，实际残差与容差同时输出。新VAE/media源码、真实VAE checkpoint、逻辑/节点本地模型view及cuDNN设置均绑定。静态检查不代表已取得原生结果。

图锚点为 F04/F06/F08R/F08/F09/F12/F13/P23/P24/G21/D09，相关新方向边在实际完成前为conditional。即使全部诊断完成，FFN/端到端省算与正式完整视频验收仍需对应独立原生时间和质量证据。
