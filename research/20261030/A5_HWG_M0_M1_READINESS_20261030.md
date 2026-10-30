# A5/HWG 的实际覆盖与下一步真实证据

M0显示旧随机行的主要缺口是跨步对应关系，而不是层数：24个HiF4来源都覆盖160位置，但1,920个同专家跨步层对仅0–8个匹配token、平均2.0354，没有完整原始M16/32/64 tile。8源BF16随机行的640对平均匹配2.0328，也没有完整tile。两源、三arm的BF16 spatial52共有480对，每对1092行完全匹配，但仍没有完整物理M16，数值族也不同。逐来源×双expert×40层结果见[CSV](A5_HWG_M0_ASSET_COVERAGE_20261030.csv)与[JSON](A5_HWG_M0_ASSET_COVERAGE_20261030.json)。

M0实际读取48,436,358字节JSON/header，核small metadata SHA及6,080个tensor文件长度/typed header；没有读取这些tensor数值，也没有重新核其完整payload SHA。旧payload identity保持继承原producer，不提升为本次独立数值证据。BF16 packet中 `state_pair=True` 一类文字不能代替实际row坐标。

原A5运行端的 `beta_row×gamma64×ternary` 重建、输入/hidden选择及gate×W2列范数已从私有归档读到；完整参数producer和compact参数未找到。后续新 `A5_INSPIRED_MEANABS64_ROW_LS_V1` 明确是新适配器，不称原A5复现。旧固定阈值A5及其历史视频/失败仍按原身份保留。

M1九个小张量代数fixture通过：线性差分/bias、GELU依赖历史preactivation、current gate、signed抵消、支持变化分解、整64组/稳定tie，以及量化与差分不交换的简化反例。简化量化fixture不等于原生HiF4验算；后续必须使用原Q1/Q2和smooth尺度。原HiF4缓存56,223,524,243字节已重新全量核SHA，且另一个HF缓存镜像也已完成复制及同SHA复验，均为 `bfb400f94ce42760070bab5916721fd942b0864907bb4e4e5207c008dc4c3df3`。

匹配采集的原计划为4个历史暴露来源，每源plain/observed两个HiF4完整Dense视频，保存全部160位置的真实fullM postQ1输入与21τ×M64匹配原生边界。898/1040/979的producer与独立VERIFY现均actual EXIT0：每源160处完整、10个状态及完整视频逐字节中性，每源独立校验342个文件、约97.8GB。来源96的最后NS_V3尝试实际EXIT1，原因是已分配GPU监控超时，视频phase数0，不计为算法数值失败，不自动重跑。之前的调度、namespace及中断失败均保留原版本。采集事实与SHA见[实际收据汇总](A5_HWG_MATCHED_CAPTURE_RESULTS_20261030.json)；即时状态以ACTIVE与实际RUN/EXIT为准。

E0的[独立预审及逐项修订](../experiment_reviews/E0_A5_HWG_DELTA_REGULARITY_20261030/AUTHOR_RESPONSE_AND_FULLM_DESIGN_20261030.md)已完成软件实现，64项fixture通过，覆盖逻辑稀疏与tile饱和、shuffle原坐标、尾部padding、signed交叉/舍入、原manifest链、原生import身份、greedy与真实选中W2闭合及source等权汇总。当前收尾范围为1040校准、898/979两个历史暴露开发来源，共3来源×双expert×40层的fullM、58臂等K比较；96失败另列，不作为缺失值补进平均。此前未运行CAL2草案保持历史身份，不登记为已做。

先用已验证1040的1344匹配行冻结整64组静态对照，开发来源不参与拟合。[CAL1_V3独立批准](../experiment_reviews/E0_A5_HWG_DELTA_REGULARITY_20261030/reviewer_cal1_v3_recheck.md)后实际完成80/80层对、actual EXIT0，耗时552.387秒，开发tensor读取数0；STATIC_K64_SUPPORT作为后续比较的冻结输入。1040首个E0范围在10/80对后因foreign compute actual EXIT1，部分结果不混入正式平均；[独立批准的资源迁移V2](../experiment_reviews/E0_A5_HWG_DELTA_REGULARITY_20261030/reviewer_source1040_v2_relocated_recheck.md)不改数值代码，已在新设备启动，范围为重新完成80对。898/979各自精确协议已准备，仍须独立复核再登记；最终CPU独立mask校验另审。当前尚无完整E0数值结论。ALG成本：每expert首次Dense、K5120的四步主MAC比例0.685185；若额外四次全M Dense W1代理，则比例1.185185（尚未含其它成本）。这不是实测速度，也不表示原A5专用sign/scale算子已经实现。

本页说明M0/M1与采集、校准的证据身份。当前用户要求完成核心比较、上传main后结束，本轮不扩展E1原生skip或E2候选完整视频。采集成功、软件测试及付费teacher重放分别登记，不构成稀疏方法视频质量或系统速度通过。
