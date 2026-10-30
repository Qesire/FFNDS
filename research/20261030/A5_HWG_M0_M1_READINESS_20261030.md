# A5/HWG 的实际覆盖与下一步真实证据

M0显示旧随机行的主要缺口是跨步对应关系，而不是层数：24个HiF4来源都覆盖160位置，但1,920个同专家跨步层对仅0–8个匹配token、平均2.0354，没有完整原始M16/32/64 tile。8源BF16随机行的640对平均匹配2.0328，也没有完整tile。两源、三arm的BF16 spatial52共有480对，每对1092行完全匹配，但仍没有完整物理M16，数值族也不同。逐来源×双expert×40层结果见[CSV](A5_HWG_M0_ASSET_COVERAGE_20261030.csv)与[JSON](A5_HWG_M0_ASSET_COVERAGE_20261030.json)。

M0实际读取48,436,358字节JSON/header，核small metadata SHA及6,080个tensor文件长度/typed header；没有读取这些tensor数值，也没有重新核其完整payload SHA。旧payload identity保持继承原producer，不提升为本次独立数值证据。BF16 packet中 `state_pair=True` 一类文字不能代替实际row坐标。

原A5运行端的 `beta_row×gamma64×ternary` 重建、输入/hidden选择及gate×W2列范数已从私有归档读到；完整参数producer和compact参数未找到。后续新 `A5_INSPIRED_MEANABS64_ROW_LS_V1` 明确是新适配器，不称原A5复现。旧固定阈值A5及其历史视频/失败仍按原身份保留。

M1九个小张量代数fixture通过：线性差分/bias、GELU依赖历史preactivation、current gate、signed抵消、支持变化分解、整64组/稳定tie，以及量化与差分不交换的简化反例。简化量化fixture不等于原生HiF4验算；后续必须使用原Q1/Q2和smooth尺度。原HiF4缓存56,223,524,243字节已重新全量核SHA，且另一个HF缓存镜像也已完成复制及同SHA复验，均为 `bfb400f94ce42760070bab5916721fd942b0864907bb4e4e5207c008dc4c3df3`。

下一项是[独立复审中的匹配采集](../experiment_reviews/E0_A5_HWG_MATCHED_CAPTURE_20261030/author_response_v3.md)：4个历史暴露来源、每源plain/observed两个HiF4完整Dense视频，保存全部160位置的真实fullM postQ1输入与21τ×M64匹配原生边界，先验收10状态/视频byte中性，再独立读取核全file/raw SHA。新采集准备好后，E0才能在同数值族、同一层/时空位置比较当前A5、增量A5、Chipmunk/static/random和paid oracle的signed输出及真实tile规则性；E1真实skip和E2候选视频仍需各自新的精确复审/登记。

本页只说明M0/M1与已核字节。采集是否运行/成功由ACTIVE和实际RUN/EXIT决定，采集成功不构成稀疏方法质量或速度通过。
