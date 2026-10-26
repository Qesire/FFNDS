# 六视频原始补评报告

本目录8份JSON是HF补评目录的逐字节副本，原视频留在HF缓存；其SHA见[结果索引](../E0_EXISTING_VIDEO_QUALITY_BACKFILL.json)。原报告中的本机绝对路径用于来源追溯，换机需显式重绑定。

**四个候选报告的 `reference_identity.model_digest` 有元数据错误：误继承候选身份，实际dense值应为 `c2396ae98e79888dd80c19eaa2a1920b501ef5f67b2e796d1253cc0d52c82329`。** 原视频/原RUN的SHA及全部像素计算不受影响。文件保持冻结字节，逐份勘误在结果索引，不把错误字段作为参考模型证明。

RUN中的core_metric_gate_pass不是官方质量或人工验收，全部quality_accepted=false。原driver仅作为此冻结分析的可审计源；未来运行须显式修复参考身份构造并另冻结，不直接复用此协议。
