# B门控后缀敏感坐标 v1：独立方法评议

**SUPPORT（ALG/SRC，限精确版本）**：审阅 [METHOD_SNAPSHOT](../../../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md) SHA256 `679f6e553bad215f1a68e935961bb34fb7b7544e25e37bce1691c540c1f644ad`。末层LayerNorm的有限归一化通路提供可证伪的head-null响应；数学、已绑定源码边界与分阶段预算相符。该支持是对方法和推导，不是原生试验、视频质量或部署通过。

令 `z=P0u`、`σ²=zᵀz/D+ε`、`T=W diag(1+scale)`，则连续 `J_u=T(P0−zzᵀ/(Dσ²))/σ`，固定gate时 `J_y=J_u Dg`。展开转置后行空间包含于 `span{Dg Tᵀ,Dg1,Dgz}`，每个条件下至多66维且Jacobian自身rank≤64。这个上界针对单状态；状态z、gate和时间调制变化会让来源间并集维数更大。64个Tu坐标加均值与平方范数足以表达固定条件的完整实数化输出，但不是可免费获得的在线表示。对于 `Tδ=0, 1ᵀδ=0, zᵀδ=0`，一阶为零，非零δ使方差增加，若原q非零则产生二阶有限响应；q为零等退化情形可无响应。更强范数保持条件是文中给定充分条件，不是所有有限不变方向的必要条件。实际BF16残差、LN、调制及head舍入使该连续推导不自动成为离散实现等式。

B-FACTORED 的 `Dh+r(h+D)`、r512时36,503,552 MAC/token和BF16中间 `31668×512×2=32,428,032` bytes均正确；第三次GEMM、两个因子驻留、分配和工作空间需要实际测量。B-FOLDED与A0主投影MAC相同，不能把秩约束当自动速度收益。σ/条件、gate和后缀敏感采集与离线训练的代价均需绑定；head-null方向确定所需状态信息不应视为上线免费。

B0是两开发暴露来源×两step的4次Dense flow和160个FFN，用于建立完整u、flow、原噪声和neutral身份；没有方向干预或训练，不能判定paired优于marginal、方向因果、运动修复或来源外泛化。两来源为独立来源簇，token、256 rows、step与专家为嵌套观测；后续B1应先固定原dtype全M neutral和有效扰动、幅度、调用/采集预算，再判定有限响应及BF16噪声下限，B2需冻结独立heldout来源和同rank/同训练成本对照。source-level同状态G与输出二阶矩配对是方法关键；固定位置CAL24 gate字节相同并不提供来源特异敏感信息。

源码资产六张量view和原checkpoint header支持两个专家 head BF16 `[64,5120]`、bias `[64]` 与modulation `[1,2,5120]`；母文件SHA来自先前完整核验，非这次header重hash。原time embed、post-write u及其匹配flow仍待producer。原文已恰当地将平均/联合代理、实数有限身份、原生数值、完整视频及执行成本分级。

## 修订快照追审

修订快照 SHA256 `8efe48542ce65708696b239b17121d23a268d3d7f7e6420349cf4b4730c47c41`；上文 `679f6e553bad215f1a68e935961bb34fb7b7544e25e37bce1691c540c1f644ad` 保留原审阅版本身份。新增五操作 fingerprint 保留离线敏感度获取及B-FACTORED第三GEMM成本；输入合同明确 `use_image_encoder=false`、CLIP不执行；历史只保存post-state而无raw head/flow，故PASS要求同次原函数对象一致和post-state对照，不虚称历史raw flow逐字节复现。核心公式未变，原ALG/SRC支持适用于此修订；q=0退化与有限不变的充分条件已在本评议明确，无需将充分条件写为必要条件。B0真实执行和独立代码审查仍待完成。本评议链接路径已检查可解析。
