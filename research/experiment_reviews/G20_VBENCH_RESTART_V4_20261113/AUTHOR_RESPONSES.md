# 作者答复与 v4 定向修订

作者 `/root`；2026-11-13。初稿三主文件由真实不同 Agent `/root/numeric_audit`、`/root/stats_audit`、`/root/scope_review` 分别静态审阅。原始审查和作者自审保留原字节。本轮只修设计，没有提交计算。

| 实际发现 | 作者答复 | 落实位置 |
| --- | --- | --- |
| NUM-B1、ST-V4-B01：57次未覆盖全部DEV的无observer Dense生产计时；scope同项提醒 | 接受。57仅为参考/质量链基础数；首轮速度主比较为候选vs新Q1，使用V5的22次无observer运行。两个plain neutral固定DEV1013/800且只在生产身份合格时复用。完整DEV11 Dense速度如需，另补9次（基础66）；若两个neutral计时均不合格，另补11次（基础68）。技术重复、额外profiling每次实际推理计次，保持条件触发。 | 路线§4–5；cohort counts/plain_controls/timing_budget；资产V5/budget/PRODUCTION_TIMING |
| NUM-B2：256行配置可能偷用512行学到的先验 | 接受。每配置的alpha、P/Q、初始化、归一化、正则尺度及读出均只读该source×prefix；CAL-CV折内重建。DEV固定512行。目标按source/row平均，同模型族/λ政策不等于共享较大配置的拟合量。 | 路线§2、§4；cohort sampling；资产fit_input_budget和V4 |
| NUM-N1：N>L32可能混入额外容量 | 接受。明确P32/Q32、线性32+tanh32，端点差与位移控制公式、b=0、scale-only处理和零位移/奇对称性。端点N只超过L32时加同bank L64/Q32，同读出参数量，不增加生成；实际在线成本仍单独测。 | 路线§3；cohort representation_controls；资产V4 |
| NUM-N2、ST-V4-N01：聚合公式尚未固定 | 接受。固定源内层平均门后MSE，再逐expert相对当前Q1取比，expert/source等权；绝对误差和均值之比另表。CAL中性误差确定可测floor，低分母保留绝对值，不删源。 | 路线§4；cohort inference；资产readout_contract |
| NUM-N3：profiling与probe-free速度身份 | 接受。无probe wall time是速度主量，FFN/projection/cache观测另作诊断；任何新增完整推理计次。 | 路线§5；资产budget |
| ST-V4-N02：元数据选择不等于实际源簇绑定 | 保持边界。配置runnable=false，image_sha256/cluster/recipe/visual_review仍pending；V0按输入证据绑定和顺延，绝不恢复fresh门槛。 | 路线§1、§6；cohort payload_binding |
| ST-V4-N03：官方指标分母 | 接受并在机器字段补明：DEV原始条件subject/motion7、background4、imaging/aesthetic11、camera_motion0；额外光流和视觉检查自报实际覆盖。 | cohort inference；资产readout_contract |
| scope：AGENTS顶部可能被旧adapter命令误导 | 接受。顶部明确新周期使用保留历史事件的版本化adapter，不直接套旧official/fresh规则，也不据当前role重写曝光。 | AGENTS当前段 |

## 请求定向 recheck 的三主文件

| 文件 | 修订后SHA256 |
| --- | --- |
| `research/20261113/G20_VBENCH_RESTART_V4_20261113.md` | `17c632a53d8621db701c835415bb7a03e6db9f8a83acff029b466ab9662451b0` |
| `research/20261113/G20_VBENCH_ASSETS_AND_QUEUE_V4_20261113.json` | `eec38fff484c9a34a176000c5affce5b9dc404f0f3ceddaa98a6bf7852a9533d` |
| `config/g20_vbench_restart_cohort_v1_20261113.json` | `8512ec30c9cbb8e1f735bca1046e750edd6942edbc1a11030724ee31812abd30` |

请只核本次修改是否实际解决发现并保持小规模设计。最终设计通过也不构成任何CPU/GPU scope的执行批准。
