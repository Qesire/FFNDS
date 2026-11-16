# Wan FFN基线：完整复现与受控重做

**2026-11-16 当前入口：[完整下一步实验方案](strict_reproduction_20261116/CONTROLLED_REPRODUCTION_PLAN.md)。** 先完整验收作者FFN机制与必要Wan适配，再作五臂DEV11比较。ICC主11源作者式校准66次，加CAL3配对诊断至多30；不生成完整CAL轨迹。阶段、产物、条件分支和预算已统一。当前仅设计整合，新模型/GPU运行0；真实clean输入与实际执行仍待落实。

过去的ICC双残差、Global-P64、author-style历史视频与A/BW/BWS都保留其实际身份；它们的完成状态不等于原作者参考已完整复现。只补记录不足以替代缺失的四状态、校准配方或作者状态机执行。

| 入口 | 用途 |
| --- | --- |
| [当前复现与重做规格](STRICT_REPRODUCTION_AND_REDO_20261116.md) | 统一对象、必要适配、消融、强基线、复用边界与完成标准 |
| [作者Wan执行链审计](strict_reproduction_20261116/CHIPMUNK_UPSTREAM_AUDIT.md) | Wan2.1 T2V/attention与Wan2.2 I2V/FFN区别；官方Ampere/Triton；计数与几何缺口 |
| [ICC作者式校准](strict_reproduction_20261116/ICC_AUTHOR_CALIBRATION.md) · [数据资产](strict_reproduction_20261116/ICC_CALIBRATION_ASSET_AUDIT.md) | 真实clean样本、前向加噪、CA单次/CD真实一步、66/132 NFE及clean数据缺项 |
| [ICC严格参考](strict_reproduction_20261116/ICC_STRICT_REFERENCE.md) | 四状态、SVD/CA/CD、全四边界统计、精度、内存与rank预算 |
| [统一实验设计与预算](strict_reproduction_20261116/CONTROLLED_REPRODUCTION_PLAN.md) | 模块完整性、必要接口与五臂DEV开发对照；视频与作者式校准调用分别计账 |
| [固定代码引述](strict_reproduction_20261116/ICC_CODE_CITATIONS.md) · [Chipmunk](strict_reproduction_20261116/CHIP_CODE_CITATIONS.md) | 函数、固定commit行号、短摘录、必要适配与最小验收 |
| [独立设计审阅](strict_reproduction_20261116/INDEPENDENT_REVIEW.md) | 实际审阅意见及修订回核；不代表任何GPU结果或已冻结RUN |
| [当前研究结果](../../../reports/20261114/g20_two_track_core/README.md) | 已完成A/BW/BWS、历史完整视频和本次新增任务的界线 |

源代码版本：Chipmunk `sandyresearch/chipmunk@b5e8d1c3da6bddc7ca9007c48e91b82087c11490`；ICC `ccccczzy/icc@dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`，均重新核实为当时默认master HEAD。代码是否可用须实际执行证明；本轮未安装或启动GPU模块。

## 历史资料

[旧资产与G19](G19_BASELINE_AND_ASSETS.md)、[旧指南](REPRODUCTION_GUIDE.md)、[20261107 ICC规格](ICC_FFN_PORT_SPEC.md)、[20261107 Chip复用](CHIPMUNK_REUSE_AND_COMPARISON.md)、[旧实验合同](EXPERIMENT_CONTRACT.md)、[候选方法](TARDIS_AND_ALTERNATIVES.md)保留溯源价值。其“仅复用”“CA可选”“r64–512”“改period2”等旧规划不再覆盖本轮完整复现要求。新CPU/GPU科学执行沿现有项目协议登记有限范围；不重新启动已经完成的旧scope。
