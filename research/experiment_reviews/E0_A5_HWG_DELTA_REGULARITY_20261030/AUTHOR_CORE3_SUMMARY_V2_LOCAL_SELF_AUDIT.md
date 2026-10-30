# 最终三来源CPU汇总 V2_LOCAL：作者执行前自审

作者：`/root`。状态：AUTHOR_SELF_AUDIT / 请求精确独立recheck，尚未登记或运行。

精确scope `E0_A5_HWG_CORE3_CLOSING_COMPARISON_20261030_V2_LOCAL`；协议`config/e0_a5_hwg_core_summary_20261030_frozen_v2_local.json` SHA256 `2f153c9867fd276e7ba7b3a43ceab4f82f85057e9f06e441ab3c076303542d43`；job SHA256 `20ad24e3a15854c3b90191ffe03381e62cb198c3cc287ec64bc8769dcb2d70c8`；fingerprint `9295dcd6b58e28837a2594f33c53973ff665401e1d9276d9053fee5690bb658c`。

该有限范围只汇总已完成E0、独立核mask/proxy的file及typed raw SHA；没有模型、CUDA、tensor数值运算、训练、拟合、分解或视频。三项汇总软件fixture已由作者及独立reviewer实际通过；软件结果不替代本次真实数据执行。

## 研究问题和能推翻结论的反例

同K5120下，增量代理可能比当前贡献、P64、静态/随机更好；也可能只改善某expert/层，或更细的逻辑支持在原M16×K64物理布局几乎满占用。最终source等权和逐site表保留两种预测，不根据前缀调整arm或预算。数学及原A5/Chipmunk身份沿用已执行E0作者审查，不将新适配器改名为作者复现。

最强反例包括：只拣成功层或prefix、将80个嵌套层对当独立n、将校准1040当新盲测、将oracle误差当在线省算、读到被改动mask、漏计某arm或operation ordinal。程序要求三父实际EXIT0、80个唯一双expert全40层site、58个精确arm、相同cache/recipe/静态支持、完整call计数和序号集合，并逐240文件独立检查whole-file与每typed raw SHA；不满足即失败。

## 实际冻结输入

- 来源898，`EXPOSED_DEVELOPMENT_NOT_BLIND`，父`E0_A5_HWG_FULLM_DISCOVERY_20261030_SOURCE898_V1`；实际RUN SHA `f245d18c20be56a48f86c2282ddf1051119f8f11fd166d753f84152d413569f2`、实际EXIT SHA `01afc9fdd2f0642adf9d367af713384b196ea7676e437c23619b37380c6ad15e`。
- 来源1040，`EXPOSED_CALIBRATION`，父`E0_A5_HWG_FULLM_DISCOVERY_20261030_SOURCE1040_V2_RELOCATED`；实际RUN SHA `0dcb2953af22d0cbb2482937bb27e6ad9043e559e3ad0eeccba31e5744c448fb`、实际EXIT SHA `b2ab6fc835d96dd3fd48354ae058473cd8752e22c32a5b66d98825af5b187de6`。
- 来源979，`EXPOSED_DEVELOPMENT_NOT_BLIND`，父`E0_A5_HWG_FULLM_DISCOVERY_20261030_SOURCE979_V1`；实际RUN SHA `d0f2857501da7b88eade45e8e13353423d80719aba3230baaebca96c74023c0b`、实际EXIT SHA `ee60d0b9763c7edab2707387a31c79828ff1c8939009d820d802ca35ca765b9d`。

1040仅CAL1校准；898/979历史曝光开发；seed42。先各来源平均80site，再image/source等权，n=3包含1cal+2dev。1024次source bootstrap只描述这些曝光来源，不宣称总体泛化或新source确认。来源96最后采集实际EXIT1、生成phase0，不补零、不算第四个数值来源；1040首E0外来进程中断的10对不混入V2完整80对。

## 真实成本、资源与失败计划

GPU0张、CPU单核（私有worker指定本地CPU0）、RSS≤512MiB、显式输入读≤96GiB、HF新输出≤256MiB、单输出文件≤128MiB，wall7200+outer120秒，单次尝试。CPU仅串行读约54GiB已存mask/proxy并做SHA，原始cache/模型不重新数值加载。实际亲和、代码、parent RUN/EXIT、protocol/job/manifest在launch前核对。

原E0的masked Dense FP32操作只作付费诊断；CPU事件CSV不是kernel active time或公平SYS证据。全行代理与M64均值输入代理的ALG成本分开；没有E1原生skip或E2候选自由视频/质量验收。

资源readahead辅助是独立有限scope；1040/898两项辅助读取可变RUN时预扣校验失败均已actual EXIT1、保留inflight UNKNOWN且不重试，不改变任何父E0输入或数值代码。CPU汇总只读取父实际退出后的不可变最终RUN，另核精确SHA。

输出SOURCE_EQUAL_COMPARISON.json、SOURCE_SITE_ARM_METRICS.csv、PAID_OPERATION_INTERVALS.csv及真实RUN/EXIT，全部HF留存；只将前两项和脱敏收据带入Git。任何失败保留实际成本/输出，不自动重试；完成后按用户要求汇总、push main并结束。

本作者记录不是独立批准；须真实不同Agent检查此精确协议/worker/job、输入actual0、源码closure及fresh输出并提供正式recheck。

V1两次SSH prestart均超时，已确认无worker目录、START_CLAIM、子进程、RUN或EXIT，保留为NOT_STARTED而非实际实验失败。此V2_LOCAL保持全部三个父输入、1039源码、算法/统计/预算不变，只改私有worker/本地CPU0和fresh输出命名；本地实际hostname与已冻结C角色profile匹配，CUDA为空。直接本地durable supervisor启动，不经过SSH。需对此精确版本重新独立批准。
