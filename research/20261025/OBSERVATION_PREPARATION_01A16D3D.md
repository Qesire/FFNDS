# 方法审计与关键观察实验准备

2026-10-25 · codex-ffnds-method-01a16d3d  
基准 FFNDS：ff6942763d8ce3e1823a4814905ad5ee61da7c31；目标环境：用户指定的原 A100。  
角色：**用户提出 idea；agent 准备可区分解释的观察、实验与证据。**  
状态：实验软件与协议已准备，原生模型采集尚未运行。

## 本次工作改变了什么

此前连续围绕量化码字、层级舍入提出方法，仍然过早限定了搜索方向。本轮不再提出新算子，也不把任何历史提案选成默认答案。先准备一套覆盖内部表征、量化、时空变化、输出几何和残差旁路的观察；用户据此提出 idea 后，再将其转换成具体干预和因果测试。

用户本次明确要求“准备实验以收集证据”，覆盖此前仅限讨论的角色限制，允许编写采集、校验和分析工具。没有把“准备”解释为已经恢复模型/GPU运行；也没有改动共享 ACTIVE_RESEARCH、QUEUE 或历史暂停记录。软件单元测试只验证采集工具的行为，不产生科学结果。

## 一、方法审计：从方案名称回到可观察前提

下表是为设计观察而做的审计，不是新增 peer-support 票，也不是选择算法。

| 现有材料 | 已成立/已有证据 | 尚未回答的关键前提 | 最有用的观察 |
|---|---|---|---|
| G6A GELU parity v2 | 理想 GELU 恒等式；量化缺陷可定义；Smooth 有效列坐标需明确 | 真实 incoming/outgoing 关系是否普遍，量化后是否保留功能关系 | 完整 z→GELU→Q2 配对，绑定有效 W1/W2；全权重几何另做离线分析 |
| G6B code-cell v2 | 完整量化状态固定时的条件编码等价 | 实际边界距离、共享状态变化、可知时机；证书成本未知 | Q2 前后原值、完整64组及状态来源；本包不冒造内部 scale/code |
| G6C FP4 orbit v1 | E2M1 条件轨道等式与容量界 | 当前 HiF4 不等于 E2M1；真实局部复用率和读取费用未知 | 保存真实 Q1/Q2 数值与坐标；未来 MX/NV 需独立同格式采集 |
| G6D metadata proxy v2 | 有条件二阶矩关系；若尺度只径向变化，排序可能静态 | 是否有超出普通幅度的变化，是否对应功能差异 | 全局/分组幅度、归一化轮廓、signed 输出；不先训练路由器 |
| G6E W2 panel v1 | 固定操作数下 W2 贡献可分解，信息时机合法 | 幅度与有符号贡献是否一致；真实执行粒度与成本 | 配对 Q2、有效 W2、gate；先研究抵消，不先删 panel |
| G6F Q1 panel v1 | 固定 decoded 权重下 W1 扰动恒等式 | 对 Q2 和后续输出的实际因果作用是 action-dependent | 用户给出干预后才采集同前缀 native 对照；观测数据不能替代 |
| 本人 QCell tile / 层级舍入 E0 | 条件共享与布局反例，属于特殊数学现象 | 真实视频中的频率与功能意义；完整等值不是必要目标 | 作为观察的一部分；不能继续以此统领全部采集 |
| 门控残差商空间 | 条件数学结构；历史 nextnorm 不是局部写回的简单替代 | 真实误差方向、旁路与后续传播如何共同变化 | pre/post residual、gate；随后按用户 idea 测真实 nextnorm/flow |

必须保留三个审计结果：

1. **CAL24 dense 与 OP2 是不同资产。** OP2 的 V1/BaseY/TeacherY/StudentY 缺少 V2；但 CAL24 dense 的原采集源明确保存 V1、V2、Y、b2、gate，历史审计记录确认 24 sources、3,840 scopes、983,040 sampled rows 和 payload SHA 复核。它可用于部分分布和几何观察，不能笼统说“旧资产没有 Q2”。本轮未接触这些 HF tensor bytes，A100 上须先核对原路径与 SHA。
2. **旧 C1 动作集合没有产生选择问题。** 三个完整动作在 72 scopes 上的 oracle 都选择 UB1；所以“增加信息仍无收益”只约束这套动作和标签，不能排除其他信息或未来用户的 idea。
3. **局部好坏与完整轨迹可以反转。** R16 报告四个位置聚合的局部 SSE 相对最好固定对照改善约 6.13%，但完整视频 pooled RGB MSE 比最好固定对照高约 8.23%。这说明要分层收证；RGB 偏差也不是感知质量判决。

这些是原记录中的事实，不是本轮新跑出的结果。C1 的 72 scopes 仍是 6 个来源；CAL24 的近百万行不是近百万个独立视频。

## 二、首轮要回答的六组观察问题

| 观察 | 必须保存的原始量 | 能区分什么 | 本轮不能推导什么 |
|---|---|---|---|
| O1 量化改变了什么 | Q1/Q2 的 pre/post 配对，原 dtype、Smooth 与源码绑定 | 普通幅度、置零、方向变化、分组异质性；量化前后可直接比较 | 内部代码/尺度分布、native FP4 特性或省算安全性 |
| O2 FFN 特征如何形成 | W1 的 z、原生 GELU 输出、Q2；固定 W1/W2 面板与完整缓存身份 | 负端、符号、样本子空间与功能读出是否一致 | 由少量 sampled rows 得到全局低秩结论 |
| O3 输出是否有抵消 | Q2、W2 面板、gate、实际完整 Y | 大幅度通道是否贡献大、组间符号交互是否抵消或加强 | 固定5个面板代表全隐藏维；量化误差就是可删除量 |
| O4 时空和四步结构 | 同来源、同 grid 的固定 row IDs；额外相邻 x/y/t 配对 | 相邻空间、latent时间、跨 denoising step 的相似性是否不同 | 相同 latent 坐标等于跟踪同一物体；蒸馏造成了该结构 |
| O5 旁路与功能几何 | cross-attention 加法前 x、attention 增量、FFN前后 residual、gate | FFN幅度相对旁路尺度、写回方向及其步/位置差异 | 真实 nextnorm/attention/最终视频的因果敏感性 |
| O6 工作到底花在哪里 | 已有 R17/UB1 原始时延；未来独立的无采样 dense profiler | GEMM/QDQ/搬运/完整FFN/全生成的成本边界 | 用 observed 运行耗时估算模型加速 |

这六组观察共用同一套配对数据，避免每个 idea 重新开一轮不完整采集。原始数据保留，报告不只给一个平均相关系数。

**蒸馏与格式比较另列。** D-Q、真正 D-BF16 和 undistilled O 必须有各自有效权重、调度及来源绑定。仅关闭 activation QDQ 而仍用 QDQ 权重，不是 D-BF16。单独观测 D-Q 也不能证明蒸馏造成了某种稀疏/平滑性。本包暂不执行这个多模型比较。

## 三、已经写好的工具

- [公共工具](../../tools/ffnds_observation_common.py)：只读 safetensors header 审查；固定坐标采样；哈希与结构化记录。无 torch/GPU 导入。
- [被动采集器](../../tools/ffnds_observation_capture.py)：包装冻结的 dense HIF4CacheRuntime、QDQ 和 FFN 边界；原调用各执行一次；采样副本在原地 residual 修改前保存；异常时恢复钩子。
- [准备与采集入口](../../tools/prepare_ffnds_observations.py)：默认只做 preflight/header；collect 为显式的原生操作。复用仓库 verified_restored_assets 和原 dense runtime，不建立新 teacher 或训练分支。
- [离线分析](../../tools/analyze_ffnds_observations.py)：直接解码 BF16 safetensors bytes；QDQ配对误差、GELU/符号统计、样本谱、相邻坐标/跨步变化、固定 W2 面板有符号贡献。只分析通过 plain/observed 对照的原生记录；legacy payload 必须提供原 SHA。
- [协议](../../config/observation_20261025_01a16d3d.json)与[软件测试](../../tests/test_ffnds_observation_preparation.py)。

固定 W2 面板贡献采用 decoded FP64 参考算术，只用于几何解释；没有冒称原生 GEMM replay。其“合计能量 / 各面板能量之和”可以大于1，表示加强，不会错误截断为 [0,1]。

## 四、按原 A100 环境执行的最小次序

### A0 · 先复用旧资产，无需模型

在原 checkout 和原已验证 Python 环境：

~~~bash
python tools/prepare_ffnds_observations.py preflight \
  --out runs/observations_20261025/preflight.json

python tools/prepare_ffnds_observations.py headers \
  --files /absolute/path/to/one_archived_payload.safetensors \
  --out runs/observations_20261025/header_audit.json

python tools/analyze_ffnds_observations.py \
  --legacy-payload /absolute/path/to/one_archived_payload.safetensors \
  --expected-sha ORIGINAL_PAYLOAD_SHA256 \
  --out runs/observations_20261025/legacy_observations.json
~~~

后两个命令中的路径/SHA 从 CAL24 或 OP2 原 receipt 取，不猜测、不重新标记来源。先看一个 payload 的真实字段，再有针对性地读其他来源。header PASS 只说明结构可读，不等于 payload SHA 通过；legacy 描述也不会假设跨 step 的 row IDs 相同。

本地 Windows 准备环境的 preflight 正确报告 ENVIRONMENT_INCOMPLETE：没有原 .runtime、torch、safetensors 或 A100。这个结果不说明原 A100 环境坏了，也不能替代在原环境的核对。

### A1 · 一来源仪器验收，然后才承认新观察

协议选择全部 4 steps × blocks [0,10,20,30,39]，20 个原生位置；每位置沿用同一组最多48个坐标：32个均匀行，加最多16个邻接面板行。均匀面板与邻接面板分开统计。采样种子只依赖 source hash 和 grid，**不依赖 step/block**，所以能做真正的同坐标比较。

未来执行命令：

~~~bash
python tools/prepare_ffnds_observations.py collect \
  --index 0 \
  --out /absolute/HF-derived/observations_20261025/input00_attempt1 \
  --execute-native

python tools/analyze_ffnds_observations.py \
  --run /absolute/HF-derived/observations_20261025/input00_attempt1/RUN.json \
  --out runs/observations_20261025/input00_observations.json
~~~

每次 invocation 仅一个来源，plain dense 与 observed dense 各一次完整四步生成；没有额外 teacher FFN，没有自动扩展或重试。两个阶段都重置 scheduler.generator，并核对：

- 全部四步 latent bytes、视频文件 SHA；
- 原 dense 1,600 次 linear 和 1,600 次 activation-QDQ、零 cache miss；
- 20个预定 scope、不重复、不缺字段、全宽、有限值；
- 模型、cache、输入图片、采集代码和原始数值路径的身份。

只有 RUN=PASS_PASSIVE_CAPTURE_NOOP 才进入新原生观察分析。软件测试通过不替代这一验收。这个 no-op 是同一已初始化 pipeline 的新 seeded 请求，不是独立进程复现。

数据预算：按所有行字段均 BF16、最多48行计算，核心 payload 约177 MiB/来源；10个 expert/block 权重面板约63 MiB，另有视频、JSON与可能的 FP32 residual。应按至少0.5 GiB/来源准备额外容量，保持协议的80 GiB空闲储备。采样期间的同步、CPU拷贝、保存会污染时延，不能把它用于性能比较。

### A2 · 仪器验收之后，建立供用户提出 idea 的观察册

建议沿用已经暴露过的6个发现来源，覆盖 subject/background/camera 各2个；不消耗未来独立质量 holdout。原 C1 记录使用的是 r17_fresh23 manifest 中索引 [0,2,8,10,16,20]，**不是 formal20 同名索引**。只有原 manifest 的 SHA 和 image/prompt/seed 复核后才可沿用。

每个来源仍单独 plain/observed，最多12次完整 dense 生成；A1来源如果与其中一个完全相同并且协议一致可复用。这里是后续工作范围建议，本包不自动启动这些来源。

观察册应包含：
1. 每来源、每step/expert/block的分布图与异常原始切片；
2. 匹配局部幅度后，方向/符号/旁路差异是否仍存在；
3. 时空邻接与跨步变化，明确 latent坐标而非物体跟踪；
4. 输出几何与量化变化是否一致，并保留反例；
5. 已有证据和新增观察各自能排除的解释；
6. 一个简短“尚不能知道什么”的清单，留给用户提出新idea。

统计单位是来源；行/组/block作为重复测量。只有6个来源时不报告虚假的大样本置信度，先给逐来源结果与一致/不一致现象。没有事后挑最像某个方案的图。

### B · 用户提出 idea 后的实验接口

届时再填写一个新的具体实验合同：
- idea 的可测主张与竞争解释；
- 哪一段计算改变、原生数值路径和完整作用域；
- 同前缀 dense D-Q、适当同成本固定对照、idea干预；
- local → nextnorm/flow → free四步视频的结果链；
- 路由、搬运、补偿和所有额外工作计费；
- 发现来源与最终独立来源分开。

不能用A阶段被动相关性替代B阶段因果结果，也不能在用户尚未给idea时预先把干预设为K-panel删除、group mask或cache。

## 五、当前证据边界与交付验收

本轮已完成方法/原始资产审查，写好采集、分析、preflight和header工具，并运行软件合同测试。测试包括BF16字节解码、非法header拒绝、采样不污染全局RNG、原地更新前快照、每个原调用只执行一次、异常恢复、signed贡献加强/抵消，以及拒绝未验收run。fake transport只模拟调用/存储合同，**不是HiF4或Wan科学实验**。

目前尚未完成：
- 原 A100 环境 preflight 和原生 plain/observed 验收；
- CAL24/OP2 原 payload 的本轮重新读取；
- 真实新观察、nextnorm因果干预、质量评估或性能测量。

本次保存的“实验准备”不改写旧实验权限，也不让旧提案自动进入执行队列。

## 原始证据索引

- [CAL24 dense 完整采集审计](https://github.com/Qesire/FFNDS/blob/ff6942763d8ce3e1823a4814905ad5ee61da7c31/reports/worker_20261020/calibration24_full_capture_audit_RUN.json)，blob 17461b22f079baff317982a272453a93e439e805。
- [CAL24 原采集源码](https://github.com/Qesire/FFNDS/blob/ff6942763d8ce3e1823a4814905ad5ee61da7c31/tools/collect_ffn_operands.py)，blob 5590c733a8659e8bf74102e64b28f8a1c381e7f6，59–104行。
- [OP2 schema/源码绑定](https://github.com/Qesire/FFNDS/blob/ff6942763d8ce3e1823a4814905ad5ee61da7c31/research/20261023/C0_coupling_offline_results.json)，blob d24d695c56fbdc58050c152073637d3b3e63fa5a。
- [C1 完整动作分析](https://github.com/Qesire/FFNDS/blob/ff6942763d8ce3e1823a4814905ad5ee61da7c31/research/20261023/C1N0_analysis.json)，blob 084da00a007fdbe2609418677fdc2ef1f2438f02；原协议另含来源分组和输入hash。
- [R16 局部/视频排序](https://github.com/Qesire/FFNDS/blob/ff6942763d8ce3e1823a4814905ad5ee61da7c31/research/20261023/R16V0_analysis.json)，blob 732dfd2a81568406d6a25d32064bf6f3db386f51。
- [R7 有限传播结果](https://github.com/Qesire/FFNDS/blob/ff6942763d8ce3e1823a4814905ad5ee61da7c31/research/20261021/R7D1_results.json)，blob 01fc7229e04bddf278ddf8abf05f513db859d900。
- [冻结 dense runtime](https://github.com/Qesire/FFNDS/blob/ff6942763d8ce3e1823a4814905ad5ee61da7c31/upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/lightx2v_wan_distill.py)，blob daf07e204175b3993c7fd848ca6d9605bb55bbac，585–792行。
- LightX2V infer_ffn/post_process 固定于 ModelTC/LightX2V@ed146db64d0b6130c669285fc1aa0b280a6a4823，366–425行；原生SHA已写进协议。

