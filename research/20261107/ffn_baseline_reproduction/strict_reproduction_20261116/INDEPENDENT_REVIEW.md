# ICC / Chipmunk 严格复现与重做：独立设计复审

审阅者：`/root/strict_repro_review`，与主文作者不同的 Agent。日期：2026-11-16（北京时间）。本轮使用 research-implementation / experimental-design 规范，读取作者源码、本地证据与重做设计；没有 torch、GPU、模型生成、因子拟合或集群执行。本评议不构成模型验收，也不把设计意见写成新的执行批准。

## 1. 当前判定与范围

**最终判定：设计层面可接受，可以发布为已独立复审的源码审计与受控重做设计。初次判定为 REVISE，实质问题经修改后已关闭，见第7节。此判定不是模型验收，也不是 ACTIVE 执行批准；仍不能称“方法已复现”或“公平实验已完成”。**

已审阅的主体是 `ffnds_review/research/20261107/ffn_baseline_reproduction/STRICT_REPRODUCTION_AND_REDO_20261116.md`，并交叉核对：

- `replication_chip/UPSTREAM_WAN_EXECUTION_AUDIT.md` 及固定作者 `modules/mlp.py`、`layer_counter.py`、`topk_indices.cu`、Triton mm2；
- `fidelity_icc/ICC_FIDELITY_AUDIT.md` 及作者 `stat_upd`、`func_decomp.py`、`utils_decomp.py`；
- `replication_icc/STRICT_REFERENCE_AND_REDO.md`；
- `replication_design/CONTROLLED_REPRODUCTION_PLAN.md` 与 `COHORT_PROPOSAL.json`；
- 本轮 counter utility、trace 与源码字节修正记录。

主文在审阅期间已经修正 CAL256、EVAL22/396、SVD 数值身份、共享 scale 的 epsilon、D4 时钟消融无辨识力，以及真实 patch-grid 等问题。以下将已关闭问题与还需合并的事项分开，避免把正在修订的旧版本当成最终方案。

## 2. 初审要求及修订依据

以下保留初审提出的问题，最终关闭情况见第7节；它们不是额外待批事项或第二份执行队列。

### R1. 主文、ICC 专项与综合计划必须只有一个执行规格

首次读到的综合计划仍是 SVD/CD 两家族、旧九臂、EVAL11/198；主文已经是 SVD/CA/CD 三家族、EVAL22/396。ICC 专项另有 EVAL22×7 臂/724 的独立建议。若三者同时作为正式规格发布，会使执行者合法地遗漏 CA、重复启动队列，或按不同样本规模拼表。

必要修改如下：

1. 以最终 `CONTROLLED_REPRODUCTION_PLAN.md` 为唯一运行数量和臂选择依据；其它专项保留算法与验收要求，明确其旧数量建议由综合计划覆盖。
2. 正式 ICC 参考包含三家族；候选为 SVD/CA/CD × {512,768,1024}，另有 CD-r1408。不得因正文写了三家族、表格却只有 SVD/CD 而遗漏 CA 的全80站点因子与完整自由轨迹。
3. 正式 EVAL 为22来源×2seed，最多九臂。相同配置去重后的上界为396条 D4；这个数仅属于 EVAL。
4. CAL256 全有效 token、四边界统计是本轮正式参考。11/22/64 是嵌套消融；不能以小前缀局部曲线“已经稳定”为理由不运行256，又称完整参考完成。
5. 实际图像 bytes、近重复 source cluster 与拟合祖先尚未绑定；metadata 选择与实体输入验收必须保留不同状态。当前文件中 image hash/cluster 为 null，不是已完成输入冻结。

### R2. 总预算必须覆盖 DEV 选点和必要消融，不能只冻结396

最终计划须给出所有完整模型调用的有限总账，至少分列 CAL、DEV 候选、DEV 适配消融、neutral、EVAL、计时重复、可选作者原工程小配对、O40、失败及重试。局部模块/SVD/统计计算另列，不伪装成“零成本”。

尤其需要检查：三家族各自选点所需的完整 DEV 视频，是否真的包含在 DEV 配置数；四态→两残差、CAL22→256、全 token→抽样不能只出现在文字要求中而没有对应父配置与计数。技术重复不增加独立来源 n。一次有效 preflight 已完成正式条件时应直接计入并复用，不能再派一次免费 smoke。

无需编造精确 GPU 小时。预估门限可保留为设计值，但 A10080GB、192GiB 主存、可用磁盘或两卡资源目前均未在本会话核实，必须写 `PLANNED_RESOURCE_UNVERIFIED`。资源不足应记录真实失败与未完成分支，不能自动删去四态、CA/CD 或全 token 统计。

### R3. 选择规则必须在 DEV 结果前具体化，EVAL 只执行已锁定配置

“按质量约束与实测成本选取”仍不是可执行规则。需要固定候选身份、主排序指标、该指标适用 source 集合、并列处理与其余指标的报告方式。若项目没有现成科学可接受阈值，可以按事先指定的主指标选点并报告完整质量向量，不必发明5%等非劣界。

局部函数风险可以筛选 DEV 候选，但不能冒充自由视频质量。用于正式比较的家族选定点必须有 DEV 完整闭环；局部与视频排序不同应保留。EVAL 结果不能触发换 rank、换 CAL、换 selector 或换 source。

当只有22个独立来源且部分指标仅适用于其中子集时，不预先承诺非劣或显著优胜。seed先在 source 内聚合；统计区间和失败范围对应真实独立来源数，不把44视频、160调用或 token 数当 n。

### R4. 将同索引算术、统计精度和有辨识力的适配对照落实到具体父配置

这些不要求扩大成因素全排列，但应在验收清单中明确：

- **Chip selector 与算术分开验。** 原 selector 有 atomic 写入顺序及补齐支持的合法非确定性，不能以原始 indices SHA 必同作为通过条件。算术核验应向原 kernel 与适配器注入同一实际 inds/counts。另检查 count 范围内每项已写入、索引范围/唯一性/对齐；上游审计给出的“13个未选列落在同一 thread 序列”sentinel 用例可覆盖具体补齐风险。此 GPU 用例尚未执行。
- **ICC 分解数值身份。** 主文已补 driver、full_matrices、TF32 和 factor action。需把这些字段落实到因子 manifest。SVD 向量存在符号/退化子空间不唯一性，不要求不同合法分解逐元素或 SHA 相同；推理对照使用同一实际因子。
- **CA/CD 累计顺序。** 原作者是先 token mean，再样本内 channel min-max，之后来源平均、sqrt、max-ratio，再一次 epsilon 和加权分解。常量幅值归一化得到全1。batch=1/source等权可以避免尾 batch 权重不一致。CA 在合法端点采幅值，CD 在同专家真实端点对采差；不能跨权重相减。
- **三个精度轴分开。** 推理、统计与分解各有 dtype/backend。BF16 原值转 FP32 后求差，与 BF16 先求差再转型不同；不要同时改统计与在线算术却称单因素。作者原 dtype 正对照和 Wan BF16 同输入对照都需要，局部通过不能写成全视频精度影响已排除。
- **D4 时钟无需重复视频。** HHLL、每专家两次调用、冷启动 anchor 且 step<10 时，global 与 expert-local 时钟在 D4 产生相同 A/H 和 mask 重算路径。主文已经修正，应在连续多hit及 step9/10/11/12 的边界轨迹比较；不能给无辨识力的 D4 对照重复收费。
- **布局/尾组/搬运。** 主端口使用作者 voxel 正逆函数；只改 FFN 局部布局时保持 native attention 不变。DENSE_TAIL52→VALID_TAIL52、voxel→raster、author arithmetic→native rounding 是不同变量。纯搬运若证明同元素同算术，可用数值与实测成本验收；有真实数值/支持变化的改动保留有限闭环。

## 3. 已实际发现并关闭的交付错误：counter 源码字节绑定

本审阅者第一次实际执行 utility，使用此次经 Git blob 验证的上游 `layer_counter.py`，得到 exit_code=1：`Source bytes differ from the reviewed author commit`。

随后逐字节比较确认，初稿使用的旧副本恰好多出一个文件末尾换行：

| 对象 | bytes | SHA256 | Git blob SHA |
|---|---:|---|---|
| 旧工作副本 | 3361 | b58bff2c407c43795ba7ad488e5da7efbf69c440dd2f38ef8dcb73c1392b2d79 | bbc3e51089425232c5f9a1856797cd5d863377d4 |
| 固定作者原文字节 | 3360 | cb9ffab57832ce69141ecd914fbb3f2c873c26bdca31eee1f7e2b41c01764288 | 6a870f07cdf05482667b6a65fe015ccd035bd007 |

主代理没有放宽检查，而是将 utility 绑定到已核实的第二份原文，并保存 `COUNTER_SOURCE_IDENTITY.md`。**本审阅者修订后重新运行，exit_code=0，三组错配数仍为 [1,1,120]。** 输出另存 `replication_root/INDEPENDENT_COUNTER_TRACE.json`。这一修正没有改变类的 AST 或计算语义，原先调度结论仍成立。

实际命令：

```bash
python ffnds_review/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116/audit_chipmunk_counter.py --source replication_chip/upstream/src/chipmunk/util/layer_counter.py --output replication_root/INDEPENDENT_COUNTER_TRACE.json
```

这只是纯 Python 接口检查。它不证明 SparseDiffMlp 已导入、selector 已编译、Triton 已运行，或作者论文发表运行发生同一错误。

## 4. 独立数学复核结果

以 d=5120、m=13824、每 token 两层 Dense 主 MAC `F=2dm`，ICC hit 主 MAC `L=2r(d+m)`，Q1 额外投影 `a=d²` 计算，`F=141557760`、`a=26214400`。

| 配置 | 完整 anchor+hit 周期 / 两次 Dense |
|---|---:|
| ICC四态r512 | 0.5685185185 |
| ICC双残差r512 | 0.6370370370 |
| ICC四态r768 | 0.6027777778 |
| ICC四态r1024 | 0.6370370370 |
| Q1r2 | 0.6851851852 |
| ICC四态r1408 | 0.6884259259 |

匹配方程为 `2r(d+m)=2a`，得到 `r*=a/(d+m)=1383.78378378`。r1408 对齐后比 Q1 周期主 MAC 高约 **0.47297297%**。主文数字准确；它是主矩阵乘预算近邻，不能称实测时间、端到端总算力或内存相等。

N=31668 时，一专家40层、BF16四态 u/z/h/y 的逻辑缓存为 **89.39483643GiB**，两残差恰为一半。CA/CD两套、全80站点、256来源、每source四边界 FP32统计向量为 **5.78125GiB**；如果同时持久化每source512行 U0/U1/Y0/Y1，BF16逻辑 payload 为 **400GiB**。这些均不含格式开销、模型、临时区和 allocator 峰值；不应将400GiB面板隐藏在“只存统计向量”的描述中。

调用计数也准确：D4全40层共160调用；ICC/Q1全层p2为80 anchor+80 hit；Chip保留每专家前2层恒Dense后，互斥分类是 **8 fixed-Dense + 76 anchor + 76 sparse-hit =160**。计数应由真实 trace 验证，不能把这项算术推导当已运行160调用的证据。

## 5. 对“Chip原本做过Wan，原代码应该能用”的审阅结论

主文现在的回答忠实于已读证据：作者确实做过 Wan2.1 T2V，官方入口存在；论文视频配置与默认 Wan 接线使用稀疏 attention，FFN未接稀疏 MLP。原作者另有可复用 SparseDiffMlp、A100官方 Triton路径和 compiled utilities，应优先直接运行这些模块建立 reference。它们尚不是当前 Wan2.2 双专家 I2V 四步 FFN-only 的完整即插即用入口。

因此不能由“原Wan没有稀疏MLP”得出“作者没做过Wan”，也不能由“有Wan脚本”得出“当前MLP端口已经完整”。I2V缺初始化/step接线、真实patch-grid、双专家与request缓存、末尾52行、plain callable内部转置权重生命周期，都是有代码依据的必要接口问题。原voxel已支持非整除几何，无需自造替代排列。

不需要追加完整FLUX、LLM或50k图像FID来证明当前FFN端口；也不能拿官方Wan attention小配对代替MLP的multi-hit/selector/kernel验收。

## 6. 接受的执行和解释边界

- 旧 A/BW/BWS 保留已完成身份，不重开同一历史问题；只复用实际兼容的 payload、因子和指标。
- 新正式参考不允许以2R代四态、Global-P64代作者Chip、普通SVD代CA/CD，或以dense gather原型速度代原Triton速度。
- 当前所有模型、GPU、编译、全图neutral与质量/速度验收仍为 `UNVERIFIED`。可接受设计文稿并不改变这些状态。
- 数据实体与硬件绑定属于实际运行必须具备的对象，不是新的审批层级。用户已经要求完成严格复现和重做；环境具备后按统一有限方案继续工作。
- 若修正后作者reference追平或超过Q1，应据实更新结论。若仍差，负结论只覆盖已验证实现、已测参数和当前Wan任务，不推广为所有ICC/Chip或所有视频模型失败。

## 7. 最终合并复核

最终复核直接读取出版候选目录 `ffnds_review/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116/`，不以工作目录的旧草稿代替发布版本。

| 初审问题 | 最终出版候选中的处理 | 结论 |
|---|---|---|
| CA遗漏、EVAL11/22、多个队列冲突 | 三家族10个离线候选，7个四态候选进入DEV；统一九臂、EVAL22×2；ICC专项明确唯一主账 | 已关闭 |
| 小CAL可跳过256 | CAL256 full-token为正式参考；11/22/64为嵌套消融 | 已关闭 |
| CAL22额外视频冲突 | CA/CD全80仅统计/action局部比较；完整CAL22 DEV只做CD与Q1共22条，未声称CA闭环校准量影响已排除 | 已关闭 |
| neutral低估 | DEV内两条hook-bypass复用；plain、ICC forced-dense、Chip forced-dense各2；Q1/GP64无法复用时各2；额外上界10 | 已关闭 |
| 精度/抽样诊断未定义或暗增预算 | 六sentinel的同CAL/rank P512-vs-full-token局部诊断；本轮明确不纳入4条FP32-FFN自由生成，也不声称排除其视频级影响 | 已关闭 |
| DEV选择规则含糊 | 固定局部筛选、完整DEV主指标、适用7来源与并列规则；EVAL不调参，不发明非劣margin | 已关闭 |
| counter utility无法使用核实原文复跑 | 绑定正确3360字节源码；独立实际复跑exit0、错配[1,1,120]，保留修正记录 | 已关闭 |
| 发布目录虚称附带39份源码 | 改为工作目录曾取回核实39份；发布包只保存固定URL/Git-blob/内容SHA manifest | 已关闭 |

### 最终预算独立重算

`256 CAL + 121 DEV主队列 + 22 CAL22闭环 + 24适配/部署 + 10额外neutral + 396 EVAL + 60计时 = 889` 次基础D4执行。

另有最多8次技术重试，故D4上界 **897**。按每条4次模型前向，上界3588 NFE；O40最多6次、按每条80次前向保守计算为480，Wan2.2合计上界 **4068 NFE**。最多4次作者原模型小配对生成另按实际步数/CFG单列。已有合法产物与未触发的条件逐条扣除，取消的额度不转给新候选。该账是待执行上界，不是已用算力或已验证GPU小时。

### 最终名单的独立机器检查

对出版 `COHORT_PROPOSAL.json` 运行纯Python断言并得到 exit_code=0：

- CAL256、DEV11、EVAL22内 record ID 唯一，三者两两交集为0；
- CAL11、CAL22、CAL64均是CAL256的嵌套前缀；
- EVAL保留首次11个元数据候选，最终11个官方类别各2个记录；
- EVAL的22个image-bytes hash和22个source-cluster字段仍未绑定，历史未见性仍为 `NOT_VERIFIED`。

因此只确认**元数据分组一致性**，没有把 record ID 不重叠提升成真实图像簇独立或历史盲性已证实。参数拟合祖先、实体输入、noise、checkpoint、数值环境和资源仍要随真正运行绑定。

### 本次复核的出版文件身份

| 文件 | SHA256 |
|---|---|
| STRICT_REPRODUCTION_AND_REDO_20261116.md | 026f82d07e9c15abedcd3d8bac8d1c2b17a96a30cdad3b2df4dfa18971bc48e3 |
| CONTROLLED_REPRODUCTION_PLAN.md | e247561641179becdb089dcc06bf1be4856a8f1ec6d4f682ca330731c3dc5e04 |
| ICC_STRICT_REFERENCE.md | 26fd5592bb6606a3816e8ee092fbcf892c814f2e9a69500c48b56290e5b663fe |
| CHIPMUNK_UPSTREAM_AUDIT.md | 8725396780d665a4a65a5c6f71bb8c6ed97a59df9d6fbfb0ddb785007e0a0c0d |
| COHORT_PROPOSAL.json | 80eae74ee993b85a6e40d10585bba3108a839b2db6946b5886367a445f5674a6 |

**最终结论：可发布此设计与审计包。** 没有发现仍需改动上述有限设计的实质算法或预算错误。实施代码、编译、真实输入、资源和所有模型验收保持未完成；这不妨碍诚实发布当前设计，也不能用本评议替代将来的执行证据。后续不重开历史A/BW/BWS，不新增审批层级；按已授权目标和统一范围继续完成必要工作。
