# UB1 泛化：先比较同源路径，再用固定预算恢复支持

2026-10-27 · 独立方法讨论 · 基于 main 3a4ef38 · 只读 SRC/RECORD/LIT，未运行数值脚本、模型、GPU或评测。

**建议最小实验**：先在保存校准行上做同输入、同组的原生路径一致性检查，确定拟合使用的 dense post-Q2 切片与缩窄 W1/GELU/Q2 的差异是否足以改变训练标签；然后固定 UB1 的四步宽度、同一 24 来源、相同 R256 预算，比较 C_mean 与 **whole64 分组的 MPC/ID 消费者恢复**，并保留只重拟合原支持消费者的对照。它直接使用已成熟的激活子集重构，不要求原创新损失。扩大来源数应作为与方法轴分开的第二阶段，而不是在新方法和更多来源同时变化时归因。

## 真实对象与第一分歧

`tools/fit_step_budget.py:34` 读取每源 256 行的 V1、V2、Y、b2、gate；V2 为 dense 已量化结果，按选中原组切片，并在该切片上拟合 alpha/b2 和 R256。`tools/step_budget_runtime.py:102` 则先 smooth/QDQ，W1 为实际缩窄 GEMM；经过模型原 GELU 后 W2 入口再次 smooth/QDQ，乘 alpha、以折叠 b2 和原选组 W2 执行，再与两次 BF16 R256 线性相加，之后原调用者 gate/writeback。参数与原组按 `step_budget_runtime.py:56` 排序整组打包。

这里有针对性的反证：`research/20261022/R12_HiF4_group_separability.json` 的 8 份检查对 whole64 选组/重排和行分割均通过，而组内置换出现不一致。因此**整组缩窄本身使其他组 Q2 scale 改变不是首选解释**。不过该验证不自动证明不同 GEMM shape、源 V1/V2 捕获边界、GELU、dtype、BF16 alpha 和加法在所有真实样本逐位相等；这些应在相同输入逐边界直接核验。已有 Q2 组内扰动连带效应的 E0 结果属于不同干预，不能嫁接为 UB1 整组选择不一致。

未来经具体登记批准时，选择预先固定的 step0/3 × 两个深度、两个校准来源及两个未用于拟合的来源，每例保存 256 行或已有可配行；在相同 post-Q1 输入、权重/偏置、选组、Q2/scale/dtype 下同时测 full W1 取组与 packed W1，再逐边比较 pre-GELU、post-GELU、Q2、alpha×Q2、W2+b2、R256 相加、gate 写回。先进行同代码重复/no-op、原 full V2 SHA/标签复现。报告逐位不等数量、分位、门控输出差以及相对原训练 residual 的比例；校准原尺寸 GEMM 再重复小样本（小M GEMM 不可冒充全M一致性）。若 V2 和 Q2 比特一致但 W2/补偿与视频仍有偏差，降低数值路径假说；若差异显著且原生路径重拟合降低跨来源门控误差，保留数值失配解释。没有可读真实 V1/V2 时先登记 MISSING，不能从哈希推断张量存在。参数阈值与预算执行前冻结；此阶段不声称质量改善。

## 同预算最小成熟方法对照

固定四步每层组数 54/108/108/162，真实原组 Smooth2/Q2，固定 24 来源×256 行及 source-level 验证折、λ、R256 训练资源，比较：

1. **A 原冻结 UB1**：原 C_mean + gain/b2 + R256，保留历史原字节/参数。
2. **B 重拟合控制**：原 C_mean 组固定，使用与候选相同的拟合数据和求解预算重拟合消费者；分别保留原 alpha+b2+R256 与允许 W2 消费者自由重构的版本。后者已有可学习参数增加，需记录额外离线拟合与储存；稠密 W2 shape 不变，理论在线 GEMM MAC 不变，权重不能视作原权重。
3. **C MPC 适配**：对真实 quantized post-Q2 激活按 64 通道块做块列选择，使用 block pivoted ID/等价块选择并拟合消费者矩阵；保留相同组数和与 B 相同的消费者与 R256 模型。MPC 原始逐列 ID 不自动具有整个 64 组的硬件/量化合法性，故这是受约束适配，不宣称原论文算法已被逐字复现。训练只用当前离线数据，部署无需在线 dense/Q2 观察器。

先以本次可读校准资产检查 V1/V2/Y 与原 W2、gate、全部所需层和来源身份；当前只确认 `.runtime/hif4_rebuilt_cache.json`、step0 RUN、原/补充20 CSV 和四原视频存在，**完整训练张量与24/48/96独立新来源的可用性未核验**。小试以 4 预定位置为诊断，不能推广为全160或自由视频。源级折叠按来源划分，且支持选择和拟合仅使用训练折；此前两批20已暴露，用作开发诊断，不称最终 held-out。要比较视频泛化则在完整160构造后使用此前隔离的新来源、同source/seed的自由 dense/候选，评估原三项、动作/背景和原速视频；需另行登记。

**决定性预测**：B 改善而 C 无额外收益 → 恢复/拟合更可能不足；C 较 B 改善且支持稳定跨来源 → 原 C_mean 的独立组能量忽略可恢复性或跨组协同是有力解释；训练提升而 held-out 恶化 → 新消费者容量/校准不足；三个局部都改善而视频不改善 → 训练边界与真实后缀/运动功能错位，而非证明某一单层造成失败。局部 raw/gated SSE、跨来源分位与负迁移、参数/pack/拟合费用和真实FFN时延均记录，最终优先来源级完整视频，不能以 256 行或 160 sites 充当独立视频。

**FLAP/OBS 竞争者**：FLAP的激活波动+bias可作为廉价评分和恢复基线；只用bias的在线成本更低，表达力也低于R256，不能把其损失单独归因评分。OBS 式协方差/二阶组序贯重构可作为下一候选，须包括曲率取得/阻尼、冻结相同数据和参数/时间预算；它与 MPC 的选择、消费者重构效果需要分开而非打包比较。将输出消费者矩阵自由拟合与 R256 保留的方案需要特别报告参数量、驻留、pack和真实 W2/QDQ 性能；虽相同名义MAC，未知真实延迟。原技术参照 [MPC](https://proceedings.neurips.cc/paper_files/paper/2022/file/f8928b073ccbec15d35f2a9d39430bfd-Paper-Conference.pdf)、[FLAP](https://ojs.aaai.org/index.php/AAAI/article/view/28960)、[OBS-Diff](https://arxiv.org/abs/2510.06751)，研究范围/模型不等同Wan视频。

## 来源覆盖与负例的区分

如能取得额外独立来源，以固定总 6144 行/位置比较 24×256 与 48×128，固定方法、参数和支持规则；另以 24×128 对照 24×256 测每源行数影响，按来源和seed独立拆分。若更多独立来源改善新的源级尾部而更多同源行效果弱，支持覆盖；若增加来源收益弱、方法 C 同预算显著改善，支持可恢复性；若训练/局部均改善而视频依旧退化，优先测试相邻block或完整真实轨迹目标。24→48 并非现有可用资源承诺，也不把 old20/new20 当盲测。

原 formal20 的 aesthetic/imaging/motion 平均相对下降约 0.874%/1.345%/0.0074%，supplement20 为 1.934%/3.166%/−0.094%；来源独立是统计单位，supplement 已暴露。最近两个 formal20 的原版 RAFT 高幅像素光流均值较 dense 低，但其 prompt 和镜头运动不同；较少动作不自动意味质量下降。`research/20261027/UB1_DYNAMIC_DEGREE_RESULTS.md` 与独立视频审阅不代表全 cohort。

**旧失败的精确覆盖**：OP3 固定原支持/gain、仅更换12来源 rank256拟合状态；在4 step×block20×两折的8比较中未超过曾用24来源的冻结 UB1，12源学生相对匹配12源teacher略改善。它不覆盖固定训练源数的 MPC 组选择/消费者恢复，也不能证明扩大来源、原生路径差异或新训练目标无用。LR2 在 supplement20 对 UB1 imaging 缺口恢复有限，不能把另一个 W2 消费者重构归入相同尝试；UB2 的时间预算调整失效只约束该 step 分配。R12/R14/C1 oracle 只覆盖冻结动作/损失，不关闭新合法固定支持。Affine54 使用 D×D+54组和不同训练设定，其局部好而视频失配提醒监督边界重要，不是UB1本身因果证据。

历史主FFN full160 3次均值 10.1776→6.5827 s、1.5461×；独立 R17 的单旧source热态完整生成中位 50.858→47.280 s、1.07568×，均不是新方案结果。任何恢复候选需计原 QDQ/GELU、BF16 R256、消费者权重驻留/pack、拟合及实际完整视频成本，尤其不能将 50% 主MAC当作50%整网速度。

## 独立代码审阅：未冻结的 native consistency runner

审阅快照 `tools/e0_ub1_native_consistency.py` SHA256 `4b539e60372428befacd67d288184ee76d009d26831adf40250468a9c3f91287`；测试 SHA `21a46c7079c7c7bf77d94b03fc9d110aa4d6838005c0312b766991553808cf93`，草案 protocol SHA `8c0ac1c3e6984e8a05cd7c819a35a2aa6ae627bf771a54ce60afe367b4e35602`。未运行 CUDA/模型/真实数据；CPU unittest 四项通过。

**必须修复 1｜系数身份不闭合（runner 73–95、103–111、154）**：`load_coeff` 只验证 chunk metadata 自己声明的 payload SHA、组数/形状和有限性，不验证其 step/block、report fingerprint、policy/cache/parent 绑定、原 W2/b2/gate 或 BF16 参数 SHA；step1/2 的 RR metadata 甚至丢弃。两个同 step 不同 block 的完整合法 chunk 对调后可通过当前检查，产生非 UB1 的支持及修正。按 `step_budget_runtime.py:41–76` 与 `trained_packed_runtime.py:32–64,75–101` 验证对应 report、identity 和每 chunk 的原始精确 SHA、fingerprint、step/block、C/centered 与张量 SHA、匹配 W2、folded b2/RR、原 b2 和 cache；对 chunk 路径与 case 做一一对应、不得仅靠所有资源的无序集合。验证实际 support 全部值/散列并写入回执。测试同形跨 block/step 交换及正确 payload 搭配错误 metadata 必须失败。

**必须修复 2｜panel、行与历史身份可错配（103–111、151–155）**：草案逐例的 `panel_sha256` / `run_sha256` 在运行时代码中未使用；检查 `panel` 在通用绑定集合仅证明一个文件在白名单内，并不保证其是当前 source/step/block 指定文件。行列表仅验证唯一范围，任意排列或合法替换也会通过，并且这个位置会影响 GEMM 形状行为。应绑定旧原生 RUN、原协议及其 case 精确 panel SHA、source/step/block/expert、原 row IDs 和顺序、权重 shard identity，核对 metadata；不能由 strict replay 推出身份，因为例如两例互换 panel 和对应权重/系数仍可能数值自洽。新增跨 case 交换、source12错绑source0及合法但错误行/顺序测试；若旧收据不存在则明确停止。

**必须修复 3｜目标中的原M/M160实际未直接比较（167–180）**：两臂分别对同一个 dense selected 报 mismatch count、norm、max；这不能推出两臂彼此相同。反例：各有1个不同元素且均距 reference 同样的范数，但差异位于不同坐标或不同符号。保留第一臂抽取后的 160 行 `z/gelu/Q2/R256/main_selected/main_narrow/combined/gated`，对第二臂同名值直接 `compare`；按原预算不增加 GEMM/Q2。还应加入 CPU 反例测试，而非只比较各臂标量。原 M 的非观测行均为0，输出是给定160行的形状控制，不代表原完整状态或视频因果。

**必须修复 4｜比较与失败记录（51–59、156–163）**：`compare` 不核 shape/dtype，会触发 broadcasting；例如 `(1,D)` 和 `(160,D)` 产生看似合法的相对误差和不匹配数，`view(int16)` 也隐含BF16尺寸。应显式要求两个同形、BF16、有限、 contiguous 或明确重排后比特比较；保留元素总数。baseline 一旦不等立即抛错且 `RUN.json` 未保存是哪个 case、各边界 mismatch/relative/max，导致结果仅是失败而非可解释的数值路径证据；先保存当前case、比较值、stage及失败边界再停止，不继续后续预算。补不等形广播、符号零、边界差异/失败留痕测试。

**源码一致性及范围**：W1的 `d['b1']` / W2 `d['b2']` 均在 BF16 且与实际原线性偏置一致时做法正确；但须从已绑定源记录验证它们就是该专家该层原偏置及原生 dtype。R256 的两层 BF16、先 BF16 alpha、W2 后 BF16 相加、gate 相乘，和现有 `step_budget_runtime.py:102–126`、`reduced_rank_residual.py:46–49` 的对应边界一致；记录的是 gate 乘积，未实现 BF16 residual 加法，不能称完整 writeback 等价。权重通过 BF16 cast 是该研究既定路径但未验证原始 dtype，也未记录 torch/CUDA、BF16 reduction flags 与环境；应显式核验/登记，避免把环境差异误归因 shape。当前预算算术正确：每case dense 2 GEMM+1 Q2，两种shape各5 GEMM+1 Q2，16例共192 GEMM/48 Q2；日志无新增模型或训练。严格 baseline 必须保持并在授权前重新冻结全部源码、资产、测试和 protocol SHA。
