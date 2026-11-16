# ICC 作者式校准修订：独立审查

日期：2026-11-16。审阅者：`/root/strict_repro_review`。

状态：**ACCEPTABLE_AS_DESIGN_DOCUMENT / NOT_EXECUTION_ACCEPTANCE / NO_MODEL_RUN**。本文件审阅本轮真实 clean clip 校准修订，不把此前 0e29 提交中的 99 条 D4 方案当成本轮新提案。最终回核于2026-11-16 05:35 UTC完成：本轮已发现的配方、数据身份、触发顺序与预算问题均已在正式规格中处理，可以作为设计修订发布；不能据此宣称校准已经完成或运行范围已验收。

## 1. 源码层面的判断

**新配方恢复了作者 CA/CD 的关键采集结构，可以作为明确记录 Wan 适配的主参考。** 它必须从真实干净内容编码，再随机选择合法时间并前向加噪。CA 在加噪点执行一次 Dense 模型调用；CD 从加噪起点调用模型，使用其真实预测执行一次原生调度器更新，再在所得下一状态调用同一专家。旧 D4 rollout 的两个端点不能替代该过程。

这一结论不表示输入视频已经存在、采集器已经实现或模型已经运行。当前资产审计只能证明旧 CAL11/CAL_EXT11 有图文条件，尚未证明所需真实视频或可追溯 clean latent 可用。

### 独立核对的作者依据

固定 ICC 提交为 `dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`。

| 主张 | 独立核到的源码 | 审查结论 |
|---|---|---|
| CA 从真实输入编码后随机加噪，执行一次模型评价 | `DiT/func/func_decomp.py` 的 `DiT_wtv_src_stat_loop_mag` | 成立；不能改成生成轨迹采样后仍称原采集程序 |
| CD 第二输入来自第一次实际 DDIM 预测与更新 | 同文件 `DiT_wtv_src_stat_loop_delta_mag` | 成立；两个独立 q-noise 端点即使用同一噪声也不是该 CD |
| 条件 dropout 为 p=0.1，CD 两次条件相同 | 上述两个函数 | 成立；CFG off 不构成删除 dropout 的依据 |
| 默认256是数据项数量，不是256条完整50步轨迹 | `DiT/gen_decomp.py` 与上述调用循环 | 成立；默认batch16、epoch1时CA/CD为16/32次批调用，即256/512次样本级评价 |
| 统计覆盖四边界，先绝对值再 token mean、逐样本通道 min–max、再累加 | `DiT/models/models_stat_svd.py` 的 `stat_upd`、`Mlp.forward_stat` | 成立；不能把两端通道均值之差当平均绝对差，也不能先按tile归一化 |
| 统计平均后sqrt，cap1000，缩放SVD及逆尺度恢复 | `DiT/func/func_decomp.py`、`DiT/utils/utils_decomp.py` | 应继续保留，不因改变采集入口而修改 |
| 公开仓库只能支持 DiT 配方的具体实现主张 | 固定完整tree与已取回源码 | 未发现公开PixArt校准分支；不能把本项目空文本映射说成已核实的作者PixArt实现 |

本次重新计算了7份 ICC 源码及4份 Wan/LightX2V 新源码的字节数、SHA256、Git blob SHA1；**11/11匹配既定来源身份**。具体结果见 `INDEPENDENT_SOURCE_CHECK.json`。这是源码身份检查，不是对这些模块进行 torch/GPU 执行。

## 2. Wan 调用合同及必须保留的细节

固定 LightX2V 提交 `9cb6929d83814ff28c4e1386532298fe93a96af1` 的 DMD2 四步配置与 `WanStepDistillScheduler` 支持以下设计。实际执行时仍需绑定安装树和解析后的配置，不能用公开示例配置替代运行配置身份。

### 时间、噪声与专家

名义列表 `[1000,750,500,250]` 经 shift=5 得到实际 sigma 近似 `[1,0.9375,5/6,0.625]`；模型时间为其乘1000。模型应使用调度器实际数组，不从显示小数重新构造。DMD2 的专家选择依据 `boundary_step_index=2`：index0/1为H，index2/3为L。

对真实 clean latent \(z_0\) 与一次新高斯噪声 \(\epsilon\)，

\[
x_{\sigma_i}=(1-\sigma_i)z_0+\sigma_i\epsilon.
\]

| 分支 | 合法采样 | 必须实际执行的输入路径 |
|---|---|---|
| CA-H | 均匀选index0或1 | 从clean latent直接前向加噪到所选sigma，H单次Dense调用 |
| CA-L | 均匀选index2或3 | 从clean latent直接加噪，不先经过H rollout，L单次Dense调用 |
| CD-H | 唯一合法pair 0→1 | fresh加噪到0 → H调用 → 原生step0 → H调用1 |
| CD-L | 唯一合法pair 2→3 | fresh加噪到2 → L调用 → 原生step2 → L调用3 |

不允许跨专家1→2，也不把terminal当不存在的第五个模型评价点。作者的 `t1=t2-1` 对应原采样器索引的相邻点，不能在 Wan 中解释为数值 timestep 减1。H-CD在sigma=1处的生成latent等于噪声，clean内容在该起点只通过I2V/文本条件进入模型；这是部署网格的真实性质。

### CD 实际更新与缓存

原生 `step_post` 将flow预测及输入转换为FP32，**先**计算 \(x-\sigma v\)，**再**加 \(\sigma_{next}v\)，最后转回输入latent的dtype。数学上的 \(x+(\sigma_{next}-\sigma)v\) 不授权更换有限精度执行次序。应直接调用绑定的原生函数，记录第一预测、实际step输出及第二次调用输入的身份。

`prepare/reset` 会重新初始化噪声latent；fresh-noised输入注入后，CD两次调用之间不得再次调用会覆盖它的初始化路径。统计缓存、方法缓存与scheduler状态应明确初始化，合法下一索引通过原生接口设置。CD第二次输入不得再次q-noise、读取旧rollout、混入其他专家状态或开启FFN近似。第二次之后无需为了统计再生成后续完整轨迹。

### 条件与 VAE 适配

1. 作者DiT的 class-null 映射为原生T5对**真正空字符串**的编码，保留p=0.1 Bernoulli抽签；CD pair重用同一个抽签结果、embedding、首帧条件及mask。CFG=false只执行一次所选条件forward。
2. 官方 Wan `image2video.py` 会把空的 `n_prompt` 替换成默认负面提示。采集器不能通过该替换后的入口取得所谓empty context；也不能用全零embedding替代空文本编码。默认LightX2V CFGoff不建立 `context_null`，故须明确替换实际使用的主context。
3. 原生 Wan VAE `encode` 返回通道归一化后的mean；主模式应登记 `wan_native_normalized_mean`。它与作者 `.latent_dist.sample() * 0.18215` 不同，是架构编码适配。禁止重复归一化、搬用0.18215或把已归一化mean和未经同样变换的logvar拼成posterior。
4. 全clean clip latent与首帧加零帧构造的I2V条件是两个对象。只对前者加噪；后者和mask保持原生合同。两者的帧来源、空间变换及文本须一致。只存在零填充条件不能证明真实clean clip存在。
5. D4网格、分专家合法pair、batch1/source等权、CAL11/22、空文本映射与VAE mean均应明示为目标域配置或必要适配，不能宣称与原DiT/ImageNet/256配置逐项相同。没有理由自动增加50时点、CAL64/256或新完整视频臂。

CA/CD允许共用source。若CA恰好抽到CD起点，且噪声、dropout、条件和完整输入身份均相同，可同一首forward同时记录mag/cache；这保持各自边缘分布，只改变估计间相关性。它是合法的可选复用，不应成为强制省调用目标。CA抽到后一时点时必须单独fresh加噪，不能拿CD第二状态代替。预算继续使用不共享的66/132上界。

## 3. 数据和公平性边界

`ICC_CALIBRATION_ASSET_AUDIT.md` 固定的旧资产表明：

- 旧CAL11及CAL_EXT11是JPG/prompt记录，缺少真实视频、帧内容和clean latent来源绑定。
- `VB_DENSE22` 是CAL11+DEV11的轨迹捕获，**不是CAL22**，也不是作者式clean数据采样。
- BWS生成视频或终态latent若存在，仍是生成数据；单图重复、首帧加零帧、生成终态均不能默认替代本轮真实clean输入。

主参考需要绑定11个真实clip或可追溯到它们的合法x0；扩展时追加11个独立来源并保持前11的RNG记录、统计及身份不变。数量沿用11/22不意味着沿用旧image record IDs。应在source/近重复组层级与DEV隔离，而非只检查文件名不同。

旧rollout的CA/CD统计和因子不能只因同模型、同shape、同seed或同CAL11名义而抵扣新采集。普通SVD同权重因子可按原规则复用，Q1旧CAL11参数可保留其原始身份。**若ICC换用真实clip而Q1继续旧图文CAL，五臂仍可进行同DEV、同seed和同部署条件的开发比较，但校准来源未配平；不能把结果差异单独归因于算法机制，或声称逐源校准已公平匹配。** 本次不要求因此扩充实验队列，只要求报告各自产物的真实校准协议与适用结论。

真实clip未绑定时可继续普通SVD、原模块与采集器静态工作；CA/CD采集及三家族局部选择仍未完成。不得因为缺少CA/CD就直接把SVD写成“三家族按预定规则选出的主ICC”。

## 4. 独立预算复算

每source、每expert使用CA1次+CD2次；两expert为6次样本级全去噪器前向。batch1时样本级NFE就是实际模型调用数。

| 项目 | 上界 | 含义 |
|---|---:|---|
| CAL11 | 11×2×3=66 NFE | 主统计，不生成完整视频 |
| CAL_EXT11增量 | 66 NFE | 新增11source，不重采前11 |
| CAL22累计 | 132 NFE | 不是另加132，亦不是132条视频 |
| 主五臂 | 5×11=55条D4 | seed42，DEV开发比较 |
| Neutral额外路径 | ≤10条D4 | hook-bypass父输出已计在主55；逐path/execution_id抵扣 |
| 同factor 4S→2R | 3条D4 | 父4S来自主表 |
| 条件CAL22闭环E5 | ≤3条D4 | 仅冻结主CA/CD满足既定局部触发；SVD时为0 |
| 唯一条件适配E6 | ≤3条D4 | 按既定优先序，oracle错误先修，不自增支线 |
| 全分支触发、重试前 | ≤74条D4 | 删除旧22条CAL完整D4后成立 |

基础视频为55+10+3=68；两个条件分支各3后为74。全局技术重试仍只有3次，视频与校准共用。正式方案已将校准重试明确为CA单调用或CD不可分的两调用pair，失败时回滚该事务统计并重用相同样本/RNG；不是整source六调用重跑。令 \(J_v+J_c\le3\)，则

\[
V\le74+J_v,\quad C\le132+2J_c,\quad
NFE_{total}\le4(74+J_v)+132+2J_c\le440.
\]

所有重试用于视频时才达到77条视频，此时总上界440 NFE；全部用于校准时上界为74条视频及138次校准调用，总计434 NFE。不能把77条视频和138次校准NFE各自极值相加；也不能给校准再开独立3次重试。失败已消耗的调用应进入实际账，不能因任务未产出有效样本而消失。

与旧99条D4账比较，完整视频上限下降，但旧最大396次去噪器调用变为新最大440次，增加44次、约11.1%。原因是忠实CA/CD每source最多6次前向，旧D4轨迹只有4次。两者工作内容不同；不能把视频条数减少说成总去噪计算已减少，更不能据此推断墙钟时间。

NFE不包括真实视频读取、VAE/T5编码、四边界缓存搬运、局部核验与矩阵分解。主CAL11为184次；可选CAL22局部诊断增加24至208，E5最多再增148；这些矩阵分解另账记录。更换校准分布不授权隐性新增网格。全tokenCD每expert的四边界首调用缓存约89.395 GiB（BF16，40层），逐source/expert卸载可减少同时常驻量，不能据此宣称实机内存已验证或把主统计降为P512。

## 5. 正式文档回核与执行证据

### 回核结果

正式方案、ICC专项、机器配方、预算及cohort元数据已统一；本轮没有需要继续修改才能发布设计的未决实质问题。以下问题已实际查到修订后的文本或字段：

- 主采集改为真实clean视频编码与fresh forward noise；CD第二状态明确来自原生step，不能再次加噪、reset覆盖或抄旧轨迹。
- 原生Euler的FP32先减后加顺序保留；实际sigma与按index选择专家均有明确合同。
- p=0.1 dropout仍保留；明确真正空文本T5与高层默认负面提示替换的区别，I2V条件保持。
- VAE mean、flow、D4 grid、分专家、统计精度等适配明示；没有把Wan接口变化写成作者DiT逐字等价。
- 真实视频绑定缺失如实保留，旧图文名单不充当clean输入；Q1保留旧身份，校准来源未配平的归因限制已写明。
- 旧rollout CA/CD统计及因子不得抵扣新配方；缺少CA/CD候选不能借SVD完成“三家族选择”。
- CAL22采集是主视频结果可见前冻结的一次可选统计诊断；E5只决定取得该诊断后是否增加视频，消除了前置条件循环。未选择CAL22时E5没有可用R22，不能凭空触发。
- 68明确包含55主视频；原22条完整CAL D4删除。共享retry仅3次，校准按CA单调用/CD双调用事务重试，联合上界440，离线成本另账。
- Neutral依真实path与execution_id抵扣，hook-bypass两条属于主55，额外路径最多10；不能把plain产物抵扣不存在的hook执行。
- 三家族、r1408、六位点局部选择规则不因这次采集修订扩大；不自动新增CAL64/256、EVAL22、多seed、O40、原系统完整生成或60次计时。

独立执行统一引述核验器得到 **exit 0，42/42源文件、91/91摘录通过，45项主张的引用结构完整**，结果为 `INDEPENDENT_CITATION_VERIFICATION.json`。该检查只证明字节、摘录和固定链接对应，关键主张的语义由本报告前述源码阅读支持；不把引用结构检查等同于全部主张的科学证明。

独立预算检查亦 **exit 0**：66/132校准NFE、68基础视频、6条件视频、3共享retry及所有允许分配的440上界通过；184/208/356分解计数相符，结果为 `INDEPENDENT_BUDGET_CHECK.json`。

### 本次审阅的正式版本身份

以下路径均相对于 `research/20261107/ffn_baseline_reproduction/`；带星号的简名位于其 `strict_reproduction_20261116/` 子目录。完整11文件身份记录另存 `FINAL_REVIEW_FILE_IDENTITIES.json`。

| 文件 | SHA256 |
|---|---|
| STRICT_REPRODUCTION_AND_REDO_20261116.md | `1edbfd908b7523f259c316ba0633565cbdfb5a891e968865ee7b4e111c32416c` |
| CONTROLLED_REPRODUCTION_PLAN.md * | `d383b778adad394ae2f0b7340443db8494d92eb94f770d90f71aa07097c3c73b` |
| EXPERIMENT_BUDGET.json * | `8038ade8cd88a2f014c37cbc507098f4e4eb76f5059dde4a65ef480f6155e3fd` |
| ICC_AUTHOR_CALIBRATION.md * | `24d15202a5af52f986055a2b6c94bbadddd852f9569fafe9ccc323b69743f1ca` |
| ICC_CALIBRATION_RECIPE.json * | `aeb24a6ae3c8e07aeab4b2223318b1e5d3bb65f1ed3cd45e335f99a6aac88189` |
| ICC_STRICT_REFERENCE.md * | `3dacc21294fbca8cfc764d6b0cdf10291b23c6cf8a0ae65cf229553ffdd48381` |
| COHORT_PROPOSAL.json * | `dedd848b90916cfab903de16abc3c636493dd9fb05c5cdf7efdf23392e70fe4b` |
| CODE_CITATIONS.json * | `3777e3ecfa0bbc8f02a2fc947360d2f07c2f3b448b60efe04d57963736701bdb` |

### 尚未取得的执行证据

本轮实际torch/模型/GPU运行：**0**。尚缺真实clean视频/x0与DEV近重复排除的实体绑定、最终producer及安装代码/权重/配置绑定、可用资源证据，以及作者模块/采集器/四状态端口的实际数值验收。当前不能说CA/CD已复现、三家族主ICC已选定、五臂重做完成或质量/速度已经比较。

这些缺口已被设计本身如实表达，不妨碍发布源码审计与精确有限方案；实际补齐后按已有实施合同推进，不额外制造用户审批步骤。本审阅只支持设计修订的科学逻辑和账目一致性，**不构成可执行scope或运行成功的证明**。
