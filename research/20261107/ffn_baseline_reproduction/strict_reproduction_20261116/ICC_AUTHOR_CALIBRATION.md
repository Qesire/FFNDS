# ICC：按作者校准方法迁移到 Wan 的修正合同

日期：2026-11-16。状态：**STATIC_ONLY / 尚未运行 GPU 校准**。本文只规定 ICC 的采集与统计合同，不形成第二套视频实验队列。主规模为 11 个 source；扩至 22 是一次可选统计预算诊断，是否执行须在主视频前冻结。E5 的局部误差条件只决定随后是否增加三条视频，不反过来决定是否采集 CAL22。不恢复 CAL256 完整视频计划。

## 1. 必须修正的主结论

上一版把完整 Dense-D4 生成轨迹中的 FFN 端点当作 ICC 的主校准分布，不符合所引用作者的公开采集程序。作者从真实数据的干净输入出发，编码后随机选时间并前向加噪；CA 调用一次未压缩模型，CD 从加噪起点调用未压缩模型、执行一个真实去噪步，再调用一次模型。**CD 第二次输入依赖第一次预测；它既不是第二次独立前向加噪，也不是从一条预生成完整轨迹挑出的端点。**

修正后主路径应为：**真实 clean clip → Wan 原生 latent 编码 → 在目标 D4 合法时间点前向加噪 → CA 一次 / CD 同专家相邻两次 → 四边界作者统计 → CA/CD 分解 → 四状态 ICC。** 当前待验证的是采集器及数据绑定，不是扩大样本数。已存在的 Dense rollout 统计只能保留为标明采集分布差异的历史资产，不能改名为本路径的校准产物。

保留 D4 时间网格是合理的目标模型适配。`num_steps=50` 是作者公开 DiT 校准命令的配置；论文实际比较的推理步数并不都等于 50，不能声称作者要求校准网格必定等于每个评测网格，也没有依据把四步蒸馏权重默认送到额外 50 个未部署时刻并称其更加忠实。原方法应保留的是 **真实数据加噪起点、随机合法时间、相邻去噪配对及统计定义**；D4 网格及专家域必须显式列为迁移配置。

## 2. 作者固定代码到底做什么

基准：`ccccczzy/icc@dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`。[统一代码引述](CODE_CITATIONS.json)中的ICC条目 已逐字及 Git blob 核验。论文为 [arXiv:2505.05829v1](https://arxiv.org/html/2505.05829v1)，§3.4、Algorithm 1、§4.1、补充 §6–7。论文 §4.1 写 256 个训练图像；下面的操作细节以作者可执行代码为准。

| 对象 | 作者真实行为 | 源码定位 |
|---|---|---|
| 数据与数量 | ImageFolder；center crop、随机水平翻转、[-1,1]；默认无放回抽 256 items，batch16、epoch1、drop_last=True；代码不保证不同文件没有共同 source | [gen_decomp.py L48–71](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/gen_decomp.py#L48-L71)，[L153–169](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/gen_decomp.py#L153-L169) |
| CA | 对 batch 内每个 item 独立均匀抽 t∈[0,T−1]；真实图像 VAE posterior sample ×0.18215；一次条件 dropout；一次高斯噪声；q_sample；一次 DDIM model call 采 mag | [func_decomp.py L54–75](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/func/func_decomp.py#L54-L75) |
| CD | 抽 t₂∈[1,T−1]；同样的真实图像编码/dropout/噪声；只 q_sample 起点一次；第一次调用缓存；其 DDIM sample 是 t₁=t₂−1 第二次调用输入；第二次统计 delta_mag | [func_decomp.py L77–101](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/func/func_decomp.py#L77-L101) |
| 条件 | p=0.1，把 class label 换为 DiT null class；CD 两次使用同一已经处理好的 y；不做两分支 CFG 预测 | 同上 L56、69–72、78、89–101 |
| 校准模型 | 当前加载的同一未压缩模型，eval 模式；不是额外一个外部 teacher；不在校准调用中启用拟评测 ICC 近似 | [gen_decomp.py L25–45](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/gen_decomp.py#L25-L45)，func_decomp.py L57、79 |
| 四边界 | fc1 输入 u、fc1 激活前输出 z、fc2 的激活后输入 h、fc2 门控前输出 y | [models_stat_svd.py L85–95](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/models/models_stat_svd.py#L85-L95) |
| 统计顺序 | 对 abs(a) 或 abs(a₂−a₁) 先 token mean；对每个 item 的完整通道向量做 min–max，分子和分母均加 1e−6；再 batch mean，累加并计 batch 次数 | [models_stat_svd.py L14–45](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/models/models_stat_svd.py#L14-L45) |
| 统计到分解 | 跨 batch 平均后开平方；left/right max_ratio=1000；分解前 scale 再加 1e−6；FP32 缩放矩阵的 SVD；恢复左右逆尺度，默认把奇异值并到左因子 | [func_decomp.py L135–173](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/func/func_decomp.py#L135-L173)，[utils_decomp.py L44–99](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/utils/utils_decomp.py#L44-L99) |

默认 50 是校准 sampler 的离散网格大小，不是每个 item 跑 50 步。CA 是 256 个 item-equivalent 模型调用，CD 是 512 个；默认 batch16 时分别为16/32次 batched model 调用。**11/22 个视频 source 的单点/双点采集保留作者方法的主要计算结构，只改变校准量级与目标数据域。** 不把 11×token 或 11×层数当成独立 source 数；也不把 256 图像机械换算为 256 条完整生成视频。

固定提交完整 tree 共83项，`truncated=false`，没有 PixArt 文件夹或校准实现。README 只给 DiT 示例，虽致谢 PixArt，并未提供可核实的 PixArt null-embedding 采集分支。因此本文选定 **公开 DiT 配方**；text-null 映射是本项目需明示的适配，不冒充作者已公开的 PixArt 细节。该版本的代码范围与本轮新增Wan接口引述均记录在[统一代码索引](CODE_CITATIONS.json)及[引述核验](CITATION_VERIFICATION.json)。

## 3. Wan 目标域和输入合同

### 3.1 真实视频、目标 latent 和 I2V 条件分开

每个 CAL source 应绑定一个真实视频 clip、时间裁剪范围、空间变换、对应文本、来源/近重复组，以及由该 clip 第一帧构造的 I2V 条件。clip 应具有目标部署的时空几何；已有 G20 配方是 81 帧、832×464。全视频 clean latent 是被加噪的生成对象；第一帧加后续零帧经原生 VAE、再拼 mask 得到 I2V 条件 y。A14B的noisy latent为16通道，条件y由16通道条件latent和4通道mask组成，连接后为36通道；应走wan2.2_moe的原生条件分支，不套用另一型号的latent首帧钳制分支。**只给目标 clean latent 加噪，不给条件 y/mask 加同样的噪声。** 官方 [image2video.py L280–325](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/image2video.py#L280-L325) 将两者分开。

视频与第一帧须接受相同空间变换。保留作者的裁剪、[-1,1]归一化和随机水平翻转（p=0.5），迁为整段视频一致变换，目标几何使用宿主832×464×81。首帧条件从变换后的clip取得；具有左右方向含义的文本也须与变换后内容相符，保存实际使用文本。不能逐帧独立翻转。

### 3.2 Wan VAE 不是 DiT posterior API

作者 DiT 显式从 VAE posterior 采样。官方 Wan2.2 的 [vae2_1.py L516–542](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/vae2_1.py#L516-L542) 虽得到 μ/logvar，但原生 encode 对 μ 做 Wan 通道归一化并返回 μ；[wrapper L647–656](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/vae2_1.py#L647-L656) 走此路径。LightX2V 实际默认 encode 也走规范化 μ。

最少改动的主路径使用 `vae_latent_mode=wan_native_normalized_mean`，这是明确的架构编码适配，**不能写成“逐字复制 posterior sampling”**。其条件是原生 VAE 与当前 checkpoint 的规范化配置绑定无误。若需要保留作者 posterior 随机性，应从未归一化 μ/logvar 采样后对样本整体施加 Wan 规范化，并单独验证；不能直接调用 LightX2V `encode_video()` 的默认 [0,1] scale，或把已规范化 μ 与未变换 logvar 拼成采样分布。此项不由本次静态审计证明已执行，也不应默认增加完整视频评测臂。

### 3.3 条件 dropout 保留作者意图，但写清映射

选定的 DiT 参考配方 p=0.1。Wan 主配方保留一次 Bernoulli(0.1) 抽样，映射为**原生 T5 对空字符串的编码**；保留第一帧 I2V 条件与 mask，CD 两次使用完全相同的条件。空文本不是零向量，也不是有语义的默认负面提示词。每条记录保存该 Bernoulli 结果及实际文本 embedding 身份，CAL11 中出现0、1或更多个 dropout 均按预先固定 RNG 接受，不事后抽种子凑恰好10%。

官方高层生成入口会将空negative prompt替换为默认负面提示词（[image2video.py L298–305](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/image2video.py#L298-L305)）。校准应直接调用相同tokenizer/T5编码真正的空字符串，不经过该默认替换，也不用零embedding。

CFG=false 不禁止单次空文本条件前向；它只意味着不做 conditional/unconditional 两次预测再线性组合。LightX2V 的默认 runner 在 CFG=false 时不准备 `context_null`，所以校准入口需主动把抽中的主 `context` 换为空文本编码。该映射的代码可执行性与其是否匹配蒸馏训练的条件混合分布是两件事，后者未证明。p=0 是可明确标注的部署条件分布消融，不能仅因 CFG 关闭就默认删掉作者0.1规则。

## 4. D4 flow 校准的精确调用合同

固定 LightX2V `9cb6929d83814ff28c4e1386532298fe93a96af1` 的配置给 nominal steps [1000,750,500,250]、shift=5、CFG=false、expert boundary index=2。实际 sigma 必须从绑定的 scheduler 读取，不按名义 t/1000 猜测。该配置得到

\[
(\sigma_0,\sigma_1,\sigma_2,\sigma_3)=(1,0.9375,5/6,0.625),\qquad e=(H,H,L,L).
\]

LightX2V [step_distill/scheduler.py L29–53](https://github.com/ModelTC/LightX2V/blob/9cb6929d83814ff28c4e1386532298fe93a96af1/lightx2v/models/schedulers/wan/step_distill/scheduler.py#L29-L53) 给出前向加噪和实际 Euler 更新：

\[
z_{\sigma_i}=(1-\sigma_i)z_0+\sigma_i\epsilon,\qquad
z_{i+1}^{\mathrm{step}}=z_{\sigma_i}+(\sigma_{i+1}-\sigma_i)v_{\theta_e}(z_{\sigma_i},t_i,c).
\]

实际实现调用原生add_noise/step_post并保留FP32中先减σv、再加σ_next v的顺序，不把公式的实数等价式改成新的融合舍入。prepare/reset会重置latent；在加噪输入注入后到CD第二次前向之间不得reset，第二输入记录为原step_post的实际输出。

这里所有调用使用**正在压缩的同一冻结 checkpoint**，包括两专家各自权重；Dense 表示关闭 FFN 近似，不表示换回原始40步 teacher。不能在 W_O 上采集后把因子作为 W_D 原法主校准，也不能用高噪声专家生成跨边界终点来计算低噪声专家的 CD 差分。

| 家族/专家 | 时间选择 | 实际输入与调用 |
|---|---|---|
| CA / H | 均匀随机选 {0,1} | 从本 source 的真实 clean latent 加新噪到该 sigma，单次 H dense forward，记录 mag |
| CA / L | 均匀随机选 {2,3} | 独立在所选 sigma 构造 fresh noised clean latent，单次 L dense forward；不先走 H 轨迹 |
| CD / H | 合法相邻对集合只有 {(0,1)}；均匀抽样退化为该对 | clean latent 前向加噪到0；H forward缓存四边界；真实 step0；H forward(step1)记录 delta_mag |
| CD / L | 合法相邻对集合只有 {(2,3)} | clean latent 前向加噪到2；L forward缓存四边界；真实 step2；L forward(step3)记录 delta_mag |

1→2 跨专家对排除。即便两个专家张量维度相同，两端也不是同一 W 的输入输出变化。没有第五个 t=0 模型调用，因此 (3,terminal) 不形成新的 CD pair。H 的起点 sigma0=1 会使输入生成 latent 等于 ε，clean clip 只通过 I2V/文本条件影响该起点；这由部署时间网格决定，应如实记录，而不是事后换时间制造更好结果。

**CD 的同噪声含义**：只抽一次起点 ε，第二点继承第一次去噪结果和同一条件，不再独立注入噪声。把两个端点都写成 `(1−sigma)*z0+sigma*epsilon` 即使复用 ε，也不是作者 CD。

**CA/CD 复用条件**：CA 随机选到 CD 起点，且真实 z0、噪声、条件/dropout、dtype、模型和上下文相同，可在同一次首前向同时记录 mag 与 cache。抽到后端点时，CA 必须自己构造 fresh q_noise 输入；CD 第二状态不可充当 CA fresh 输入。不能为了省调用把 CA 总固定在起点；不共享也完全合法，最便于直接验收。

## 5. 量级、最小统计与旧资产复用

不共享调用时，每 source 两专家 × (CA1次 + CD2次) = **6 NFE**：CAL11为66、CAL22累计132。这里 NFE 是全网络模型前向次数，VAE、文本编码、SVD另记成本；不是66/132条生成视频。只有同输入首调用共享时才减少实际 NFE；保留原抽样计划并逐项记账。CA和CD未要求彼此独立数据集，可以使用相同真实source；它们各自的噪声/时间分布要按冻结计划生成。

CAL11与CAL22采用嵌套 source 集和稳定的source派生RNG，扩集不重抽前11条。batch1，避免直接继承 batch16/drop_last 丢掉全部11条或把22条静默缩成16。统计以source等权，每个source内完整有效token均值；全通道 min–max 必须在 token mean 完成后计算，不能先按tile归一化再平均。四边界都采，CA/CD分别累计，每专家独立保存。

CA 可以流式累加绝对值。CD 需要第一调用四边界与第二调用对应 token 相减后取绝对值，不能用两次通道均值之差代替；可按层/块卸载首调用激活并做精确流式差分，不必保存11个source全量激活。若使用P512等token子集，那是另一个统计采样适配；不能在主合同中无声替代作者全token统计。四状态线上主臂、CA/CD统计完整性都不能因内存不足被两残差或缺少边界替换。

主规模沿现有量级，名称为ICC_CLEAN_CAL11（可扩为ICC_CLEAN_CAL22）。当前VBench CAL11/CAL_EXT11仅为图文条件名单，不能自动获得这两个视频数据集的身份。若能找到其原始真实clip，按来源簇绑定；否则在相同11/22量级上登记新的真实视频来源，并排除DEV及近重复。Q1保留其原CAL11图文/轨迹拟合身份；两方法校准源未匹配前，五臂只是同DEV的pipeline开发比较，不能把差异单独归因于算法或声称逐源校准公平。

最小无需新增完整视频的检查是：按每source保存的四边界归一化向量重组CAL11/22；报告两组对角权重差异、固定局部评估集上加权/未加权误差、候选家族/秩选择是否变动。source数量与时间/condition配置是独立因素，不把CAL22结果用于CAL11分解或选参。旧rollout统计不能补进author-style统计；相同shape、同seed或同模型均不足以跨此采集分布复用。由旧rollout统计导出的CA/CD因子也不能抵扣新配方分解；同权重、足够rank且数值身份匹配的普通SVD因子可复用。

## 6. 输入身份与证据边界

| 已有资产 | 可作主路径 clean 数据吗 | 正确用途 |
|---|---|---|
| 有来源、完整真实视频、对应首帧及文本 | 可以，需时间/空间/来源检查 | 真实数据 author-style calibration |
| 仅 I2V 首帧图片与prompt | 不足以定义真实 clean video | I2V条件，不能冒称全视频目标 |
| Dense生成末端 latent/生成视频 | 不可以冒充真实数据 | 明确命名 pseudo-clean generated-data 变体；生成末端latent即使不经VAE也保留合成数据身份 |
| 单图片复制81帧 | 不是原真实视频分布 | 明确命名 static-repeat 变体，不能用其成功/失败覆盖视频校准结论 |
| 现成 Dense-D4 rollout端点 | 不符合作者 fresh-noise 采样 | 单独的 rollout-distribution ablation/历史证据 |

本次没有证实当前CAL11对应的11条真实clean视频已经可用；首帧与prompt清单不自动证明视频存在。若资产审计只找到图片，应登记 `MISSING_REAL_CLEAN_CLIP` 并继续完成 SVD/采集器静态规范，不能通过把生成终点改称 x0 来消除缺口。允许替代数据时必须换方法标识并保持上述主校准未完成状态。

## 7. 最小验收（尚未执行）

1. 输入身份：clean clip来源、source组、crop/time span、首帧一致性、文本与empty-text编码、VAE版本/latent模式、两专家权重、调度器真实timesteps/sigmas可复查。
2. CA trace：每source每expert恰有一个合法随机时间；目标latent满足对应flow前向加噪；I2V y/mask保持原生条件；四边界来源是真实dense FFN。
3. CD trace：每pair固定expert与条件；第二输入逐值对应首预测执行的真实step输出；无第二次q_noise、无完整rollout替代；时间对恰为0→1与2→3。
4. 统计oracle：四边界、token轴、abs-before-mean、min–max eps、source计数、sqrt、cap1000、SVD尺度eps与左右反缩放均对照作者函数；至少覆盖非零min、常量通道向量、两条不同source，能辨别错误排序。
5. 复用trace：共享CA/CD首调用须证明完整输入身份相同；后端点CA必须fresh noised。RNG按source保存，扩至22不改前11。
6. 数值与资源：真实BF16采集和分解/累计dtype、offload实现、CUDA库及实际调用计数写入manifest；统计对齐与线上四状态验收分别报告。缺GPU证据仍是UNVERIFIED，不写已复现成功。

只有上述校准合同与四状态线上实现都通过，才可以用新CA/CD结果讨论“作者方法在本目标域的表现”；仍需带上CAL11/22、D4-grid、条件映射和Wan编码模式这些适配限定。旧两残差或旧rollout产物的负结果不能升级为新主臂结论。

## 8. 资源计数与执行资产

本配方用单点/双点校准替代旧E1的22条完整D4。主55视频不变；完整视频合计基本68条（55+10 neutral+3同因子2R），两个条件分支全开后74条。技术重试全局最多3次：视频一次为同条件D4，校准一次为CA单前向或CD不可分两前向pair；失败时回滚该统计事务，再用相同样本与随机记录重试。r_video+r_cal≤3，视频≤74+r_video，校准≤132+2r_cal次去噪器调用；不能各自另获3次重试。全部条件打开时两类去噪器调用合计保守≤440，不代表等时长，模块、VAE/T5编码、SVD和搬运另计。

源数与D4视频分开计账；主11源最多66次校准调用，可选22源累计132次。旧99次D4总账已失效，以[统一预算](EXPERIMENT_BUDGET.json)为准。本次不为校准新增完整D4/O40轨迹。

当前数据清点见[资产审计](ICC_CALIBRATION_ASSET_AUDIT.md)。其结论是尚无可核的11条真实clean视频或其编码x0；这是输入缺项，不能通过使用旧rollout、复制首帧或换成普通SVD规避CA/CD复现。SVD模块与采集器准备可以继续，但缺少CA/CD时不能把SVD写成三家族完整选择后的主ICC；三家族候选齐全才可启动预定选择和主五臂比较。所有新参考路径的真实执行状态仍为未运行。具体参数与缺项见[机器可读配方](ICC_CALIBRATION_RECIPE.json)。
