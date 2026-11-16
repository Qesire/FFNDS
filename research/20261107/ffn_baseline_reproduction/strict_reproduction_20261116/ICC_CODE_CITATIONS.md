# ICC：固定源码引述、实际端口映射与校准预算纠正

状态：STATIC_SOURCE_VERIFIED；没有GPU/模型运行。本文件不改算法，也不新建执行队列。作者固定commit为 `dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`，FFNDS固定commit为 `e7a1e6b7a6babb5cf742072cc07a9961af854416`。

**最重要的修正：论文的256是随机选取的校准图像条目数。作者CA对每图只调用一次denoiser，CD只调用相邻两次；不是256条完整生成轨迹。它也不是作者证明过的256个统计独立source。因此，“完整正确移植ICC”不要求机械增加为256个Wan视频条件。上一版将CAL256当成必须完成的主reference门槛，混合了算法忠实度与一个经验校准预算，这个约束过强，应撤销。**

可以缩减校准规模，同时保留四状态、SVD/CA/CD、原统计公式与因子路径；缩减必须明确为目标域校准配置，并用预先冻结的小消融展示敏感性。小校准负结果仍不能推广为所有ICC配置失败。

## 1. 字节与引述的核验方法

作者8个相关文件已从固定commit重新读取，以原始CRLF/末尾换行保存临时核读副本，逐一与固定git tree的blob SHA1和size匹配；其中7个进入下列引用，cache辅助文件仅作交叉核对。此前研究缓存中的部分换行被规范化，不把其哈希冒称原文件哈希。下表FFNDS文件的完整字节SHA256/size与BWS公开SOURCE_MANIFEST一致。每条引用的quote是其行区间内连续、逐字存在的短片段，没有ellipsis或拼接。

[统一索引CODE_CITATIONS.json](CODE_CITATIONS.json)中，ICC前缀对应本页20项主张。它记录固定仓库/commit、仓库内路径、原字节SHA256/size/Git blob、函数、行区间、短摘录字符位置、永久链接和最小验收；行号1-based，字符偏移0-based/end-exclusive。校验器不做换行归一化。

## 2. 作者代码：12处决定性依据

### U01 · linear_decomp; SpMlp.forward_upd; SpMlp.forward_inc

[固定源码 DiT/models/models_sp_step_decomp.py，L69–L109](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/models/models_sp_step_decomp.py#L69-L109)；补充 [L24–L29](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/models/models_sp_step_decomp.py#L24-L29)。短摘录：`x = linear_decomp(x, self.fc2_wa, self.fc2_wb)`

**支持的主张：**anchor保存两级linear的输入/输出四态；hit先更新W1、执行真实激活，再校正W2。hit只读anchor，不更新它。linear_decomp先右因子再左因子；bias由anchor保留。

**最小验收：**同输入/权重/因子/dtype/backend与原SpMlp逐边界比较；零增量回anchor；连续两个hit仍读取同一anchor。

### U02 · SpDiTBlock.forward_inc; SpDiTBlock.__init__

[固定源码 DiT/models/models_sp_step_decomp.py，L280–L283](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/models/models_sp_step_decomp.py#L280-L283)；补充 [L232–L239](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/models/models_sp_step_decomp.py#L232-L239)。短摘录：`gate_mlp.unsqueeze(1)`

**支持的主张：**原作者每步重新算当前调制与gate；FFN前做norm与shift/scale，FFN后乘当前gate。模型实例用tanh-GELU、drop0。

**最小验收：**Wan接收当前pre-W1输入，返回pre-gate输出；gate恰好一次；验证GELU模式及不存在隐含drop/norm差异。

### U03 · sched_PERIOD.__init__; clr; step

[固定源码 DiT/adapters/sched_sp_step.py，L7–L23](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/adapters/sched_sp_step.py#L7-L23)。短摘录：`self.gate_list[i] = 1 if (i % period == 0) else 0`

**支持的主张：**p2为A/H交替，无另设warmup或强制末步dense；clr重置step游标。

**最小验收：**四步输出A/H/A/H；请求边界重置；Wan新expert首步必须anchor。

### U04 · SpMlp.forward_stat

[固定源码 DiT/models/models_stat_svd.py，L85–L95](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/models/models_stat_svd.py#L85-L95)。短摘录：`stat_upd(self, x, stat_flag, "fc2_x")`

**支持的主张：**CA/CD统计四边界为u、pre-GELU z、post-GELU h、pre-gate y；并非只统计输入或只处理W1。

**最小验收：**producer保存4边界键、shape、expert/layer/step与配对ID；检查W1左右对应z/u，W2左右对应y/h。

### U05 · stat_upd

[固定源码 DiT/models/models_stat_svd.py，L14–L45](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/models/models_stat_svd.py#L14-L45)。短摘录：`tmp_mag = delta.abs().mean(dim=1)`

**支持的主张：**CD使用同一边界差的mean-abs，CA使用原值mean-abs；随后每sample跨通道min-max（分子分母均加1e-6），batch均值及计数累计。

**最小验收：**直接调用作者函数作oracle；覆盖常量幅值、零delta、不同batch大小；常量min-max结果为1而非0。

### U06 · DiT_wtv_calc_mag; DiT_wtv_calc

[固定源码 DiT/func/func_decomp.py，L135–L173](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/func/func_decomp.py#L135-L173)。短摘录：`alpha = 0.5`

**支持的主张：**累计平均取平方根；输入/输出两侧都经shift2ratio；mag与delta_mag走同一后处理。

**最小验收：**分别核4个累计量、count、sqrt后向量及cap后向量；显式记录左右cap，不以函数存在推断实际调用。

### U07 · wsvd; wsvd_merge_lowrank; shift2ratio

[固定源码 DiT/utils/utils_decomp.py，L44–L84](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/utils/utils_decomp.py#L44-L84)；补充 [L95–L99](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/utils/utils_decomp.py#L95-L99)。短摘录：`u, s, vh = torch.linalg.svd(x)`

**支持的主张：**scale先各加1e-6；FP32分解LWR再撤销scale；原调用不显式指定driver/full_matrices；默认奇异值合入左因子。cap为max/max_ratio下限。

**最小验收：**保留作者因子拆分；比对有效矩阵action与实际数值路径。绑定实际PyTorch/backend，不把本仓gesvd说成作者显式指定。防多rank循环重复原地加epsilon。

### U08 · main: calibration dataloader

[固定源码 DiT/gen_decomp.py，L48–L68](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/gen_decomp.py#L48-L68)。短摘录：`RandomSampler(dataset, num_samples=args.num_samples, replacement=False)`

**支持的主张：**校准来自ImageFolder训练图像，经中心裁剪/随机水平翻转/归一化；按条目无放回抽样，batch尾部drop_last。原代码不做source-cluster去重/独立性验证。

**最小验收：**记录实际图像索引/样本数与随机状态；小CAL不能照搬batch16+drop_last而丢样本；说明Wan采集分布变化。

### U09 · argument defaults: sample/batch/epoch/step/cap/rank

[固定源码 DiT/gen_decomp.py，L156–L177](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/gen_decomp.py#L156-L177)；补充 [L83–L95](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/gen_decomp.py#L83-L95)。短摘录：`parser.add_argument("--num-samples", type=int, default=256)`

**支持的主张：**默认256条目、batch16、epoch1、50步离散scheduler；左右cap1000、rank128。rank由统一手工参数设置；256不是源码中的正确性断言。

**最小验收：**保存resolved args和实际计数；普通SVD无activation校准依赖；不得把256默认解释为256条完整50步轨迹。

### U10 · DiT_wtv_src_stat_loop_mag

[固定源码 DiT/func/func_decomp.py，L54–L75](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/func/func_decomp.py#L54-L75)。短摘录：`model_kwargs = dict(y=y, stat_flag="mag")`

**支持的主张：**CA将真实图像VAE采样成latent，随机选一个离散t，q_sample加噪，再调用一次DDIM；另有0.1类别dropout。

**最小验收：**核实每图1次denoiser调用、t/噪声/VAE/类别随机状态。Wan原生轨迹采集须另标分布适配，不冒称同一q_sample分布。

### U11 · DiT_wtv_src_stat_loop_delta_mag

[固定源码 DiT/func/func_decomp.py，L77–L101](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/func/func_decomp.py#L77-L101)。短摘录：`t1 = t2 - 1`

**支持的主张：**CD每图随机t2，真实图像VAE latent先q_sample；第一次DDIM缓存，第二次相邻DDIM累计差。共两次模型调用，不运行完整采样轨迹。

**最小验收：**两调用必须同图/latent/标签和同一采样续步；Wan用同expert合法native pair，严禁跨high/low权重造delta。

### U12 · main: dtype/TF32 and scheduler attachment

[固定源码 DiT/sample_ddp_sp_step.py，L49–L98](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/sample_ddp_sp_step.py#L49-L98)；补充 [L153–L155](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/sample_ddp_sp_step.py#L153-L155)。短摘录：`torch.backends.cuda.matmul.allow_tf32 = args.tf32`

**支持的主张：**DiT采样代码设置TF32且模型未在此转BF16；每batch重置scheduler，首步full覆盖旧cache。不能说作者代码逐batch一定调用物理cache.clear。

**最小验收：**原dtype原工程reference与Wan BF16分别标识；绑定TF32、权重/activation dtype、初态及首步覆盖行为。

## 3. FFNDS实际入口：8处映射与已知缺口

### F01 · ICCFFN.anchor; ICCFFN.hit

[固定源码 reports/20261114/g20_two_track_core/BWS/v1/source/icc_ffn.py，L22–L57](https://github.com/Qesire/FFNDS/blob/e7a1e6b7a6babb5cf742072cc07a9961af854416/reports/20261114/g20_two_track_core/BWS/v1/source/icc_ffn.py#L22-L57)。短摘录：`(z - self.first.apply(u)).detach()`

**支持的主张：**本仓同时有author_four_state与ffnds_two_residual；后者anchor预计算q1/q2，hit用当前u/h再加q。实数等价但BF16与anchor成本改变。

**最小验收：**四状态作主reference；同一因子/输入/dtype下单因素比较2R，记录zero-delta与anchor额外投影。不得用2R负结果替代4S。

### F02 · Factors.to; Factors.apply

[固定源码 reports/20261114/g20_two_track_core/BWS/v1/source/icc_ffn.py，L7–L20](https://github.com/Qesire/FFNDS/blob/e7a1e6b7a6babb5cf742072cc07a9961af854416/reports/20261114/g20_two_track_core/BWS/v1/source/icc_ffn.py#L7-L20)。短摘录：`return F.linear(F.linear(x, self.right), self.left)`

**支持的主张：**本仓因子顺序与作者一致；to明确允许FP32/BF16并检查finite。

**最小验收：**记录实际调用的dtype，不从接口可支持FP32推断实际运行用了FP32。

### F03 · delta_stat; DeltaAccumulator

[固定源码 reports/20261114/g20_two_track_core/BWS/v1/source/icc_decompose.py，L5–L24](https://github.com/Qesire/FFNDS/blob/e7a1e6b7a6babb5cf742072cc07a9961af854416/reports/20261114/g20_two_track_core/BWS/v1/source/icc_decompose.py#L5-L24)。短摘录：`def weights(self, max_ratio=None):`

**支持的主张：**基础CD函数实现token mean-abs、逐sample min-max、累计均值/sqrt，但默认max_ratio=None不截断。

**最小验收：**取回实际producer调用和4边界输入；必须确认显式1000或披露修改值，不能只读基础函数就认证旧CD因子。

### F04 · factorize

[固定源码 reports/20261114/g20_two_track_core/BWS/v1/source/icc_decompose.py，L26–L47](https://github.com/Qesire/FFNDS/blob/e7a1e6b7a6babb5cf742072cc07a9961af854416/reports/20261114/g20_two_track_core/BWS/v1/source/icc_decompose.py#L26-L47)。短摘录：`torch.linalg.svd(w, full_matrices=False, driver=driver)`

**支持的主张：**FP32缩放分解、奇异值左并与作者数学一致；显式reduced SVD、可指定driver、无原地epsilon累计是工程差别。

**最小验收：**检查scales/rank/dtype/driver及action；不同solver因子不要求凭空逐位相同，原路径与改写分别保留。

### F05 · icc_predict

[固定源码 reports/20261114/g20_two_track_core/BWS/v1/source/b_common_replay.py，L127–L143](https://github.com/Qesire/FFNDS/blob/e7a1e6b7a6babb5cf742072cc07a9961af854416/reports/20261114/g20_two_track_core/BWS/v1/source/b_common_replay.py#L127-L143)。短摘录：`q1 = c['z0'] - first(c['u0_bf16'])`

**支持的主张：**BWS实际预测入口直接执行BF16 two-residual，不调用author_four_state。

**最小验收：**以该入口命名旧局部结果为2R；新版4S需真实执行receipt，不因文件有4S类而认证已经跑过。

### F06 · make_w_o_factors; load_w_d_factors

[固定源码 reports/20261114/g20_two_track_core/BWS/v1/source/b_common_replay.py，L146–L158](https://github.com/Qesire/FFNDS/blob/e7a1e6b7a6babb5cf742072cc07a9961af854416/reports/20261114/g20_two_track_core/BWS/v1/source/b_common_replay.py#L146-L158)。短摘录：`factorize(w1, RANK_ICC, driver='gesvd')`

**支持的主张：**W_O用普通SVD/gesvd并转BF16；W_D读取已存因子。此调用没有CA/CD统计。

**最小验收：**核W_D文件SHA与checkpoint；不能将W_O ordinary-SVD执行证据当作CD producer证据。

### F07 · fit_and_score_site: ICC identity/reload

[固定源码 reports/20261114/g20_two_track_core/BWS/v1/source/bws_runner.py，L404–L423](https://github.com/Qesire/FFNDS/blob/e7a1e6b7a6babb5cf742072cc07a9961af854416/reports/20261114/g20_two_track_core/BWS/v1/source/bws_runner.py#L404-L423)。短摘录：`fobj['method'] == 'svd'`

**支持的主张：**运行检查W_D checkpoint/method=svd/rank512/文件SHA/shape/BF16，W_O新分解后保存重载；这闭合普通SVD局部调用身份。

**最小验收：**新reference继承同级别identity guards，另绑定cache mode与CA/CD实际统计manifest；不能仅沿用旧scope。

### F08 · icc.W_O_factors_note

[固定源码 reports/20261114/g20_two_track_core/BWS/v1/source/B_BINDINGS.json，L1644–L1648](https://github.com/Qesire/FFNDS/blob/e7a1e6b7a6babb5cf742072cc07a9961af854416/reports/20261114/g20_two_track_core/BWS/v1/source/B_BINDINGS.json#L1644-L1648)；补充 [L1506–L1510](https://github.com/Qesire/FFNDS/blob/e7a1e6b7a6babb5cf742072cc07a9961af854416/reports/20261114/g20_two_track_core/BWS/v1/source/B_BINDINGS.json#L1506-L1510)。短摘录：`"W_O_factors_note"`

**支持的主张：**绑定文件明确W_O是ordinary SVD r512、没有CD或rank search。历史CD视频另有汇总/结果SHA，但当前入口不能认证旧CD全生成链。

**最小验收：**补回历史CD producer/统计参数/因子与视频receipt；找不到时只重建受影响资产，保留旧CD结果原身份。

## 4. 256校准图像：应怎样精确解读

原论文[§4.1及Algorithm 1](https://arxiv.org/html/2505.05829v1)说明从训练集中随机选256张图做CA/CD校准；输入构造与调用次数应进一步依固定实现U08–U11判断。论文没有把该数量写成正确性定理或最小有效样本数。

| 项目 | 固定作者代码实际含义 | 不能据此推出 |
|---|---|---|
| 数据条目 | 训练ImageFolder的图像索引；默认单epoch、无放回256条目 | 256个经过近重复/来源聚类验证的统计独立source |
| 数据预处理 | 中心裁剪、随机水平翻转、归一化；VAE从posterior抽latent | 使用确定的VAE均值或只有输入图像ID即可重现 |
| CA调用 | 每图随机一个scheduler离散时间，经前向加噪得到状态，执行一次DDIM调用收集幅值 | 每图完成50步生成；先生成完整图像再统计 |
| CD调用 | 每图随机一个非末端索引，经前向加噪，连续两次相邻DDIM调用收集差 | 完整轨迹、任意两个未配对输入，或任意跨权重的差 |
| 默认计数 | 256条目、batch16、epoch1：16个batch；CA 16次batch DDIM调用，CD 32次 | 每个样本执行50次或100次模型调用 |
| 其它随机性 | 时间、噪声、VAE latent采样及类别dropout均参与输入分布 | 样本数一样就保证校准分布一样 |
| 50这个参数 | 定义校准所用离散scheduler及随机时间索引的支持集 | 校准循环为每图执行50步 |
| 小CAL的实现 | 必须记录真实消费数量；默认batch16会使11条目全部被丢弃，22条目只消费16条目 | 改了num-samples便一定实际校准了11/22个样本 |

按每图等效计数，默认CA是256次模型评价，CD是512次；这些调用通过batch并行完成。无放回在常规训练集大于256、默认单epoch条件下避免重复索引，但代码没有检查同一物体、近重复内容或来源聚类。增加epoch还会改变实际观察数量和可能重复情况。因此256既是有依据的论文经验设置，也不是Wan必须原样复制的统计单位定义。

**准确的复现标准是：原法四状态算子、CA/CD估计公式、四边界、配对、因子与执行路径不被偷换；样本预算及目标数据分布必须显式披露并检验敏感性。** 若要复现原ImageNet实验表，256及其原输入构造应照原设定；若要把同一方法应用到Wan，不能把256张图机械升级为256个完整视频前向来定义“忠实”。

## 5. 当前校准修正：遵循作者状态构造

主校准改为真实clean视频→原生Wan编码→随机时刻前向加噪。CA单次Dense调用；CD第二输入来自第一次Dense预测执行的真实native step。旧D4 rollout两端退出主参考，保留历史身份。数量沿11/22来源，分别至多66/132模型调用，不生成完整CAL轨迹。

原生Wan VAE normalized mean、flow加噪/更新、D4合法时间与专家域、class-null到真实空文本T5的p=.1 dropout均有明确映射，见[作者式校准合同](ICC_AUTHOR_CALIBRATION.md)和[参数](ICC_CALIBRATION_RECIPE.json)。新Wan接口引述合入[CODE_CITATIONS.json](CODE_CITATIONS.json)的ICC-CAL前缀；旧作者引述仍有效。

当前CAL11/CAL_EXT11只是图文名单，VB_DENSE22为11 CAL+11 DEV的U/Y/gate，不是22个clean校准视频。主11条真实clip/x0尚未绑定，见[资产审计](ICC_CALIBRATION_ASSET_AUDIT.md)。不能由旧统计形状一致推断采样分布一致，也不通过只保留SVD跳过CA/CD。

主55视频与原四状态/三家族模块要求保持，执行数量只依[统一方案](CONTROLLED_REPRODUCTION_PLAN.md)。Q1现有参数保持其旧CAL11身份；ICC新视频来源未配平时，结果为各pipeline在同DEV上的比较，不独立识别算法因果。
