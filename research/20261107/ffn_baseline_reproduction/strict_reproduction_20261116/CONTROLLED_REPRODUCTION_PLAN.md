# 第一轮精简复现设计：算法完整，实验按问题取舍

状态：DESIGN_ONLY / NO_NEW_MODEL_RUN。**本轮ICC修正：采用作者的真实clean样本→前向加噪→CA单次/CD真实相邻两次；撤回native-D4端点主校准。**本轮仅完成源码核对、引述核验与设计更新；模型/GPU运行0。此规格取代上一轮“897次D4”的提案，不与其队列、CAL256或EVAL22叠加。本轮使用experimental-design/research-implementation的原则，目标是完成原FFN算法与Wan移植的可执行核验，判断关键适配是否影响当前比较；不承诺原论文全模型、FID或VBench榜单复现。

## 1. 为什么必须撤掉上一轮大部分生成

“算法完整复现”要求公式、统计、状态机、数值边界和实际调用链完整；并不要求每个可选配置、数据预算和实现变化都跑一套视频。上一轮把三层事情混在一起：原模块验收、Wan方法比较、跨来源确认。结果把独立问题重复乘成大网格，而且误把作者256图像统计当成256完整Wan轨迹的前置。

新方案只有三层：

1. **原模块与同输入回放**：完整覆盖ICC四状态/SVD/CA/CD、Chip作者proxy/selector/多hit状态；多数证据不需要新native生成。
2. **必要宿主接口验证**：真实pre-W1/current gate、专家/请求缓存、global-step时钟、voxel与尾52、offload；以最小neutral和同输入对照为主。
3. **五臂同源视频**：Dense、Q1、一个预定局部准则选出的四态ICC-r1408、作者式Chip、Global-P64；DEV11×seed42，最多55个生成槽位。

这是一轮**研究开发对照**。ICC家族在DEV局部结果上选择，再观察相同来源的完整视频，不能叫来源留出的确认性实验，更不能把11来源当全部视频模型的总体结论。

## 2. 用代码纠正“256”的含义

固定ICC源码为 ccccczzy/icc@dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4。

- DiT/gen_decomp.py 第59–68行使用ImageFolder及RandomSampler(..., num_samples=args.num_samples, replacement=False)；第158–161行默认num_samples=256、batch_size=16、num_epochs=1。它数的是dataset items，不是经过原图/近重复簇审计的“256个独立来源”定理。
- DiT/func/func_decomp.py 第54–75行，CA/mag对VAE编码的真实图像加噪，在随机t只执行一次ddim_sample模型调用。
- 同文件第77–101行，CD/delta_mag先在随机t2加噪，执行cache调用得到x_t1，再在t1=t2−1执行delta_mag调用；每个样本参与两个相邻时刻的模型评价，**不是走完50步生成轨迹**。严格按调用数，默认batch_size=16、256items、1epoch时，CA为16次批模型调用，CD为32次批模型调用；样本级则是256/512个item-evaluations，不能混用。

作者256是数据项数量；本轮沿现有11/22量级，改为真实视频源而非先生成完整轨迹。主配方固定ICC_CLEAN_CAL11，CAL22只作一次可选预算诊断，是否执行在主视频前冻结；E5条件只决定诊断完成后是否增加视频。具体[作者校准合同](ICC_AUTHOR_CALIBRATION.md)与[参数](ICC_CALIBRATION_RECIPE.json)是ICC采集唯一入口。

保留作者随机时间和配对语义：CA从干净视频latent直接加噪到随机合法时刻；CD从加噪起点作第一次Dense调用，执行一次原生采样器更新，再作第二次Dense调用。CD第二输入来自第一次预测，不能再次前向加噪或抄已有rollout端点。

Wan必要适配明确为原生VAE编码、flow加噪/一步、目标部署D4时间网格、按专家分别统计，以及class-null到空文本条件的映射。作者p=0.1条件dropout保留，I2V视觉条件保持，CFG仍关闭；不把50个原DiT校准时间点直接套给四步学生。

原始链接：
- [gen_decomp.py：数据入口与默认样本数](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/gen_decomp.py#L59-L68)，[默认值](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/gen_decomp.py#L158-L169)。
- [func_decomp.py：CA和CD实际模型调用](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/func/func_decomp.py#L54-L101)。

固定版本、行号、短引述与字节身份见 [ICC代码引述](ICC_CODE_CITATIONS.md) 和 [统一引述索引](CODE_CITATIONS.json)；以上不构成256或11个样本的充分性结论。

## 3. 数据、配置和选择时点

目标模型保持当前BF16全权重LightX2V Wan2.2 I2V A14B四步，timestep=[1000,937.5,833.333…,625]、HHLL、CFG off、832×464×81、16 fps、torch_sdpa与相同block offload。FFN维度d5120/m13824、有效N31668、patch grid=21×29×52。输入为当前真实norm/modulation之后的pre-W1，输出为pre-gate，宿主当前gate仅乘一次。checkpoint、预处理、初噪与精确scheduler沿当前已核绑定取回并逐项验证，不能只凭模型名称复用。

A10080GB是拟用设备背景，当前GPU/host/编译/集群入口均未验证；代码、输入实体与资源绑定完成前，本文不是已派发scope。

### 3.1 复用当前身份，不另造确认集

Q1/旧bank的VBench CAL11图文条件，seed42：
vbi_1005,1039,0895,0937,1094,1082,0966,0304,0648,0456,0032。

旧VBench CAL_EXT11图文保留条件，seed42：
vbi_1024,1064,0901,0922,1107,0967,0958,0152,0760,0448,0040。
这些图片名单不自动构成ICC所需的clean视频。ICC_CLEAN_CAL11及可选EXT11按同样11/22量级绑定真实clip/x0，优先追溯已有CAL来源；否则另登记与DEV无重叠的真实视频来源。不得把DEV挪入CAL。

主视频DEV11，seed42：
vbi_1013,1040,0886,0947,1109,1084,0950,0176,0800,0392,0056。

固定小消融DEV3：
vbi_1013、0886、0800，seed42。

Neutral的两个来源：
vbi_1013、0800，seed42。

既有Q1/DEV沿当前v4/BW/BWS绑定核验。ICC真实clip/x0、第一帧条件、实际prompt及预处理须独立绑定，不能只继承旧图像ID。实际bytes/cluster或fit祖先未知则标PENDING，不把metadata名单说成全部已绑定。旧七源和新EVAL22本轮都不另生成。

### 3.2 主ICC的固定选择合同

- 所有SVD/CA/CD分解与四态算法都先通过L0；不能只实现获胜家族。真实clean输入未绑定或CA/CD候选未产出时，选择及主ICC保持未完成，不能删去候选后仍声称完成三家族选择。
- 主预算固定r1408，不做r512/768/1024全套视频搜索。四态完整anchor-hit主MAC比约0.688426，接近Q1的0.685185；这是算术预算接近，不是实测时延相同。
- r512只作为历史参照，从同一分解取前缀，做同输入风险与四态/两残差局部对照，不产生新的完整视频队列。
- SVD/CA/CD三个r1408候选先在既有DEV11、固定六个位点high/low×layer{4,19,34}、同P512行、同当前gate上评分。每位点用F64门后绝对MSE；层等权→专家等权→来源等权。选择平均风险最低的一家，数值严格相同则按SVD→CA→CD固定顺序。保存另外两家分数，不删负结果。
- 在主视频开始前冻结家族、CAL11、rank、全部80处导出因子、四状态缓存实现和数值路径。全部80处只构造选定家族；另外家族保留六位点完整算法及局部证据，避免重复造三套全模型参数。
- 这叫“按预定局部准则选出的、r1408预算合理的四态ICC参考”，不能不加限定称“最强ICC”或“完整Pareto最优”。

本轮只把被选择家族放入完整视频表。CA/CD其余分支的完整视频表现仍未测；“算法已验收”和“终端质量已评价”分别报告。

### 3.3 Q1与Chip

主Q1保留其原VBench CAL11校准协议，复用已绑定的Q1r2参数；只有fit祖先/公式/数值身份不符时才按同一CAL11重建，不能引入旧评估来源的A0后仍称当前CAL11。固定PCG、lambda、导出与选择规则，不加新的Q1改进。ICC使用真实视频作者式采样，Q1使用原图文条件下的轨迹配对；在来源未配平前，主表只能解释为同DEV上的各方法pipeline比较，不能声称逐源校准公平或单独归因算法。两者数据与成本分别披露，不为此自动扩大队列。

Chip作者参考保留原compiled selector、proxy、voxel、BF16作者kernel顺序、首2层dense；global scheduler context与expert cache隔离分别核验。D4共8 fixed-dense+76 anchor+76 hit。ICC/Q1保留原全40层策略，即80 anchor+80 hit。逐实际调用计总成本，不为了“看起来同范围”再多造共同38层主臂。

Global-P64固定现有算法、K5120、64真实行及原40层策略；不得将其写成作者实现的另一后端。

## 4. 视频与校准调用分别计数

| 实验 | 唯一问题 | 当前动作 | 完整D4上限 |
|---|---|---|---:|
| E0 原模块reference | 是否完整执行作者定义 | 四态、三家族、selector和状态机 | 0 |
| E1 ICC作者式校准 | 真实clean加噪与相邻一步统计是否正确 | 11源最多66 NFE；可选22源累计132 NFE | 0 |
| E2 Wan neutral | 接口/缓存是否保持Dense语义 | 两来源的独立真实执行路径 | 10 |
| E3 五臂主视频 | 修正参考后的同DEV自由生成表现 | 五臂×DEV11×seed42 | 55 |
| E4 ICC 4S→2R | 缓存改写的闭环影响 | 同因子DEV3，复用4S父输出 | 3 |
| E5 条件CAL22闭环 | 主CA/CD的预算变化是否影响视频 | 同家族/rank/4S，DEV3 | 3 |
| E6 条件适配闭环 | 一个合法非等价适配的闭环影响 | 一个预先冻结的变量，DEV3 | 3 |

视频基本68=55+10+3，两条件全开后74。技术重试总共最多3次，由视频和校准共享：r_video+r_cal≤3。视频≤74+r_video，校准≤132+2r_cal次去噪器调用。校准重试单位为CA单调用或CD不可分的两调用pair，失败统计回滚并使用相同随机记录；不重跑整源6次来扩大额度。

全部重试用于视频时最多77条D4；所有条件都打开时，视频与校准合计保守≤440次去噪器调用。这不包含模块检查、VAE/T5编码、SVD和搬运时间，不代表等GPU时长。旧99条D4总账已被此两类计数替代；不再请求22条完整CAL轨迹。

输入真实clip的读取/解码及VAE编码保留；无需输出VAE解码或生成MP4。新O40、额外整模型计时和原系统完整生成仍为0。合格旧产物按真实(source, seed, path, execution_id)抵扣；复用全部11个Dense主槽还须两条必需hook路径已可复用，plain不能冒充hook。

### 4.1 离线分解也计入预算，不能用“0条视频”隐藏成本

| 真实Wan大矩阵分解 | 计算方式 | 上限 |
|---|---|---:|
| CAL11局部候选 | 6位点×SVD/CA/CD三家族×W1/W2 | 36 |
| 主ICC全80参数 | 所选家族80位点×2，复用上述6位点的12个结果 | 新增148 |
| CAL22局部预算消融 | 仅CA/CD×6位点×2矩阵；SVD原因子不变 | 新增24 |
| **CAL11主路径** | 36+148 | **184** |
| **含可选CAL22局部诊断** | 184+24 | **208** |
| E5条件闭环的CAL22全80参数 | 仅所选CA/CD家族，复用其已有12个局部分解后补齐 | 条件新增148 |
| **包含E5的上限** | 208+148 | **356** |

这是完整矩阵分解次数，不是来源×token样本数。每次按作者约定分解并保存足够因子，r512/r1408取同次分解前缀，不重复分解。旧rollout导出的CA/CD统计及因子不符合新校准身份，不能抵扣；同权重普通SVD因子符合数值/秩条件时可复用。旧资产若只存r512截断因子，不能据此构造r1408；若存有足够完整且身份一致的分解才抵扣。选中SVD时CAL22不改变该方法，E5相关额外148为0。

E0另限作者原shape的3家族×W1/W2共6个矩阵分解；小代数fixture不计入Wan大矩阵数量，但实际CPU/GPU耗时照录。Q1优先复用有效CAL11参数；若身份不符，最多80位点固定规则PCG拟合，单列迭代/耗时/峰值，不伪装为免费的“复用”。不做CAL22 Q1新拟合，因为本轮主Q1始终固定。

统计精度、P512减采样、CA端点与来源重加权首先比较同捕获统计向量；已有因子的作用只作为固定参照，不自动为每条敏感性轴再生成三家族新因子。需要新分解时必须在运行前列为单独有限增量，不能隐含突破184/208/356。首次有效W1/W2分解就是正式产物；用其真实时间和峰值预测剩余已冻结矩阵的成本，超出可用资源则报资源限制，不自行换近似SVD、降rank或增加机器来维持“已经复现”的表述。

## 5. E0/E2：保留机制，减少大模型调用

### 5.1 不削弱的模块验收

ICC保留：
- anchor u/z/h/y四缓存、真实GELU与bias、右因子先乘/左因子后乘、hit缓存策略；
- 普通/CA/CD统计入口；CA绝对激活、CD四边界delta，归约与min-max顺序、sqrt、max_ratio=1000；
- same-input、zero-delta、precision、full-rank/合法rank与因子重载；
- 四态/两残差同因子，在r512与1408均做局部对照。

Chip保留：
- 真实SparseDiffMlp callable及其转置权重生命周期，不能靠nn.Module.to自动迁移；
- compiled selector的阈值/随机补充/256对齐；合法atomic索引顺序可能不同，不能以mask SHA必同判错；
- 算术对照给两实现注入同一实际inds/counts，分别检查proxy、cache前后和输出；
- 连续多hit与step9→10→11→12，旧proxy/indices复用及request reset；
- 使用原voxel_chunk_no_padding/reverse，核对真实grid与有效行；
- N=31668尾52采用保守dense tail，主体247×128原核不改；
- 显式global scheduler context，expert/layer/CFG/request分离；保留作者原计数器缺陷trace，不把缺陷复制到端口再惩罚算法。

这些可以在作者原shape/dtype和已有Wan输入上做，不需要先跑完整FLUX/DiT图像榜单。D4 global-clock与expert-local-clock在cold-start、各段<10时不能形成有辨识差异，故不额外生成一套D4。

### 5.2 Neutral计数不合并方法名

两个来源各需要真实执行路径：
1. native plain；
2. hook-bypass/instrumented native；
3. ICC forced-dense；
4. Chip forced-dense；
5. 条件性Q1 forced-dense；
6. 条件性Global-P64 forced-dense。

主表Dense的这两条若需新生成，明确作为hook-bypass capture，已经包含在55槽位中。plain2+ICC2+Chip2是额外6。Q1/GP64仅在相同已验收Dense-fallback producer代码、精度和调用绑定时复用；否则各2，额外neutral最多10。不存在把两个adapter的forced-dense合成一次调用的做法。

执行抵扣键是(source, seed, path, execution_id)。同一plain产物可以支持质量和neutral两种用途，但不能冒充hook执行或把两个缺失路径抵扣。若现有Dense视频已经可复用，仍按实际缺的路径计数；已有效的hook/原生对照可抵扣，不因改报告名称重跑。零差异验收比较实际逐步/最终latent，确定性环境要求一致；合法非确定性先独立固定自身包络，不能看近似结果后放宽。

ICC四态约89.4GiB缓存先用无损host offload/有界staging保留原数学。资源实机UNKNOWN；不够就报RESOURCE_BLOCKED，不悄悄用2R替代四态。A100/host等是运行前待核对象，本设计不宣称当前已有GPU能力。

## 6. E1/E5：遵循作者校准，只缩放样本量

### 6.1 真实clean输入及作者式采集

主ICC_CLEAN_CAL11为11个真实视频source；当前这些输入尚未绑定。现有VBench CAL11/CAL_EXT11只有JPG+prompt，VB_DENSE22则是11 CAL+11 DEV的U/Y/gate面板；均不能充当clean video/x0。详见[资产审计](ICC_CALIBRATION_ASSET_AUDIT.md)。若缺真实clip，就补数据身份，不回退为旧rollout统计或仅SVD来跳过CA/CD。

编码后的完整目标latent与I2V首帧条件分开。目标latent使用原生Wan归一化mu；noise只加到目标latent。CA在H的{0,1}、L的{2,3}分别均匀抽一个索引，从clean latent直接加噪再调用对应Dense专家一次。CD对H的0→1、L的2→3，从clean加噪起点调用一次，使用同一专家的native step_post推进，再调用第二次。每源最多6 NFE，一组调用同时收全部80处统计，无需逐层或整轨迹重跑。

保留作者p=0.1条件dropout，改为空文本T5编码；视觉条件保持，CD两次固定相同条件。原生VAE编码、flow算术和D4网格均是明确的宿主适配，不宣称与DiT原latent分布逐字相同。采集器实际调用、条件和数值验证见[详细合同](ICC_AUTHOR_CALIBRATION.md)。

batch1、epoch1、不丢尾批，CAL11的随机记录和统计在扩至22时原样保留。四边界全有效token统计仍按作者归约顺序；P512用于原局部评估，不能替代主统计。可复用的CA/CD统计和因子必须由本作者式采集配方产生，并匹配真实输入与数值身份；旧原生rollout统计及其CA/CD因子不能抵扣。

### 6.2 预算消融与触发

主参数始终ICC_CLEAN_CAL11。可选ICC_CLEAN_CAL22只追加11个真实视频到同一作者式采集/统计合同；SVD是无CAL负对照，复用同一因子，不能因为改样本数量而重新引入随机分解差。

固定六个位点和DEV11 P512上，比较同家族、r1408、4S的CAL11/CAL22风险、逐源变化及因子作用；必要时直接由per-source统计重加权/减采样。保留Q1 CAL11固定作为参照；不把这个预算干预变成同时换Q1、换家族或换rank。

E5只在主家族已选中CA或CD且以下预登记条件之一出现时触发，最多DEV3三条。若主家族是SVD，E5=0；CA/CD候选排序敏感性仍报告，不能生成相同SVD视频冒充样本量干预：
- 已冻结主ICC家族相对固定Q1的平均风险差在11→22时变号，且超过独立neutral数值包络；
- 对主ICC，|R22−R11|≥0.5|R11−RQ|，且|R22−R11|超过同量纲、预先固定的数值包络，DEV3中至少2源的R22−R11同号。这里R为来源等权门后绝对MSE，RQ固定为Q1-CAL11；改进或恶化均可触发。该阈值只判断预算效应是否足以改变解释，不是质量非劣标准。

其它家族的排序变化单独记录为局部选择敏感性，不足以触发主家族CAL22视频。触发后仍只给主ICC固定家族/同rank/4S增加CAL22视频，父CAL11视频来自主表。不因这3条表现较好就悄悄替换主表。

若可选22仍明显不稳定，结论是当前校准预算未解决，而非自动扩至64/256。只有这个未决问题足以改变机制判断时，才提出下一批固定新增来源的单独有限设计；本轮视频/校准两类上限不自动增长。未触发不等于证明11/22与作者256或原校准分布等价。

## 7. E3/E4/E6：少量视频如何形成有效判断

五臂主表中，四态ICC选定r1408后不再调参。Q1与ICC在全部40层，作者Chip保留首2层dense，Global-P64保留既有覆盖；用实际all160调用成本比较，明确范围差异。输出为各自自由轨迹，绝不把teacher replay当视频。

主分析单位11个source cluster，每源一个seed42。报告逐源配对差、均值、范围和leave-one-source-out；不把token/layer/视频帧当独立n。这是当前已暴露DEV上的开发证据，不构造“确认性显著”结论。

质量轴分别为适用来源的I2V subject/background、consistency、MUSIQ/imaging、aesthetic、运动方向与连续幅值；静态Dense分母低的相对流幅记不可稳定估计，不把运动平滑当运动保留。指标实现/帧采样/分辨率沿已有版本固定，缺项不填0。没有事前质量margin就不宣布非劣。

E4必保留：仅把主ICC4S换成2R，完全相同因子、CAL11、rank和端点，DEV3补三条。它回答原算术正确但部署改写是否改变闭环；局部接近不能替代这3条。

E6不是“三条任意搜索额度”。任何作者oracle不匹配都应先修实现，不能用E6视频质量裁决代码是否正确。主reference保留作者rounding和dense-tail；valid-tail、native-rounding都是非等价候选分支，默认不进入主表。只有必要局部检查证明合法但非等价、且影响主适配解释的方案才eligible；在任何主视频结果可见前，按固定优先顺序tail→precision选择第一项，记录精确算术/接口差异，只在DEV3补三条。无eligible项则E6=0。若tail与precision都需要完整视频才能解除限制，本轮最多评价其中一个；另一个明确UNRESOLVED，不能写“两项都通过”。没有相关问题则E6=0。

删除60次ABBA：主55槽位中未插observer的真实执行记录首次生产时间、cache/CPU/GPU峰值及all160成本。已有视频只复用质量，不复用不匹配的旧timing；Dense若只有两条新的plain timing，则端到端速度对比只在这两个真实配对来源上描述，不伪造11源时间。微小速度差记未分辨，不在本轮追加无限重复。

## 8. BW/BWS与原模型完整系统：本轮不再扩native生成

既有BW/BWS的状态、参数与风险足以继续限定普通SVD/GP64的结论：它们不是新CA/CD/author-Chip的替代证据，但也没有理由因为报告重写而重采6条O40。

本轮允许在充分资产上添加四态/两残差或新family同输入回放；若作者Chip需要full-M块统计而既有O40只有P512，则标DATA_MISSING，相关O40作者Chip结论不做。新O40=0，不把不充分P512伪装成true-M作者算法。

同样，原Wan2.1官方attention配置跑通并不能验证本任务FFN。原模块正对照必须有，原模型完整系统生成不是本轮前置，新增原模型完整生成=0。任务没有因此少掉FFN机制；少掉的是与当前问题无直接辨识力的整系统跑分。

## 9. 留/删清单与失败记账

| 上一轮内容 | 决定 | 理由 |
|---|---|---|
| 四态、SVD/CA/CD完整模块 | 留 | 直接解决“实现是否为原算法” |
| 作者Chip proxy/selector/多hit/counter/voxel | 留 | 这些是方法定义，不能为了省计算省掉 |
| CAL256完整Wan轨迹 | 删 | 误将原图像items/一或两次调用等同完整视频；当前无必要性证据 |
| ICC真实clip CAL11→22单点/双点统计 | 留 | 按作者生成校准状态，无完整CAL轨迹 |
| 7个ICC点×DEV11完整视频 | 删 | 同输入有限候选先选择，原算法不是每点都要全视频 |
| 9臂×EVAL22×2seed | 删 | 本轮是实现/迁移开发判断，不承担大规模确认任务 |
| 四态→两残差同因子DEV3 | 留 | 未被既有局部接近替代的关键闭环问题 |
| 共同38层、多个layout/selector全视频网格 | 删 | 原策略保留；多数可先局部解释，不值得全套生成 |
| 60ABBA与4原模型系统生成 | 删 | 非第一轮必要；时间精度和整系统复现主张相应收缩 |
| 新O40/native weight×state交叉生成 | 删 | 既有数据先用，缺失问题不转成默认采集任务 |

全局技术重试最多3次（D4视频或CA单调用/CD两调用pair共享），适用于已定位可修复的基础设施/实现失败；失败首次执行也计数。不能因质量不好重试、换seed或换source；不能把省下的calibration/neutral额度变成新候选。若更改算法/数值定义，原未完成条目INVALID并版本化，超出各类有限预算则报告未完成原因，不借“新版本”归零预算。

第一条检查可复用即记为正式样本；旧BWS曾额外smoke重采的问题不能重复。每项保存实际producer、input/noise/config/weights、cache与selector状态、exit与产物绑定。L0/neutral未通过不进入大批视频；质量负结果在实现正确后如实报告；资源不够不以2R偷换4S。

## 10. 本轮能与不能得到什么

完成后可以说：三家族原FFN机制已可执行验收；在明确11/22真实视频、作者式加噪配对及Wan宿主适配下，选定预算合理的四态ICC、作者式Chip、Global-P64和Q1得到同源闭环对照；必要的缓存改写与被触发的适配有配对证据。

仍不能说：原论文整个系统或榜单已重现；未上主视频的ICC家族完整视频必然相同；11/22视频足以代表目标数据分布，或ICC与Q1校准来源已配平；新O40作者Chip/CA/CD已验证；Q1跨模型/全质量/全速度胜出。减少实验同时缩窄这些主张，保留本轮真正需要的机制判断。

