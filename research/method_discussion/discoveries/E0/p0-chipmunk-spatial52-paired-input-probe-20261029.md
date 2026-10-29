# P0 Chipmunk · 已存 spatial52 同层跨步输入与固定单函数 probe

作者：s2_theory_design_independent；**仅 metadata/SRC/ALG，REVIEW_REQUESTED；无 tensor 数值读取或新 GPU/模型实验**。此文更正此前对128真实行必须新采集的判断。可先用已存采集检验真实输入上的选择与两核局部数值；其组由离散抽样行构成，不是原raster group或全视频成功结果。

## 实际资产核查

[已闭合结果](../../../20261029/S2_SPATIAL52_DENSE_STUDENT_CAPTURE_RESULTS_20261029.md)为 source898/96、Dense/value10/dcac10、各全160位置，共960包；每包21τ×52xy、M=1092行。两源是历史暴露开发来源。此处读取并校验了 **两源 Dense block20，step0/1/2/3 共8个 metadata 文件**相对 GENERATION_OBSERVATION 中的 SHA：全部匹配；每源内部四步 `selected_rows` 全1092项相同且唯一，step0/1为high、step2/3为low，维持各expert内部配对。未逐包重新读取数值或验证其他block的文件；collector 源码以固定geometry给每位置同一列表并在保存时检查实际rows，提供更宽范围的生成时证据，但本次逐文件核对范围仅上述8个。坐标顺序为τ优先、每τ同52xy，grid `[21,29,52]`，flat=`τ*1508+y*52+x`；2个远点+25个中心点+25个高flow几何点，不是语义分割。两source各自列表不同，严禁跨源配对。packet metadata明确 V1 BF16 `[1092,5120]`、h `[1092,13824]`、Y `[1092,5120]`、gate `[5120]` 和 int64 rows，V1来自实际完整M原生调用；由已完成producer记录tensor SHA与finite，不是本次重读值。

固定首个可审probe为 source898/seed42/Dense/high/block20/step0→step1；采用两packet相同的列表首128行，原flat行号从0至3873，跨约两个完整τ和第三τ一部分，中间大量间隔。数据路径前缀 `ffnds-derived/e0_s2_spatial_student_full160_portable_v2_20261029/part0/attempt1/worker/results/spatial52_ord898_seed42_dense/`；两个 `step{0,1}_block20.json` 及同名 safetensors 在HF。其身份：

| step | metadata SHA256 | payload SHA256 | V1 raw SHA256 |
|---|---|---|---|
| 0 | `be8f1401edb64b63e958bc52f4cbf03b01a2e8dd86916a604633b107d3ade9f0` | `a9a5679622c4e8949874e34a83a537598f7e129ce18e97b559a4d91eac79bab6` | `48942b4c500164b472210f7e2b33b85841bfbc5a56629497f0c3c32fc6e0e40f` |
| 1 | `d9d70142cb0c68fbd3da1997fdd8870efd1683b1eb7e66e5d71c85a26fdde99a` | `f4a408ffc6202040eebbbb60273750a9addd42ee45ea5a15aa11636056babeaf` | `fc3889ee70ba7b59a70d15f9357047986147181907d1b5c9058205c4d1a7024f` |

每包 rows raw SHA同为 `755da38636b6e4a7e3b2e5f3390d3411afa66ba0f0bcf5ae8d2d49c800e6d396`。按生成记录固定同源的protocol、generation receipt、两元数据、两个完整payload和原始checkpoint身份，在获批执行时实读验证payload SHA、shape、rows和实际值；目前只核metadata。W1/W2/bias 从 `config/e0_s2_new12_dense_baseline_part0_20261029_v2.json` 固定原high checkpoint与实际header读取 `blocks.20.ffn.0.*`、`blocks.20.ffn.2.*`，核查实际布局/配套transposition和SHA，不以硬编码shape代替对象证明。

## 重排合法性与真正的比较对象

FFN对每行独立，置换 P 满足连续算术 `F(PX)=PF(X)`；可把这128条行打包为一组，结果再用 P⁻¹恢复对应原行位置。但**选择策略依赖分组**，该组的均值、排名与真实连续raster组不同，不能由行置换等价推出整个组选择策略等价。P仅是选定128行的子集注入，不能把它当对全部31668行已执行的全排列。未来全M方案需要定义完整bijective P及P⁻¹，付费统计创建索引、搬运 V1、缓存h/Y、回散输出和构造/变更组；实际测量wall与GPU读写，不可假定免费或假定它改善质量。此probe对128行打包与回散也计费并报告。

对每step分别做 **M128重算 Dense**，与该step保存的 **原M31668捕获 h/Y** 在同坐标比较；两者作为不同参考，不将M128 GEMM的舍入差认作压缩误差。三种量均保存：M128 Dense与native满M anchor的差，稀疏输出与同M128 Dense的差，稀疏输出与native满M anchor的差；各报告h、Y、当前gate后结果（原BF16写回顺序）max/相对L2/finite，并记录signed cross，防止误差相消。初始化用M128 Dense h/Y的路径作为定义内控制；另用已捕获fullM h/Y初始缓存的路径做native-anchor敏感性，单列结果。全更新连续代数是Dense但BF16差分、GEMM归约和加法可有偏差，需预冻结neutral阈值与失败条件，不能声称byte-neutral。

## 最小固定实施合同，供独立审稿

实现新独立A100/Triton BF16工具，不改变作者源码或原生runner：kernel A 对128行按选择indices直接gather W1权重，FP32累加、bias、明确与原路径对齐的tanh GELU及BF16 h写回、计算差分；kernel B 直接gather W2权重，用差分乘积FP32累加，写入包含b2的缓存Y，禁止重复bias。两个kernel必须有实际dispatch/launch和有效/填充tile计数，输出经原gate处理，索引唯一有效且缓存按 `(source,seed,expert,block,rows,step)` 校验。精度路径需明确 delta BF16 或 FP32 的落盘和tensor-core输入；不能将不支持的高精度路径偷偷标为 FP32。

执行条件冻结 source898 high block20 step0→1和首128行；先验证assets，创建同M128 Dense step0/1参考，逐独立副本依序做：**全更新**、**零更新**、作者式proxy选集、同count固定seed随机选集；缓存与输入完全相同。前两者检验bias、缓存和数值闭合，后两者比较选择价值；记录实际chosen count、padding、选择掩码SHA、选集重叠、proxy/pack/cache/两核/回散完整边界。参考初始cache策略的变化单列，不混入速度比较。固定本次仅这一个source、一个function、一个组、两步；真实输入是teacher历史，不是自由student。128行实验检验局部物理动作与反例，不用于全M或整机速度估算。

作者源码 `chipmunk_mlp.py` 使用 `bmfc1=fc1(block_mean(x, MBM))`，与缓存的 `block_mean(mid)` 相减取绝对值，并对 `BM//MBM` 个小块求和，再通过作者 topk_indices/random/counts对齐及copy_indices维护被选参考。128行作为本受控port的明确单组，但 `MBM`、`BM`、实际 tie/random/256对齐和复制语义必须在实现时与作者配置和调用源码逐行冻结；若不能保持作者选择，标签改为明确的非作者策略，不冒称忠实。历史缓存按实际BF16操作写回，不能使用当前 full h/Y生成 oracle mask；预先确定随机seed。两种竞争预测：proxy超越同count随机并保留局部输出，或均值抵消/舍入造成proxy不优甚至误差大。若全更新或随机控制失败、identity/source/rows不一致、非finite或实际物理kernel未运行，停并留证；若质量/性能不优，记录有界负结果，不改选源或阈值重试。

另立 **full-M 合成性能诊断**：仅在独立review/ACTIVE批准后，固定 M31668、D5120、H13824、组大小128和固定p及group掩码，含248个组尾mask；比较原Dense与两核加代理、pack、cache、回散的热/冷成本，合成输入不可称真实功能证据。应预声明真实kernel dispatch、physical tile、GPU peak、warmup/重复、编译/setup、H2D、workspace、随机seed以及所有账本上界；物理HBM事务需要 profiler 实测才可报告。可先复审并实现128行单函数原型；若小M无法验证全M访存/共驻风险，不扩展其结论。

建议下一步：独立审稿先核对固定对象和真实作者 proxy/计数规则，再批准有界单函数原型的资源与neutral阈值；若需真实原raster分组或全M，另立采集与诊断scope。所有现有设计不授权科学执行。
