# 独立 v5：paired driver 的计数、收据链、失败扣账与解释范围

`/root/independent_e0_audit`，2026-10-26。只读源码/数学审查；未运行模型、GPU、张量、测试或Git。保留v3/v4及原始回执，不修改冻结实现。此文不授予执行、方法支持或质量验收；实际授权由用户与ACTIVE的精确scope决定。协调者告知v2已获授权且首pair开始加载，本审阅不将加载当作完成证据。

## 精确读取版本

| 文件 | SHA256 |
| --- | --- |
| `tools/run_e0_paired_full_depth.py` | `82df60878f7a010e1682977aa404049d2d217573ac30b352545c030821ed9b2e` |
| `config/e0_paired_full_depth_20261026_v2.json` | `dfefff076066c766766897ab76bb161b1e82fbfb74c5a0031c3f91c68ef95ce0` |
| `tools/ffnds_paired_video_strict_gate.py` | `59d6596cff43cc6197d4b1ab76eb90e9ca92855901b6d3394b7eabc257c9a6d3` |
| controller / observer / residency | `bd99bdba0651dc59fcc5a6cdf72922906a5c26eb08dfb4d4650f1d6cefa560bd` / `691a2e66f3326ce1f6b65536d576f8b89c9861db217fdee4885463a54d196476` / `75ecc7a5cc8d338c2e0afee8c301d3eefaabb3ddb6daa48814eb0b9853052bc1` |

亲自核对协议driver/native全部列出的SHA与当前文件一致；未调用driver preflight或任何模型入口。此前v3设计推导与v4接口审查原字节保留。

## 1. 科学对象与理论总账

仍是离线Wan `affine54`：原Q1后保留54×64 hidden，经原tanh-GELU/Q2及新BF16读出，加新BF16输入affine支路，再通过真实原gate/residual。四类解释保持分开：部署native/参数差、student前缀输入变化、signed下游传递、四步flow-state变化。`P23/P24`的原生记录需实际完成，不能由`P22/P25`终端指标或本静态审阅替代。

协议固定顺序为DD source0/seed42、DC source0/42、DC source12/54，共3pair、6完整四步轨迹、每pair两次解码、单GPU串行。source0/12已暴露，不是新source泛化。DD仍计算candidate shadow，故三个pair的四函数总算量相同，仅main/shadow分配不同。

每pair的target-linear账本可独立重建：160site×两actual前缀×每前缀8个非FFN target linear = **2560 nonFFN**；两dense函数×两个GEMM×160 = **640 dense FFN**；两candidate函数×三个GEMM×160 = **960 candidate FFN**。因此runtime只记的3200为2560+640；全部native target GEMM为3200+960=**4160**，其中FFN native为**1600**。FFN native QDQ为8×160=1280，另nonFFN2560，合计 **3840**。两个32行continuous每site各3GEMM，合计 **960**。

三个pair对应协议 **11520 native target QDQ、12480 native target GEMM、2880 continuous GEMM、480 paired sites、1920 native FFN函数求值**，算术一致。两DC pair保存8行×10个5120宽BF16字段×320site = **262144000 bytes**，不含safetensors header。driver会核actual observer计数、runtime3200/3200、cache misses=0、当前block FFN转移320等；目前只是断言存在，不是断言已通过。

这些是target模块计数，未涵盖VAE/T5/patch embedding/head等所有物理算术；不能称整模型全部GEMM。计数也不能替代不同矩阵shape的MAC或真实时延。

## 2. 来源与模型身份

默认仅做文件读取/hash；execute另要求ACTIVE内相同scope id、精确protocol SHA及允许执行状态。必须绑定mandatory driver/native来源、canonical input、cache manifest、旧结果及80个固定affine pack。执行前还校验冻结的canonical study root和私有executor配置；本记录不摘取私有节点/资产路径或硬件标识。

实际sample的image SHA和seed须等于pair冻结身份；generate使用同一个sample.prompt与显式空negative prompt。成功路径逐branch核四步legacy latent值hash/shape和MP4 SHA与各自原dense或原candidate RUN完全一致。因此DD双轨必须复现原dense，DC的teacher/candidate分别复现原对应分支，而不只是两轨内部相等。

新版本evaluate_pair读取原dense RUN，检查其phase seed/video SHA，并逐项对比原/新checkpoint SHA、cache manifest SHA、contract。old reference模型digest用原assets构建；student reference使用实际teacher dense身份，避免先前补评复制candidate model digest的缺陷。新strict gate另从实际generation receipt的video inventory及D/S mode重建模型digest，与report交叉检查；不只接受报告自报身份。

prompt身份的真实性来自被冻结的driver、canonical输入及实际调用参数的共同链；strict gate自身没有从独立的原RUN prompt原文再次计算。引用时应保留这一来源区别，不把所有身份语义都归功于hash gate。

## 3. 无SHA环的收据顺序与全视频指标

成功路径先完成native、TRACE、两MP4与最终monitor记录，再写独立的 `NATIVE_RUN.json`。之后metrics绑定这个已经稳定的native receipt SHA；BUNDLE绑定视频、metrics、native receipt及dense reference；STRICT_BUNDLE再绑定core BUNDLE和独立native inventory。最后final RUN写native SHA并更新ledger状态。NATIVE_RUN不回写metrics SHA，因此没有“报告hash必须包含自己的未来hash”的环。

每个新teacher clip对原dense作配对全81帧评价；每个student clip对本pair teacher作配对全81帧评价。原evaluator及core gate之后，再过strict inventory/model gate。strict gate要求本native目录实际MP4集合恰好等于两产物清单，检查遗漏/重复/额外clip和实际dense reference身份。这是真正比原v1 gate更强的native证据入口，不代表官方质量通过。

初读未冻结草稿在native检查失败后会直接跳过metrics，我已即时提出挑战。当前绑定版本新增 `collect_failed_video_evidence`：冻结FAILED_NATIVE_RUN，对每个实际clip尝试全帧自身诊断；文件缺失或解码/评价失败分别保留状态与SHA，metadata不全也列明failures，不伪造PASS。FAILED_VIDEO_EVIDENCE明确core_gate_pass=false、quality_accepted=false。它是失败取证，不是把失败结果补成完整成功bundle。

最终RUN未直接记录BUNDLE/STRICT_BUNDLE SHA；两bundle内部已有向下绑定，原文件仍可独立核验，但最终RUN的字节本身没有证明哪一个bundle被通过。已作为非阻断审计边界报告协调者；可在公开结果索引另绑精确bundle SHA，保持已冻结/正在执行文件不动。

## 4. 失败扣账、不重试与成本

初读草稿的ledger可通过更换study_root重置；我已提出scope级预算缺口。现版本校验冻结canonical root，锁保护STUDY，按协议固定pair顺序，每次预留2轨迹，任何既往非PASS状态阻止新尝试；不自动重试，也不在失败后自动继续其他pair。private executor绑定与GPU reservation负责实际设备范围，不能只靠“同一目录有锁”保证GPU隔离。

仍有一个保守的初始化失败边界：reserve之后的out.mkdir及首次RUN写入在主try之前，若它们失败，ledger可停在STARTED而非结构化FAILED收据；两轨迹预算仍已消耗，后续被阻止，不会无声释放额度。此处不能描述为“所有失败都一定已有最终RUN”。

成本字段是 **whole paired offline observation + two decodes**，包含双轨、shadow、continuous、FP64 reduce、hash/保存/同步；不代表单候选部署成本。两视频重复引用同pair成本，并带cost_deduplication_key；研究总账按唯一pair去重，不能将6clip的cost求和。FFN event是primary+shadow边界，不含postgate/finish_site分析，与旧部署FFN不是同一计时对象。

最终cost peak从closed monitor的资源记录统一填入，关闭初读草稿的早期peak/最终资源记录可能不同的问题。report所述是采样物理峰值，仍非连续显存上界。软件payload限制不能证明原生一定适配显存；失败不得追加自动重试。

## 5. 两个关键解释限定

`native_vs_continuous`仅取固定32行，以这些行continuous能量为分母；full-M teacher/student operator SSE使用完整M及dense-teacher能量。缺少同32行D-vs-S差及相应cross term，不能用两者数值大小宣布H-NUM主导。相同覆盖上应先定义 `e_op=(S_native-S_cont)+(S_cont-D_native)`，其SSE包含交叉内积；native-cont差还包含设计中的Q2作用，不自动表示实现bug。

DD的actual modes虽为dense/dense，shadow candidate仍可使teacher_operator、student_operator乃至四函数total_SSE非零。若U_T=U_S，则prefix_response为0，而这些operator SSE等于相同的`||S-D||²`；DD中性看actual q1、block_output、flow/latent轨迹与历史native指纹，不能要求四arm全部SSE=0。

对当前成功的冻结普通I2V pair，controller验证每step两轨embed0相等；c_gate_msa只来自共享block.modulation+embed0的chunk，与latent前缀无关。因此同step/block `gT=gS`；当前writeback prefix term仍含不同residual base、dense函数输入响应与BF16写回，却不应把gate变化当成本次观测解释。泛化公式允许不同gate，不代表此配置实际发生。

以上两点已另以短反馈直接交给协调者；没有为检验它们启动任何计算。最强可支持结论是冻结driver的计数/身份/收据顺序在明确范围内静态一致，并修复了已指出的失败证据与预算命名空间缺口；原生计数、hash中性、数值分解、指标bundle与质量状态仍需实际完成的结果核验。
