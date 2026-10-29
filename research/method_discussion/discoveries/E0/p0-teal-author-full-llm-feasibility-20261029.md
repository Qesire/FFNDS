# TEAL 完整LLM作者路径：可执行性与有限复现设计

作者 `/root/s2_theory_design_independent`；2026-10-29；LIT/SRC/metadata，**REVIEW_REQUESTED**，未下载模型权重、读取科学tensor、运行模型/GPU或授权实验。已闭合单个FP16 kernel测试及作者式benchmark是有限内核证据，不是完整 LLM。固定作者 [commit fb7373c93ac3594817c9ee64d4e08b47430a1822](https://github.com/FasterDecoding/TEAL/tree/fb7373c93ac3594817c9ee64d4e08b47430a1822)，本次18个真实源码文件保存HF，逐文件版本/字节/SHA见 [source manifest](../../../literature/20261029/P0_TEAL_FULL_LLM_STATIC_SOURCE_MANIFEST_20261029.json)。

## 作者路径与决定性的质量区别

作者 `gpt-fast/generate.py`：FP16、model+KV cache、seed1234；CLI默认5个sample、每sample最多200新token、temperature 0.8、top-k 200；`--compile`时额外一次warmup并编译单token decode，prefill默认不编译。`model.py` SwiGLU有 gate、up、down 三个 FFN GEMV，每层还在作者默认完整patch里替换 attention QKV和O：作者全模型稀疏并非FFN-only。完整模型保持词嵌入、norm、RoPE、attention/KV、SiLU乘法、LM head和采样；不能以每矩阵独立kernel的加速当作系统token/s。加载hist后作者 icdf 用0.5+0.5p；在p=0时并未特殊保证threshold恰好0，因此**零阈值neutral必须直接明确强制所有阈值等于0**，标注为测试仪表控制，不将p=0一律当精确neutral；真实FP16 split-K atomic顺序与dense仍可能不同，冻结数值/logit/token容差和失败处理。

`kernels/sparse_gemv.py` 的 seq_len=1 用实际作者阈值稀疏GEMV；**seq_len>1返回未经mask的普通 dense matmul**，QKV同理。这是推理prefill和decode混合语义。另一方面作者 `teal/mlp.py` / `self_attn.py` / `utils/utils.py` 的HF质量路径对多token默认将**后半序列逐元素mask**，再用dense线性层计算；单token全部mask。它能评估作者指定的稀疏函数，但没有证明稀疏物理速度。不能在未mask的gpt-fast整段prefill算出PPL后称稀疏PPL；也不能把HF质量路径的矩阵mask当实际稀疏kernel速度。论文 §5.4.3明确讨论此差异：WikiText等似然任务后半prefill mask，生成任务prefill不mask。

实际作者示例 `grab_acts.py`: `tatsu-lab/alpaca` train流式前300条拼接，tokenizer截断并溢出为最多前10×2048、max-length padding，逐层取激活并建立hist（默认10000 bins、1%尾部边界）；`ppl_test.py` 默认同train前250条、`eval_ppl`按context2048/window512切连续文本求窗口loss后取均值的exp。这两个示例有**同一前缀重叠**，不是held-out。仓库 `models/Mistral-7B/histograms/layer-{0..31}/{mlp,self_attn}/histograms.pt` 提供预计算阈值来源，但这些文件的实际校准数据版本/随机身份未由静态源码证明；下载固定commit并逐文件SHA、受信pickle策略/来源和32层×2文件完整性需要独立确认。论文报告的是[WikiText validation 128个随机样本、2048 context+512窗口以及六任务](https://arxiv.org/html/2408.14690v3)，主要50%表为block-wise greedy而非这里的uniform。仓库默认示例不等于论文完整评测；如复做作者原示例须标注数据重叠，held-out另用已固定WikiText validation。数据版本、随机样本索引、tokenizer revision和所有评分token需冻结；原论文128样本的具体随机种子/选择资产尚未找到，不假称论文数字可逐点复现。

## 可获取对象与真实缺口

[官方Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/27d67f1b5f57dc0953326b2601d68371d40ea8da) 公共、非gated、Apache-2.0；HF API当时revision `27d67f1b5f57dc0953326b2601d68371d40ea8da`，两BF16 safetensors分别9,942,981,696与4,540,516,344 bytes、LFS SHA `9742cb4764964155b7a5f35eefad651f590006091ddeb536863d6c5865cca1b9` / `9bcf56354ec0c68b5f8e97b4f3b02d16af899a65b0868d6dba5a51c1b30f01cb`；tokenizer.model SHA `dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055`。仓库配置有 Mistral-7B 32层、D4096、H14336、8 KV heads，但README将FP16 inference正式支持列为Llama-2/3、Mistral列为accuracy；gpt-fast模型配置本身不证明Mistral全推理或权重转换正确，需先逐项检查完整架构（尤其sliding window、RoPE、tokenizer/权重转换）。**若未证明，Mistral sparse decode是明确的受控移植，不能报告作者原条件已复现。** BF16 checkpoint转换/加载到作者FP16须记录权重转换、逐文件SHA、时间与峰值，不能把原权重dtype说成FP16。

[Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) 是README明确指定的作者推理目标，当前API标manual gated、revision `01c7f73d771dfac7d292323805ebc428287df4f9`；许可与实际账号访问状态未验证，不假设可下载。当地 `/home/xiang/.cache/huggingface/hub` 的Mistral-7B目标目录不存在；查见Llama-3.1少量缓存目录但未证明包含权重，不能称可用；其他节点资产未核。公开数据 [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) revision `dce01c9b08f87459cf36a430d809084718273017`；[WikiText](https://huggingface.co/datasets/Salesforce/wikitext) revision `b08601e04326c79dfdd32d625aee71d232d685c3` 包含wikitext-2-raw-v1 validation。两者revision是本次API所见，非作者历史数据快照；正式scope获取实际文件、索引及SHA，重新核对文本/采样顺序。

作者要求 Python≥3.11、torch2.4.0、transformers4.44.1、eval额外flash-attn2.6.1等；本地 `.venv` metadata为torch2.10.0+cu128、transformers5.6.2、triton3.6.0，datasets/tiktoken/blobfile/msgspec/flash-attn未安装。已有kernel环境不能代替完整LLM环境；单独审查隔离安装、编译、license、仓库闭包和真实GPU/driver兼容，不能宣称已满足。作者转换脚本把各分片累积于CPU再写model.pth，约14.48GB原权重与转换副本、临时布局/loader可能超过32GiB；≤32GiB host应事先以实现证明峰值及受限方案，否则失败并另审转换scope，不能靠声明预算保证。原代码更改、转换或格式重排产生另一个SHA与成本。下载约14.48GB权重、tokenizer/config及作者64个hist均是独立受审获取范围，不能虚构已闭合。

## 拟议有限两段实验（分别review/登记）

**A 原条件模型路径优先**：先确认账号可获优先目标 Meta-Llama-3-8B 及安全kernel tile、作者 pinned源码/环境/hist和小权重定位；该目标当前config/tokenizer实际HTTP403，保存阻断，Mistral做明确的“作者机制+Mistral受控适配”，不代替原Llama速度复现。两者均先验证所有线性层shape及作者kernel输出tile的尾部边界；Llama-2 H=11008可能不整除某个512输出tile，未经安全证明的自动调优候选不得运行；屏蔽/修正候选即明确补丁，作者未改核的性能复现保持另一个身份。

在选定并验证后，固定同模型/tokenizer/FP16/输入prompt/initial KV/sample RNG；三臂 **Dense**（不patch）、**显式全部阈值0的稀疏kernel neutral**、**作者uniform 0.5**（all attention+FFN，按实校准hist阈值，非每次恰好50%）。额外 FFN-only 是不同定义需另臂及预算，不得把它与作者all-matrix混同。固定相同一条多token prompt、三次每臂各64新token（含1 warmup+2有序测量），除非真实EOS导致提前结束则按实记，不伪造完成；共≤9次prefill、≤576新token、≤567个单token decode model调用。每decode 32层×3 FFN投影，预算全臂≤54,432 FFN投影调用，neutral+candidate≤36,288稀疏FFN调用；作者全注意力每层每decode另1 QKV+1 O调用，≤36,288全臂attention projection调用。prefill多token dense fallback、首次sample及KV更新单独计；逻辑调用、physical kernel launch、编译/autotune额外调用和timing必须分别记录，后者没有观察时 UNKNOWN。若compiled graph使每次Python调用计数不能代表实际dispatch，则加独立trace而非把Python次数当真实launch；trace额外预算冻结。

以同给定token teacher-forced逐token比较logit误差、top1和缓存；t=0 neutral先通过预声明有限容差及finite，再用相同随机流自由decode记录实际文本/采样差异，不能要求生成文本字节一致代替数值控制。报告全部7个投影实际门限和观测零率、作者kernel调用、FP16 dtype/布局、外层splitK、cache、编译与预热、模型加载/转换、整体墙钟+同步event token/s、GPU peak和CPU RSS；时间必须包括真实selector、prefill、decode、采样、KV和同步，分别给cold/warm，交错多次、保存每次值；此小scope不是论文速度结论。操控sparsity还影响attention，因此FFN-only节省须在独立对照中测量。

**B 质量路径**：同checkpoint/tokenizer与hist，HF作者 `SparsifyFn` +原支持的transformers/flash-attn环境，固定WikiText validation随机种子与8个独立长度足够的样本窗口2048+512（比论文128小，报告开发诊断而不称论文全复现），Dense、0阈值neutral、uniform50%按作者后半prefill mask与同评分token做teacher-forced NLL/PPL；记录每窗口 token、NLL、拒绝短窗口和权重汇总。避免作者示例把最后一个不足stride的loss等权当完整512令牌；如采用作者原窗口loss均值则单列“作者程序口径”与正确按有效token加权口径，说明可能不同。论文greedy50%、128随机样本与六任务是另行有限完整scope，涉及固定数据版本/源码任务列表与多次正式评审；本scope不冒充全复现。

两段各最多1张80GiB A100、CPU≤4、host≤32GiB、时间/HF/output/read/转换和编译预算以具体producer预扣冻结；当前用户全局最多4 GPU，由协调者安排空闲卡且不覆盖已有scope。正式设计必须给实际完整调用预算（包含图编译、warmup、decode、teacher reference、评测窗口、并行和输入下载），失败留RUN/EXIT不自动重试。权重和hist下载、环境构建、转换、A及B须各自独立审查和授权；本设计没有完成这些准备，也没有批准票。

## 相反预测与决定

H1：neutral通过、已付费阈值让decode真实访存降低，50%层间实测稀疏率与真实完整模型吞吐改善同时，WikiText后半mask NLL变化有限；才值得扩大数据/论文条件复现。H2：kernel层速度在LM head、attention、KV和compile之外被吞没，或固定阈值在不同层/文本的实测稀疏率偏离、quality显著恶化；零阈值数值不稳、Mistral架构差异或tail潜在越界也可先否决该具体实现。若单kernel快而whole-model不快，不推断作者论文错误；硬件/模型/预热/口径不同。当前Mistral无门禁但推理支持未被README承诺；原Llama路径账号或安全shape未确认，最有辨别力的下一动作是**只读获取许可/资产元数据并对模型架构和全kernel tile做静态闭合**，之后把作者LLM路径与Mistral适配分别提交独立评审。

## 优先对象：Meta-Llama-3-8B 的实际门禁与作者配置

协调者要求优先作者明确支持且 FFN 形状匹配已测kernel的 `meta-llama/Meta-Llama-3-8B`。只读官方API显示 revision `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`、manual gated；本地HF credential存在，但使用该credential访问此精确revision的 `config.json` 与 `tokenizer_config.json` 均实际HTTP **403**。同样方法对Llama-2-7b-hf这两个文件也实际403；未输出、传出凭据，未尝试大权重下载。因此当前账号无法取得官方受门控对象，元数据可见不等于访问批准。取得访问后另核固定revision的config/tokenizer/许可证/实际完整SHA；非官方镜像如被提出，须核逐文件原始内容身份、许可与路径，作为不同来源，不假称官方访问通过。

公开API提供四个Llama3 safetensors LFS元数据：`model-00001-of-00004.safetensors` 4,976,698,672 B SHA `f2c144103072514542e327fa8080bd375cb300f2d453fba9ca3aea81d0d4cf33`；`00002` 4,999,802,720 B SHA `d9eee5f23d94405d90b7e9ff88b9443fee42f8528a658f54214c2aba7530d80c`；`00003` 4,915,916,176 B SHA `4b8fbc5e113f69768dd8de84661ea20af8a32b734a9976144b4236c447b40ccc`；`00004` 1,168,138,808 B SHA `5dc34e6bdf2da9e35f0d93b5c333c870f3677dc43dc3a91ea3a8ad28a1fe1acb`。这只是服务端LFS元数据，未下载/校验实物。作者 gpt-fast `model.py` 自身的 `llama-3-8b` 配置为32层、D4096/H14336、32 query/8 KV heads、vocab128256、rope500000；与先前kernel测试形状有关，但不是对门禁内官方config的验证。先检查完整作者config匹配及安全tile，不能由形状就断言完整LLM能跑。

作者原Alpaca train前300与评测前250重叠，是可在另行授权后按原方式忠实复刻的历史recipe，结果必须以“校准前缀重叠”标记；另开固定WikiText validation及不重叠Alpaca后续样本作counterfactual，不悄然替换作者路径。无论文128随机索引身份时不得写精确重现。

## 公开再分发对象：权重匹配，作者 tokenizer 仍未闭合

匿名公开HF API比对已保存在[机器元数据](../../../literature/20261029/P0_TEAL_PUBLIC_REDISTRIBUTION_METADATA_20261029.json)，原始API响应及公开模型卡/小config在 `ffnds-research-sources/p0_teal_full_llm_20261029/redistribution_metadata/`；未下载权重。原官方端点对现有凭据的 config/tokenizer 403 事实不变。

[NousResearch/Meta-Llama-3-8B](https://huggingface.co/NousResearch/Meta-Llama-3-8B/tree/315b20096dc791d381d514deb5f8bd9c8d6d3061)，revision `315b20096dc791d381d514deb5f8bd9c8d6d3061`，公开非gated；四个 safetensors shard 的**服务端 LFS SHA256 和大小**均与官方 `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920` 逐项相同。`config.json`、generation_config、safetensors index、tokenizer.json、tokenizer_config、special_tokens_map 和 LICENSE 的 Git blob ID也逐项相同（文件实际读取及强SHA/安全转换另验），该公开config实际可读，SHA256 `2430cee764b6530ff8673cf9ba8561e1d5a33152d503cd0de909ff5718261441`。模型卡指向 `license_name: llama3` 和 LICENSE，该 LICENSE blob 与官方相同；可公开读取并不自行证明具体用途满足 Meta Llama 3 Community License，应按条款及实际权限核对，不把再分发仓库改称官方endpoint。

**决定性缺口：** 作者 `gpt-fast/generate.py` 要求 checkpoint同目录 `tokenizer.model`，`tokenizer.py` 的Llama3路径用tiktoken读取其BPE格式。公开再分发仓库虽有与官方blob一致的 HF `tokenizer.json`，其文件清单无 `original/tokenizer.model`；官方原始文件有LFS SHA `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55`、2,183,982 B，但该模型官方访问仍受门控。HF tokenizer的blob相同可用于HF质量路径身份核查，**不能据此声称作者原 tiktoken 文件已取得或token IDs/BOS/EOS完全等同**。合法获得匹配原始文件，或者另立清晰标注适配的 tokenizer 及全词表/特殊token和冻结样例编码对照审查，方可准备该推理范围；后者不应称无改动作者tokenizer。作者可公开提供的 `models/Llama-3-8B` 直方图仍需逐文件获得、验证并核实对应模型。

[NousResearch/Llama-2-7b-hf](https://huggingface.co/NousResearch/Llama-2-7b-hf/tree/8efe6c9b93655b934e27bd9981e3ec13e55aee9d) 两个权重shard LFS SHA/大小及safetensors index Git blob与官方相同，`tokenizer.model` 的LFS SHA/大小也相同；然而config、generation_config、tokenizer.json、tokenizer_config和special_tokens_map Git blob**不同**，不具备完整recipe字节相同证明。模型卡写Meta许可且要求接受许可，API没有明确license字段，公开仓库也未列单独LICENSE；许可及差异均需核实。Llama2 `H=11008` 的某些kernel输出tile尾部风险仍需解决；不能因权重SHA匹配就跳过。

这些对比仅证明服务端元数据声称的对象身份，不能证明下载后实物完整，不能绕过原官方许可/门禁，更不能代替注册前原模型架构、tokenizer、训练数据、运行环境和source SHA复核。公开仓库是新资产locator，未来下载需独立scope和逐文件全SHA。
