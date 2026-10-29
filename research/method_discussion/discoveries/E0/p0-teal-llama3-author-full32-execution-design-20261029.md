# TEAL 作者 Llama3 32层完整LLM：分scope执行设计

提出者 `/root/s2_theory_design_independent`，2026-10-29；**SRC/ALG、REVIEW_REQUESTED，未执行新校准/模型/质量/计时**。目的先忠实验证作者公开32层LLM推理，再以单独FFN-only作用范围作解释；Wan2.2适配、单GEMV和完整LLM保持不同证据身份。TEAL作者代码固定 `fb7373c93ac3594817c9ee64d4e08b47430a1822`，已读源码与SHA见[原始源码清单](../../../literature/20261029/P0_TEAL_FULL_LLM_STATIC_SOURCE_MANIFEST_20261029.json)及[完整路径审查](p0-teal-author-full-llm-feasibility-20261029.md)。

## 现有状态与前置身份

[原始 tokenizer.model 字节桥](../../../20261029/TEAL_LLAMA3_TOKENIZER_BYTE_BRIDGE_RESULTS_20261029.md) 已实际EXIT0并closed，产物 `ffnds-derived/e0_teal_llama3_tokenizer_byte_bridge_20261029/attempt1/worker/results/tokenizer.model` 为 2,183,982 B、SHA256 `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55`，与官方原始文件公开LFS一致；不是完整tokenizer runtime测试。先前官方端点403事实保留；公开Nous再分发的四个权重分片服务端metadata与官方SHA/size一致，**实际16.07GB获取仍以真实EXIT/closed逐项SHA为准，不预填成功**。同时冻结公开config、index、tokenizer相关文件、许可notice、revision及取得路径，校验两种tokenizer所需资产；权重许可/下载、环境构建、转换、校准、推理、评测各自独立审查与ACTIVE，不能用前次byte桥或单kernel批准替代。

作者 `gpt-fast/model.py` 的 `llama-3-8b` 配置32层、D4096/H14336、32 query/8 KV heads、vocab128256、RoPE base500000。必须核对收到的config/各tensor header、32层完整键及原项目逻辑；特别确保checkpoint目录名含 `llama-3-8b`，否则 `Transformer.from_name(checkpoint_path.parent.name)` 的模糊匹配可能选择错误架构。作者代码 `generate.py` 固定目录内 `tokenizer.model`，转换脚本也尝试复制 `<checkpoint_dir>/original/tokenizer.model`；把已closed原始文件以经审核的映射/复制放在这两个确切相对路径并记录SHA及存储成本。公开镜像与官方是不同资产locator，即使字节匹配也不改写来源身份。

## 0. 原作者转换与环境（先行CPU scope）

作者 `gpt-fast/scripts/convert_hf_checkpoint.py` 读取HF safetensors索引与所有分片，在CPU累积字典，逐键映射为gpt-fast名字，将Q、K按head维度重排，与V连接成 `wqkv`，写 `model.pth`；其它投影直接映射。**转换不主动cast FP16**，须从实际BF16源实测记录产物dtype；作者生成 `_load_model` 在加载后调用 `model.to(device,dtype=torch.float16)`，因此FP16是运行时转换。量化、其他布局改动都属于新scope。对四分片、index/config、tokenizer、代码/依赖、产物做真实SHA、header、expected keys与转换前后count/shape/finite抽样检查；如果必须流式改写以满足host预算，该转换是标明的受控实现，需与原脚本逐tensor等值/字节校验，不可声称未经更改运行原脚本。

13个公开源资产共约16.07GB，原脚本同时驻留分片、重排QKV和序列化；host≤32GiB与可用磁盘在完整分配/复制峰值未测，必须先对源码、allocator和可用空间作实证或在受审有界实际执行时硬限制并允许失败；转换结果另占约模型大小，不计入权重下载scope。作者 Python≥3.11、torch2.4.0、transformers4.44.1、safetensors、tiktoken、sentencepiece及其固定lock，质量路径要求flash-attn2.6.1等。已存在项目venv版本不同且缺若干包；独立构建/验证环境、Wheel/包SHA、CUDA/driver/GPU compatibility，网络与编译另记。转换无模型forward但可能用Torch CPU，仍须作为新CPU实验review。

## 1. 原作者校准scope：实际原300条及直方图

固定公开 Alpaca `tatsu-lab/alpaca@dce01c9b08f87459cf36a430d809084718273017` 作为**当前固定revision，不声称与作者当时同一快照**，取得可验证dataset文件SHA及行顺序。按 `teal/grab_acts.py` 原方式流式train `skip(0).take(300)` 拼接各 `text + '\n\n'`，tokenizer `truncation=True,max_length=2048,return_overflowing_tokens=True,padding='max_length'`，取最多前10段；实际段数、pad/attention_mask、token IDs SHA及被截断余量必须记录。作者将 `attention_mask=None`、显式位置索引，逐32层调用 layer 并在每层保存输入状态，分别在attention/MLP抓h1与h2；抓取是dense、无训练。

`ActivationModule.find_histogram` 展平对应实际激活、GPU sort，默认10000 bins、按1%/99%附近构造边界，然后 `torch.histogram`，保存每层 `self_attn/histograms.pt` 和 `mlp/histograms.pt`，共64个，每个各含h1/h2 centers与counts；作者保存32层输入激活另约 `32*10*2048*4096*2=5,368,709,120` B的理论FP16 tensor payload（不含文件/metadata，实际以段数、dtype为准）。大中间数组、GPU sort及CPU缓存/持久化会增加峰值，预算由producer实际精确预扣，非可忽略的免费阈值。`Distribution.icdf(.5+.5p)` 基于有符号histogram的对称分布假设，不等同直接精确 `|x|`经验中位数；p=0.5只指定校准target，实测每层每输入实际零率可能不同。源版本、hist 64文件SHA、h1/h2、格式与出入库、生成成本和真实RUN/EXIT独立闭合。已有作者预发布hist可另做“作者随仓库预计算hist”范围，不冒称本轮按原脚本重采。

实施前按具体实际代码预算：一次来源获取、一次tokenize、32 layer执行、32输入状态持久化、64 histogram输出及各实际Tensor sort；把模型加载、FP16 cast、激活/hist内存、GPU峰值和所有文件写入记账，非一次开放式脚本。任何依赖/FA2调用签名/内存/输入不足导致失败就保留真实状态，不换分片或自动重跑。首次单GPU≤80GiB、CPU≤4、host≤32GiB为候选上限，实际能否共存UNKNOWN；精确wall/磁盘/网络限制及操作predebit由producer自审和独立review后确定。

## 2. 完整32层作者生成与控制

同固定模型/原权重/转换产物/FP16、同tokenizer、prompt ID、seed1234、sampling top-k200及temperature0.8、同decoder KV策略、同环境与实际GPU，先做真实3臂：

1. **Dense**：`hist_path=None`，不patch，原32层FFN与attention均为dense。
2. **零阈值 neutral**：作者patch路径启用真实稀疏kernel，显式将每层所有threshold设为0（**作者按p=0计算有符号median不保证真正为0**，故这是有标签的测试控制）；必须记录mask零值、投影对照和FP16 split-K原子求和差异，teacher-forced同token逐step logits、top1和KV语义用预冻结容差验neutral，失败停止，不以自由生成随机碰巧相同替代。
3. **作者uniform50%**：作者 `--hist_path ... --sparsity 0.5 --compile`，每层七个逻辑投影q/k/v/o、gate/up/down均使用阈值，h1/h2分别来自64 hist；qkv融合是一项kernel dispatch但三个门限，阈值选择及observer开销必须记。这里是单token LLM，不使用Wan空间分组假设。原kernel为FP16输入、矩阵列主序和split-K累加；编译、转置、warmup、阈值/布局准备与首次allocation作为setup单列，不省略。

同一多token prompt预先冻结token序列并强制长度>1；作者 `SparseGEMV` 和QKV在seq>1执行**未经mask的dense fallback**，仅seq=1 decode才执行mask sparse。记录每投影position/seq长度、七投影数、底层物理dispatch（fusion与compile可能使Python调用计数不可代表物理kernel），分别输出prefill、decode、采样、KV、LM head、E2E墙钟/CUDA event及setup。此范围是作者**all-matrix**；用户研究FFN-only须单独受审定义：attention沿原Dense、仅gate/up/down按各自hist稀疏；使用同FP16、同prompt/noise/seed、相同调用/时间并记录差异，这一隔离不是默认作者完整配置。后续用四臂等条件更有辨识力但需另增预算与审查，不将两种的收益合并。

具体首个有限范围可固定每臂1 warmup+2测量、64个新token，9个真实generation上限；每次第1 token来自prefill，后续最多63次单token decode，所以≤567个decode model调用；每decode 32×3 FFN逻辑投影，合计≤54,432（其中neutral与candidate≤36,288）；每层qkv fused+o共两项attention projection dispatch口径，3×3×63×32×2≤36,288；多token prefill另≤9×32×3=864 FFN逻辑投影及≤9×32×2=576 attention调用。实际EOS/停止时按实记。编译/autotune、可能的CUDA graph replay、tuning kernel和资源监控额外上限须在实际producer中细化；完整调用次数是上限而非实际物理launch。真系统速度对比应同配方、warmup、batch1、固定长度或报告实际token数、同 GPU独占且交错顺序，生成/重排/阈值setup与trace单独计；额外独立trace单独预算，避免在有侵入observer的时间中假称论文速度。mask实际零比例记录各层/投影而非假设50%。

## 3. 作者质量与同前提比较

作者 `teal/ppl_test.py` 使用HF `SparsifyFn`、FP16、flash attention路径，对多token **后半token显式mask**（各q/k/v/o及gate/up/down），线性算子本身仍dense；step=1时完整mask。它评估相同阈值函数但与gpt-fast速度路径多token**不mask**的prefill语义不同。先执行原示例的 `Alpaca train` 首250条与上述300校准的前缀重叠，作为**忠实历史recipe且非held-out**；明示作者 `eval_ppl` 将逐窗口平均loss再平均exp，记录实际有效token、最后窗口和原程序口径。

独立补充WikiText validation固定revision/8预选独立长窗口、2048 context+512评分token，或另外不重叠Alpaca区域；token化后冻结完整ID、窗口indices与样本hash，Dense/0/50全用相同token和mask计划。作者论文是WikiText validation **128随机样本、block-wise greedy及六任务**，8窗口uniform仅开发质量试验，无法宣称论文表复现；完整作者任务需另审数据、代码版本、样本身份和预算。另一公平对照将 gpt-fast dense-prefill +单token sparse decode 的自回归teacher-forced log-likelihood用于相同目标token（限步数、KV与逐token成本），应与HF后半mask PPL分别命名，不能把未mask全序列prefill的PPL称实际稀疏评测。两条评测实现、开销、backend和输出解释均保留。

## 可证伪预测与启动门槛

H1：neutral通过，真实稀疏dispatch跨32层闭合，作者50%在完整模型稳态decode有净加速且评测NLL退化可接受；H2：原prefill/LM head/KV与稀疏layout成本抵消单核收益，或阈值/attention误差使相同teacher token质量和自由生成明显偏离；均可能出现单核1.85×但整机无收益。两个expert Wan/图像视频事实不能为语言模型结果背书。kernel尾shape/autotune、FP16累加及实际批大小须在GPU前校验，非finite、mask不覆盖、neutral失败、超资源或真实输入闭合失败即停，完整失败收据保留，不自动重试或用猜测改阈值。

每项实验先自审、不同Agent对抗复核、修订与精确指纹recheck，再注册ACTIVE、目标节点复核GPU1与当前**全局≤4卡**；上面是可实现设计，预算和脚本仍需实际producer、review和登记，既无实验通过也无速度/质量结论。
