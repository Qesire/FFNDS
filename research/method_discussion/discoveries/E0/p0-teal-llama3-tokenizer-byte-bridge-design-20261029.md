# TEAL Llama3：公开HF tokenizer到作者tiktoken文件的字节桥

作者 `/root/s2_theory_design_independent`；仅 SRC/ALG 与拟议scope，REVIEW_REQUESTED。**没有下载 tokenizer.json、转换、执行模型/GPU或获取权重**。官方端点config/tokenizer此前使用本地凭据返回403；本桥使用独立公开再分发locator，不宣称官方端点访问成功。以官方公开LFS全文件SHA与大小作为严格通过条件，只有实际闭合才可称重建出相同字节。

## 可检验推导

[Transformers v4.44.1 `TikTokenConverter`](https://github.com/huggingface/transformers/blob/ca56cd7b31b2e1691235dd7f273eb3db139498de/src/transformers/convert_slow_tokenizer.py) 的 `extract_vocab_merges_from_model` 将tiktoken byte token通过`bytes_to_unicode`逐字节映成unicode串，保留原rank写入BPE vocab；[GPT-2实现](https://github.com/huggingface/transformers/blob/ca56cd7b31b2e1691235dd7f273eb3db139498de/src/transformers/models/gpt2/tokenization_gpt2.py#L33-L54) 定义映射：byte `!`–`~`、`¡`–`¬`、`®`–`ÿ` 自映射，剩余byte按0..255顺序映至U+0100起的连续码点。建立该256个字符到byte的逆映射，给每个 `model.vocab` token逐unicode码点反查并拼成原bytes，不用UTF-8直接编码字符。`added_tokens` 为独立特殊token机制，不直接写入mergeable BPE：只使用 `model.vocab`，并核对新增token与rank集合的关系，遇到重复/冲突即停；**不凭token字符串长相删除**可能真实出现的字节序列。

[tiktoken 固定源码 `dump_tiktoken_bpe`](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken/load.py#L134-L154) 将 `dict[bytes,rank]` 按rank升序写 `base64(token_bytes) + ASCII空格 + ASCII十进制rank + LF`；`load_tiktoken_bpe`按行解析base64/rank。该公式给出**候选canonical字节串**；原始Meta文件是否按同一顺序和换行写出目前未知，不能由字典同构证明byte identical。最后必须检查 `2,183,982` bytes 且 SHA256 `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55`：任一不符就停止、保留失败凭据、不把产物提供给模型，不宣称等价。

作者 gpt-fast [tokenizer.py](https://github.com/FasterDecoding/TEAL/blob/fb7373c93ac3594817c9ee64d4e08b47430a1822/gpt-fast/tokenizer.py) 用 `load_tiktoken_bpe` 并自行创建256特殊tokens；`generate.py` 读取与转换checkpoint同目录 `tokenizer.model`。完全SHA匹配将验证这一文件的原始字节身份；整条生成tokenizer路径还须之后结合作者源码、特殊token映射、路径和依赖校验，不能把本文假设变成实际编码测试。

## 精确输入与身份

匿名 HTTPS 仅取 [NousResearch/Meta-Llama-3-8B 固定revision 的 tokenizer.json](https://huggingface.co/NousResearch/Meta-Llama-3-8B/resolve/315b20096dc791d381d514deb5f8bd9c8d6d3061/tokenizer.json)：公开API文件大小 **9,085,698 bytes**，Git blob SHA1 **`b197f72effb9d5ed16ee0f5663e11e4cfac2ba62`**，同时匹配官方公开metadata。blob校验须实际对 `b'blob ' + ASCII(length) + bytes([0]) + contents` 求SHA1；Git blob不是文件SHA256，producer另记实际整文件SHA256。禁止带认证header、自动缓存凭据和非限定下载，验证最终文件长度、blob和source repo/revision，HTTP不成功直接FAIL。输入JSON已公开但尚未在本任务下载或解析；其 `model.type`、vocab 规模、结构、特殊tokens与rank范围是运行前检查对象，不能凭预测预填结果。Meta-Llama-3官方gpt-fast配置vocab为128256，常见128000 mergeable加256 special只是待验证假说，不拿假设替代解析。

官方模型的 `original/tokenizer.model` LFS metadata作为最终目标身份，尚未读取原文件；公开再分发LICENSE文件blob与官方相同、来源模型卡声明Llama3 Community License，记录完整许可/notice的locator与SHA并按其条款使用，公开访问不自动证明任何额外许可。原HF metadata证据见 [公开对照](../../../literature/20261029/P0_TEAL_PUBLIC_REDISTRIBUTION_METADATA_20261029.json)。

## 可实现的独立有界 producer 合同

1. 独立新ID、确定唯一HF输出/attempt、严格禁止覆盖/重跑；冻结代码、Python版本、source URLs与revision、输入期待blob/size、目标LFS SHA/size、许可/notice身份、启动spec和review fingerprint。执行前实际独立review→recheck→ACTIVE；此文不是授权。
2. 单CPU、CPU内存峰值≤512MiB、wall≤120秒/outer≤180秒、下载≤10MiB、HF新输出≤16MiB、0 GPU/0模型/0Torch/无权重。用stdlib HTTPS client保持匿名，禁止任意origin网络/超过上界的redirect与未知长度无界读取；若真实网络/资源证明预算不合理，保留失败并另审版本，不能默默加大。许可/notice与source metadata须作为预先冻结或限定的独立输入，不在10MiB tokenizer下载之外暗中获取。
3. 先创建confined fresh输出及STARTED RUN；逐块下载并预扣、记实际字节/超时与中断，输入封存SHA256、Git blob、期望size核对；输入损坏/多余字节/不合法UTF-8或JSON重复key、未知结构、分数/负数/布尔rank、非唯一rank/bytes、未知unicode码点、空token或异常特殊token集合即失败。应冻结合理token数/单token字节/总生成量上限，防解析膨胀；核对预期连续rank与256个单byte覆盖，不满足则停止并记录，不能人为修补数据。
4. 内存里构建逆映射、转换并按数字rank升序输出候选到限额临时文件；流式计量和fsync，读回base64/rank解析并核对唯一bytes/rank及完整映射（标为stdlib语义检查，不引入模型或tiktoken runtime）。按精确原始LFS预期size和SHA同时比较；不相等时只报告候选size/SHA和FAIL，保留证据，不重试猜排序/改文本。匹配后以原子rename发布至本scope专用路径；权限不扩大、禁止外链/路径逃逸。最终RUN记录来源版本、输入blob+SHA、输出hash/length、read/write/wall/CPU/peakRSS、校验与真实EXIT；在独立 `verify-closed` 检查真实EXIT0和RUN SHA、输出再哈希与路径绑定，不能以候选计算值代替已写实体。
5. 生产与下一作者模型任务分scope。实际匹配才可在后续独立审查后将输出复制/链接到明确checkpoint目录并计成本；如匹配失败，则原合法路径仍须获得原tokenizer，或独立评审HF tokenizer适配，不能继续假称作者完整复现。预声明不从文件“恢复”私密数据或绕过受门控的endpoint；输入是已公开再分发JSON。

小型软件控制可用自造0..255字节与控制字符、非ASCII bytes、rank无序、碰撞、JSON重复、特殊token隔离、换行和大小/SHA反例，证明序列化与边界；这些不是实际模型tokenizer转换。独立代码作者与reviewer还应核查 `model.vocab` 真实schema、原文件规范是否允许额外注释/BOM、输出失败保留与路径 confinement。

原始三份短源码保存HF `ffnds-research-sources/p0_teal_full_llm_20261029/tokenizer_bridge_sources/`，含固定上游commit和字节SHA见[源码台账](../../../literature/20261029/P0_TEAL_TOKENIZER_BRIDGE_SOURCE_LEDGER_20261029.json)。与TEAL方法的联系仅是作者LLM路径的资产身份，C20/G14仍为文献/源码映射，不是模型实测。
