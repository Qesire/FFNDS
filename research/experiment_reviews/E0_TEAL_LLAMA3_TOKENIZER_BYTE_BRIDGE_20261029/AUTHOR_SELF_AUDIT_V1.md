# TEAL Llama3 tokenizer 字节桥：作者自审，待独立评议

状态 REVIEW_REQUESTED；未下载、解析或转换真实 tokenizer，没有模型、GPU、Torch 或权重运行。目标是核对一个公开镜像 JSON 中的 ByteLevel BPE rank/字节表能否按固定 tiktoken dump 规则生成与官方公开 LFS 元数据完全相同的文件；即使完全匹配，也只证明文件字节身份，不证明作者 tokenizer 的编码路径或完整 LLM 复现。

## 证据与可推翻预测

输入限定匿名 NousResearch/Meta-Llama-3-8B 的 revision `315b20096dc791d381d514deb5f8bd9c8d6d3061`、tokenizer.json，精确 9,085,698 B、Git blob SHA1 `b197f72effb9d5ed16ee0f5663e11e4cfac2ba62`；运行时还计算文件 SHA256。Git blob 使用 `blob <length>\0 + content`，不误写成文件 SHA256。唯一通过目标为 2,183,982 B 和 SHA256 `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` 的同时匹配；候选输出、readback 和最终发布文件分别重算，任何差异保留失败、不重排猜测或重试。候选路径不是可供模型使用的 `tokenizer.model`；成功后在同目录以 `link` 的不可覆盖方式发布并移除候选。

竞争解释：JSON的 BPE byte→rank 字典可能可逆且与 Meta 原文件使用同一排序与换行，得到完全相同字节；也可能即使 rank/字节集合一致，原始文件排序、换行、注释等使 SHA 不同。因此最终逐文件字节验证是决定性证据，不能把字典等价或本自审当 PASS。固定 Transformers ByteLevel 源码、tiktoken dump 源码、公开/官方 metadata、公开 LICENSE/README 和 review 设计均按现存小文件哈希绑定；不额外下载许可或metadata。许可与notice身份已记录，但不将公开可访问解释成额外许可。

## 实现与失败边界

- 仅 stdlib `http.client.HTTPSConnection` 直连 `huggingface.co`，不使用代理、凭据、Authorization、Cookie或HF客户端；只有初始固定revision路径及同host同revision resolve-cache路径可访问，最多2次redirect、3次HTTP请求。失败或未知其他CDN路径停止不重试。Content-Length若存在须精确匹配，若缺失也逐块限额且逐字节核大小和blob。
- 在每次读取64KiB时记预扣、实际应用层接收/落盘字节；进度在每块后持久化，硬中断未持久化一块至多64KiB；传输层物理网络字节 UNKNOWN。下载<=10MiB，输入后再次实际读取比对 SHA。候选输出按256KiB进度记账，未持久化上界为间隔加最大一行8192B；解析、input/candidate/target的重复逻辑文件读取也逐次预扣和完成记录。严格 fresh output、O_EXCL临时文件及非覆盖 link，已有目标不覆盖。总HF输出<=16MiB，包含worker日志/回执及64KiB余量。
- JSON严格UTF-8、拒绝所有对象重复key和非finite；只用 `model.type=BPE` 的 object vocab，要求每个rank是非负非bool整数、0..N-1连续，token字节/排名均唯一、token非空及单token<=4096B，且覆盖全部256个单字节。added_tokens 仅承认唯一、非空、special=true 且 id 在词表外的独立集合；数值与名字冲突停止，不根据名字形状删除词表成员。max vocab 200000、added 1024；不预设实际词表数和特殊token数。若真实JSON结构或某项特殊token语义不满足，范围失败，不能事后放宽来伪造通过。
- rank升序，`base64(bytes)+space+decimal rank+LF` 流式写；完整readback用stdlib检查canonical base64、顺序及全部byte/rank一一映射。安全失败保留真实RUN、错误和非标准候选文件；没有自动重试。
- CPU1、GPU0、Python `/usr/bin/python3 -I -S -B`、CPU0、空CUDA_VISIBLE_DEVICES，512MiB RLIMIT_AS、RSS后验检查、worker120s/outer180s。资源记录 CPU process time、wall、峰值RSS、download/转换/file-read及全worker输出，所有运行计数和 EXIT 实证；网络与资源错误均保留失败。stdio/日志与状态均仅声明一次attempt。解释边界：单次内存上限会因系统/库开销使合规输入失败，固定host路径白名单也可能拒绝真实重定向，均需新scope独立复审，不增加本次预算。

## 审阅和限制

9项合成单测涵盖全部256字节、控制字符/高字节、逆映射、乱序rank、canonical LF/readback、重复JSON/rank、未知码点、special冲突、输出量限制、禁止越域及模拟匿名HTTP失败；在隔离 stdlib Python3.10.12 上 draft 也通过，无真实tokenizer或网络访问。尚未读取真实 JSON，实际 added_tokens 结构、排序是否符合目标、TLS/网络可达与120s资源均未知。独立 reviewer 需审计允许redirect、失败时 byte账与硬中断、目标文件防覆盖、schema验证、stdlib/worker/supervisor身份及不使用用户凭据，提出可推翻的反例；作者回应与独立recheck后由协调者登记 ACTIVE并启动，本草案无授权。

待审 draft SHA `58190a23d8cf27907ea49dd93128a9d50baac0157607a4d410913f5fdb4c09f6`，fingerprint `9760f868a001f10a36779a1be1af8a782d100322388a0c14a03baa98c03fb0da`；producer SHA `12baeddf9200cb8db5f5264b28ef997b044b10dd625109a4412bc16f618b7706`，test SHA `ec077ef327c6138600890d0b42afc41564ad2c9487a3034cf96ad705275b7132`。
