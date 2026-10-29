# TEAL 原作者32层激活与直方图校准：独立初审 REVISE

2026-10-30，评审 `/root/sm120_full_quality_build_terra`；作者 `/root/teal_full_llm_calibration_build_terra`。审阅 producer SHA `35366ec4c29c47cc7777a757419fa1d9a2ca1d69bd1841cb738e49137f75ea22`，protocol SHA `5f9360ca6c349e3acd64da799d0203444e2bbeea1fbb26ff7af00db8ddbdc1ae`，fingerprint `3fccb8aa6191c7493c967ada0b8dcaab80b3ffe555bf2859bdd84559fb5fcc59`，私有 GPU_JOB SHA `0265ff04e8a35c090a2cac4068d2b5f1be378bb0d50ed3d202ad5382260a0fb1`、自审及作者 fb7373c 源码。未执行模型、GPU、校准或数据记录。初审不授权 ACTIVE。

原作者 `grab_acts.py` 把 Alpaca 前300条文本拼接并取前10×2048，按层保存输入、执行原层、分别为 MLP/attention 的 h1/h2 建直方图。`ActivationModule.find_histogram` 对每个激活作GPU sort，再 CPU histogram；最终64文件、128 sort 和128 histogram 的成功路径与拟议计数对应。作者 `SparseModelMixin.build_sparse_layers` 将模型参数 `requires_grad=False`，故不能仅因外层未包 `no_grad` 断言必然持有跨层训练图。固定公开数据 revision 是项目适配，不假称作者历史数据。flash-attn 2.6.1/cu123 wheel 与 torch2.4/cu121、A100的真实兼容尚待实际运行；32 GiB RSS/GPU采样是可失败的范围，不是无超峰值保证。

**可推翻实验的反例：** 同一300条文本若 tokenizer 对象配置、special token 或 padding 发生差异，两次 token 化产生不同前10段；现有 launcher 只比对文本 SHA，仍会把预检 token SHA错误标为作者实际输入。另一个反例是第32个 `act_31.pt` 已成功保存，而第32层 forward 后失败：32次保存发生在调用前，不能当32次实际成功 layer forward。

需修订并重新核定：

1. **实际作者输入 identity。** 代理作者真正使用的 `get_tokenizer` 返回对象，记录实际 token 化结果的前10×2048 IDs、attention mask SHA、实际长度及 padding；在首层执行前与预检值精确相等，不相等即失败。预扣/完成两次数据与 tokenize，保持作者真实 tokenizer 参数与前300行顺序；不声称 preflight 对象就是 author 对象。
2. **真实 layer forward。** 32层各加 pre/post hooks，持久预扣、完成次数、层号及输入/输出形状；最后一层保存不等于 forward 完成。任何额外有限性或统计计算须预先冻结实际 observer 的元素数、次数与时间/显存成本，不能把抽样称全量。sort/hist/save 当前在调用前递增，失败时应称 attempts/predebit，完成计数须在调用成功后记录或明确账义。
3. **实际代码与运行链。** 校验 supervisor 的 `protocol_path`、`hf_root` 与已绑定路径，确定 exact argv/私有 profile 和 actual EXIT。`utils.utils`、`utils.data`、`teal.model`、`teal.mlp`、`teal.self_attn`、`utils.eval_ppl` 等动态导入必须在 import 后核实际模块文件与绑定 SHA，防名称冲突/环境 shadow；只比较列表路径而不比实际已载模块不能证明闭包。实际13文件 acquisition 和验证 RUN/EXIT、Alpaca获取EXIT以及各输入最终 SHA继续绑定。
4. **实际资源隔离。** 当前 monitor 仅按 UUID读取全卡显存，没有核独占 GPU 的进程；长达16GB输入哈希发生在本模型进程建立CUDA上下文之前，仅启动时检查空卡不能防止期间 foreign compute 进入。使用既有单卡 UUID 锁与本进程预占、持续外来进程检查和失败监控；核模型所有参数及实际 `device_map` 均在指定 GPU，而不是每层 `next(parameters())` 的首个参数。资源峰值采样间隔/子进程RSS不可观测要继续标 UNKNOWN，不要求把32 GiB声称硬限。
5. **CDF 可用性和失败账。** 64个真实 hist 需 key/shape/finite 外检查 count非负且总量正、centers有序；逐文件哈希完整 predebit/完成与限额；任何额外归约计入成本。保留日志/partial 和真实 supervisor EXIT；后续生成scope自行对64文件再次预扣重哈希，不把静态计划冒充闭合。

作者逐项回应、修改并给出确切 producer/test/protocol/spec 后，本 reviewer 对确切版本 recheck；任何批准仅限本次采集，不表示 LLM 质量、性能或 FFN-only 稀疏复现。
