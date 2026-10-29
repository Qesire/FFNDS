# TEAL Llama3 作者校准：静态执行审计

状态：SRC/ALG，REVIEW_REQUESTED；没有执行数据下载、模型、校准、GPU、生成或质量实验。本文不注册运行权限。作者源码固定 fb7373c93ac3594817c9ee64d4e08b47430a1822，核对本地原文 `ffnds-research-sources/p0_teal_full_llm_20261029/{teal,utils}`。

## 实际数据与资源路径

Hugging Face 官方 dataset revision API `https://huggingface.co/api/datasets/tatsu-lab/alpaca/revision/dce01c9b08f87459cf36a430d809084718273017?blobs=true` 返回相同 revision；唯一数据文件 `data/train-00000-of-00001-a09b74b3ef9c3b56.parquet`，24,246,638 bytes，LFS SHA256 `06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073`。这是服务端 metadata，并非本地获取或内容验证。作者 `get_dataset` streaming `skip(0).take(300)`；作者未固定 revision，本项目固定该 revision 是可追溯适配，不能声称与作者当时快照相同。原 `load_dataset` 不传 revision，真正运行须以受审 wrapper 显式传入 revision，并禁用网络后确认实际输入文件 SHA；单独评议并登记数据获取 scope。

`grab_acts.py` 将前300条 text 连 `\n\n`，tokenize 为最大2048、padding=max_length，取前10段；position_ids 却固定 repeat(10)，实际段数不足10会维度不匹配，不能把“最多10段”当成该原脚本能接受任意较少段。记录段数、padding、input IDs SHA、截断情况；模型层调用不传 attention mask。32层输入文件若恰为10×2048×4096 FP16，payload 是 5,368,709,120 B，实际以结果为准。

作者每层 MLP 和 attention 两个模块，各保存 h1/h2，故128次 sort、64个 histogram 文件；MLP h2 为 14336维，其他三个为4096维。`grab_activations` 先 `.detach().squeeze(0).cpu().float()`：采集副本在 CPU FP32，`combine_activations` 的 `torch.cat` 会另建 CPU tensor；每个 key 复制到 CUDA 并 `torch.sort(acts)[0]`，sort 同时生成 values 和 int64 indices；随后 sorted values 回 CPU，`torch.histogram` 在 CPU。最大 MLP h2 的单份 FP32 是 1,174,405,120 B，sort index 是 2,348,810,240 B；CUDA sort 另需实现相关 workspace，同时有模型和其他存活张量，不能以此值作为峰值保证。CPU 亦有原始采集、cat 副本和回拷排序值重叠；按源码建立资源预留并实际监测。每模块保存10,000 bin counts 与 centers，不包含完整尾部分布。

作者 `get_sparse_model` 使用 `device_map="auto"`、FP16、`attn_implementation="flash_attention_2"`；project 核心 requirements 是 torch 2.4.0、transformers 4.44.1、accelerate、sentencepiece、tiktoken、blobfile、safetensors、pandas、datasets、triton、msgspec；eval extra 指定 flash-attn 2.6.1。作者抓取实际上调用 FA2，不能因为 flash-attn 列于 eval extra 而忽略它。仅凭导入不能证明 CUDA/FA2/GPU 兼容。原脚本调用 `layer(...)`，预先计算 `position_embeddings` 但未将其传入该调用；其 Transformers 4.44.1 运行兼容性需真实执行验证，不能静态预判成功。

## 可执行前必须解决的工程边界

1. 新建独立环境而不修改既有转换环境或旧kernel环境；固定 wheels 和 SHA，核对作者依赖、FA2 官方构建与实际 torch/CUDA ABI，避免误称 CPU import 为 GPU 兼容。
2. 按独立审核的 acquisition 获取 parquet 并验证实际 SHA；使用精确 revision 的离线缓存。原脚本未暴露 revision，因此须审核有限 wrapper 对 `datasets.load_dataset` 的参数注入，保留原始算法并输出 dataset、source、token、code、checkpoint 和环境身份。
3. 原作者 `grab_acts.py` 没有固定最大墙钟、硬内存、磁盘、数据输入和审计回执的机制；必须实现 fail-closed wrapper、资源预算与结果检查，再以 exact code SHA 做独立审查和 ACTIVE 登记。不得绕开已有审查以直接调用此脚本。
4. 原 gpt-fast `generate.py` 的速度路径多token prefill 是无mask dense，后续单token decode 为稀疏；HF PPL 的后半 prefill mask 是另一语义。作者 uniform 50% 为全矩阵，FFN-only 是单独范围；显式零阈值 neutral 与作者 p=0 取 histogram 中位数不同。

尚无完整运行 wrapper、隔离环境、数据本地副本、模型运行或全32层验证结果；依赖和源文件身份、执行预算与生产者需进一步完成和审查。竞争预测、neutral 和质量/计时设计见 `p0-teal-llama3-author-full32-execution-design-20261029.md`。

## 隔离环境工程与采集 producer 更新

新独立环境位于 `ffnds-derived/teal_author_calibration_env_20261029/venv`，以两个 `.pth` 只读引用已有转换环境和 kernel 环境。FA2 从作者官方 GitHub release v2.6.1 获得 `flash_attn-2.6.1+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl`，198,483,594 B，实际 SHA256 `4b2423dd55121b991a306677d3409db1fe2f0415adbe72faf573323baee6f9ac`，`uv pip install --no-deps`；cu123 wheel 对 cu121 实际 GPU 运行兼容未验证。原环境 pyarrow 26 与 numpy 1.26 CPU import 冲突，新环境安装 PyPI `pyarrow==17.0.0`（no-deps）后，`CUDA_VISIBLE_DEVICES=''` CPU import torch、transformers、datasets、accelerate、flash_attn、pandas、blobfile、msgspec、safetensors、tiktoken、sentencepiece、pyarrow 成功；分别为 torch 2.4.0/cu121/ABI false、transformers 4.44.1、datasets 2.21.0、accelerate 0.34.2、flash-attn 2.6.1、pandas 2.3.3、blobfile 3.3.0、msgspec 0.22.0、safetensors 0.8.0、tiktoken 0.14.0、sentencepiece 0.2.2、pyarrow 17.0.0、numpy 1.26.4。`.pth` SHA256 `5abaf247b0cd512dbe643390629938782e0d2d60c29fe534490e9a9c22096725`。这个环境只验证导入；FA2、模型和数据加载尚未执行，后续运行前需冻结真实环境内容及资源身份。

采集 producer `tools/e0_teal_alpaca_acquisition.py`、自审 `research/experiment_reviews/E0_TEAL_ALPACA_PARQUET_ACQUISITION_20261029/AUTHOR_SELF_AUDIT_V1.md` 和3个小合成测试已准备；测试通过。已请求独立初审。draft runtime 闭包生成需要大量文件 metadata/hash 读取，为避免与当前 checkpoint 转换争用存储 I/O 已中断，未生成或冻结 protocol。须在转换完成后重新生成，reviewer 回应并完成精确复核后由 root 登记 ACTIVE。单文件获取只含进程内 SHA 和真实退出，独立二次验字节须另审另登记。

2026-10-30 补充：官方固定 commit 的 `gpt-fast/tp.py`、`gpt-fast/distribution.py` 原本不在保存的源码子集；从作者原始固定 commit URL 获取至单独 `ffnds-research-sources/p0_teal_gptfast_support_20261030`，对应 SHA256 `d1d88eb60d2b985341ea64bf92c55a4d689b379decbf4baaedc6d787ddfc3be4`、`94100f32155358451b1b121b0340f94b5d19c271dd8e8e41e038d3bcc7c89a16`，进一步导入闭包尚未验证。Alpaca 获取 draft 的环境指纹改为 METADATA、RECORD 和 Hub 关键入口，并明确非全包/ELF身份；单scope预算包含两次全文件 SHA 读取和进程 EXIT，需独立评议同进程双读的充分性。`tools/e0_teal_author_llm_scope_planner.py` 构造静态有界配置，禁止执行；不是已经完成的校准或生成launcher。
