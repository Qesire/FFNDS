# TEAL Llama3 校准 V2 新范围：独立初审 REVISE

2026-10-30，独立评审 `/root/sm120_full_quality_build_terra`，作者 `/root/teal_full_llm_calibration_build_terra`。精确审阅 producer `28c4e7b59982d31a3b2c932850fdd2f60e994acc2e9fc276345a1bcbd53317df`、tests `3bd88819d51481215242d5961e55b13c562136e16a30ee4abe9bed2b1aa551ba`、protocol `a9ba8c240a6e955b44dca1592b76d831c445813a73d77ec423c8828dc724644a`、私有 spec `367cd353c316e92e82ba8ea11b01d1b3f41e83324cc4972b4692e574fd724397`，fingerprint `7ae6c40d0f99abd1efd84bcccb17e19f6840aaa7729b849a9f19758476321b7c`。这是新范围；旧 V5 校准实际失败及旧复核不授权此范围。

代码真实将作者和 FA2 模块前置到模型 hash 之前；原 `utils.data` 在前置导入时已绑定 `load_dataset`，修订显式替换 `author_data.load_dataset`，同时保留 streaming、skip/take、源前300行和两次 tokenizer 比较。六项非模型测试通过，其中目标 venv 的无 GPU 子进程实际模拟 Transformers 的 FA2 可用分支并导入六原模块、FA2 及 einops。模型/数据运行、CUDA编译和 GPU kernel 没有执行。作者层、排序、直方图、实际 token、finite observer 和来源资源账仍为明确有限范围。

**可推翻的反例：** 假设父环境带 `PYTHONPATH` 指向同名 `torch` 或 `transformers` 模块：实际 supervisor argv 使用 `python -B` 而非测试的 `python -I -B`；当前只核分发 metadata 和部分作者模块文件，故实际导入可能与被测环境不同或加载未绑定代码。要求把实际 job 和 producer 的精确 argv 改为隔离 `-I -B` 并核 `sys.flags.isolated`，对非隔离拒绝及 exact spec 加合成测试；如必须非隔离，应提供可审阅的等效环境清理/身份约束。另要求 protocol 明确绑定旧 V1 的实际失败 RUN 和 EXIT，核实真实失败身份及零模型调用，使新有限范围和历史失败不可混淆。

固定 CPU 导入只证明分支可导入，不证明真实 A100 的 FA2 二进制兼容；32 GiB host 上限、全卡 GPU/RSS/磁盘五秒监控是采样限额，短时峰值及非合作输入写者仍不能排除；本 scope 未加入 FA2 tiny GPU smoke，首个实际作者 forward 仍可失败并保留账，不自动重试。作者回应修订并提交 exact source/test/protocol/spec 后复核；初审 REVISE 不登记或启动。
