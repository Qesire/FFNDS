# A0 六视频质量 v2：原评价环境隔离修复

身份：SRC／小 JSON 核验／tiny CPU controls；本次未加载模型、未执行 official compute、未调用 GPU。v2 仍为 draft，真实 imports-only、官方分数和 RAFT 尚待协调者冻结登记后执行。

v1 的首个 `imaging_quality` 模块在 `pyiqa → imgaug → np.sctypes` 导入处失败。实际 RUN `dfd8f972cb4c52b1ac2708d1f6c59cb3a88b4428dd367abaf491ee67854ae4a2`、actual exit1、outer418.662s；`official_dimension_calls/new_video_dimension_results/RAFT_forwards` 的 predebited、invoked、completed 均为0。旧 scope、协议 `bf571e1bdb319bba69a0126d7decdf05af0cfe8ba3d77a45cade067fdd6b0a2a`、HF 输出及 Git9782dc2 保持失败身份。

## 根因与真实环境

v1 把 metric site `append` 到已初始化的项目 Python 搜索路径后面，NumPy 先从项目 `.venv` 的2.4.6导入。实际 legacy metric site 已有 NumPy1.26.4、SciPy1.13.1、pyiqa0.1.10、imgaug0.4.0；其中 Torch2.10.0+cu128／torchvision0.25.0+cu128 是指向项目原运行时的符号链接。

原 v8 的真实 `IMPORT_RESULT.json` 记录 NumPy 来自 legacy metric site、版本1.26.4；原流程用 metric venv 的 Python `-I -S` 后显式导入 site 和 setuptools80.9.0 overlay。旧 Dense **评分** RUN／LAUNCH_EXIT 没有记录模块实际 origin／NumPy版本；v8 imports receipt 不能替代这一缺失字段。因此 v2 将六条现有视频全部在同一隔离进程重新评分，旧 Dense 分数只列 historical same-byte sanity delta，候选分母取本次新 Dense 分数。

## 新的精确软件范围

`E0_COMPACT6016_CONSUMER00_12_QUALITY_ISOLATED_V2_20261028`；source0/12、seed42、Dense/energy94/random94 原六条视频，0新生成、N=2旧暴露图像簇。原八个作者指标、采样、RAFT40对和五个固定8pair窗口保持原数学身份。两 I2V 组各3视频，其余六维各6视频，共42逐视频结果；8次 official compute，RAFT240次forward／4800iterations。原6条 CORE 报告绑定复用，0重新计算；旧 Dense score复用数为0。

主进程必须由 `/home/xiang/.cache/huggingface/derived/ffnds/E0_official_metric_environment_v2/venv/bin/python -I -S` 启动。仅提供 original source view → overlay → legacy site → repo tools → interpreter stdlib 路径；项目 site 不作一般 fallback。Torch 前先核 NumPy／pyiqa／Torch spec，实际先导入并检查 NumPy1.26.4，再导入 Torch。native RUN 记录17个真实模块的 origin／distribution version／入口文件 SHA，以及完整119项环境 metadata closure。

现有 byte-only CPU 证明外新增 `import-preflight`：CUDA全部隐藏、8个原 official 模块真实导入，只检查 callable/signature，不调用 compute；复用冻结 v8 的 checkpoint／Parameter／Buffer／NN forward／CUDA／网络／越界写入 guards。仅原文已绑定的8个无参数 activation wrappers 与1个无参数 MaxPool wrapper 是允许的导入副作用，另计数。预算 CPU2／300s／outer360／新HF32MiB／0GPU。native 在 Torch 前分别要求 `scope.cpu_preflight` 与 `scope.cpu_import_preflight`，两者均须绑定本次 protocol SHA、RUN bytes/SHA 和实际 exit0 receipt，拒绝任何旧 v1 或旧 v8 receipt 代用。

协调者冻结协议后，CPU 两个独立 durable worker 分别使用：

```text
<metric-python> -I -S <repo>/tools/e0_compact6016_consumer_quality.py preflight --protocol <new-frozen-protocol> --hf-root <HF> --out <fresh-byte-proof.json>
<metric-python> -I -S <repo>/tools/e0_compact6016_consumer_quality.py import-preflight --protocol <same-new-frozen-protocol> --hf-root <HF> --out <fresh-import-proof-directory>
```

两个 worker 均设置 `CUDA_VISIBLE_DEVICES=''`；第二个的 RUN 为 `<fresh-import-proof-directory>/RUN.json`。两个实际退出收据完成并登记后，native worker 才允许在获分配的单卡上用同一解释器和 `-I -S` 进入 `run`。单作业GPU1／本会话最多2卡／CPU2／1800s／outer2100／新HF256MiB，0自动重试。其它指标内部模型调用次数保持 UNKNOWN。

## 本次实际软件检查

- 37项 tiny CPU controls：exit0／0.344s；覆盖六条新分母与历史分数隔离、240次 fake RAFT 实调用、路径优先级、前置 numeric import、wrong spec／origin／version／bytes、两CPU证明的 stale protocol／wrong RUN／非零NN或CUDA／不全8导入／预算等语义负例。
- py_compile、draft inspect：exit0。17个 repo source binding、17个真实模块入口绑定、原 v8 import receipt 与失败 v1 RUN／actualexit1 身份核验通过；无 NumPy／Torch 导入。
- 使用真实 metric Python `-I -S` 的 stdlib-only spec 检查 exit0：NumPy1.26.4／pyiqa0.1.10来自旧site，Torch来自原项目symlink；检查后 NumPy／Torch均不在 `sys.modules`。此项是路径证据，不冒称真实八模块 imports-only 已完成。

交审版本：driver `0e047e79cc77b91fa0bb45b88c065867f181315c92558ab56957c5f9920c187f`；tests `064b9d2f374cda9fe3a86796b2bd4766a54d23a2e93ab3f64fddb71722c5ca21`；v2 draft `f63296d1902fde0e9647437d6558bc5f0ddfea037a19458f478e52fb889059e4`。未改全局 `.venv`、八个作者评分源码、任何 BH／B-F 冻结源、ACTIVE 或旧实验结果。
