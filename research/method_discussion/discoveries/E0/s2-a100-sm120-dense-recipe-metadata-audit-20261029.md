# A100 与 SM120 新12 Dense：配方和状态元数据核对

2026-10-29，元数据审计。最直接的差异是：**12/12 来源的 raw FP32 初始噪声与 initial BF16 噪声的记录 payload SHA 均跨两次运行不同**，虽然 seed 都是 42。每个来源后续四个 flow 和四个 latent 的记录 payload SHA 也均不同；各 safetensors 文件 SHA 同样都不同。这是已记录哈希比较，不是新读张量或数值差异计算；无法从 SHA 判断差异幅度或生成差异的原因。

12/12 的 source ID、输入图片 SHA、prompt SHA、seed 和实际 timetable 匹配。两组所有协议声明的四模型 checkpoint SHA、整个 runtime overlay、runtime versions、local transformers 清单一致，实际记录的 TF32/FP32/BF16 reduction 标志一致；overlay 三类 attention 均配置 `torch_sdpa`。共同冻结的 984 项 repo_sources SHA 在每组协议比较中均匹配；SM120 每份清单另有 16 项，包括新的固定层 runtime、producer 及对应依赖/协议，所以不能把两个实际执行包装称为相同源码闭包。上述对比是记录的身份与配置；它不证明逐算子执行或输出相等。

以 source 911 为例，A100 / SM120 的 raw FP32 payload SHA 分别是 `3242709241cda0361478d30e37f6126baa4b7f15bcff14e971393bfb1fcbaec0` / `70696daf319d842a74499fee026669cf2ba086f1dc84126bed19195966766bb7`；初始 BF16 payload SHA 分别是 `c7d5e0585dca82cff5d49427d9a2623516eacb11b9f4c5fc4bd07a17b1e17945` / `e2753e314da102a3b23e1a6f73e362e27207afb43def7d33d92f2806f7481d8f`。机器文件逐源保存两方 GEN、视频、10 状态路径和文件及 payload SHA、所有来源比较、协议与 RUN/EXIT 绑定、八组源码清单比较。

先前的 A100 source 911 连续接触表中约帧 58–68 嘴部事件，与 SM120 V1 部分页面看起来近静止的观察，仅是两次不同输入噪声的 Dense 视觉现象；后者只是帧 42–80 的 AI 缩略图观察。无法把该区别直接归因于 GPU 硬件、attention kernel 或压缩，也不能称其为 neutral 失败。四臂评分应使用同 SM120 范围内实际匹配噪声的新鲜 Dense；跨硬件比较若需因果解释，须另行冻结并证实相同初始噪声及执行边界。

实际 `torch_sdpa` 内部选择了哪些 kernel 未在本次审阅元数据中观测，保持 OPEN；不同初始噪声的生成原因也未由这些记录确定。此审计验证两方 RUN/EXIT 的哈希身份与退出状态，并读取、重新哈希各小 GEN 收据；状态文件与视频本身没有读取、解码或重新计算，payload SHA 沿用已有记录。两方闭合 RUN 是生成证据，不是跨硬件字节中性证明。

[完整机器审计](s2-a100-sm120-dense-recipe-metadata-audit-20261029.json)
