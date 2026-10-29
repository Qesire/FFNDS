# S2四角reader启动修复V2：冻结前审阅

2026-10-29 · coordinator ROOT。原V1两个CPU预检均actual exit1，run_sha=null，writer首次写CPU/RUN.json.tmp前未创建其parent；0 Torch/NumPy、0 GPU/模型/reader GEMM。两份实际exit和完整stderr按原HF身份保存；不是数值候选失败，没有覆盖或重跑其旧scope。

V2 wrapper独立namespace更换BASE/output，只在未变CPU预检前创建受限的新输出parent，draft/CPU预检/原生run均校验旧失败proof绑定；原V1数值核心、数学helper、固定两source/52xy/160site/两个候选因子、资源与精确计算账不改。两项软件控制actual exit0：真实缺parent→原writer成功且拒绝越界/覆盖，namespace身份分离且原数值算法/预算相同。

真实独立Agent `/root/s2_theory_design_independent`静态读wrapper及测试并回复未发现阻断问题，提出CPU/run前重查旧exit/stderr SHA；已加入verify_repair。其原理论/频点修正审阅文档不修改，V1绑定保持。新V2各part单次固定范围，原CPU失败扣账0模型；新CPU proof实际exit0后再启动原生。失败保存、无自动重试。原生结果仍需新RUN/EXIT，软件修复不作科学成功。
