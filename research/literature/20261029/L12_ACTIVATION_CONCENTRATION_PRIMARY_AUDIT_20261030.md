# L12 原文复核与 Wan W2 判别

2026-10-30，`/root`；LIT 读至 [v2 §2–6](https://arxiv.org/html/2606.00567v2)，未核作者 profiler/simulator 源码。论文以全 token 后激活阈值定义冷列；性能来自 16×16 systolic＋Ramulator/GDDR6 模拟。FFN-Reuse 的旧输出复用与本文质量测试的冷列清零需分别识别；都不同于本仓实际 TEAL 逐元素 mask。其模拟收益不作为 A100/SM120 Wan 原生速度证据。

ALG 复核：原文 §4.2/4.5 同时声称热列集合增加与相邻集合 Jaccard=1。若集合严格增加，`A⊊B`，则 `J(A,B)=|A|/|B|<1`；二者不能同时精确成立，可能存在四舍五入或实现定义差异，需作者源码才能解释。因此不把这段稳定性解释作为免费静态布局的证明。

本项目判别由真实操作数决定：同一状态的采样行集合 S 是全行集合 R 的子集，则 `cold(R)⊆cold(S)`。在 S 上已热的列，在 R 上必热；这给无需独立采样假设的上界。正上界不证明全 M 真冷；不连续的256行不提供真实M16 tile。实际生产者、来源与审阅见 [TEAL设计](../../method_discussion/discoveries/E0/p0-teal-wan-fullm-sparse-w2-physical-reachability-20261030.md)。图锚为 F09 与 G14 T05/T07/T13；真实结论另依 ACTIVE、RUN/EXIT，不由文献批准。
