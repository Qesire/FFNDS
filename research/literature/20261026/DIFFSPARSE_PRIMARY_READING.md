# DiffSparse · ICLR 2026原论文公式与训练/实验审计

**官方原论文**：https://proceedings.iclr.cc/paper_files/paper/2026/file/5e66b9f710a8e38351da690f3806cba9-Paper-Conference.pdf ，共19页。已查阅论文§3.2方法PDF页4–6、§4.1实验PDF页6–7、附录A.5.2与A.5.3。该方法是token feature caching加**离线训练的跨step/layer稀疏配置**，不是直接FFN神经元/硬件N:M算法。

- Token selector在每层聚合self-/cross-attention影响、cache reuse frequency等当前可观测分数，可用空间bonus，排序并保留K个需要重新计算的token。作者主张预算学习器与具体token ranking启发式基本正交。
- 离散预算候选 S={0,.25,.5,.75,1}。成本矩阵大小为(T×L)×|S|，通过训练期DP递推F(i,r)=min_s[F(i−1,r−s)+C(i,s)]选择满足全局保留率的配置。实际推理只用预优化的配置；不会在线跑DP。
- 离散mask通过STE参与训练；完整生成teacher/student最终图像使用LPIPS作为监督，而非所有中间FFN张量的SSE。先预热full-step与layer两部分，再把full-step prior融入layer配置，联合优化。
- 原§4.1列模型：PixArt-alpha=20 steps；DiT-XL/2=50；FLUX.1-schnell=4；**Wan2.1-1.3B=25 flow-matching steps**。这是对先前文献地图中“Wan few-step”泛化说法的必要纠正。
- Wan训练在WebVid-10M约10000 caption条件上进行；每实验8张AMD MI250 80GB、约4–10h。Wan视频评价为950 prompts×5=4750视频，每个2秒8fps、256×256，在VBench 16个维度评价。PixArt-20步减54%计算的摘要数据不能用作Wan4步结果。
- 本轮未审计公开作者实现commit、显存wall拆分与实际Wan2.2四步适配，不能将source级的完整质量结论移植。

该论文的真正启示是通过终端生成观察**训练计算分配策略**；这不是证明FFN hidden里存在哪种天然的可执行冗余。与FFNDS的“功能结构发现”是互补但不等价的研究目标。
