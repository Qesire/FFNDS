# 新12 Dense启动V2：修正已被替换的全局权限字段读取

2026-10-29，ROOT。V1两路CPU实际EXIT1，错误为`Required current two-GPU authority differs`，RUN未创建，0模型/0GPU/0图像及状态payload读取。V1把上游已替换为dynamic、值为null的历史全局字段硬判等于2；这是启动守卫错误，并非数据或模型失败。原V1源码、协议、EXIT、stderr不更改。

V2在独立private namespace复用V1全部生成、Dense neutral、数值与预算，仅修订authorize、版本ID/输出目录并绑定V1失败。核当前动态资源政策和持续授权，本固定pair仍仅用两个实际asset-ready卡位；ACTIVE分配总数需≤2。此pair的有限选择不重设项目全局GPU上限。新协议附两条真实EXIT1/run_sha=null/timeout=false、原stderr摘要与零模型事实，预检及原生启动均重新核验，没有自动重跑旧scope。

11项新旧软件控制通过，涵盖动态null政策合法、额外分配第三卡被当前pair拒绝、未授权继续被拒绝、版本输出独立，以及原12源/neutral/预算。每part仍7条完整Dense视频（6新来源+1旧source中性控制）、28flows、1120FFN、70状态；无候选/factor/拟合/FFN行张量/官方评分。单GPU/CPU4/40GiB/host128GiB/2400秒/外层2700秒/HF4GiB，原CPU4/600秒/外层900秒。先CPU gate真实通过，等当前V3重放退出释放GPU后再启动。
