# 主 Agent 修订回应：原生匹配采集与独立文件核验

作者 `/root`。回应真实不同Agent的 `first_review_v1.md`；本文件不预设reviewer同意。最终对象为四个 PART0–3 的 `config/e0_a5_hwg_matched_capture_20261030_part{N}_frozen_v1.json` 及其私有 RUN/VERIFY job，不借用任何旧scope许可。

1. **ACCEPT：真实M64几何。** `matched_rows()` 使用全局flatten索引、起点整除64，整块处于同一τ；每τ1508行，64行会跨52宽的图像行。这是每τ一个raster物理tile，共1344行，既非单一二维行也非完整空间覆盖。固定来源/seed/τ哈希选择，全40层×4步同坐标。fixture验证全部21τ、首尾合法和错误grid拒绝。全M支持规律必须后续独立付费重放。
2. **ACCEPT：真实生产边界。** `MatchedObserver` 继承现有HiF4被动观测，不改原生FFN算术。在原Q1唯一真实调用返回后保存全部V1；1344行U/Z/H/Q1/Q2输入输出、Y/gate/writeback均从原对象切片。fullV1与被动捕获的sample V1当场 `torch.equal`。D2H为同步 `.cpu()`，在生产stream/buffer释放前完成。没有子集GEMM/QDQ、量化delta或模型干预。
3. **ACCEPT：重放边界。** fullM preQ1 rawx未保存；记录真实postQ1输入和其hash，后续原生W1/Q2/W2重放需对每site实际 Z/H/V2/Y逐bit先验收。本scope没有重放/选择/拟合/质量或性能实验，原Q1及其成本不能在未来省掉。
4. **ACCEPT：十状态与视频中性。** 新 `StateTap` 对每个plain/observed分别保存原FP32随机噪声、实际initial BF16噪声、4个原FP32 flow及4个原BF16 post latent，reset专属generator并重新初始化原有双buffer预取标志。全部10个状态和完整81帧视频文件SHA必须相同。各phase检查1600 linear/1600 QDQ、cache miss0；观测全160位置的Q1/Q2/W2和原postwrite断言仍强制。
5. **ACCEPT：预算实算。** fullV1 51,884,851,200B；9个1344×5120 BF16字段各2,202,009,600B；4个1344×13824字段各5,945,425,920B；b2和gate各1,638,400B。每源payload合计95,487,918,080B（约88.93GiB），加2GiB状态/视频/元数据/失败余量为97,635,401,728B（90.93GiB），低于128GiB硬上限。每tensor写前预扣、同步保存，无额外tensor temp副本；写入/文件SHA重读与raw-memory SHA分别计数。每site完成后持久化manifest，失败保留已完成与未知边界。
6. **ACCEPT：并发内存竞态。** `HostReservation` 在同host `/tmp` 统一flock串行准入、最多2模型slot；检查各已准入模型的160GiB预测减实际RSS的未消费预留，再要求新160GiB加128GiB余量。PID/startticks用于移除真正stale slot。GPU原有物理UUID锁、实际空卡检查和持续foreign-process monitor保持；单job1卡、会话≤4，每source4核，wall14400s，无重试。
7. **ACCEPT：单独独立文件验证。** 本次请求还包含4个 `_VERIFY` CPU范围，每个GPU producer真实EXIT0后才启动：1 CPU、0 GPU、7200s、读取≤160GiB、RSS≤512MiB、输出≤16MiB。依实际supervisor receipt验证最终RUN/protocol/spec SHA；逐file单pass同时核完整file及各typed raw tensor SHA，元数据/source/全部160坐标、plain/observed十状态与视频SHA核对，并从fullM V1重新提取21个M64真实字节，独立核与sample V1 raw SHA相同。此验证不重新执行模型，不把producer自己的自报hash当独立核验。新tiny fixture对真实小safetensors的file/raw/header及数据损坏拒绝进行验证。
8. **ACCEPT：来源与研究结论。** source898/96/1040/979均为历史暴露开发图像，seed42。四来源不等于四个blind确认；行/block/τ不作为独立n。本scope完结只确认新真实资产可重放、被动中性及文件保全，H1/H2/H3均留待新E0。原A5 producer未恢复的事实保留，后续适配器不称原A5复现。

实现：`tools/a5_hwg_matched_capture_20261030.py`、`tools/a5_hwg_matched_capture_prepare_20261030.py`，以及对应capture fixtures；M1与capture合计16项小型软件检查通过（其中9项为M1）。这些检查不代表原生模型成功。代码绑定还包含原LightX2V/HiFloat4/native Wan `quant.hif4_backend` 和编译扩展身份；实际import时显式设置三个源码根和原HiF4扩展入口，避免依赖被launcher清空的PYTHONPATH。

最终精确protocol与8个私有job SHA见 `/home/xiang/.config/ffnds/a5_hwg_capture_20261030_v1/BOUND_JOBS.json`（私有文件不入Git）。仍需reviewer核查最终版本和反例、返回真实recheck后，root才会提交本地冻结、登记ACTIVE、逐job启动。
