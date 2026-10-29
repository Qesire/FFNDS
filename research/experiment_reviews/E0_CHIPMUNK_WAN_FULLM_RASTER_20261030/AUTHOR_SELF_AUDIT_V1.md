# Chipmunk 真实 full-M raster 被动采集：作者自审

当前问题是：相同expert两步之间，真实连续 raster 组上的付费 proxy 选择及其实际动作有何误差、代价？此前 source898 的1092行非raster首128组与合成重复行不能判断这一问题。本范围采集两个已暴露开发来源898/96、seed42、原始 BF16、block20/step0–3；不是heldout、也不把block20作为后续160位置的永久边界。每source一张卡单独scope，最多两路全模型并发，全节点最多4卡。

这两个来源现有 spatial52 Dense 位于不同历史数值/设备链，不能预先声称与SM120逐字节相同；因此每source新生成两条实际完整四步×40层视频。首条无额外operand tap，第二条仅被动拷贝，二者同一SM120模型、checkpoint/seed/recipe；要求 raw noise、initial noise、4 flow、4 post各完整state SHA/raw SHA/shape/dtype和输出视频 SHA及大小全部相同。差异立即失败并保存，不能以近似或历史A100结果替代neutral。

第二条在原native W1输入捕获V1、原W2输入捕获 post-GELU H、原W2返回捕获 Y，实际gate在原infer_ffn处捕获；张量复制同步到CPU，不新增GEMM、不重排、不更改计算/权重。所有capture绑定同一个FFN对象和gate/写回对象、step对应high 0/1 low 2/3、block20和实际grid `[21,29,52]`。行号按 native `row=tau*29*52+y*52+x` 编码于typed header；不使用空间重排或采样。每step V1/H/Y各324280320/875556864/324280320 B，gate 10240 B；每source四步原始payload 6096510976 B。每对象一个safetensors文件，1GiB硬单文件限制、每scope新增8GiB软/增量总额；两视频、20state、16个tensor文件、4header及日志都计入总额。实际序列化header大小、独立文件SHA和原始tensor SHA在运行时记录。

预扣捕获16次D2H、16次raw SHA、16次文件写及16次文件全文hash（每次按payload+1MiB预扣上限）、20状态写与20次文件hash、4 grid/header；实际正常路径每项需全部完成。视频、模型原始40层4step、状态D2H与prefetch保持原负担；observer的同步与磁盘读写 wall 均含在生成时间中，非性能结论。文件、tensor hash及额外的元数据读分别计算；模型mmap物理读、包内库隐式读及启动前小收据/元数据读取 UNKNOWN；总输出为采样监控加增量记账，并非跨所有库调用的硬实时文件系统quota。

每源须预先审核并登记CPU preflight、GPU job；仅预检实际EXIT0后可启动GPU。GPU actual EXIT0后另登记独立CPU VERIFY，对20state、2视频、16 tensor、4 header及2 journal合计44个确定文件逐项预扣实际字节、读取并核文件SHA；tensor同一遍流式核原始数据SHA、BF16 dtype、shape及safetensors data offsets，验证 header与真实来源、expert、step/raster契约。VERIFY实际EXIT0后才 closed。若中断，在途读/拷贝保持UNKNOWN，保留partial，不重试。脚本和两个真实父 RUN/EXIT、源库/环境profile SHA、资源、独立Agent结论和三个私有job均按每part精确绑定；声明不会运行本体、GPU或读取大tensor。

预先反例：捕获误挂其他FFN、对象复用/被异步覆盖、错行/错expert、保存过程改变生成数值、H文件超过1GiB或整scope超8GiB、验证只检查文件名而无真实dtype/SHA。具体捕获事务、同步、typed-header、同设备neutral、GPU/CPU资源及独立逐文件校验应由不同Agent核对；任何一项失败不能报告采集成功。
