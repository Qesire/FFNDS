# 最终实现修订自审

作者 `/root`；回应reviewer对V2代码的具体阻断。尚未执行模型；以下为代码/合同修订和小型fixture证据，批准仍由真实reviewer决定。

- **写入持久预留与失败扣账**：Writer现在保留per-file holds，在外部序列化前向逐事件journal写 `FILE_WRITE_BEGIN_UNKNOWN`；finally统计实际partial文件与成功文件，未完成位置/文件保留UNKNOWN。sample/fullV1按准确payload+64KiB header预留；两个opaque MP4在生成前各保留512MiB，进程及其ffmpeg后代均继承512MiB `RLIMIT_FSIZE`。19项软件检查中包含注入save_file失败后partial被扣账的反例。元数据替换同时保留旧文件与tmp的预算，完成/失败都写journal。上限仍128GiB/源；95.49GB原始张量+两视频≤1GiB+状态/元数据/失败余量在预定2GiB余量内。
- **读取预扣与UNKNOWN**：VERIFY的完整tensor、视频及每site固定21块fullV1切片都在读取前按已知完整长度预扣，完成读取字节另行累计；失败保留inflight UNKNOWN，而非把尚未完成文件写成已验证。metadata另有256MiB全文件预扣账，包括authorize源码/registry/review/protocol、parent receipts与各site JSON的SHA和读取。CPU命令入口先设7200s SIGALRM；报告单次≤1MiB、原子tmp，CPU验证全部进程文件4MiB RLIMIT，读取与RSS持续检查。160GiB payload上限包括95.49GB原始tensor和约2.2GB从fullV1重新提取sample行等读入，metadata另计。
- **禁止伪全层和自描述schema通过**：独立验证meta自己的(step,block)必须等于packet，且source/seed/expert/grid/数值族符合固定合同；160个metadata、320个tensor都要求具体site文件名、结果目录内路径、不同路径且不同inode。固定15个sample字段/shape/BF16、固定fullV1[31668,5120]、固定10状态schema、Q1/Q2真实两次调用参数、6个native checks、W2 cache_record_key/shape/dtype/实际缓存hash检查均独立构造。gate仅在序列化时reshape成5120，另保留native shape；原始gate计算完全不改。fixture覆盖重复site、错block、删H、改dtype/shape、native check false及payload改字节。
- **实际数值/导入身份**：启动全量重新读取12项原模型文件SHA和56GB cache SHA，不只看size/hardlink；合计125,270,591,557B，另加代码/小型metadata及42MB Transformers，在128GiB验证读取预算内。Transformers2302个文件逐一按旧原版inventory核SHA，安装限制实际origin的finder。原生HiFloat4、Wan quant、LightX2V/platform及编译扩展actual `__file__`均必须位于本次已绑定源码树，关键backend实际加载后再核查；避免local包目录遮蔽受绑源码。
- **真实资源可用性**：M0历史HiF4文件完整SHA通过；56GB新HF镜像也已完整SHA通过。工作节点B的缺失Transformers已从同一原字节源复制，实际启动前仍逐文件再核。4个source映射到4张空闲A100、三个worker；真正占用由原GPU UUID锁和原生监控决定。每host共用flock、最多2模型slot并扣未消费的160GiB/RSS预测余量，runtime host余量≥128GiB。

下一步：重绑定当前修订后的源码/protocol/8个private job，记录本地不可变commit，reviewer对精确版本recheck。M0/M1与工程fixture仅为准备，E0科学比较/E1原生skip/E2完整候选视频均仍未运行。本scope只是补齐必要的真实全层全M输入及匹配观测，维持原有A5复现缺失与后续A5-inspired adapter边界。
