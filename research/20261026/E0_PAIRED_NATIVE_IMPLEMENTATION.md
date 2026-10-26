# E0 · 原生双轨采集实现：软件验证与原生执行边界

**记录层级：本页描述实现与初始软件验证，并非实时执行状态。之后用户另行授权了[固定paired-v2协议](../../config/e0_paired_full_depth_20261026_v2.json)；实际启动/结果以[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)与原生RUN为准。本实现阶段零GPU/零原生观测是历史统计值。**

本阶段把前轮静态账本接入实际 Wan 调用路径：[执行入口](../../tools/run_e0_paired_full_depth.py)、[四步双轨控制器](../../tools/e0_paired_wan_controller.py)、[四臂FFN观测](../../tools/e0_paired_ffn_observer.py)和[当前block驻留](../../tools/e0_paired_weight_residency.py)均已实现。**实现与CPU软件测试完成，不是原生数值、显存或视频验收。此处零GPU计数仅针对初始实现阶段；后续固定v2已单独登记并启动，实时结果必须以活动状态与原生RUN为准。P23/P24只有获得真实合格数据后才改变证据身份。**

## 这次实现解决的问题

两个真实scheduler分别维护latent、step和flow；同一GPU block buffer依次执行两轨，四函数与四次原生写回在当前block内立即归约，然后仅swap一次。保留原pre/head上卡和下卡顺序，显式设置并恢复HiF4的expert/step/block context；原非FFN算子仍由原runtime执行。FFN求值复写原Q1/W1/caller GELU/Q2/W2顺序，candidate使用原冻结BF16参数包；不复用旧hook不可重入的pending状态。

旧全驻留策略会永久保留所有影子dense FFN，新策略仅让non-FFN常驻，dense FFN限制在当前block、两轨和observer消费结束前保留。释放前显式fence，成功和异常都恢复hook/调度器；上传pre/head中途失败也走同步及卸载。资源能否容纳仍待实测，不把约40.70GiB已知权重payload当峰值上界。

有一个影响失败归因的有限精度反例：BF16、g=1、dense输出0、candidate输出2^-9时，residual基底r=1会使两次写回都舍入到1，误差0；r=0时误差为2^-9。**相同raw FFN差也可因残差基底改变而有不同写回差**，不必先发生FFN输入失配。这是常数软件夹具/算术反例，不是Wan已测机制；因此新工具记录真实四writeback而非用g×FFN误差推算。

完整M标量另按预登记的3个时间段×3个竖向区域分层，区域由实际21×29×52栅格确定，不冒称主体/背景分割；仅保存固定8行完整宽度BF16向量作为佐证。native-continuous检查切取实际full-M native输出32行，对比同一已部署BF16系数升FP32、去掉Q2和中间BF16 cast的函数；它不测原拟合FP32系数丢失，因为后者未保存在bank。

## 固定模型范围

初始[未执行v1草案](../../config/e0_paired_full_depth_20261026.json)规定以下3个串行fresh process；当前有界执行依据是后续独立冻结的[paired-v2协议](../../config/e0_paired_full_depth_20261026_v2.json)，保留了三pair与单张空闲A100的核心范围：

| 顺序 | 原来源/seed | 两条完整四步轨迹 | 控制目的 |
| --- | --- | --- | --- |
| 1 | 0/42 | dense、dense | 每block两轨一致；再核对原dense四latent和视频hash |
| 2 | 0/42 | dense、affine54 | 熊视频的真实student-prefix与同输入部署数值 |
| 3 | 12/54 | dense、affine54 | 小镇视频的异质时序表现 |

总计6条完整路径/视频、480个配对site；每site额外两个shadow FFN和两次32行FP32比较。预计每pair3840次实际QDQ、4160次native目标linear GEMM，另960次小M连续诊断GEMM。runtime3200 QDQ/GEMM与observer计数含dense重复部分，不能直接相加。预加载、真实归约、同步、hash、双VAE解码及峰值单列；每pair两视频共享一次成本，按协议SHA/pair去重，不当candidate部署速度。

新输出上限0.5GiB，保留80GiB磁盘余量；不训练、拟合、分解或调用官方评分。两轨都必须与各自原RUN四latent/视频SHA对齐，不因D/D内部一致便宣称中性。任何失败扣账、保留、不重试且阻止后续pair；输出目录和私有executor配置绑定协议，不能换目录或节点重置额度。每个实际clip无论native通过或失败都产生全帧指标；缺/坏视频显式标记，外部质量仍NOT_RUN。

## 已完成验证与剩余边界

初始实现阶段的17项新CPU软件测试覆盖四臂主/影子隔离、真实写回反例、完整160处调度顺序、每block一次swap、native dtype哈希、上卡/中途失败恢复、缓存上限/释放fence、有限次数与失败clip报告。夹具不加载Wan/HiF4或GPU，保持SYNTHETIC。完整静态preflight验证了16份driver/23份native源码和80个真实bank文件字节；这份**初始preflight快照**当时报告`model_scope_authorized=false, GPU_initialized=false`；它不用于判断后续v2授权或执行状态。私有worker的当时只读检查确认仓库/资产可见且有空闲A100，并未预占卡。

独立审阅：[控制器生命周期](../method_discussion/discoveries/E0/independent-full-depth-native-interface-20261026_v3_review.md)、[四臂语义与成本](../method_discussion/discoveries/E0/independent-paired-observer-runtime-audit-20261026_v4.md)、[执行入口最终审查](../method_discussion/discoveries/E0/independent-paired-driver-final-audit-20261026_v4.md)。审查发现并促成修复实际config键名、上卡失败清理、同步异常跳过hook恢复、换目录重置次数和失败视频跳过评估等问题；独立审查仍是读源，不是原生实测。

HiF4量化实现与原.so通过原cache manifest的identity.source_sha256及HistoricalSources逐项绑定；完整cache SHA仍由verified_restored_assets在执行前核验。旧模型/协议/原视频/原RUN不改，24个既有scope保留。初始实现阶段尚未获得本次固定范围的授权；之后另一指令已授权v2，且活动状态记录了实际启动。不得再把初始阶段的授权缺口写作当前待办，也不能据此扩展其它实验额度。

**执行协议边界**：[v2严格身份/产物清单检查](E0_PAIRED_NATIVE_STRICT_EVIDENCE.md)已另冻协议，原v1草案未执行；本页保存初始软件实现与17项测试的证据范围。v2授权、正在运行或运行结束后的实际产物都需以[活动状态](../ACTIVE_RESEARCH.json)与真实回执核查，不能用本页固定快照替代。
