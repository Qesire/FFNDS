# E0 全 M 实现第二轮独立预审

评议者 `/root/sync_abba_actual_review_terra`；状态 **REVISION_REQUESTED**。本次只读静态源码及已有纯软件 fixture 设计，不读实际采集 tensor、不启动科学任务，也不对未来尚未物化的协议、资产或机器 job 签发运行批准。

源码将两个 expert 各40个相邻同 expert pair 展开，支持的 K5120、原始 raster 与 shuffle 后恢复坐标、全 M 的付费 FP32 W2 诊断、真实采样 Q1/Z/H/Q2/Y 逐字节门、current gate 和 BF16 sample postwrite 的证据身份总体有明确实现。static 使用两来源采样上的 signed Gram、每来源归一化和固定80组 forward greedy；它是一种已规定的有竞争力可行基线，不是全局最优支持。全 M 实际原生 Y 未采集，sample 匹配即使通过也不能变成整矩阵逐字节证明；当较大 M 的原生 GEMM kernel 舍入使 sample 不同，须保持失败并另行审定等价重放或采集，而非自动放宽。

## 阻断与必须修订

1. **已验证 parent 到实际科学输入的传递链缺口。** `validate_parent` 核 producer/VERIFY 的 RUN、EXIT 及状态后只检查新协议中 `source['metadata']` 长度；后续的 `bound_json` 用的是新协议自带的 metadata SHA，并未核它等于 producer RUN 已绑定的 observed capture manifest 中对应 site 的 metadata 引用。反例：独立 VERIFY 已结束后，同一个合法 site 路径的 metadata 和 payload 被替换成同形状、同 source/step 的新文件，而待冻结 E0 协议引用新 SHA；当前的路径、source、schema、签名检查仍可通过，但新内容不是 VERIFY 核验对象。须按已绑定 producer RUN 读取并复核其 capture manifest，将全部160个 step/block 对应的 metadata 完整 ref 与 E0 输入逐项比较；随后依被绑定 metadata 中 payload/full-V1 的 ref 哈希读取。需测试同路径、同 schema、不同 SHA 的替换被拒，并检查 producer RUN 的 id、两 phase 及 manifest 本身的关联。static 和 discovery 共享此缺口。
2. **在执行前验证真实原生代码来源。** discovery 现在直到80对全部完成后才 `validate_import_origins(p)`；此前已经导入和执行 HiF4 QDQ。异常路径可先执行未绑定代码、消耗完整预算，最后才被拒。须在首次科学重放前显式导入真正使用的 HiF4 backend、核验包/namespace/原生扩展来源及文件 SHA，再保留结束时复核；同时限制 import path、为错误目录 shadow 编写不占 GPU 的拒绝 fixture。需把上述代码及其导入闭包绑定具体协议和独立 review。
3. **完整成本和内存生命周期。** ledger 对主 GEMM 和57臂投影逐项计数，但调用外的 `v1.cuda()`、sample/坐标/gate 的传输、proxy 的 full-M 差/除/绝对值/乘法、mean-score 的 repeat 和结果参数 D2H 尚未明确独立收费。`signed_output_metrics` 在每臂还创建 `weight2.float()`，并多次对 full-M 输出进行 FP64 norm/dot；`support_summary` 对完整 mask 做邻居 Jaccard、转换和原物理布局 occupancy，逐 token K1 argsort 含巨大 int64 临时。其开销、CPU/GPU 同步、内存复制和输出不可由4560次 FP32 W2 数量代表。冻结逐类次数、shape、dtype、字节及明确计时归属；按同时存活的两状态 Z/H/Q2/V2、full-M scores、mask/indices、临时转换与 matmul workspace 推导并验证76 GiB GPU 及96 GiB host 的峰值上界，另将整个实际 wall 和输出/读取费用记账。资源不够时先修预算/算法并重新审，不在运行中热改。
4. **校准目标的有限精度。** signed Gram 和 target dot 是 FP32 GEMM，随后才转 FP64；forward greedy 对这个已舍入的二次式作精确边际选择，不是对真实实数目标的精确求解。在强抵消或近相等边际时，FP32 Gram/相关项可改变排序。冻结 backend、dtype 和 tie 规则，保存所选支持的真实采样向量直接重构损失、Gram 二次式预测值和闭合残差；增加 signed 近抵消/接近 tie 的 fixture，报告不稳定性，避免称全局 oracle 或精确实数最优。

两个校准源自身结果和两个历史暴露开发源须分开报告；source/prompt 是独立单位，80层对及31668行用于来源内结构统计，不能充作独立样本。四个开发来源不足以证明泛化。真正的 native W1/W2 跳算、全FFN性能和自由视频质量分别需后续独立审查和实际运行。本次全部候选投影均是付费的 zero-masked Dense FP32 诊断；量化、采样、打包、缓存与预处理成本不产生真实跳算结论。

最终复审还须以真实采集与条件 VERIFY 的 actual EXIT0、两校准 artifact actual EXIT0、精确 frozen code/source/SHA、完整资源与私有 job 作为输入；开发 source 的科学读取不能先于静态支持冻结。新 mean-input-per-M64 proxy 是独立命名新动作，不是原逐 token A5，也不能把不提供原参数 producer 的历史方法当作已复现。
