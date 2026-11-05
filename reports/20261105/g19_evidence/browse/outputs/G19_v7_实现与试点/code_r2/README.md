# G19 v7 实现与有限试点

研究目的：辨认真实 FFN 的信息、输出值域、高阶可分离性及条件结构，寻找改变真实计算量的表示。G19 与输入低秩只是候选，不是必须证明正确的结论。

本版本已经实现可单测的数学／统计核心与一个可提交 Slurm 的 **S0 资源／数值试点**。完整全层普查、PH1 配对映射、正式候选确认和原生状态验证尚未接入该 runner。不要将试点成功理解为整个 v7 实验完成。

## 数学实现

- Gaussian 正交投影与固定子空间 pick-freeze；固定线性控制变量针对最终比值的影响函数选型，128/128/256 独立用途分割。
- Hermite 正交展开、含偏置 tanh-GELU 系数、A1、Q2/Q4 固定多项式残差。任意固定低阶多项式 G 满足 E||F−G||² ≥ 真正高阶尾能量；样本残差本身不是上界证书。
- 完整渐近 Fieller 集合，保留无界、分裂、空集和退化情形；不是有限样本保证。
- 输出值域候选与独立完整输出误差；高维小样本产生的零特征值不能证明真实低秩。
- 矩阵自由随机二次切片：M_t X = Pᵀ diag(a2 ⊙ Vt) P X。在探测子空间内取绝对值较大的谱方向，构造 μ + Aᵀz + Σ c_k h2(q_kᵀz)，以独立样本检查完整函数误差。

最后一项受到 [He–Kressner 随机共同对角化](https://epubs.siam.org/doi/abs/10.1137/22M1541265)启发。本实现只做有预算的随机切片候选，尚不构成全矩阵族共同对角化验证，其稳健恢复定理不能直接套用。控制变量独立拟合和成本比较参照 [Owen 的方差缩减分析](https://artowen.su.domains/mc/Ch-var-basic.pdf)。

## 真实数据与试点范围

层 0、12、23、37 来自预先四层段的首次随机抽取；high/low × O32/Oq/Dq 共24单元。每个 expert-layer 使用64个独立(z,w)块；四个符号样本属于同一块。三臂共用输入与二次随机探针。开发64个独立样本与留出块使用不同 DEVELOPMENT 流；CONFIRMATION 流未使用。

O32 是原始 F32 权重提升，Oq 是其显式 BF16 RNE 后提升，Dq 是蒸馏 BF16 位模式解码后提升。全部实际偏置纳入。FP32 sampled forward 关闭 TF32；FP64 A1、Hermite 与单行 forward 抽查均是连续函数数值探测，不等同原生 BF16 程序。

输入 manifest 冻结 header SHA、size、mtime、dtype、shape、offset；本次读取建立实际 tensor SHA。FIRST_READ_IDENTITY 不意味着已有历史 raw hash 匹配；后续确认必须冻结这些新 SHA。

GH192/96 差与单行 FP64 差仅标为 UNASSESSED 指标。64块不是正式确认样本量；不计算或声称已达到90%功效。rank32 是资源探针，不能把其失败推广到所有输出或二次表示。

## 安全运行

只能用 `sbatch job_v7.sh <scope.json的精确SHA256>`。脚本与 Python 双重拒绝登录节点；Python 用 `scontrol` 核验当前用户、RUNNING 作业、当前主机确实属于分配、1节点1GPU、至多4CPU/64GiB/30分钟。真实权重读取再次检查分配。没有跳过参数。

启动必须同时满足：精确 scope、代码／config／input SHA 匹配，以及不同作者的 INDEPENDENT_REVIEW.json 对该 scope 明确 ACCEPTED。修改文件后旧批准失效。独占输出目录、START/EXIT、信号和软截止保留失败证据；硬 kill 缺少 EXIT 必须判未完成。

本次 scope 只有1个作业，禁止自动重试或扩到全层。采集后回到机制判断与分层覆盖，不围绕rank32局部调参。

## 文件

`core_math.py`、`statistics_v7.py`：数学与统计参考核；`structure_v7.py`：可单测结构候选；`weights_v7.py`：字节级读取；`runtime_guard.py`：计算节点和精确scope核验；`runner_v7.py`：CUDA试点；`scope.json`/`pilot_config.json`/`job_v7.sh`：有限执行合同。

三个 test 文件覆盖34项微型反例和拒绝路径。运行 `python -m unittest discover -p 'test_*.py' -v`；本地需要 NumPy/SciPy。该命令仅用合成微型数据。GPU路径另需PyTorch，并只能在通过上述分配检查的计算节点执行。

R2补充：两个运行守卫测试使总数增至36项。已在计算节点完成一次24单元试点，实际结果和边界见上级报告。内存请求回退核验不等于cgroup硬限额，另监测进程RSS。
