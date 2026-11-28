# 均值与尺度运输：真实执行与完整成本复核

证据提交 ec8a39b8a6c8c6a6cb25badfe11e8b6a38b09f63，学校 A100 作业 201500。本次独立工作读取运行源码及原始收据，用 CPU 复算配对时间、状态指纹、数值见证和显存范围；新增模型运行、GPU 作业与视频解码均为 0。

## 1. 已经完成的执行层次

轻量构造已进入真实 Wan：先在两条冻结 C 轨迹上记录 low 第 0、13、26、39 层的对照，然后运行两来源的 centered 自身完整轨迹，并对原 C 与 centered 做每源两轮、顺序交错的计时。原 C 指纹与历史 job 194188 的 C 分支一致。

计时之前分别预热 raw 和 centered。正式比较共四对、八次完整生成；两次带诊断固定回放与两次预热另列。两来源的每个方法在两次正式重复中，初始状态、四步 flow、四步 latent、RGB、MP4 以及专家身份均逐字段相同。centered 预热也与对应来源的正式 centered 轨迹相同。

四对比较的 initial、前三步 flow/latent 与专家序列均相同；第四步 flow/latent 确实改变。该证据定位了构造的作用：step2 编码新的历史状态，仍返回原生 attention 输出，因而不改变当步隐变量与后继 FFN anchor；step3 才读取新状态并改变生成。

## 2. 真实公式与现有接口相符

原 low OUT-D 历史为

\[
u_a=A_a-(1-\rho_a)B_a.
\]

当前实现用 key 轴的 value 均值和总体标准差，构造

\[
C_a=\operatorname{BF16}\{[A_a-(1-\rho_a)B_a-\rho_a\mu_a^\top]D_a^+\}.
\]

这里直接从现场原生 A、selected B 编码，未先写旧 BF16 u 再重新编码。命中底座为

\[
B_s^{new}=(1-\rho_a)B_s+\rho_a\mu_s^\top+C_aD_s.
\]

完整 C_a 占用原 u_a 槽，shape、dtype 和持久全场数量保持相同。原 rho、selected 支持、selected 投影差特征、CAL、P64 与 lambda=0.1 继续使用；ridge 对角加项仍是 64×0.1=6.4。high OUT-ABS 调用直接走原实现。

统计和编码路径为：

- moments 对 value.float() 在 key 轴执行 var_mean(correction=0)，尺度 ≤1e−12 时将 scale 和 inverse 置零。
- encode 和 decode 使用当前均值、rho、尺度执行上述逐元素公式。
- moments、encode、decode 均通过 torch.compile(fullgraph=True, dynamic=False) 调用。
- 正式 raw 模式的 layer=None，low_cache / approximate 直接返回父类旧 C 路径，未携带诊断用的新统计或 shadow 分支。

运行时断言验证冻结 attention 源码、FIT.sha256 与 CAL 配方。独立复算绑定的旧源码 SHA256 为 8781a7983461e5c0042e1e0f4d58e545c2cab5b7508d35dc450e2de87b07dcee，和冻结配方一致。此处核对了归档主文件和已有冻结资产；未将这些指纹扩写为学校环境全部导入依赖的新逐字节验证。

## 3. 固定轨迹与生产轨迹分别提供什么证据

固定诊断在实际 C 输入上进行。raw 为现场旧 C；centered 是同输入下的 shadow。H448 的原生 attention 教师通过 448-query、全部 key 的 SDPA 求得；P64 复用现场 C 的短 query、全 key 教师。它们与上轮 N-query 原生输出取 P 的诊断路径不同，但本轮两分支的教师相同。

JointTail 的 Gram 仍相对同输入 native attention 经固定 theta 的 wide8192 尾部构造。LowTail.live_arm 被设为 raw，因此 live_relative_l2 的含义是 **raw 回放与现场 C 块出口的差**。最大值为 3.042922116947011e−6，不能把该字段重新命名为 native 回放误差。

现场 FFN P64 输入和标签只捕获一次，三个分支使用同一拟合 theta。8 处 P 输入对 live 的相对差均为零。Dense 参照固定同一 C-trajectory 块输入，在原生 H448 attention 所产生的 FFN 输入上额外计算 Dense FFN。诊断额外执行 8×448=3,584 行 attention query 和 3,584 行 Dense FFN；它们不进入正式计时循环。

固定诊断的 anchor C_a 由 compiled encode 生成，hit 的 H448/面板底座调用 eager decode；完整 centered 生成使用 compiled decode_base。归档没有同输入 eager-panel 与 compiled-fullfield decode 的位级对照。因此局部数值诊断和候选自身完整轨迹各有直接证据，而不是一份已证明逐位一致的全场 replay。

8 处 anchor/current 通道都高于活动阈值；本轮验证到了正常活动通道，没有触发零尺度分支。有符号 rho 原样沿用，最小值为 −3.051804378628731e−5。舍入前固定恢复更新恒等式最大相对 L2 差为 2.3423067525083823e−6。源码、常量通道的数学参考及这些实测数值分别说明其适用范围。

## 4. 完整单次复用周期已有实际收益

每源先对两次正式重复取均值，再对两来源耗时比取几何平均。独立复算与归档全部计时字段一致。

| 测量区间 | 原 C 均值／秒 | centered 均值／秒 | centered / C | 平均节省／秒 |
|---|---:|---:|---:|---:|
| low anchor step2 | 8.980628 | 8.855057 | 0.9860175 | 0.125570 |
| low hit step3 | 4.360389 | 4.292389 | 0.9844051 | 0.068000 |
| low 两步周期 | 13.341017 | 13.147447 | 0.9854906 | 0.193570 |
| 四步 GPU | 25.458827 | 25.273122 | 0.9927055 | 0.185706 |
| 完整请求含保存 | 33.257629 | 33.057210 | 0.9939724 | 0.200420 |

四对的 low 周期分别节省 0.176213、0.184042、0.217530、0.196496 秒；anchor、hit、low 周期和四步 GPU 均为 4/4 更快。完整请求为 2/4 更快。统计单位是两个来源，重复用于计时稳定性，不能当四个独立来源。

low 周期包含当前统计、anchor 重编码、cache 写读、hit 解码、原有恢复和后继模块，已覆盖每个 anchor 仅服务一次 hit 的实际生命周期。因此本批执行支持：**新增运输可以在现有系统中以相近且稍低的完整复用成本运行。**

### 执行方式也是当前候选的一部分

原 C 正式路径使用原 eager 数值实现，centered 路径对 moments、encode、decode 使用 compile。这个配对测量的是“运输构造及其融合实现”的整体效果。它尚未分别测量“只把 raw 编解码也融合”或“centered 使用旧 eager 执行”。

因此低周期节省是当前实现的真实结果，可以用于决定继续推进；它不单独证明增加均值/尺度操作会在任意相同编译条件下产生同样速度差。当前源码与峰值数据共同提示，中间数组的物化方式是成本得到控制的重要路径。

## 5. 整片时间存在可定位的顺序差异

| 运行顺序 | 对数 | low 周期变化／秒 | 四步 GPU 变化／秒 | 完整请求变化／秒 | GPU 事件之外变化／秒 |
|---|---:|---:|---:|---:|---:|
| raw 先运行 | 2 | −0.186354 | −0.162430 | +0.079684 | +0.242115 |
| centered 先运行 | 2 | −0.200786 | −0.208981 | −0.480524 | −0.271543 |

变化均为 centered−raw。两种顺序下 GPU 和 low 周期都节省，完整请求的正负变化则随本批运行顺序分开。额外差主要处在四步 GPU 事件之外；归档没有逐次保存时间戳，不能直接把全部差命名为写视频开销。

顺序交错使均值包含两种顺序，但样本量仍小。完整请求均值省 0.2004 秒是本批配对结果；目前更稳定的工程判断来自四对一致的 low 周期和 GPU 变化。无需把约 0.60% 的本批整片均值先提升为广泛加速结论。

## 6. 峰值下降对应工作存储，历史状态容量保持相同

四次正式 raw 的峰值分配均为 70,232,717,312 字节，即 65.409315 GiB；centered 均为 68,933,642,752 字节，即 64.199458 GiB。差值为 1,299,074,560 字节，约 1.209857 GiB。

真实尺寸 N=31,668、H=40、d=128 下，一个全场 FP32 张量占 648,560,640 字节，两个占 1,297,121,280 字节，与峰值差相近。这与融合点运算减少大临时量的代码结构一致；目前没有分配时间线或生成内核清单，可以把原因限定为有依据的执行解释，不能将差值精确指派给特定两个缓冲。

两个方法在同进程持有相同 wide8192 常驻后端；FFN anchor_gpu_capacity_bytes 都是 32,428,032,000，且没有 D2H/H2D 锚点搬运。attention 的完整历史槽也是同形状 BF16 替换。因此这里测得的是**实际峰值工作存储下降**，并非历史状态压缩了 1.21 GiB。

## 7. 可复算资产

recompute_mass_runtime.py 从原始收据计算时间、顺序分层、峰值、前缀和重复指纹，最后才与归档计时汇总核对。mass_runtime_verified.json 保存逐对结果与所有检查范围。

~~~sh
OPENBLAS_NUM_THREADS=1 python recompute_mass_runtime.py \
  --source /path/to/mass_centered_runtime \
  --historical-quality /path/to/ONLINE_QUALITY_194188.json.gz \
  --frozen-code /path/to/attention_online_structured.py \
  --frozen-recipe /path/to/C_CONFIRM_METHOD.json
~~~

历史质量、冻结代码与冻结配方参数可省略，省略时输出不会宣称完成对应检查。运动代理仅有归档评分，本次没有重新解码视频；两来源完整视频仍由用户判读其动作和外观。
