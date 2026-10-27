# C5 · 保持原CUTLASS字节表示的直接选择与打包

2026-10-27 · SRC/ALG/CPU_TEST；新GPU结果尚未产生。

**同一个压缩格式可以有不同的构造计算图。** C4在相同BF16输入、相同top2选择下，通用Torch CUTLASS converter占约10/27ms，而预打包乘法加连续输出约14–15ms。格式本身要求一半数值和metadata，并不要求先创建完整masked矩阵、多个非零mask/int64索引、gather/stack结果和全局重排索引。因此本候选直接把稳定top2、数值写出与metadata swizzle合为一次kernel，同时严格保留原converter的packed/meta字节。

这对应两个不同分歧：`通用converter的中间表示/调度成本`与`既定格式的最低构造成本`；即使前者可大幅消除，`SpMM本体与输出布局的物理效率`仍独立限制在线收益。C5不调整selector、不拟合参数、不改变数学支持或后端，也没有真实Wan Q2/W2、完整FFN或视频证据。复用既有激活2:4/C线讨论，不新增文献或图节点。

## 1. 源码身份与独立对照

本机Torch2.10 `_semi_structured_conversions.py` SHA为`1759d16ca5dbdcf0a5cef676eafbd4f1477af5e961b18a8fc8dd83a1c1c6728e`。关键边界为L87–145的非零判断/位置编码/gather、L149–177的四nibble打包和metadata scatter。其[官方v2.10源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/sparse/_semi_structured_conversions.py)也把该布局对应到CUTLASS行重排；NVIDIA的[固定v4.3.3 `reorder_meta`](https://github.com/NVIDIA/cutlass/blob/v4.3.3/tools/util/include/cutlass/util/host_reorder.h)给出行内置换与2×2低位交换。实际执行以本机已绑定文件为准，不能用另一版本转换器替代参照。

CPU对照直接调用安装包中的原函数，未重写“参考converter”。新增代码位于：

- [packer与标量代数](../../../../tools/activation24_fused_cutlass_pack.py)
- [独立probe](../../../../tools/e0_activation24_fused_cutlass_probe.py)
- [CPU测试](../../../../tests/test_e0_activation24_fused_cutlass_probe.py)
- [新scope草案](../../../../config/e0_activation24_fused_cutlass_probe_20261027_v1.json)

原C1–C4代码、protocol、RUN和ACTIVE均未改。

## 2. 原选择函数与零占位有两个独立步骤

对四个有限BF16数，设其16位存储为b_i，绝对值编码为a_i=b_i&0x7fff。有限BF16的非负编码单调对应其数值绝对值；+0/−0给出同一个a=0，正负subnormal的符号也被去除。定义

\[
r_i=\sum_j[\,a_j>a_i\;\lor\;(a_j=a_i\land j<i)\,].
\]

于是r_i<2与原`argsort(abs, descending=True, stable=True)[:2]`的数学选择相同。被剪元素写为原生+0，保留元素按原BF16位复制。该实现不先把subnormal提升为FP32做算术；GPU原版排序的实际语义仍由本次byte gate核对。

**metadata必须依据选择后的实际非零，而不是直接写r<2的支持索引。** 原converter对y_i执行`y_i != 0`。令m_i为该布尔值，它对±0均为False。其编码可以写成

\[
e_0=m_0m_1,\ e_1=\neg m_0m_1,\ e_2=\neg m_0\neg m_1;
\quad i_0=e_1+2e_2,
\quad i_1=(e_0\lor e_2\lor m_3)+2(e_1\lor\neg m_1).
\]

输出两个数值为(y_(i0),y_(i1))，nibble为i0|(i1<<2)。这就是原converter的canonical补零规则，m2由合法2:4情形的其它标志隐式决定。CPU测试也对原函数的全部16种非零mask核对了这个映射，而非只测试恰好两个非零。

| 选择后非零位置 | 原converter存储位置 | nibble |
| --- | --- | --- |
| 无 | 2,3 | E |
| 0 | 0,2 | 8 |
| 1 | 1,2 | 9 |
| 2或3 | 2,3 | E |
| 0,1 / 0,2 / 1,2 | 同这两个位置 | 4 / 8 / 9 |
| 0,3 / 1,3 / 2,3 | 同这两个位置 | C / D / E |

具体反例：x=(1,0,0,0)的top2支持是0,1，正确converter却编码0,2。全零quad编码2,3，四个全零quad的int16 word是0xEEEE，即−4370。直接编码原top2索引会在这些情况下产生不同metadata，虽然解压后的浮点数值可能一样。

signed-zero也保留该区分。选择阶段可以保留−0；原converter的补零位置却落在该合法0/1非零情形中被剪为+0的位置，因此最终packed零占位是+0。C5复制原规则，测试用字节相等而非`torch.equal`数值相等；后者无法区分±0。Inf/NaN不在有限BF16输入合同内，probe在调用前检查有限性。

## 3. 数值排列与metadata置换是一一映射

每个group的两个packed值直接写到flat offsets2g和2g+1。这正是原`stack(sparse0,sparse1).view(Mp,K/2)`的行主序位置。每四个相邻group组成一个int16 metadata word：

\[
w=\sum_{j=0}^3 c_j2^{4j}.
\]

各nibble位互不重叠，sum等于OR，没有进位；中间int32足够，最后截取int16两补码与原Torch的int16 shift/OR一致。固定K%64=0、Mp%32=0、CTA=256groups，使每四group不会跨row或CTA，合法几何的group总数还可被256整除。

逻辑metadata坐标为(r,c)，M=Mp、C=K/16。先做

\[
r_1=32\lfloor r/32\rfloor+4(r\bmod8)+\lfloor(r\bmod32)/8\rfloor.
\]

若r=32b+8u+v，其中u∈[0,4)、v∈[0,8)，则r1=32b+4v+u，因此这一步交换两个有限索引域，是32行块内双射。再将r1=2a+b1、c=2d+e表示为低位与高位，2×2 swizzle实际交换b1与e：r2=2a+e，c2=2d+b1。存储地址为

\[
f(r,c)=2M\lfloor c_2/2\rfloor+2r_2+(c_2\bmod2).
\]

每个相邻column pair覆盖连续2M地址；r2与最低column bit又覆盖其中全部位置。因此f在合法偶数C、M%32=0时无碰撞、无空洞。各线程仅写它唯一的int16地址，不需要atomic，也无需先物化完整scatter-offset矩阵。

CPU测试在多组row/column尺寸（包括320/864 metadata columns）核对该映射的完整置换，且逐项等于原`_calculate_meta_reordering_scatter_offsets`。

## 4. 实现与证明状态

新Triton kernel固定256 groups/4 warps，无autotune。`MASK_ONLY=True`分支使用完全相同的原始bit加载、rank比较和选择，仅物化masked矩阵；这个分支加原converter构成同选择机制的baseline。`MASK_ONLY=False`直接写packed values与swizzled metadata。二者都原地生成Mp行输出，虚拟padding不读取不存在的源row。

最终存储为BF16 `[Mp,K/2]` 加int16 `[Mp,K/16]`，共9MpK/8 bytes，是BF16 dense容量的56.25%。低于两个非零的quad依然存两个值与metadata，硬件不会因为其中含零就进一步少做MAC。

已完成的CPU证据：

- 原converter SHA身份核对；16种非零mask编码；所有关键0/1/2非零、ties、±0、subnormal及有限极值。
- 8个有限BF16位模式的四元组穷举，共4096组，与原稳定选择+converter的packed/meta逐字节一致。
- 两个固定edge矩阵(35,64)、(33,128)的标量CPU实现逐字节等价；包括padding rows。
- **实际Triton DSL在CPU interpreter上**运行这两例，packed/meta/masked三个输出都与原函数逐字节相等，CUDA未初始化。
- 十项CPU测试通过；新CLI的31项source/backend bindings静态预检通过。CPU解释器结果不能通过probe中的“实际GPU前置”守卫。

以上不代表已经通过GPU编译或测到新性能；实际GPU差异会在同一新scope的前置门槛中暴露。

## 5. 固定判别scope与真实代价

新草案ID是`E0_ACTIVATION24_FUSED_CUTLASS_PROBE_20261027`，尚未登记/运行GPU。两fullshape与C4完全相同：M31668→Mp31680，(K,N)为(5120,13824)/(13824,5120)，A/W随机seed顺序仍20261027/28/29/30。

顺序固定：先做两例实际GPU小矩阵的原packed+meta byte gate；每个fullshape又做它自己的GPU byte gate，随后才允许该shape的SpMM与timing。任何字节/非有限/资源错误都停止，不换后端、不调selector、不自动重试。完整输出必须是M×N contiguous。

每个shape的七项计时各2次warmup+5次记录：dense、materialized selector、原Torch converter、fused selector/pack、prepacked functional SpMM、materialized selector+converter+SpMM+contiguous、fused selector/pack+wrapper+SpMM+contiguous。GPU event与host wall同时记录，allocations/wrapper在对应完整计时内；首次编译/setup单列。

上限为随机初始化4、edge上传2、有限检查6、原stable参照4、materialized selector30、fused pack32、原converter32、wrapper48、packed/meta byte check4、masked byte check2、dense GEMM16、CUTLASS SpMM44；model/fit/video/search均0。所有public operation在调用前预扣，失败不退，成功返回计数另列；不声称这些数等于Torch converter内部CUDA kernel数。

CPU身份预检（source hash/worker/prior C4）在ATTEMPT与480秒compute alarm之前，未初始化CUDA；outer600秒包括该预检。ATTEMPT之后才加载GPU运行时、申请一张空闲A100、执行上述gates及计时，失败保留单次attempt。总输出100MiB，2 CPU线程，tensor allocated上限64GiB，用户全局GPU上限仍4。本候选不继承原C4的速度或任何模型质量结论。

## 6. 打包与SpMM余量是两组竞争解释

**H-converter：构造表示的中间计算是主要在线损耗。** 预测：同一packed/meta字节下，fused pack明显快于materialized selector+原converter，完整在线耗时向prepacked functional成本靠近。若直接metadata scatter或内核layout低效使fused pack仍慢，该具体实现的预测受削弱；“格式本身不可快构造”尚未由一个实现证明。

**H-SpMM：即使构造近乎免费，Sparse kernel物理吞吐仍限制收益。** 两shape dense工作量均约4.48e12 FLOPs；约17ms对应约260TF/s。半MAC乘法14–15ms只对应约150TF/s的实际算术吞吐，所以2×MAC减少没有转成2×时延。输出流量、metadata访问、tile/warp组织及accumulation/reduction路径都可竞争解释。记录的BF16 reduced-precision flag本身不能证明CUTLASS的具体accumulator dtype；不能把差异先归因给某一种accumulation机制。

**H-output：raw乘法与规范输出之间可能存在大代价。** [已完成C2布局结果](../../../20261027/E0_ACTIVATION24_BACKEND_LAYOUT_RESULTS_20261027.json)已给出cuSPARSELt实际反例：B-view、transpose_result=True时，第一shape默认raw14.155ms但functional21.989ms；第二shaperaw15.524ms而functional20.009ms。transpose_result=False的searched配置则raw/functional约16.412/16.415ms和16.627/16.631ms，后处理几乎不收费。这个真实区别说明“raw更快”不足以选执行布局。

C2是另一后端/配置的有界证据，不能用它直接确定C4/C5 CUTLASS的raw/contiguous拆分。C5保持现有固定scope，只报告完整functional输出与组成项，不新增raw-product实验；若将来需要隔离CUTLASS输出布局，应另登记对应计时。one-pass降低converter价格并不等于完整FFN大幅加速，下一系统主张仍需真实Q2/W2及其它FFN操作的同条件成本和自由视频功能证据。
