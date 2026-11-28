# 当前 wide8192 的联合计算图审查

审查基准：Qesire/FFNDS，PR65 head 94b7aa8eef1e7268741bdfadc25970980261e474。本文件记录源码审查，新增 GPU 执行数为0。

## 源码身份

生产绑定的 LightX2V transformer 和 step scheduler 与 ModelTC/LightX2V@ed146db64d0b6130c669285fc1aa0b280a6a4823 的原始文件 SHA256 精确一致：

| 文件 | SHA256 |
|---|---|
| lightx2v/models/networks/wan/infer/transformer_infer.py | 5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225 |
| lightx2v/models/schedulers/wan/step_distill/scheduler.py | 2d560449e0559aea44d80afb539f9e97bd544e5e26abcc06236e61918d01e1db |
| lightx2v/models/networks/wan/infer/pre_infer.py | f3f88ca8ccbbd778206f8f04805c124beb84c19dcd22c4c38a6028b42c1502b5 |
| lightx2v/models/networks/wan/infer/post_infer.py | 6567c439e2f16f322a9f10cd2ca326d3f924e9da75b3ae2ae442ec1327a327a2 |

FFNDS upstream_audit/AUDIT_SUMMARY.json 记载这个 reference commit 与存档 ZIP 的全部 1663 个 scoped 文件匹配；没有假定已恢复原机器的 git HEAD 元数据。

## 真实计算图

模型宽度 d=5120，40 heads，每 head128，FFN hidden13824。当前视频 patch 网格为21×29×52，N=31668，latent [16,21,58,104]。

固定层与步的逻辑实数表达（以下按单query列向量书写；self attention读取完整X，线性投影含相应偏置）：

1. U1 = (1+s_a)⊙LN1(X)+b_a。
2. A = self_attention(U1)，含 Q/K RMSNorm、RoPE、全 token K/V、head concat 与输出投影。
3. R = X + g_a⊙A。
4. T = R + cross_attention(LN3(R), context)。
5. U2 = (1+s_f)⊙LN2(T)+b_f。
6. Y = T + g_f⊙[W2 GELU_tanh(W1 U2 + b1) + b2]。

LN1、LN2 无学习 affine；LN3 有学习 affine。均沿最后一个5120维轴计算 LayerNorm，eps=1e-6。head norm 也是 LayerNorm、无 affine。gate、scale、shift 来自 timestep embedding 和层 modulation；当前 distilled runtime 使用每一步一个全局时间，gate是5120通道并对 token 广播，与当前残差X无关。

源码位置（原生源文件固定于[LightX2V参考提交](https://github.com/ModelTC/LightX2V/tree/ed146db64d0b6130c669285fc1aa0b280a6a4823)）：
- transformer_infer.py: pre_process、infer_block、infer_self_attn、infer_cross_attn、infer_ffn、post_process、infer_non_blocks。
- transformer_weights.py: norm1、norm2、norm3 与 head norm 构造。
- lightx2v/common/ops/norm/layer_norm_weight.py: LNWeight.apply。
- lightx2v/models/schedulers/wan/scheduler.py: step_pre（单一时间输入，当前 distill 变体不使用固定前缀掩码）。
- q1_fused_post.py: gate.numel()==5120；gate按 i % d 取值。

BF16 真实实现包含 norm、modulation、线性算子与 residual 的 cast/rounding 边界。公式描述数学对象；数值声明应由原生 dtype 路径验证。

## Cross attention 和位置

通用 LightX2V 源码允许在 I2V 且 use_image_encoder=True 时，将 context 的前257行作为 image context，独立计算 text 和 image cross attention，求和后使用同一个 cross_attn_o。当前生产契约 use_image_encoder=false，因此此配置只有 text cross attention。图像条件通过 VAE 编码结果与 noisy latent 在输入通道拼接，进入 patch embedding。

cross attention 只在 fixed context 轴求和，不在视频 query 轴混合；LN/AdaLN/FFN/residual 都逐 token。FFN 自身不混合位置，但其输入已经由 self attention 聚合全视频、由 RoPE 影响，并受文本 cross attention 影响。

## P64 闭包与 teacher 成本

从当前 self attention 输出 A 的任一行，到同一 block 的输出Y，是逐 query 的确定函数（固定 context）。所以在同一个当前 block 输入X上，先用全N当前K/V准确计算 self P64，再沿 cross/LN/FFN 计算P64，可以获得原生 block 的64行 teacher。不能只用P64 K/V替代全N K/V。

有三种实现口径，需分别计算成本：

A. 保持 attention P64 精确覆盖，再做 wide8192 FFN 独立在线 fit。
P64 在进入 cross/LN/FFN 前已被 exact attention 覆盖，因此现有 FFN probe 实际输入是 U2_star_P，所算 F(U2_star_P) 可以作为整个 block 的 teacher FFN 标签复用。此时不一定需要第二次 dense FFN。不过，联合残差还需要“未覆盖近似 attention 输入”的 cheap block 输出：必须以覆盖前 Ahat_P 为输入，额外执行 cheap shadow tail，并固定原 FFN 已拟合的读出参数，才能构造这个条件反事实。只保存覆盖前 attention 值，不能直接获得该非线性 tail 输出。固定读出参数的 shadow 不等于把整个独立 FFN fit 改到另一套输入后重新运行算法。

B. 保留未覆盖的 Ahat_P，并在该输入上执行 FFN 独立在线 fit。
这时原 FFN probe 计算 F(U2_hat_P)，而 exact-attention 联合 teacher 需要 F(U2_star_P)。两个输入不同，因此保持这两套标签时需要两次 P64 dense FFN，并计算相应的两条 cross/LN tail。此成本判断适用于本情形，不能套用于情形 A。

C. 一次 block-joint fit 替换 attention/FFN 独立在线 fit。
cheap 全行路径保留未被 teacher 覆盖的 attention 和 FFN raw baseline；独立 P64 teacher 路径做 exact selfP→crossP→FFNP；在 block 末端拟合一次联合修正。移除原 FFN 独立在线 fit 后，可以设计为每个 P 仅一次 dense FFN teacher。此为待实现的重组方案，不是已证实的无额外开销；仍须实际计算和保留两支 tail 输出。

当前 wide8192 是按 P64 观测拟合共享系数：固定系数后每行读出逐行，但原始全算法含跨 P 行依赖。对 probe 输入的扰动会改变系数并影响全部 token。因此固定系数的条件 shadow 与重新拟合后的完整算法是不同问题；若实验声称后者，就必须实际重拟合，不能把现有 wide8192 整体当作固定的逐 token 函数。

## 下一块和终端

下一块 self attention 是闭包边界：状态误差会同时改变每行query和全体K/V；只追踪P64不能给出真正全视频多层teacher轨迹。可选：
- 固定全N baselineK/V、计算选定query的条件读取误差（只回答条件问题）；
- 保存全N状态并重算下一层K/V与必要query；
- 对源端K/V扰动建立全局summary及可验证误差；
- 完整suffix或完整rollout验证。

终端 infer_non_blocks = LN→timestep modulation→linear head，5120→64。post_infer unpatchify只reshape/permute，64=16×1×2×2，对应latent [16,21,58,104]；之后decoder才进入像素视频空间。因此最终hidden的欧氏误差不能直接替代flow或视频质量。

四步 sample_shift=5，steps[1000,750,500,250]对应sigma[1,15/16,5/6,5/8]（实际FP32有舍入）。latent更新的数学式 z_next=z−h*flow，h依次为[1/16,5/48,5/24,5/8]。hit step1与step3的直接flow→latent幅度权重相差6倍、平方36倍；完整rollout还受后续Jacobian和误差相关性影响。当前scheduler没有high→low切换再加噪声，也没有固定latent前缀覆盖。

生产契约的固定仓库证据为 Qesire/FFNDS@94b7aa8eef1e7268741bdfadc25970980261e474 的 upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json。该文件明确给出 sample_shift=5.0、denoising_step_list=[1000,750,500,250]、out_dim=16、use_image_encoder=false、model_cls=wan2.2_moe_distill、scheduler=Wan22StepDistillScheduler。链接：https://github.com/Qesire/FFNDS/blob/94b7aa8eef1e7268741bdfadc25970980261e474/upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json 。这份契约的目标尺寸字段是较早配置，不应替代当前执行几何；这里引用的是上述四步、输出通道与 image encoder 开关字段。

## 可证明的后缀不变方向（实数算术）

由于当前所有 residual branches 都先作逐token LayerNorm，且 gate只依赖时间，逐token的通道常数平移 E_i=c_i*1 满足每块 B(X+E)=B(X)+E。末端head LN消除E，所以原生suffix输出满足 S(X+E)=S(X)。RoPE不会破坏此性质，因为其Q/K输入在LN后已经相同。

它给出真正全suffix不读取的残差状态方向。attention输出扰动首先要乘当前gate，只有门控后的残差扰动属于该空间才能套用。BF16舍入、非当前配置的未归一化旁支，以及未按当前norm输入构造的近似器需要单独核查。此性质不意味着一般误差都会被LN消除。

## 结论范围

代码证明了同block query closure、联合teacher可计算性、norm平移对称与scheduler权重。它没有证明联合修正已改善质量、P64足够覆盖可见方向、下一层K/V误差可忽略，或任何新增方案的运行速度。这些应由分叉实验和终端rollout判别。
