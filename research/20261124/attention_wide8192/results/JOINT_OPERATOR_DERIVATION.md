# 从注意力恢复到残差块及采样读出的联合恢复

本文件是数学推导与可执行构造；真实模型证据仍采用 FFNDS 当前采集结果。附带两份 NumPy 程序只核验代数与架构结构，使用合成权重，不代表新增 Wan 运行。

## 1. 对象与实际边界

固定当前去噪步、专家、层、模型权重、文本条件及调制。单个视频 token 采用列向量，宽度 D=5120。真实 block 可写为

\[
z_i=x_i+g_i^A\odot a_i(X),\quad
u_i=z_i+C_i(z_i;\mathrm{context}),\quad
B_i(X)=u_i+f_i(u_i),
\]

其中 a 是 self attention 经输出投影后的分支，C 含 cross 的 pre-LayerNorm、Q/RMSNorm、固定 context K/V attention 与输出投影，f 含 FFN pre-LayerNorm、AdaLN、两线性层/GELU和最后的FFN gate。

关键结构：self 读取所有视频 tokens；固定当前X后，C和f对视频query逐行独立。于是“精确self query→cross→FFN→block出口”在64个query上闭合。下一块self会重新通过K/V读取全部视频token，闭合范围到当前block出口为止。

## 2. 注意力误差如何经过后续模块

把self写入误差记为 e_i，已包含W_O与当前gate。在相同当前条件下：

\[
\delta u_i=(I+\overline J_{C,i})e_i,\qquad
\delta B_i=(I+\overline J_{f,i})\delta u_i,
\]

\[
\boxed{\delta B_i=(I+\overline J_{f,i})(I+\overline J_{C,i})e_i.}
\]

每个割线矩阵是相应两端之间的积分Jacobian；C的积分路径在z空间，f的积分路径在u空间。这是有限差分恒等式。端点只约束算子在实际方向上的作用，不唯一确定整张矩阵。实际求值可只算方向作用。

### LayerNorm具有闭式有限割线

令 P=I−11ᵀ/D，u_0=Px_0，u_1=Px_1，s_j=sqrt(||u_j||²/D+eps)，u_bar=(u_0+u_1)/2。则

\[
\mathrm{LN}(x_1)-\mathrm{LN}(x_0)
=\alpha P\Delta x-\gamma\bar u(\bar u^\top\Delta x),
\]

\[
\alpha=\frac{s_0+s_1}{2s_0s_1},\quad
\gamma=\frac{2}{D(s_0+s_1)s_0s_1}.
\]

affine LayerNorm再左乘对角weight；固定条件AdaLN再左乘时间scale。此式包含eps，不要求无限小扰动。

### Cross与FFN可沿用已有有限变化工具

固定K/V的cross每头满足

\[
\Delta o=\eta\operatorname{Cov}_{\pi}(v,k)\Delta q/\sqrt{d_h},
\]

令两端softmax概率为p0、p1，逐键对数平均为 ell_j=(p1_j−p0_j)/(log p1_j−log p0_j)，相等时取连续值；eta=sum_j ell_j，pi_j=ell_j/eta。上式Cov_pi(v,k)为值与键的交叉协方差矩阵，q已经经过实际norm3、Q投影和Q-RMSNorm。FFN满足

\[
\Delta f=D_{g_F}W_2\operatorname{diag}(\beta)W_1\Delta M,
\quad
\beta_j=\frac{\phi(h_{1,j})-\phi(h_{0,j})}{h_{1,j}-h_{0,j}}.
\]

零位移取连续值，M是实际AdaLN后的输入。这些公式使“注意力写入→条件重新读取→非线性FFN响应”成为可测的有限输运。

## 3. 注意力与FFN近似的四臂精确分解

固定入块X、条件、anchor/cache；定义u_0为完整attention进入cross后的状态，u_1为近似attention进入cross后的状态。令

\[
B_{00}=u_0+f(u_0),\quad B_{10}=u_1+f(u_1),
\]
\[
B_{01}=u_0+\hat f(u_0),\quad B_{11}=u_1+\hat f(u_1).
\]

则

\[
B_{11}-B_{00}=E_A+E_F+I_{AF},
\]

\[
E_A=B_{10}-B_{00},\quad E_F=B_{01}-B_{00},
\]

\[
I_{AF}=B_{11}-B_{10}-B_{01}+B_{00}
=e_F(u_1)-e_F(u_0).
\]

I_AF衡量注意力扰动改变了FFN近似误差多少。最终能量应保留EA、EF、I_AF的完整带符号Gram；模块误差能量相加会遗漏补偿与共同放大。

如果FFN近似含在线P64拟合，完整算法会通过共享系数跨query耦合。冻结cache与读出系数的四臂先隔离固定读出的输入响应；正常重拟合则还测量系数反馈。现方案若attention先在P精确覆盖，在固定X下两条attention臂的P输入可能相同，因而仅由P决定的FFN系数也相同。必须记录实际覆盖和拟合时序。

## 4. 一个整个后缀都不读取的方向

实际绑定的LightX2V生产配置满足：self、cross、FFN分支均从逐token LayerNorm读取残差；gate只依赖时间/条件；终端head也先LayerNorm；生产未使用未归一化adapter旁路。

因此对任意token常数c_i，实数算术下

\[
B(X+c\mathbf1^\top)=B(X)+c\mathbf1^\top.
\]

每个分支忽略通道共同平移，残差旁路将它原样传送。对剩余全部blocks归纳，再用终端head的LayerNorm，得到

\[
\mathcal R(X+c\mathbf1^\top)=\mathcal R(X).
\]

这里R是从当前残差位置到当前去噪flow输出的整个后缀。因此，若联合恢复满足 P_channel B_hat=P_channel B，对当前后缀即可得到相同flow输出。BF16/FP32实现应测量相应舍入底噪。

这不是“归一化把所有尺度误差都消掉”。径向缩放经过残差相加一般不保持该对称。也不要把hidden的通道共同平移混同成像素亮度变化；后者是不同空间中的量。

## 5. 末端head的65个充分统计与64维真实读出

当前head为5120维LayerNorm、时间scale/shift、线性5120→64；64对应16 latent通道乘2×2空间patch。固定时间条件，令

\[
q=W_hD_sPx\in\mathbb R^{64},\qquad
v=\|Px\|^2/5120.
\]

则精确有

\[
\boxed{\mathrm{flow\_patch}(x)=\frac{q}{\sqrt{v+\epsilon}}+W_hb_t+b_h.}
\]

64个投影与1个方差足以求head。此构造有意依赖当前时间条件；当D_s变化时，旧q不能自动成为新条件的充分统计。65统计也不声称最小表示，直接64维flow本身是最终读出。

因此最后一个block的P64 teacher可以继续经过head，得到64维flow标签。候选可在head之后直接拟合flow校正，不必先恢复5120维hidden再投影。读出目标维度减少80倍；总模型计算量仍由attention、FFN及数据移动决定。unpatchify是索引重排，不增加跨token混合。

从第39层开始提供一个严格闭合的联合构造。向前跨下一次self attention时，teacher P会依赖全部teacher K/V；此时应增加所需状态或执行真实后缀，不能自动延用P64逐行闭合。

## 6. 一次联合读出的具体可执行构造

### 便宜主分支

在全部query上计算不含当前probe拟合校正的attention底座a0。通过原生cross得到U0。用现FFNDS的anchor/折叠/选中非线性底座F0计算

\[
B_i^0=U_i^0+g_F\odot F_i^0.
\]

现FFN的在线probe读出由下面的联合读出替换。anchor步仍走原生block，建立其正常缓存。底座可以复用现有anchor资产；本构造没有预先承诺额外缓存减少。

### 精确teacher分支

在相同当前入块X上，以P64全键self attention得到a*_P，然后运行P64 cross和P64完整FFN，得到B*_P。teacher与便宜分支在同一当前状态比较；teacher不是另一个历史Dense轨迹。

### 一次联合拟合

从便宜attention、U0、FFN选中非线性、anchor差分等在线可得量构造Phi_i∈R^r，r≤64；具体特征族留给联合目标检验。下列批处理采用行存约定：B为N×D，Phi为N×r，Theta为r×D。拟合

\[
\Theta=\arg\min_\Theta
\|(B_P^0+\Phi_P\Theta-B_P^*)P_{ch}\|_F^2+\lambda\|\Theta\|_F^2.
\]

输出

\[
\hat B_i=B_i^0+\Phi_i\Theta,
\]

最后才在P覆盖为B*_P。修正位于FFN gate和残差相加之后；若复用pre-gate FFN接口，必须将gate纳入字典与可达空间，不能用G_F^{-1}暗中撤销可能很小或为零的gate。

若位于最后一个block，直接把B目标换成64维flow_patch，在该空间直接比较误差，并在head之后加Phi_iTheta_flow。hidden通道居中投影P_ch不应用于64维flow。这时teacher流程不变，联合目标直接面向采样器实际读取的向量。

在真实中心化block创新恰属于Phi的列空间且Phi_P满列秩时，无正则精确标签可恢复该族，随后由后缀平移对称得到相同flow；一般近似情形继续用H448和最终读出检验。

### 三种实施成本应分别记录

1. **现方案的渐进影子诊断**：A先在P覆盖、wideF的nativeF(P)在teacher输入上，已有teacher block标签可复用。额外保留覆盖前a0_P，跑cheap crossP和cheap F_P（使用已拟合参数），得到未覆盖底座。若再加jointfit，它是第三个拟合，需单列成本。
2. **同时保留两套原生FFN标签**：若另要求nativeF(Uhat_P)来测注意力扰动经过完整FFN的响应，同时保留teacher F(U*_P)，则需要第二次native FFN(P)。完整四臂诊断属于这一情况；已有原生输出可按身份复用。
3. **上述一次联合读出**：便宜主分支在全行保留；一次nativeFFN(P)专供teacher分支；替换独立attention与FFN在线读出，末端只拟合一次。额外teacher crossP、底座P、特征/求解与后处理均计入。P64完整self query数与FFN行数可以分别维持64。

## 7. 恢复对象与最终任务的联系

对候选写入空间U，probe观测映射M_P，最终任务读出线性化R，恢复最终作用的条件是

\[
\ker(M_P|_U)\subseteq\ker(R|_U).
\]

它比逐模块重建全部输出要求更弱。若试图由FFN写入方向V补偿经后续模块输运的attention误差e，则在线性任务读出T下可达到的最小风险为

\[
\min_c\|T(e+V c)\|^2
=\|(I-\Pi_{\mathrm{range}(TV)})Te\|^2.
\]

gate、共享系数、空间特征都应纳入V；“FFN通道宽”本身不决定可达补偿。联合block后处理允许选择更直接的写入空间。

当前flow误差还要经过后续采样。在Euler形式 z_{t+1}=a_t z_t+b_t v_t(z_t) 下，实际误差满足

\[
\Delta z_{t+1}=(a_tI+b_t\bar J_{v_t})\Delta z_t+b_t e_t,
\]

其中e_t是候选在其当前状态上的直接flow误差，割线连接真实与候选状态。对带缓存的方法，持续演化的状态应包括latent、anchors/cache、专家与调度位置。多处误差传到最终视频后的带符号交叉项仍需保留。

## 8. 只补决定路线的观测

优先在已有观察层39、两个专家、现有DEV状态上做四臂局部分叉，固定入块X/cache。把评分从attention写入延长到block出口和64维flow patch，并保留完整误差向量及带符号Gram。

三个竞争解释可由同批分叉区分：

- **状态分布改变**：wide背景使attention本身的创新更易恢复；固定后续算子时收益已经出现。
- **原生后续处理提供补偿**：同一attention误差经过cross/FFN后范数或任务读出显著改变，EA与输入写入的差明确。
- **联合近似发生相互作用**：EF与EA具有稳定的相消/相长方向，或I_AF显著；联合目标应保留这种关系。

第39层的联合flow拟合可以直接复用这批P64/H448标签。它检验新的恢复对象是否改变方法排序。只有当结果要求更早层的全局效应时，再对少数候选执行真实后缀或自身轨迹；这些是任务作用验证，不是重复局部参数搜索。

## 数值核验

`verify_joint_identities.py`：有限LN/cross/FFN/block输运与四臂恒等式最大误差≤5.6e-16；三block后缀平移不变性误差约2e-15。缩放产生非零末端变化。

`verify_terminal_statistics.py`：合成权重采用真实5120→64维度，65统计重建误差3.1e-15；在保持q和方差的有限旋转中hidden相对变化77.3%，head变化4.9e-15。这是说明任务等价方向的代数例子，不是当前Wan误差分布的测量。

两份程序只依赖NumPy和Python标准库，以脚本所在位置写对应JSON，无临时绝对路径依赖。
