# Wan I2V 14B W4A4 MXFP4 量化：代码路径全链路分析

> 项目: `wan-quantization-study/Wan2.2-I2V-A14B-W4A4`
> 目标模型: Wan2.2-I2V-A14B (14B, 40 layers, dim=5120, 双模型)
> 量化方案: MXFP4 W4A4 + SVDQuant 低秩分解 + GPTQ 误差补偿
> 分析方式: 严格按照源码执行路径追踪

---

## 一、总览：PTQ pipeline 5 阶段

入口文件：`tools/ptq_wan_i2v_opens2v_svdquant.py` (871行)

```
[Stage 1] 加载校准样本 (OpenS2V-Eval 数据集)
    ↓
[Stage 2] 构建 Wan I2V pipeline（high_noise_model + low_noise_model）
    ↓
[Stage 3] 收集激活统计 (per-channel absmax)，同时可选在线收集 GPTQ Hessian
    ↓
[Stage 4] 逐模块生成 SVDQuant PTQ state (Smooth + SVD + GPTQ-MXFP4)
    ↓
[Stage 5] 输出 ptq_stats.pt (merged artifact)
```

---

## 二、核心问题 1：DiT 激活的离群值导致量化误差爆炸

### 问题表现

Wan DiT 每层有 10 个 Linear（4 self-attn + 4 cross-attn + 2 FFN）。不同 timestep 下的输入激活分布显著不同，且存在**通道级离群值**（channel-wise outlier）——某些通道的 absmax 是其他通道的 10-100 倍。

### 解决方案：SmoothQuant 风格的通道平滑

**代码路径：** `quant/wan_smooth.py:392-406` → `_smooth_scale_from_act_and_weight`

```python
# wan_smooth.py:392-406
def _smooth_scale_from_act_and_weight(act_s, w_s, alpha=0.5, eps=1e-8):
    """SmoothQuant-like scale: act^alpha / weight^(1-alpha)."""
    scale = act_s.pow(alpha) / w_s.pow(1.0 - alpha)
    return scale.clamp_min(eps)
```

**做了什么：**

1. **收集激活统计** (`wan_smooth.py:116-196` `LinearActStatCollector`):
   - 在每个 `nn.Linear` 上注册 `forward_pre_hook`
   - 校准阶段：在完整的 denoise trajectory 上前向传播，记录每个 Linear 输入的 **per-channel absmax**
   - 高噪声和低噪声模型分别统计

2. **计算平滑 scale** (`_smooth_scale_from_act_and_weight`):
   - `act_s = max|X_input_channel[j]|`  (激活通道 j 的最大绝对值)
   - `w_s = max|W[out][j]|`  (权重输入通道 j 的最大绝对值)
   - `scale[j] = act_s[j]^α / w_s[j]^(1-α)`
   - 目标：将激活的 outlier 部分迁移到权重

3. **应用转换** (`wan_smooth.py:409-419` `_build_smoothed_weight`):
   - `W_s[:, j] = W[:, j] * scale[j]`  (权重放大离群通道)
   - 推理时: `x_smooth[:, j] = x[:, j] / scale[j]`  (激活缩小离群通道)
   - 数学等价: `x @ W = (x / scale) @ (W * scale)`

**为什么 Smooth 对量化有帮助：**
- 量化误差主要由 absmax 决定 scale
- 激活通道的 outlier 被缩小 → 量化范围变窄 → 更多有效格点 → 更小误差
- 权重被相应放大，但权重是提前离线量化的，放大不会增加额外的推理成本

---

## 三、核心问题 2：直接量化 14B 模型精度损失大

### 问题表现

直接对平滑后的权重做 MXFP4 RTN (Round-to-Nearest) 量化，精度有限。FP4 E2M1 只有 8 个正数值 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，表示精度远比 FP16 差。

### 解决方案：SVDQuant 低秩分解

**代码路径：** `quant/quant_linear.py:500-576` → `QuantLinearWithBranch.from_smoothed_weight_and_bias`

**核心公式：**
```
W ≈ Q(W_main) + W_branch
W_main = W - W_branch
```

**执行步骤（quant_linear.py:642-669)：**

1. **SVD 分解** (`quant_linear.py:748-766` `LowRankWeightBranch.from_weight`):
```python
# 对平滑后的权重 W_s 做截断 SVD
U, S, Vh = torch.linalg.svd(W_s, full_matrices=False)
sr_sqrt = torch.sqrt(S[:r])
# A = S_r^{1/2} V_r^T  →  Linear(in, r)
# B = U_r S_r^{1/2}    →  Linear(r, out)
W_branch = B @ A       # [out, r] @ [r, in] = [out, in]
```

2. **主分支残差**:
```python
W_main = W_s - W_branch   # 用低秩分支减掉大部分能量
```

3. **仅量化主分支残差**:
```python
Q(W_main)  # MXFP4 量化
```

4. **推理时** (`quant_linear.py:707-714`):
```python
def forward(self, x):
    x_smooth = x / smooth_scale      # 激活平滑
    x_main = QDQ_MXFP4(x_smooth)     # 主分支：MXFP4 Q→D→Q
    out = self.dequant_weight(x_main) + self.branch(x_smooth)  # W4A4 + BF16低秩
    return out
```

**为什么 SVDQuant 对量化有帮助：**
- `W_branch` 提取了权重的大能量成分（低秩结构），以 BF16 保留——零量化误差
- `W_main = W - W_branch` 的数值范围显著缩小 → 量化误差大幅下降
- 低秩分支 `in→r→out` 的参数开销较小：rank=16 时，对 `[13824, 5120]` 的 FFN 权重，分支参数量为 `13824×16 + 16×5120 ≈ 303K`，仅为主分支的 0.4%

---

## 四、核心问题 3：RTN 量化未充分利用校准信息

### 问题表现

RTN 逐组量化：每组 (group_size=32，即 MXFP4 标准的 32 个元素) 独立计算 scale，然后 round 到最近 FP4 值。它不知道列与列之间的误差如何传播。

### 解决方案：GPTQ-MXFP4 误差补偿

**代码路径：** `quant/mxfp4.py:252-393` → `quantize_linear_weight_groupwise_gptq`

**算法流程（逐列处理 + Cholesky 误差传播）：**

1. **Hessian 累积** (`quant/gptq.py` `GPTQHessianCollector`):
```python
# 校准过程中，每个 Linear 的输入激活 x 累积 Hessian
H.add_(x.T @ x)   # [in_features, in_features]
# 使用平滑+MXFP4 QDQ 后的激活 → 匹配推理时的实际输入分布
x_main = QDQ_MXFP4_activation(x / smooth_scale)  # 先平滑再量化
collector.add_batch(x_main)                       # 在线累积 H
```

2. **Cholesky 分解** (`mxfp4.py:314-327`):
```python
h = h + damp * diag_mean      # 正则化（damp_percent=0.01）
L = cholesky(h)                # 下三角分解
Hinv = cholesky_inverse(L)     # 逆
Hinv = cholesky(Hinv, upper=True)  # 上三角 Cholesky of inverse
```

3. **GPTQ 列补偿** (`mxfp4.py:339-369`):
```python
for col in range(in_features):
    # 3a. 计算当前列的 FP4 E2M1 scale（E8M0 格式）
    group_id = col // 32
    block_scale = compute_mxfp4_scale(w_groups[:, group_id])  # PoT E8M0

    # 3b. 量化 + 反量化当前列
    codes = fp4_e2m1_encode(w_col / block_scale)       # → FP4 codes
    q_col = fp4_e2m1_decode(codes) * block_scale       # → dequantized

    # 3c. 误差传播：未量化列减去当前列的量化误差
    err = (w_col - q_col) / Hinv[col, col]             # 缩放误差
    w[:, col:] -= err[:, None] * Hinv[col, col:][None, :]  # Hessian-weighted 传播
```

**为什么 GPTQ 对量化有帮助：**
- RTN 的量化误差主要集中在每一组：每个 group 独立做 round，误差不会传递
- GPTQ 根据 Hessian 逆矩阵，将当前列的量化误差**按最优比例传播**到尚未量化的列上
- 本质上是在做 `min_w ‖WX - Q(W)X‖²` 的二阶近似最小化，等价于逐列最小化量化损失
- 结合 SVDQuant：GPTQ 作用于 `W_main`（低秩残差已减去），进一步缩小误差

---

## 五、核心问题 4：激活量化比权重量化更困难

### 问题表现

权重可以离线量化——每个 Linear 的 weight 固定不变。但激活每次 forward 都变化：
- 同一样本在不同 denoise timestep 下分布不同
- 不同样本之间分布也不同

### 解决方案 1：动态 per-group 量化

**代码路径：** `quant/quant_linear.py:86-148` → `_qdq_mxfp4_activation_groupwise`

```python
def _qdq_mxfp4_activation_groupwise(x, group_size=32, ...):
    # 将输入 reshape 为 [..., num_groups, 32]
    xg = x.reshape(-1, num_groups, 32)

    # 每个 group 独立计算 MXFP4 scale
    amax = xg.abs().amax(dim=-1)               # per-group absmax
    block_scale_exp = ceil(log2(amax / 6.0))   # safe_ceil 规则
    scale_code = block_scale_exp + 127         # E8M0 编码
    block_scale = 2^(scale_code - 127)          # E8M0 解码

    # 量化+反量化（QDQ）
    codes = fp4_e2m1_encode(xg / block_scale)  # FP4 编码
    x_dq = fp4_e2m1_decode(codes) * block_scale  # 反量化
```

**问题**：这种动态量化每次 forward 都要计算 per-group scale，有 overhead。

### 解决方案 2：per-module per-expert per-timestep 激活量化策略

**代码路径：** `quant/quant_linear.py:279-326` → `get_current_act_quant_params`

```python
def get_current_act_quant_params(self):
    """支持三级策略查找：module → expert → timestep_bin"""
    # 1. 查 module-specific 策略
    module_policy = policy.get(self.module_name)

    # 2. 查 high/low expert-specific 策略
    expert_policy = module_policy.get(self.act_policy_context["expert_id"])  # "high" or "low"

    # 3. 查当前 timestep_bin 的策略
    timestep_policy = expert_policy.get(timestep_bin)  # 如 "t_0", "t_100", ...

    # 4. 应用 clip_ratio, exp_offset, scale_method
    clip_ratio = timestep_policy["clip_ratio"]
    exp_offset = timestep_policy["exp_offset"]
    scale_method = timestep_policy["scale_method"]
```

这允许为不同的 timestep 区间设置不同的激活量化参数，解决 "t=900 和 t=100 时激活差异巨大但只能共享同一量化参数" 的问题。

---

## 六、核心问题 5：双模型独立量化

### 问题表现

Wan I2V 14B 使用两个独立模型：high_noise_model (t ≥ boundary, boundary=900) 和 low_noise_model (t < 900)。两个模型的权重完全独立、分布不同，需要分别进行 PTQ。

### 解决方案

PTQ pipeline 对两个模型分别执行全部步骤：

```python
# ptq_wan_i2v_opens2v_svdquant.py
# Stage 3: 分别收集激活统计
act_stats["high_noise_model"]  # 高噪声模型在 t≥900 时的激活分布
act_stats["low_noise_model"]   # 低噪声模型在 t<900 时的激活分布

# Stage 3b: 分别收集 GPTQ Hessian（仅对指定 model_name 的模型）
high_hessians = _collect_gptq_hessians(pipe, calib_samples, target_model_name="high_noise_model")
low_hessians =  _collect_gptq_hessians(pipe, calib_samples, target_model_name="low_noise_model")

# Stage 4: 分别生成 PTQ state
high_state = _build_one_model_ptq_state(model=high_model, model_name="high_noise_model", ...)
low_state =  _build_one_model_ptq_state(model=low_model, model_name="low_noise_model", ...)

# Stage 5: 合并为单个 artifact
torch.save({"high_noise_model": high_state, "low_noise_model": low_state}, "ptq_stats.pt")
```

**推理时的模型切换**（与原始 Wan I2V 一致的 `_prepare_model_for_timestep` 逻辑）：
- t ≥ boundary → 使用 high_noise_model（含 QuantLinearWithBranch）
- t < boundary → 使用 low_noise_model（含 QuantLinearWithBranch）

---

## 七、MXFP4 格式细节（源码 mxfp4.py）

### E2M1 FP4 编码（mxfp4.py:16-19, 63-73）

8 个可表示正数值：`{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`

```python
# FP4 E2M1 编码：clamp → 阈值 → bucketize → sign
_FP4_POS_THRESHOLDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
# x ∈ [0, 0.25) → 0.0, [0.25, 0.75) → 0.5, ...
# x ∈ [5.0, 6.0] → 6.0, 负数同阶梯
```

### E8M0 Scale 编码（mxfp4.py:76-90）

```python
block_scale = 2^(code - 127)   # code ∈ [0, 254], 255 = NaN
```

### OCP Floor vs Safe Ceil 对比

```python
# ocp_floor: floor(log2(amax)) - 2
#   → scale 偏小，amax 可能被 clip
# safe_ceil: ceil(log2(amax / 6.0))
#   → scale 偏大，保证 amax 可表示（≤ 6.0 × scale）
```

激活量化默认用 `safe_ceil`（保证不 clip），权重量化默认用 `ocp_floor`（更精细）。

---

## 八、LongLive 2.0（NVFP4 W4A4 推理系统）对照

项目：`wan-quantization-study/LongLive/`

### 核心差异

| 维度 | Wan2.2 W4A4 (PTQ) | LongLive 2.0 (训练+推理) |
|------|-------------------|------------------------|
| **目标模型** | Wan2.2-I2V-A14B | Wan2.2-TI2V-5B |
| **量化方式** | MXFP4 PTQ | NVFP4 全训练 |
| **Block Size** | 32 (MXFP4标准) | 16 (NVFP4标准) |
| **Scale 格式** | E8M0 (2^E) | E4M3 (更精细) |
| **训练** | 仅 PTQ | AR teacher-forcing + DMD 蒸馏 |
| **推理** | BF16 反量化 + F.linear | NVFP4 fused Triton kernel |
| **保持精度的层** | SVDQuant 低秩分支 BF16 | embedding, time_proj, head.head, norms 保持 BF16 |

### LongLive 量化配置 (quant.py:44-58)

```python
# 不量化的模块列表
DEFAULT_GENERATOR_FILTERED_MODULES = [
    "text_embedding.0",    # T5 投影第一层
    "text_embedding.2",    # T5 投影第三层
    "patch_embedding",     # 3D Conv patch embedding
    "time_projection.1",   # 时间投影
    "time_embedding.0",    # 时间嵌入
    "time_embedding.2",
    "head.head",           # 输出头 Linear
    "head.modulation",     # 输出头调制参数
    "re:.*norm_k$",        # QK RMSNorm
    "re:.*norm_q$",
    "re:.*norm1$",         # Block 归一化层
    "re:.*norm2$",
    "re:.*norm3$",
]
```

这个清单对 Wan W4A4 项目也有参考价值——量化时这些模块的精度更敏感。

---

## 九、量化全链路速查表

| 阶段 | 解决什么问题 | 方法 | 关键文件:行号 |
|------|-------------|------|-------------|
| 激活平滑 | 通道级 activation outlier → 量化范围过大 | SmoothQuant: `scale = act^α / wgt^(1-α)` | `wan_smooth.py:392-419` |
| 低秩分解 | 直接量化 14B 权重精度损失大 | SVDQuant: `W = Q(W_main) + W_branch` | `quant_linear.py:500-670` |
| 误差补偿 | RTN 未利用二阶梯度和列间相关 | GPTQ-MXFP4: Cholesky + column-wise error compensation | `mxfp4.py:252-393` |
| 激活量化 | 激活每次 forward 分布不同 | 动态 per-group MXFP4 QDQ + per-timestep policy | `quant_linear.py:86-148, 279-326` |
| 双模型 | high/low noise 权重独立 | 分别执行全部 PTQ 阶段 | `ptq_wan_i2v_opens2v_svdquant.py:757-798` |
| 精度评估 | 量化误差是否可接受 | block/linear 级 BF16 vs Quant 输出对比 (MSE/cosine) | `eval_hooks.py:72-123` |
