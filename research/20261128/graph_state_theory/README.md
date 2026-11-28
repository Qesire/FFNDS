# 从计算图推导当前方法的极限、表示与缓存

2026-11-28。当前研究阶段为**理论推导与 CPU 代数核验**。依据用户最新方向，先分析当前方法可达到的资源边界与新表示；本轮没有启动新的消融、生成、评分或计时。

主文：[THEORY_AND_GRAPH_REFORMULATION.md](THEORY_AND_GRAPH_REFORMULATION.md)。

在保留两个原生锚点、QKV/输出投影和 cross-attention 的图范围内，当前方法已利用约75%的乐观可删主算量。新的具体构造包括：low anchor 单次互补分区归约，P64 前置的完整块流式调度，FFN 规范变换及因子化输入缓存，以及 attention 的联合历史/动态读出表示。详见主文中的适用条件、成本和恢复后误差等式。

## 资产

| 目录/文件 | 内容 |
|---|---|
| `limits/` | 真实形状的完整周期算量、最新时间敏感度、状态/重算关系 |
| `ffn/` | 真实GLS结构、恢复后压缩误差、16模式规范变换、因子化缓存 |
| `attention/` | 单遍分区、C/A联合状态、动态读出、真实CAL矩阵分析 |
| `verify_block_streaming.py` | 完整块P64先行与LayerNorm后线性读出的CPU见证 |
| `CAL_ASSETS.json` | 三个冻结CAL资产的精确字节与SHA-256，仅供本地CPU复算 |
| `RESEARCH_STATUS.json` | 证据提交、当前方向、结果身份和未执行事项 |
| `MANIFEST.json` | 本目录文件哈希，不包含自身 |

## CPU 复算

依赖 Python 3 与 NumPy。以下命令从仓库根运行；`analysis_20261128/SPEED_MEMORY_RECOMPUTED.json` 是既有108次请求的复算结果。输出写入单独临时目录，保留本文归档结果。

```bash
theory_dir=research/20261128/graph_state_theory
repro_dir=$(mktemp -d)
python "$theory_dir/materialize_cal_assets.py" --out "$repro_dir/cal"
OPENBLAS_NUM_THREADS=1 python "$theory_dir/limits/recompute_limits.py" --timing research/20261127/centered_medium_eval/analysis_20261128/SPEED_MEMORY_RECOMPUTED.json --out "$repro_dir/LIMITS_RECOMPUTED.json"
OPENBLAS_NUM_THREADS=1 python "$theory_dir/ffn/verify_ffn_graph_rewrites.py" --out "$repro_dir/FFN_GRAPH_REWRITES_VERIFIED.json"
OPENBLAS_NUM_THREADS=1 python "$theory_dir/attention/verify_attention_graph.py" --cal-root "$repro_dir/cal" --out "$repro_dir/ATTENTION_GRAPH_VERIFIED.json"
OPENBLAS_NUM_THREADS=1 python "$theory_dir/verify_block_streaming.py" --cal-dir "$repro_dir/cal/calibration" --pv "$repro_dir/cal/plan_assets/bases/P_v.npy" --out "$repro_dir/BLOCK_STREAMING_VERIFIED.json"
```

CAL来源和运行形状是真实资产。随机hidden、权重与状态是明确标记的合成数值见证，用于验证推导和代码操作一致；它们不构成Wan质量或GPU性能结果。有限精度缺陷与实数等价分别记录。不同BLAS环境的浮点尾差可有变化。

FFN脚本会在旧本地来源路径存在时附加源码哈希；换机器复算时，该辅助来源字典可能为空。数学核验与预算独立于该字典，归档的源码身份见结果和 `attention/SOURCE_IDENTITIES.json`，当前实现身份由冻结方法绑定固定。
