# 当前方法的数学合同与时间因果图

2026-11-28。以研究分支 `72280c1a264bf018343e746a2df2993dafe89149` 为证据起点，整理冻结 **BF16 wide8192 + high OUT-ABS + low mass-centered OUT-D**。本轮只完成源码与实际小资产读取、推导和CPU数学核验；新模型前向、GPU作业、成片与评分均为0，方法默认配置不变。

## 阅读入口

| 需要理解或核对的对象 | 资产 |
|---|---|
| 方法整体、数学覆盖、跨模块和跨步误差 | [METHOD_AND_CAUSAL_GRAPH.md](METHOD_AND_CAUSAL_GRAPH.md) |
| FFN实际余项、16模式可辨识性、半幅核、P64点边 | [ffn/FFN_FORMAL.md](ffn/FFN_FORMAL.md) |
| high/low注意力、ridge、动态遗漏项、未来读出状态 | [attention/ATTENTION_FORMAL.md](attention/ATTENTION_FORMAL.md) |
| HHLL执行顺序、原生块、缓存生命周期、实际step时间 | [graph/CAUSAL_EXECUTION_GRAPH.md](graph/CAUSAL_EXECUTION_GRAPH.md) |
| 机器可读的覆盖与缺口 | [METHOD_CONTRACT.json](METHOD_CONTRACT.json) |
| 完整4×40双图，gzip格式 | [graph/CAUSAL_GRAPH.json.gz](graph/CAUSAL_GRAPH.json.gz) |
| 图输入、生成和验证 | [GRAPH_INPUTS.json](graph/GRAPH_INPUTS.json) · [build_graph.py](graph/build_graph.py) · [verify_graph.py](graph/verify_graph.py) |
| 各交付字节身份与复算入口 | [MANIFEST.json](MANIFEST.json) |

## 两套图分别代表什么

冻结图包含5,561个算子边界节点、12,668条信息依赖边；已实测combined_graph09图包含10,801个边界节点、26,268条信息依赖边。两者均展开4步×40层。节点数不等于GPU kernel调用数；`tau`来自源码提交顺序，不是GPU时间戳。微节点时长保持null，真实时间只附到已有收据中的四步宏节点。

两图检查当前全key依赖、P64教师/拟合输入、原始坐标、同expert/同层缓存、无未来读取以及write→take→last_read→borrow_end→下一次写入顺序。40个动态层槽与80组expert/layer离线常量独立计数。图没有把合法调度自动当作速度优化；已测09的整请求耗时增加22.55%，仍按最近结果保持原基线。

最新pilot没有公开数值sigma表。这一项保留`OPEN_BINDING`，源代码已确认的step-distill更新语义与该数值缺口分开。全图包含跨步缓存；scheduler自身没有UniPC多步历史。

## CPU复算

以下命令从本目录运行。使用Python 3.11或更高版本；两个局部数学脚本及传播见证需NumPy，图生成/验证只使用标准库。命令不调用模型或GPU。生成的复算JSON写入临时目录，保留本次已发布原始证书。

```bash
python ffn/verify_ffn_formal.py --output /tmp/ffnds_ffn_formal_recomputed.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python attention/verify_attention_causal.py --cal-assets ../graph_state_theory/CAL_ASSETS.json --out /tmp/ffnds_attention_formal_recomputed.json
python verify_causal_transport.py --out /tmp/ffnds_causal_transport_recomputed.json
python graph/build_graph.py --output /tmp/ffnds_causal_graph_recomputed.json
python graph/verify_graph.py --graph /tmp/ffnds_causal_graph_recomputed.json --output /tmp/ffnds_causal_graph_verification.json
```

attention脚本从`graph_state_theory/CAL_ASSETS.json`中的原始base64字节重新校验3份资产SHA后加载；该包已在同一研究分支。也可传`--cal-root`使用实际解包目录。FFN的真实observer元数据和两套数学脚本的source manifests随本目录交付。默认复算会明确记录metadata模式，`--source-root`仅在具有对应源文件布局时额外验证源文件指纹，不能把读取已存SHA说成重新读取源文件。

展开图可直接解包，或者由小输入确定性生成：

```bash
python -c 'import gzip,pathlib; pathlib.Path("/tmp/ffnds_causal_graph_unpacked.json").write_bytes(gzip.decompress(pathlib.Path("graph/CAUSAL_GRAPH.json.gz").read_bytes()))'
```

完整JSON为6,908,206 bytes，SHA256为`1eec6bd46f4ee7d2347e737841dcb48deae19deedb04a6edd26f2d0e9d428683`。图输入与生成器移动到独立目录后已复现相同字节。gzip使用确定性mtime=0；文件身份见MANIFEST。

## 证据边界

- 真实P64几何：42条同帧边，点值加边差秩仍64，覆盖19/21 latent帧。
- 真实CAL：high/low合成投影rank31、条件数1.068/1.066；它们不代表当前X_P或完整缓存状态的谱。
- 合成代数：FFN16项、attention7项，以及注意力微分/扩展状态传播核验。其误差量是公式见证，不是新Wan误差测量。
- 既有GPU：源于`5f83e1c590b6cb93b81dbe88a21101ca1b9260a6`五源pilot；参照为冻结centered wide8192，不是Dense。统计范围仍为固定stress/median筛查。

根目录方法交接仍承担其原FFN默认范围；本文件登记当前attention研究分支的数学解释与执行边界，不修改历史回执或重新授予默认升级。
