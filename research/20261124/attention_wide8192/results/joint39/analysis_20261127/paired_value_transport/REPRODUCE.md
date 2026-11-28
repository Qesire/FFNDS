# CPU 复算与参考实现

日期：2026-11-27。Python 3 与 NumPy 即可执行下列三个独立入口。所有输入均在相邻目录中；脚本按自身路径寻找输入及写出结果。

```bash
python evidence/recompute_confidence.py
python value_transport/verify_value_pair_transport.py
python discriminator/verify_risk_conditions.py
```

`evidence/` 复算真实归档指标与 Gram，并保留来源、哈希和统计口径。`value_transport/` 提供每头 value 配对识别、centered 状态编解码、固定 P64 校正以及论文定义的 SparsePR v2 完整通道恢复函数；核验使用合成 FP64 attention 和真实 Wan 尺寸的静态资源计算。`discriminator/` 检查信息风险、P64 后更新和非线性后继作用的恒等式。

三个目录的 `*_VERIFIED.json` 分别记录真实旧数据复算、合成代数核验与静态算量。新增 Wan/GPU 执行数为 0；原始 full-S128 学校面板未在本次环境中取得，故没有新的 SparsePR v2 实测胜负。参考恢复函数只实现给定 S、P 与 coverage 权重后的输出恢复，不包含论文的响应分组、路由或 probe 选择。

候选的固定设置与范围见 [METHOD_SNAPSHOT.json](METHOD_SNAPSHOT.json)，整体判断见 [FRONTIER_MECHANISM_AND_CONFIDENCE.md](FRONTIER_MECHANISM_AND_CONFIDENCE.md)。
