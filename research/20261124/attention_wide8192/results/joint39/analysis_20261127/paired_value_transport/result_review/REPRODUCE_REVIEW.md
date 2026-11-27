# 新结果的独立复算

在本目录使用Python 3与NumPy运行：

```bash
python recompute_recovery.py ../experiments RECOVERY_RECOMPUTED.json
python recompute_e2e.py ../experiments E2E_AUDIT_VERIFIED.json
python verify_operator_semantics.py
```

所有输入均来自相邻experiments中的固定学校收据与归档源码。局部复算重建216个Gram的指标与780个归档数值；E2E复算检查已发布误差的一致性、配对差、聚合与极化恒等式；算子核验检查执行源码SHA256、primal/dual ridge以及配对矩阵的正规方程。

这些程序不运行Wan、Torch内核或GPU，也不重新提取MP4/张量指标。原片与原始完整张量仍保存在本次环境未取得的学校资产中；其公开数值、哈希与提取源码可在experiments核对。新增模型/GPU执行为0。
