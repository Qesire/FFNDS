# 0392 六位点预检修订自审

两个 V1 作业在历史 `wan_ffn_port.py` 预检处退出。`bind_native_ffn` 是 `contextmanager` 装饰的函数，Python `inspect.getfile` 取得标准库装饰器代码路径。两个实际作业皆在构建视频推理器前退出，未产生 NFE 记录。`173973`、`173974` 的 `ATTEMPT_EXIT` 原样归档，静态臂 V1 没有提交。

V2 路径通过 `inspect.unwrap` 定位真正的函数定义。它仍要求实际函数文件、历史 `bundle_sha256` 和本次冻结 `bundle_sha256` 同时匹配。模型、数值路径、适用位置、对照及三个有效视频的总预算沿用上一版。两个零计算预检退出和未提交 scope 原样保留。每项一张卡，同一时刻最多两张，没有自动重跑完整视频。[范围问题与对照](scope6_author.md)
