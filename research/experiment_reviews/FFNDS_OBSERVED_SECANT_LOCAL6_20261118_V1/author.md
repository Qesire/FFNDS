# LOCAL6 作者自审

本次研究比较当前 64 行观测能否辨识对同源另 256 行有用的 FFN 修正。观测提供可迁移信息时，固定在线分支应在留出的开发来源上胜过各自静态控制，且接近可执行 query oracle。观测不足时，有信号的特征方向可能在 64 行上接近零，连续及可执行 oracle 胜过静态，固定在线却无法复现增益。连续 oracle 也缺少增益时，本轮表示的局部能力有限。连续与可执行 oracle 之间的落差用于定位编译或量化问题。

- 范围为旧原始 BINDINGS 对应的 CAL11、DEV11 和 high/low 各 04、19、34 层，264 个已存在的 P512 文件。共同矩阵为过去接受的六份 BF16 L64。逐端点核 SHA256、时间步、真实坐标、source/image-cluster、checkpoint、gate 字节及归档 producer。上传包绑定计划、代码、Python、LightX2V FFN/gate 源码以及上轮归档身份。
- 数学轨以冻结 BF16 值提升至 FP64。部署轨使用 CUDA A100 BF16 matmul、真正 64 行 `MMWeight.apply → GELU(tanh) → MMWeight.apply`、含 bias 的同权重路径。将真实当前 BF16 gate 喂给原生 `post_process` 并核对一次写回，残差加数是为见证选用的 P512 输入。模式模板和编译使用关闭 TF32 的 FP32，解为 FP64。共同门控损失采用当前 gate 的实数提升。
- 每个来源以 global ID 固定切分，换存储次序只改变本地索引。A/B 互为 probe 池和共同 query。64 行 BF16 observer 只接收选中输入。最终标签为已有 full-M 捕获 BF16 Y。结构筛选只用 CAL，CAL 留一来源时冻结传感器或模式基，重新拟合 prior 并用实际舍入输出计分。两来源解释的主要判别依赖各自静态、连续 oracle 和已编译 oracle。
- panel-P64 以同折固定 64 行计算 FP64 GELU 隐层差的 RMS，选 K=5120 并保留实际符号。旧视频的严格 Global-P64 使用不同的全 raster 固定 token、BF16 范式和原生 packing。此次的 panel-P64 参与 FP64 方向分析。
- 一项 Slurm 作业，1 张 A100 80GB、4 CPU、64 GiB 主存、4 小时 walltime，无自动重排队，0 D4、0 NFE。逐位置释放 GPU 工作集，GPU allocator 限 78 GiB，归档读取不复制 57 GB 权重。PCG 上限每迭代约 0.145 TFLOP、66 次拟合合计最多 0.77 PFLOP，并分别记录源收敛和耗时。一并计入 CAL 模式编译及读出筛选。半途故障的进度、site 参数、作业退出证据保留。
- 开发准入依据 11 来源均值、来源胜数、最差相对退化、共同 A 和 BF16 合同。固定在线读出居预定主位。科学结论限制在同端点 P512 与六个位置，首轮的见证不覆盖完整视频、学生轨迹或物理性能。
