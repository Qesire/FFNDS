# Chipmunk · W1前/后的真实信息与代价路径（实际源码检查）

**代码版本**：检视Git tree 177fdefc4803e621fd8a02aad491812e9d1ecc94的四个原始文件，分别有本页列出的文件Blob SHA。作者官方仓库为 https://github.com/sandyresearch/chipmunk 。

## MLP调用流程（不是后验oracle）

full step：先调用完整fc1(x)、activation、fc2，再保存postactivation转置、输出cache以及blockmean(mid)。

sparse step：从x算block_mean(x)，在这个缩小token集上运行fc1，和old blockmean mid求绝对差；按分块reduce和topk_indices得到当前该计算块需要处理的indices，再自定义W1/GELU/cache diff、稀疏W2和scatter-add改写历史输出。

函数路径可见于 https://github.com/sandyresearch/chipmunk/blob/master/src/chipmunk/modules/mlp.py ，blob 1415c033df8af9ec388b315036b450a7d2ea87ee，尤其第50–119行。当前代码中有独立的first_n_dense_layers跳过支路及条件 inference_step>=10 才考虑复用已缓存mask，4步生成不能按原默认设计简单推算有效稀疏。

真实kernel分别位于：
- https://github.com/sandyresearch/chipmunk/blob/master/csrc/mlp/csp_mlp_mm1.cu ，blob 7fe07b98462f748af5405e94ce3f430a1f7bf89d
- https://github.com/sandyresearch/chipmunk/blob/master/csrc/mlp/csp_mlp_mm2_and_scatter_add.cu ，blob 5048cb5ac007ff0c93f77abaf4e76a7607c5a68a
- https://github.com/sandyresearch/chipmunk/blob/master/src/chipmunk/ops/mlp.py ，blob a42074fd41477cf666ae298a994baa09f68214c7

W1支持BF16和源中指定的FP8，kernel融合与线程SM分配已显式计费；原HiF4 Q2并不属于直接支持路径。作者PartII给出连续“静态W2 value向量×动态post-GELU系数”的数学解释，但用户当前四步Wan2.2的专家切换意味着跨expert W2与bias不同，不能套简单上一步输出校正公式。

**下一项研究是确定同expert条件下这个结构何时出现，而不是立即移植kernel。** 需要将真实新旧post-Q2系数差、Q2 shared-scale组依赖、低成本前置信号、cache容量和打包时间同时绑定；仍允许不同于缓存delta的其它机制。
