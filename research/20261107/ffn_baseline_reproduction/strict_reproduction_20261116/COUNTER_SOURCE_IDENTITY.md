# Counter接口检查的原文身份修正

独立审阅实际复跑utility时发现：初稿固定了旧审计工作副本的SHA256，而不是作者Git blob的原始字节。差异只有文件末尾额外一个换行；类的源码、AST和所有调用语义相同。

| 文件身份 | bytes | SHA256 | Git blob SHA |
| --- | ---: | --- | --- |
| 初稿使用的旧工作副本 | 3361 | b58bff2c407c43795ba7ad488e5da7efbf69c440dd2f38ef8dcb73c1392b2d79 | bbc3e51089425232c5f9a1856797cd5d863377d4 |
| 固定作者commit中的实际原文 | 3360 | cb9ffab57832ce69141ecd914fbb3f2c873c26bdca31eee1f7e2b41c01764288 | 6a870f07cdf05482667b6a65fe015ccd035bd007 |

作者commit：`b5e8d1c3da6bddc7ca9007c48e91b82087c11490`；文件：`src/chipmunk/util/layer_counter.py`。

修订后utility严格要求第二行的原文字节，重新执行并更新`UPSTREAM_COUNTER_TRACE.json`，三种调用序列的错配数量仍为1、1、120。审阅中在正确原文上触发的checksum异常属于初稿交付错误，已修正；没有模型/CUDA运行，也没有以更换校验值掩盖源码算法变化。旧工作trace仅保留为审阅过程记录，不用于当前39/39作者原文一致性声明。
