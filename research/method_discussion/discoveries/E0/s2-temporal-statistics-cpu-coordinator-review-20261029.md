# S2 时间统计 CPU 协调者审阅

协调者完整阅读上游统计核、producer及全21τ读取/原生写回neutral边界。静态启动发现两处 report.update 的字典键语法错误，均在任何真实数据运行前修复；新增producer导入/投影测试。8项合成测试实际exit0：Parseval、Nyquist、低高频、rank、anti-phase复相干及零AC缺测。独立Agent服务额度耗尽，本次没有同行SUPPORT票。

只在已采十图×160包的原始BF16 Dense teacher tensor上做T1–T4统计，全部21τ及去τ0两视图，每包全文件SHA/9raw SHA/row校验及原u byte neutral；source先160位置等权再十图等权。不是student/实际flow对应，也不新调用模型或拟合。频率是latent槽单位；8固定xy的空间覆盖保持原身份。compare()具备相位接口，但本producer不读取候选tensor，T6 candidate频谱尚未测。

新有限scope，四个CPU线程、隐藏CUDA，最多7200秒、32GiB输入流/256MiB新增输出。先绑定实际已闭合采集和CPU父协议；一次运行失败保留，不自动重试。
