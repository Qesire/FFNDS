# 协调者启动核对：全32层生成有限诊断

协调者 `/root` 已核对真实校准V2的EXIT0与64个histogram文件、原作者checkpoint转换及独立复验的四份收据。原Llama3为32层、D4096、H14336，SparseQKV的4096/1024/1024分段和列主序数学经源级独立核对；完整FP16原子累加与编译行为仍是本次实际问题。Dense、作者patch后的真正零阈值、作者uniform50七投影是三个独立模型；原noninteractive生成遇EOS继续64个token。

先注册的CPU阶段完整hash64个小hist，实际EXIT0后才能进入单A100 GPU阶段。GPU阶段再校验输入并做两遍checkpoint SHA、三次模型加载、256次原hist加载；预算9次生成各64 token（9prefill/567decode），另10个teacher-forced和4个observer，共590个逻辑模型调用。编译/图捕获/autotune的物理kernel次数和mmap读取仍UNKNOWN，不将逻辑预算作实测物理次数。

eager math-only teacher-forced neutral要求有限、relative L2≤0.03且top1相同，失败即停；该控制不证明compiled logits或逐元素KV完全相同。三臂KV容量相同，三模型同时驻留、逐token CUDA event插桩使计时仅为开发诊断，不作原论文独占单模型速度、PPL、FFN-only或Wan质量结论。

采用真实不同Agent初审REVISE、作者修订和精确recheck APPROVE；协调者启动前核源commit字节、原作者source、runtime、worker/profile及CPU/GPU两份spec，GPU使用UUID及共享锁。新范围最多单卡，当前会话同时最多4卡。失败/超时保留实际扣账，不自动重试；输入、源和本范围代码冻结后保持不变。
