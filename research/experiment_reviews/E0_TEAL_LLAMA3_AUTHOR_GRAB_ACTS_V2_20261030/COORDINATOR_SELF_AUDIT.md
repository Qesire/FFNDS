# 协调者核对：校准新 V2 有限范围

协调者 `/root` 已核对 V1 的真实 EXIT1、26.563秒与零模型/forward/sort/hist账；失败由CUDA条件导入下缺少einops触发，旧源码、环境、协议和输出保持原样。V2使用独立新环境、文件、scope ID与输出，绑定旧失败双收据；不将其写作旧范围的自动重跑。

新环境补齐einops，完整可见环境pip check通过；CPU直接导入FA2 padding/interface/native扩展与模拟Transformers FA2条件分支的工程检查覆盖了旧预检缺口。作者及FA2导入移至16GB输入哈希之前。实际进程精确使用`-I -B`，源码/模型/数据与三份真实独立评议必须匹配，随后由root唯一登记ACTIVE和核对单张空闲A100。八项测试和独立精确APPROVE是软件前提，A100上的FA2和模型正确性仍由实际范围测量。

原作者300条前缀、10×2048、32层、64hist/128sort/CPU histogram及原额外observer预算不变。32GiB host RSS/16GiB输出采用采样监控，不宣称瞬时硬配额；实际GPU kernel数与内部临时分配为UNKNOWN。该范围没有文本生成、PPL、训练或速度结论；完成或失败均保留实际RUN/EXIT，不自动继续另一次模型尝试。当前用户会话最多同时4GPU，本范围1GPU。
