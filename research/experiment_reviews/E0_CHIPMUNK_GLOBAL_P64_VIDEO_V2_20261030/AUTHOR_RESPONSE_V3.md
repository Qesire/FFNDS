# 作者回应工程初审 V1

失败finally先查询 `torch.cuda.is_initialized()`；false时直接记录 `UNKNOWN_CUDA_NOT_INITIALIZED`，不会查询memory或初始化CUDA；读取其他异常记UNKNOWN和类型。新增隔离Python子进程真实 `run()` 合成注入：在原scope已通过合成启动门后，真实执行到charged_assets时注入已import但未初始化CUDA的torch替身并抛异常；替身除is_initialized外访问均直接报错；最终实际RUN保存FAILED和UNKNOWN，原异常保留。15项CPU合成通过。其他边界、76GiB和不reset保持不变；无GPU/model新运行。必须对最终两part的协议和全部六job真实独立recheck。
