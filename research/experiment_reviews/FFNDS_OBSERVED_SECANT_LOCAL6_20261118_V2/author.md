# LOCAL6 启动修订作者自审

[V1 作者自审](../FFNDS_OBSERVED_SECANT_LOCAL6_20261118_V1/author.md)确定了问题、反例、范围、数据和科学判据。本次允许同一配置重新启动。Slurm `173585` 在 `native_runtime()` 的源码路径见证处退出，原生入口 `GET_DTYPE` 经 `functools.lru_cache` 包装，`inspect.getfile` 拒绝 wrapper 类型。`EXIT` 为 1，`output/` 尚未创建，CAL/DEV 拟合、回放、NFE、D4 皆为 0。

代码仅在路径见证时对被导入对象采用 `inspect.unwrap`，仍核对原始源码的物理路径和已冻结的 SHA256。dtype 验收继续调用实际的 LightX2V 入口。新范围一项 P107-A100 作业，无自动重排队。过去的失败记录保留于 `local6_v1`。
