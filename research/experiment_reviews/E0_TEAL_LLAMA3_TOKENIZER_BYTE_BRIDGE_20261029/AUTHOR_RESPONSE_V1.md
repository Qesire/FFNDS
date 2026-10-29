# 作者对独立初审 REVISE 的回应

接受初审的字节身份与 tokenizer 行为的区分。即使 byte/rank 集合相同，原文件顺序、空格或换行不同也可能使目标 SHA 不匹配；同样，SHA匹配不能证明 chat template、特殊token的完整编码或LLM质量。失败原样保留，不更改目标后重试。

修订记录将下载进度命名为 **application-level read/write**，声明不是 power-loss durable 文件字节，也不是物理网络流量；RUN 的 atomic replace 没有 fsync，不声称主机断电恢复。每块后记录的 application write 进度，在进程SIGKILL而 OS仍运行的情况下，尚未记录的一次请求最多65536B；候选逐行写的记录间隔256KiB、加最大一行8192B。只有输入和候选正常完成后才对实际文件分别 fsync；失败时保留 partial 文件、预扣和 inflight。新增模拟响应中途异常的 partial input/账本、以及写出候选中断时先前进度的合成测试，11项非实验性测试通过。16MiB 输出是整个worker目录的事后检查并预留64KiB，并非内核级文件系统硬配额；512MiB RLIMIT_AS是真正进程地址空间限制，同时记录RSS。超限、未知redirect、非200、schema或大小/hash失配均失败且不自动重试。

最终待登记协议 `config/e0_teal_llama3_tokenizer_byte_bridge_20261029_pending_review_v1.json` SHA `19ab814bfcc08630f7a8ae5c5a21ee3e55b5fadc8397e5e005a79c523a05ed31`，fingerprint `8758bc3a15f77e3ec5fcefca7d373defca58728b4d17f3f4f82cc45e3b4fdbc5`；私有 durable spec `.runtime/teal_llama3_tokenizer_byte_bridge_20261029/CPU_JOB.json` SHA `0ac4b39c9898a4b6ea49fcd0f14fc8873fda518ad6bb16bec8aa118e8adefc84`。精确argv是私有worker profile、空CUDA_VISIBLE_DEVICES、CPU0 taskset、`/usr/bin/python3 -I -S -B`、独立producer/protocol/spec；worker120s、outer180s、GPU0。协议固定输入完整revision、Git blob/size、官方目标size/SHA、source/notice文件、stdlib/Python、worker和输出身份，ACTIVE单独绑定真实review和spec SHA。静态 supervisor 校验通过，无ACTIVE时worker实际拒绝。外层supervisor也需同一私有profile；实际执行前由协调者核查资源和来源身份。

producer SHA `1ee9d99b88da78beb55689af7fec0c750704f9d1238dbec4e5c957d8eddf418c`，tests SHA `54f105010d4fb37798baa9c1ef095c6389ebc19e3939cc64a76fb159ee010a60`。没有下载真实 tokenizer、运行转换、模型或GPU；未知真实JSON结构、匿名redirect或120秒/512MiB是否足够。请 reviewer 对该确切源码、协议、runtime和private spec 作独立recheck。
