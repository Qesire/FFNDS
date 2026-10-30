# AUTHOR_SELF_AUDIT · 当前E0的CPU资源预读

实际作者 `/root`。用户要求加快当前比较。本范围只是两个节点各1 CPU的有界文件预读提示，现有三张GPU的58臂科学任务、输入、原SHA/rawSHA验证及数值操作全部不变，不重启父进程，不产生新的模型候选。

| 范围 | protocol SHA256 | job SHA256 |
| --- | --- | --- |
| E0_A5_HWG_RESOURCE_READAHEAD_20261030_SOURCE1040_V2 | 867a441f9e79e555dbbcf827021a7599f85a9fbe5340c876b153251999c8447a | 15808e5c438159346e3dc1f670975dbdc050a9c57547ed78193d1e1c673ace2f |
| E0_A5_HWG_RESOURCE_READAHEAD_20261030_SOURCE898_V2 | 05501e51e88fa03ca23a419586ce55ce279e150fbdfae38f16d2573a511d0f84 | 2549b970c14e0506553b422dfc70574a8dee4e1eb95286160066e94cfe88e73a |

每个helper只读取已完成producer/独立VERIFY链、原metadata及动态父RUN等小文件；对下一两对的既有payload/fullV1，先检查HF路径、长度与只读fd身份，再分4MiB调用POSIX_FADV_WILLNEED。**进程返回tensor payload字节数0；内核物理I/O未知**，不能声称物理读取为0或把提示范围当真实落盘字节。原GPU reader仍照原协议真实验证每个完整file和raw tensor SHA。

每节点CPU0单核、GPU0张、RSS≤256MiB、小metadata显式读≤256MiB、HF新输出≤4MiB、每文件≤1MiB输出、最长7200+120秒；提示文件长度累计预扣≤96GiB，系统调用≤25000，最多预读后两对。每文件及每4MiB前检查父EXIT与host128GiB余量；父EXIT正常停止，低余量停止当前批次。已部分提示的文件不补发，部分状态保留；全文件字节预扣不回滚。fsync每文件提示前的预扣记录，fd采用O_NOFOLLOW/O_CLOEXEC并在首次调用前及结束核设备/inode/长度/mtime/ctime/regular。父80对或EXIT后结束，不向父发送信号。

第一未提交/未登记草案仅做静态review，发现内层退出/余量及打开文件竞态两项问题；已修订，四项fixture由作者及独立评议者分别通过，覆盖中途EXIT、余量下降、打开inode不符、有限range且无os.read。草案未执行，私有归档；本V2才请求正式独立审批。

缓存建议可能隐藏I/O等待，实际速度效果待观察；不作控制变量测速或SYS因果加速主张。无训练、模型、CUDA、数值诊断、视频、源文件改写或自动重试。请独立核精确parent、worker/CPU0、fresh输出、全部源码及协议/job/真实冻结commit；本作者记录不构成批准。
