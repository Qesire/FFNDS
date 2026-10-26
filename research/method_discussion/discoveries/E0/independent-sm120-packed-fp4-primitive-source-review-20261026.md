# SM120 packed NVFP4 primitive 独立源码与失败路径审阅

日期：2026-10-26。作者：`whole_flow_native_review`。证据类型：源码／构建回执／只读哈希预检与 mock 软件测试，尚非原生运行结果。

**DISCOVERY — 正向研究对象。** 官方 SM120 block-scaled E2M1 GEMM 提供了一条可实测的物理数值表示：已打包的低位值与 scale 直接进入 Tensor Core，FP32 累加并输出 BF16。先用两个很小的精确 host-reference 对照，能够把“当前工具链／硬件可执行这个表示”与“如何从 Wan BF16 操作数获得合适的低位表示”分成两个可辨识问题。它是可选的物理比较线，保留了 intrinsic FFN 表示研究的独立空间。

**竞争预测与廉价判别。** 成功编译仍可能得到不支持当前架构的 no-op 可执行文件，或在 scale/layout/epilogue 路径上出现数值错误；真正原生通过应同时具备零退出码和 upstream `Disposition: Passed`。最便宜的决定性下一步是按冻结协议运行两个既定小 case；本审阅没有执行它们。

## EVIDENCE：精确对象与 source→build→binary 链

| 对象 | SHA-256／revision |
| --- | --- |
| CUTLASS v4.3.3 HEAD | `d55f6beeebb6df501a250dc82827db97660f06e0` |
| 官方 `79a_blackwell_geforce_nvfp4_bf16_gemm.cu` | `c4c4db11c85795bdd38c6df294e013b57be54eb5ff8d583281cfb90f6a54295f` |
| `tools/run_e0_sm120_fp4_primitive.py` | `a46720162e4ebe7ad7f4fa24db44b3f509bf48165877a12e59ef7469b52174db` |
| `tests/test_e0_sm120_fp4_primitive.py` | `661c1a9b8720e1f29c81412727fd4fd8a320bb634c3f49c6707dbcd28b5cab1f` |
| `config/e0_sm120_fp4_source_build_20261026_v2.json` | `00ca7eb897bb387272d5c88251f621e4b9ce8b2e7759a108a6ee5a5b1fc43967` |
| `source_build2/RUN.json` | `56130cd6290a393ab9fa1681825e0f17ebacb799a29f7d0680ad8bba122c102e` |
| 已构建 79a binary | `76d3f50c28c341da27d4c89faac45391aa25b21100ed0261bd9900f4fc6ba5f0` |
| `config/e0_sm120_fp4_primitive_20261026.json` | `ad944b803937e7e7deeaf40af9f4c426d897165504a8c2fadc6ae490c52dd08a` |

我阅读了完整的固定 revision 官方 79a、示例 CMake、驱动、编译协议／实际 RUN、UUID 空卡 launcher 和 worker/source helpers，核对本地 tracked CUTLASS source 无修改，并独立哈希上述实际回执和 257,072 字节 binary。默认元数据入口成功输出 `BINARY_PROVENANCE_PREFLIGHT_ONLY`、`GPU_calls=0`。

最终 preflight 关联编译 RUN 的 id、状态、协议 SHA、source revision、binary 相对路径；再检查编译协议的 target、输出目录、configure arguments 和原始协议 SHA；最后核对实际 binary SHA。编译配置是 `120a`、CUDA12.8、Release，启用官方 tools 的 interface include，关闭 tests／library／profiler。build2 回执仅包含 configure 与指定 79a target 的编译命令；原先 tools include 关闭导致的失败属于另一保留尝试，不能用 build2 成功覆盖它。

## 原生数值命题与预算

官方 source 的 A／B 都是 `nv_float4_t<float_e2m1_t>`，各有自己的 scale tensor，输出 C／D 为 BF16；`initialize` 直接生成已量化类型的 host 输入再上传。全局静态 `uint64_t seed` 为 0，host 数据依次使用 2021–2025。`verify` 用 host `Gemm3x` 构造同一 scale/layout 与 alpha/beta 的参考，要求 BF16 `TensorEquals` 且两端 norm 均为正。

两个 case 固定为：

| M×N×K | alpha／beta | profiling iterations | 包含 correctness 的 GEMM 数 |
| --- | --- | --- | --- |
| 128×128×128 | 1／0 | 0 | 1 |
| 128×256×512 | 0.5／0.5 | 3 | 4 |

合计保留 5 次 GPU GEMM，即 69,206,016 个乘加位置；host reference 各做一次，共 18,874,368 MAC，与协议一致。`gemm.initialize`、内存搬运和运行时辅助工作并未被这项 GEMM 计数冒称为零 GPU 工作。profiling 分支的短时结果只作为这个小 primitive 的日志，缺少匹配 BF16 comparator 和 Wan shape，不能支持 FFN 或端到端提速结论。

upstream 对旧 toolkit、错误架构和缺少支持宏的某些 no-op 路径返回 0。最终 `native_pass` 同时检查退出码 0、出现 Passed 且没有 Failed；只检查 exit code 的伪阳性已由测试阻断。两个小输入分布的精确比较仍然不同于 BF16→NVFP4 量化验证、所有输入的通用正确性证明或 Wan FFN 等价。

## GPU 归属与已修复的失败账本

launcher 对一张无 compute occupant 的卡取得 `flock`，以 NVML UUID 设置可见性并 exec 父 Python；父进程持锁，不创建 CUDA context。驱动核对物理索引与 UUID 一致及 SM120 Blackwell 条件。每个 native case 前要求没有 compute PID；运行中 NVML 仅放行当前 `Popen.pid`，其他 compute PID 出现即 fail-stop；它不会终止外来 PID。两次 C++ binary launch 串行，遥测 `nvidia-smi` 属辅助子进程。NVML 是约 0.2 秒加查询延迟的抽样，不是精确峰值或不间断的资源证明。

初稿在 Popen 前写 `CHILD_STARTED`，而 foreign／timeout 异常后虽终止子进程，case 没有实际 PID 或终态。我用全 mock 反例得到“实际 returncode=-15，case 仍 CHILD_STARTED”，将问题交给根代理修复。最终代码区分 `launch_attempted`、`child_started`、`child_PID`，保存 command；清理后填 terminal 状态、returncode、时间及错误；在明确 native PASS 前不把保留 GEMM 冒算已完成。

按根代理单独委托，我只在原测试文件增加了三项 mock 失败路径测试，未修改驱动或协议。foreign 情况核对仅实际 child 被 terminate／wait、PID／终态与 log SHA／bytes 保存、第二 case 未启动；timeout 情况同样清理并保留 TimeoutError；Popen 自身失败明确记录 `FAILED_BEFORE_CHILD_START`。原有三个 guard 测试保留，合计 6 项通过。所有 Popen、worker 绑定和 NVML 查询均被 mock，没有执行 native binary 或 `nvidia-smi`。

## EXECUTION 与研究解释

执行边界：本审阅仅做源码／一手文档／回执哈希及合成软件检查；零 GPU 初始化、零 native GEMM、零模型／Wan FFN／视频。根代理负责精确 ACTIVE 注册、空卡分配与一次固定运行。最终版本无阻塞该范围的剩余实质问题；输出上限 1 MiB、每个 native child 的 120 秒失败门禁、日志保留和不重试语义均已核对。最大 native launch 数为两次，辅助遥测另计。

图锚为 F06／F08／F09／F12／P14，连接属于 conditional 可选物理表示，而不是已测 Wan dense execution fact。若此次原生通过，新增的是 packed block-scale operator 的可执行性证据；下一阶段的 BF16 量化／scale 生成／packing、必要 FFN 功能、真实工作量和完整视频仍需要相应独立对照。质量状态维持未接受。

实际读取的一手来源：[固定 commit 的 NVIDIA 79a](https://github.com/NVIDIA/cutlass/blob/d55f6beeebb6df501a250dc82827db97660f06e0/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu) 与上述完整本地 source；另查阅 [CUTLASS Blackwell 官方文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) 时明确区分其 SM100 `tcgen05` 页面与当前 SM120 block-scale MMA 示例，未把前者的架构吞吐结论移用到本实验。
