# E0 paired native driver — independent final static audit, 2026-10-26

结论：在下列冻结源码范围内，未发现仍阻断接线的 driver 缺陷；实现可以进入单独授权审议，但未获得模型执行授权，也未被本审查证明能够运行、保持 DD 数值中性或容纳于单卡显存。本文记录已经完成的源码审阅，不新增实验。

| 冻结对象 | SHA-256 |
| --- | --- |
| `config/e0_paired_full_depth_20261026.json` | `6e9e6a39898d12c90f8378cec7ee77a7f26a909e5690d8e1547896bdaf67c188` |
| `tools/run_e0_paired_full_depth.py` | `093324bfb700678120176354ea05adb0ee57248f32c92f7be88d0a16ad1bb977` |
| `tools/e0_paired_wan_controller.py` | `bd99bdba0651dc59fcc5a6cdf72922906a5c26eb08dfb4d4650f1d6cefa560bd` |
| `tools/e0_paired_ffn_observer.py` | `691a2e66f3326ce1f6b65536d576f8b89c9861db217fdee4885463a54d196476` |
| `tools/e0_paired_weight_residency.py` | `75ecc7a5cc8d338c2e0afee8c301d3eefaabb3ddb6daa48814eb0b9853052bc1` |
| `tools/e0_full_depth_trace_contract.py` | `28154f7d340cf164f07bbb908e31b37f04414b3fb7ef4f9f03e45fbc377ba98e` |

审阅范围为 driver 全路径、上述三组件与 trace consumer，并沿原生 pipeline、Wan runner/pre/transformer/post、scheduler、offload 与 adapter 核对调用语义。协议的 16 个 driver 与 23 个 native 绑定在审查时均与实际文件 SHA 相符；本次收尾复核的六个实现 SHA 仍与已审快照一致。更早的原生生命周期结论保留于 `independent-full-depth-native-interface-20261026_v3_review.md`，SHA `701dc6e3306a0f015afbc4948c6166cd2ef157501e4cb11588148d3e1a20a680`。

- 执行门在 torch/native 导入前核对 ACTIVE 中精确协议 SHA、固定 canonical study root、私有 worker 配置摘要和 worker identity，再用加锁账本预约 pair。DD0→DC0→DC12 固定顺序、每项两条路径、总计六条完整四步路径；失败或未闭合 attempt 阻止下一项及重试。静态 preflight 本身不启动 GPU，协议仍声明 `NONE_UNTIL_EXACT_PROTOCOL_REGISTERED_AFTER_EXPLICIT_USER_AUTHORIZATION`。
- 每个 fresh process 只调用一次 `pipe.generate`；controller 在各 step 内执行两个真实前缀，student 由原生 generate 解码。teacher 使用保存的最终 latent 另调 `run_vae_decoder`，赋给 `gen_video_final` 后调 `process_images_after_vae_decoder`。原生 `end_run` 已清空 `input_info`，driver 暂时恢复 controller 保存的信息并改 teacher 保存路径，finally 恢复原值；与原生最终解码入口语义一致。
- 每 pair 的 160 个 site 使用当前同一 block buffer 完成两轨和 shadows 后再 swap，预期 FFN 权重传输 320 次。角色计数 DD 为 dense-main 640、dense-shadow 0、candidate-main 0、candidate-shadow 960；DC 为 320、320、480、480。observer 各记录 320 个 primary-plus-shadow timing 与 160 个完成 site，原 runtime 线性/QDQ 各 3200、cache miss 0。
- 计数不可直接相加：observer 的 native QDQ 1280 中有 640 次 dense QDQ 已计入 runtime 3200，因此每 pair 实际 native QDQ 为 3840。native GEMM 为 3200+960=4160，另有连续 FP32 诊断 GEMM 960；三 pair 对应协议 11520 native QDQ、12480 native GEMM 与 2880 诊断 GEMM。观察开销均须计入，不能用此 wall time 宣称部署加速。
- 源绑定覆盖输入、缓存 manifest、旧结果、80 个 affine pack、实现和协议副本；`HistoricalSources/verified_restored_assets` 继续核对原 checkpoint/cache 与 manifest 中 HiF4 Python/native `.so` 来源链。新旧 reference 的 seed、视频摘要及 checkpoint/cache/contract identity 分别交叉校验，未用当前资产身份替代旧 receipt 身份。
- 两次解码后仍检查资源监控；最终 `close` 后再次 `check`，最终采样物理峰值写入共享 cost，按 protocol SHA/pair ID 去重。成功输出先保存不可变 `NATIVE_RUN.json`，再逐视频执行全帧 evaluator 与 bundle gate；teacher 对旧 dense，student 对新 teacher。官方 VBench-I2V 与人工评估未运行，质量接受始终为 false。
- 失败路径保留 attempt 和 traceback，并为已存在的 teacher/student 视频尝试逐帧诊断；缺失、坏文件或评估失败明确报告。失败诊断不冒充完整 core gate 通过；失败 scope 不自动追加生成。

数学解释仍须按四项分开：同输入上 deployed partial-native affine54 与 dense 的差；student 前缀改变 FFN 输入的差；经 gate/residual 及后续 block 传输的有符号差；四步 4D flow/latent 状态差。full-M FFN 输入是二维 panel，scheduler 状态是四维 tensor，不能混作一个误差空间。四个交叉 writeback 应读作 `W(rD,yDD,gD)`、`W(rD,yAD,gD)`、`W(rA,yDA,gA)`、`W(rA,yAA,gA)`；prefix 项包含 residual/gate 改变。双轨观察不能单独识别某一 site 的因果后缀效应。

dtype 只能条件推断：冻结 worker 的 BF16 与 sensitive=None 经原生 env fallback 得到相同 dtype，且未见 Wan22 强制 keep FP32，因此静态路径预测外层 step-pre BF16；旧 `.float().cpu()` hash 不能证明真实入口 FP32。consumer 记录 step-pre 入口 cast 与 step-post 边界，实际 dtype/shape/SHA 仍须由将来原生执行给出。

有界 FFN residency 已有实现，但尚无原生运行证据。已知权重 payload 40.6972 GiB 不含全部激活、buffer、workspace、allocator 保留或监控遗漏，绝非峰值上界或单卡可跑承诺。80-pack checksum 与 CPU 软件测试也不能替代内存、流同步、DD 中性或 native API 实测。

独立验证限于源码、数学计数、SHA 与普通静态元数据；本审阅未运行模型、GPU、SSH 或软件测试。root 报告完整 80-pack preflight PASS（`model_scope_authorized=false`、`GPU_initialized=false`）及 17 个 CPU 软件测试 PASS，此为作者报告。协议的六条路径与全部观察开销仅为拟议有限 scope，不是本文件授予执行许可。

图锚点为 G02、F03/F04/F09、F12/G20、F13/G21、P22/P24、D09；producer 接口仍是 conditional，trace 关联是 posthoc。本文没有新增文献机制或外推 streaming 任务；独立重建的竞争预测是 DD 应保持原生轨迹一致，而 DC 的局部同输入误差可与 prefix/flow 扩散分离，是否成立留待另行授权的 native 证据。

保留最初 219 行源审计 `independent-full-depth-native-interface-20261026.md`，SHA `9ed02bc4e3299b95e3881d3c0a5dcf161afce106c7a079991ac2ca6c317e0faf`；本文不覆盖任何旧审阅、ACTIVE 或注册表。后续任一绑定源码变更须视为新快照，不自动继承此结论。
