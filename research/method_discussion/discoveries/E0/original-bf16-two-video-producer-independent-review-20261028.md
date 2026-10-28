# Original BF16 两视频 producer：独立源码审阅

审阅者：`/root/compact6016_six_video_build`；作者：`/root/completion_lineage_review`。结论 **SUPPORT（确切软件版本，SRC/合成 CPU）**。本审阅未执行模型、权重张量装载、Torch/NumPy 导入或 GPU；真实 raw payload CPU proof 和原生视频结果由随后注册的 RUN 决定。

## 确切版本与实测控制

| 文件 | bytes | SHA256 |
| --- | ---: | --- |
| `tools/e0_original_bf16_videos.py` | 58635 | `5e14e3bc1d0ed84ea51cff31cb77551c55455ff4b5226a5d2d1ec1d5227a24d0` |
| `tests/test_e0_original_bf16_videos.py` | 19721 | `070560eed2eecea2a495dcf60c0b822ca204e55c3a80ea1c336412927e8b5e3c` |
| `config/e0_original_bf16_videos_20261028_draft.json` | 203049 | `780ac79f76f5b2d1632a9537b2a894cb2e4a7a8f8e5b843e9fc4349cf856acb7` |

独立执行 `.venv/bin/python -m unittest discover -s tests -p test_e0_original_bf16_videos.py -v`：18 项 controls 实际 exit 0，测试体 1.186 秒；同版本 `inspect` 实际 exit 0。另用 stdlib 逐一核对 draft 的 900 项源码 bytes/SHA，全部一致，两个 Python 文件 `compile()` 成功；进程中无 `torch`、`numpy`。7 项 tokenizer/config 元数据合计 21,454,628 bytes，与独立 CPU proof 字段一致。

## 主路径核对

- **真实 raw BF16 身份**：两 expert 的原始 safetensors 公共 SHA、1095 个 BF16 tensor 的 header/offset/shape 与实际 `_load_ckpt` 返回 CPU tensor 均有门。不是仅关闭旧 HiF4 的 activation QDQ；新 pipeline 不导入 HiF4 adapter，不使用旧 decoded FFN cache。四个原模型 payload 共 69,025,362,058 bytes，CPU proof 两条 hash stream 完整读取；native 入口要求独立实际 exit 0、RUN SHA、协议/source/model 摘要和当前文件 stat 匹配。
- **原算术和布局**：实际原类 `lightx2v.common.ops.mm.mm_weight.MMWeight`；runtime input/weight/bias 为 CUDA BF16，权重为原 K×N 转置布局 `(1,K)`，不带 LoRA/diff 分支。observer 只检查 metadata，委派原 `MMWeight.apply` 的 `torch.mm`/`torch.addmm(..., out=...)`，未额外调用 `_get_actual_weight` 或增加 teacher 计算。
- **T5 与 DiT 分账**：此前全局 MM observer 会误拒 T5 的接缝已修复。真实 T5 offload 保留：168 个 CPU + 7 个 CUDA MM load，两次正 prompt 共 48 个 block buffer load、336 次七模块有序 apply。DiT 为 812 个原 MM load + 40 个 CUDA buffer MM load、320 FFN/640 FFN linear、2560 个 block 非 FFN linear、40 pre MM、8 head MM。T5/DiT 错 phase、错 expert、跳序/重复及错误 stride 都在原 GEMM 前拒绝。
- **完整轨迹与预取**：两个已开发暴露的 consumer source 0/12，seed42；每 source 全四步、40 blocks、81 帧。两 expert 原 double-buffer/prefetch 继续执行；每 video 的新 expert 首步 41 load、同 expert 次步 40 load，总计 324 load/4 reset，未删除原权重预取成本。严格核对完整十模块顺序和实际 160 个 FFN/site。
- **噪声与时间表**：原 scheduler 的一次 FP32 draw 和 step_pre0 的 BF16 初状态分别保存，分别与旧 same-source/seed 的输入控制逐字节核验。原 denoising indices `[1000,750,500,250]` 和 shift5 得到的真实 sigmas `[1,.9375,.8333333134651184,.625]` 分别记录；没有混入 NVFP4 recipe 的另一时间表。新 BF16 数值路径的 flow/post 不要求与旧 HiF4 trajectory 相同。
- **真实扣账与失败**：操作先 append 预扣、原调用返回后 append observed；失败保留预扣且无自动 retry。320 对 FFN CUDA events、encoders/decode/整体耗时和峰值分别记录。finally 的 monitor 关闭异常或最终 violations 会把 RUN 降为 FAILED，保留原异常；真实 supervisor 应绑定 finally 完成后的最终 RUN SHA。

## 结论边界

该固定版本适合进入独立 69 GB raw-byte CPU proof，再进行单卡两视频的 BF16 原生可用性实验。它提供原始 BF16 全流程与旧 HiF4 输入控制的明晰身份；两个 source 均为已暴露开发数据。正式质量、完整运动、人审与速度结论仍须读新实际结果，旧 HiF4 视频不充当 BF16 原生输出对照。单 job 1 张 A100、4 CPU、64 GiB reserved 上限及本会话两卡约束独立于仓库通用 dynamic/null 政策。
