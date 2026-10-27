# 激活2:4 C4 CUTLASS / cuSPARSELt：独立冻结前审阅

**结论（STATIC/CPU-MOCK）**：修订版为两个指定 Wan 形状分别增加一次稳定 top2 参考选择，对完整补齐后的 BF16 张量以字节比较 Triton 输出，差异立即失败；此前提出的当次选择语义验证缺口已修复。独立运行5项CPU测试通过。该工具可在获授权后比较 PyTorch 2.10 的 CUTLASS 与 cuSPARSELt 预压缩、压缩和在线组合；实际GPU结果仍待运行。

审阅工具 SHA256 `7972ff6dbaf09cf5e8bb09a3a803a7ae33ed11dd5816d62fdcfde589ffca9f7a`、测试 `f47e1ac8a9e5feb3afc9ef9935ef4493251d169621fcb0ffe04453ea4a2cb3b7`、草案协议 `0c348736d3088c84e1235fd503bcc3da0ee43c2d400411e0d34df0b96cc8c209`。协议要求原用户继续授权及本scope精确SHA注册、1卡、2CPU、480秒compute/600秒outer；尚未执行GPU、Triton JIT或实际后端性能。

原 PyTorch CUTLASS BF16 行/列约束为32/64；M31668补齐31680符合32，两个K均为64倍数。`SparseSemiStructuredTensorCUTLASS.from_dense`使用安装版CUTLASS压缩，`to_dense`在二维metadata路径调用专用解码；当前 `torch.equal` 是数值等价检查，不保证正负零位模式相同。两后端输出都对相同mask的dense进行相对L2和finite/shape/contiguous检查；该对照不测未掩码模型质量。cuSPARSELt选固定alg1/split1/mode1，非转置后输出连续化；CUTLASS `torch.mm` 后也截断并连续化。实际映射libtorch_cuda、libcuSPARSELt和libtriton由运行时核验，静态审阅不能预先证明实际加载。

每形状含初始一次选择和一次稳定参考、两种压缩、一次CUTLASS解码、一次masked dense和各一次sparse正确性调用，再对七项路径各2 warmup/5计时；两形状总预算4随机、2稳定参考、30 Triton选择、30+30压缩、2解码、16 dense、30+30 sparse、0搜索/模型/视频，逐操作预扣。压缩和在线包含各自组件；首次Triton调用含JIT单列，输出上限100MiB。捕获式监测失败会将RUN标失败并传播异常，硬杀在途完成量仍未知。CPU样例验证安装版CUTLASS二维压缩/解码，但不等于真实GPU backend或性能通过。
