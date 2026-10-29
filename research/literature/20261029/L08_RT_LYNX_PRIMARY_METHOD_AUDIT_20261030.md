# L08 · RT-Lynx 原文方法复核

2026-10-30，`/root`。LIT：已读 [PMLR 原文](https://proceedings.mlr.press/v306/cong26a.html) §4、Algorithm 1、Eq.4–5、Table 2、实现说明及 Appendix F.5；作者代码未核。PDF 仅存 HF `ffnds-literature/rt_lynx_primary_20261030/cong26a.pdf`，不是模型实验。

论文给出每四个激活保留两个的规则，并用 `s=sqrt(||X||²/(||Top2(X)||²+1e-8))` 恢复范数。完整方法另有训练的 rank64 残差 `X(L_A L_B)^T`；无 LoRA 的 SA-NC 是单独消融，不能继承完整方法的质量。Qwen-Image/MJHQ FID：Dense 21.98、SA 35.85、SA-NC 25.28、加 LoRA 21.25。实现在 H20/image DiT 上；原文某些 linear 保持 dense，并非整层免算。主文的四维示例与矩阵算法尚不足以确定代码的补偿归约轴、排序和 tie 规则；保持 CODE_NOT_VERIFIED。

ALG：输入范数恢复可以修正整体幅度，但不保证 W2 输出。令 `x=(2,2,1,1)`，保留前两项；任意标量补偿后，对 `w=(0,0,1,1)` 都有 `xw=2`、`S(x)w=0`。对 `w=(1,1,0,0)` 则补偿放大已保留贡献。竞争机制是“幅值衰减主导”与“被删方向/有符号组合主导”，两者在相同 mask 的 SA 与 SA-NC 输出、学生自由视频上预测不同。

Wan FFN 迁移仍是 OPEN：原 GELU 后 2:4 只能影响 W2；W1 需在其输入形成合法 mask。需先核实际代码与 A100/SM120 对应 BF16/NVFP4 算子、元数据和在线代价，再单独审同模型全160视频。此原文阅读不注册实验、不训练 LoRA，也不挤占当前 P0 SharQ 完整视频。G14 保留 T04/T13 与 K13–K15；它们是文献作用映射。

独立[作者代码检索](../20261030/L08_RT_LYNX_AUTHOR_CODE_SEARCH_20261030.md)保留实际检查的官方入口及未知项；未找到可信实现不等于证明其不存在。
