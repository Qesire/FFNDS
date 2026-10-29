# Chipmunk FLUX MLP-only：作者机制受控复现设计与自审 v1

**状态 AUTHOR_SELF_AUDIT / INDEPENDENT_REVIEW_REQUESTED；作者 `/root/s2_theory_design_independent`；无 reviewer 投票、无 ACTIVE 注册、无模型/编译/GPU运行。** 这是 FLUX 作者实现的有标注 bookkeeping 修补，不是未经修改的作者程序，也不是 Wan 迁移或论文结果复现。用户请求的原版 Wan2.1 系统复现另立独立 scope；不能用本草案替代。审查合同见 [执行前独立复审](../../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md) 与 [issue #57](https://github.com/Qesire/FFNDS/issues/57)。

## 实际源码与两处否决反例

原论文 [arXiv 2506.03275v1](https://arxiv.org/html/2506.03275v1) §3.3.2 以跨步 MLP 激活缓存与选择性更新组成输出，作者报告 Wan 2.67× 是含 STA 与 step caching 的整系统结果。原论文附录 D.1 明确说由于视频模型 MLP 占较小的 wall-time，HunyuanVideo 和 WAN2.1 只使用 attention sparse delta；这与 Wan 示例 MLP 未接入相互印证。论文 benchmark 使用 H100-SXM5、CUDA 12.8、PyTorch 2.5.0。原码固定 [177fdefc4803e621fd8a02aad491812e9d1ecc94](https://github.com/sandyresearch/chipmunk/tree/177fdefc4803e621fd8a02aad491812e9d1ecc94)，L09 已登记；本次新增的 FLUX 接线和计数器静态深读先列于此，待协调者按 G14 同步而不虚构 NATIVE。`examples/wan/wan/modules/model.py` 的 `ffn` 是普通 Sequential，forward 直接用它，`LayerCounter.build_for_layer(...is_mlp_sparse=False)`，故 Wan 改 config 为 true 不会调用 MLP 稀疏核。相反 [FLUX layers.py](https://github.com/sandyresearch/chipmunk/blob/177fdefc4803e621fd8a02aad491812e9d1ecc94/examples/flux/src/flux/modules/layers.py) 的 `sparsify` 在 double 和 single stream 构建 `SparseDiffMlp`；前者仅图像 MLP 稀疏，文本 MLP仍 dense；后者切分联合 projection，不能把整个 FLUX 或 Wan 的 FFN 操作数等同其 MLP核。

[MLP源码](https://github.com/sandyresearch/chipmunk/blob/177fdefc4803e621fd8a02aad491812e9d1ecc94/src/chipmunk/modules/mlp.py) 的首步/周期 dense 缓存 `pa`、`out`、`block_mean(mid)`；稀疏步先在128 token均值上做 `fc1`，选组内前 `top_keys=0.3` 加 `random_keys=0.05`，`counts_multiple_of=256`，再调用真实 `chipmunk.ops.mlp`（BF16 mm1、fused mm2+scatter/cached out）。选择是**上一步已缓存与当前块均值 W1 付费代理**，不是免费 oracle；cache/选择/非均匀 tile 以及编译、W2 转置均须计时。索引从第10步之后才可按 block_mask_cache=2 复用；前2层仍 dense，full_step_every=10。W2 bias 由 dense cached output 承载；稀疏是差分更新不能每次重加 bias。随机 keys 和计数取整意味着声明30%并不等于真实30%工作量。

**关键反例 1：MLP-only 配置错误计数。** FLUX `sparsify()` 注册 attention + MLP 两个 submodule，`SparseDiffAttn.forward` 在 `is_enabled=false` 时直接返回 SDPA，未 increment；`SparseDiffMlp` 开启时仍 increment，故两个调用被记成一个，层/step 归属错、缓存及 full-step cadence 错。只改 YAML 不可执行为可信 MLP-only。

**关键反例 2：末次调用提前 reset。** `LayerCounter.increment()` 在加计数后、返回前判断计数已位于最后 step/layer/submodule 并 reset；因此下一真实最后 submodule 会读取归零后的 step，而不是预期最后位置。必须实测逐调用轨迹和独立层标识；不能预填理论 55×45 个调用为实际事实。另复核 Dense baseline 分支两个 disabled 都不自增，不能将其视作同一个 counter trace。

**可审修补草图（未实施）：** 分支 `SparseDiffAttn.forward` 在 disabled 返回 SDPA 前，仅当已构建 counter 且 MLP enabled 时调用 `layer_counter.increment()`；保留 SDPA 原值。把 `increment()` 的 reset 改为在完成最后坐标的自增进位后检查 `cur_inference_step == GLOBAL_CONFIG['steps']` 才 reset，返回原本调用坐标。先以独立 CPU 纯整数合成测试覆盖 57层×2子模块×50步（每坐标正好一次，最后一项后才回0），以及 attention-only/MLP-only/两者关闭和连续两次生成；再由另一 reviewer 对 patch 与实际 profile recheck。任何 patch 的准确源文件、diff、SHA 在独立 review 前提交；本段不是已运行代码。选择统计必须通过各 module 的固定 layer_num 与 step observer 对照而非靠自增器自证。

建议供 reviewer 对照的最小 patch（仅设计，不在 author checkout 实施）：

```diff
--- a/src/chipmunk/modules/attn.py
+++ b/src/chipmunk/modules/attn.py
@@
         if not GLOBAL_CONFIG['attn']['is_enabled']:
-            return F.scaled_dot_product_attention(q, k, v)
+            out = F.scaled_dot_product_attention(q, k, v)
+            if GLOBAL_CONFIG['mlp']['is_enabled'] and self.layer_counter.has_mlp_sparsity:
+                self.layer_counter.increment()
+            return out
--- a/src/chipmunk/util/layer_counter.py
+++ b/src/chipmunk/util/layer_counter.py
@@
-        if self.cur_inference_step == GLOBAL_CONFIG['steps'] - 1 and \
-            self.cur_layer == self.num_layers - 1 and \
-            self.cur_layer_submodule == self.num_submodules_per_layer - 1 and \
-            self.cur_model_invocation_per_step == GLOBAL_CONFIG['num_model_invocations_per_inference_step'] - 1:
+        if self.cur_inference_step == GLOBAL_CONFIG['steps']:
             self.reset()
```

该 patch 假定 FLUX 中 attention / MLP 每层各调用一次且以该顺序调用；不是通用 counter 修复。需检查双流/单流全路径、torch.compile 的状态更新和多次生成，然后才绑定可执行 producer。模型环境实际缺失，当前既不能编译验证，也不称该 patch 已通过。

## 条件、资产和执行命令

本机只读可见 4×A100 SM80；另有项目既知 SM120，当前未证明作者 `sm_90a` kernel 可在这两者运行。源码 `setup.py` 单一 target h100、`-DKITTENS_HOPPER -arch=sm_90a`，无官方 sm80 后端；README 指 Hopper、CUDA≥12.4（推荐12.8），作者 FLUX CLI 明确拒绝 SM<9。**需真实 Hopper SM90a 与足够可用显存；硬件不可得就停在源码结果，不在 A100/SM120 擅自改架构冒充复现。** 本机 `/usr/local/cuda-12.4/bin/nvcc` 存在，但环境 PATH 的 nvcc 不存在；实际可用版本、编译环境/驱动和 GPU 空闲需在获准 worker 再确认。source checkout 有精确 commit，ThunderKittens submodule 必须为 `0c44d2c7262fdd94273a18420a5861bc96a335e3`。作者 requirements 指 `torch==2.5.0`、flash_attn 等，和现有 `.venv` torch 2.10 不相同；单独环境，锁定真实全部版本及构建产物 SHA，禁止声称兼容已测。

作者入口 `python -m flux.cli --name flux-dev`，默认 50步，模型 BF16、3072→12288、19 double+38 single blocks。需冻结访问许可和完整文件 hash：`black-forest-labs/FLUX.1-dev` 的 `flux1-dev.safetensors` 与 `ae.safetensors`（HF 模型标记非商业许可/访问条件），`google/t5-v1_1-xxl`、`openai/clip-vit-large-patch14`，以及 CLI 无条件加载的 `Falconsai/nsfw_image_detection`；tokenizer/config 同版本。当前本机指定 HF hub 中仅看到 CLIP 目录，未证明权重完备；FLUX/T5/NSFW 对应目录未见，其他 worker 存在情况未知。要先做新独立 metadata/下载 scope，确定 revision、所需文件、字节、可访问许可和 SHA；不下载或拟用 Wan2.2 权重替代。

获得独立审查和 ACTIVE 后，原作者整体系统先单独执行（只读命令草案，不是运行授权；须从 pinned clean checkout 建 isolated environment，下载/构建各属经审 scope）：

```sh
git checkout 177fdefc4803e621fd8a02aad491812e9d1ecc94
git submodule update --init --recursive
# 在审核后的 Hopper 环境：先锁定实际 torch==2.5.0、CUDA Toolkit 和所有依赖，再编译作者扩展。
python -m pip install --no-build-isolation -e . -e ./examples/flux
cd examples/flux
python -m flux.cli --name flux-dev --width 1280 --height 768 --seed 42 --num_steps 50 --prompt 'A small bird perched on a branch in a forest' --chipmunk-config ./chipmunk-config.yml --output_dir ./output_author_system
```

该默认组合有 attention 稀疏，**不作 MLP-only 性能**；只是一图开发 smoke，不等于论文质量复现。checkpoint下载/编译失败停止，记录 stderr/实际 EXIT，不自动重试。命令中的联网下载应预先完成、哈希锁定，生产阶段禁未登记联网，scope 禁止交互临时变更。

## MLP-only 最小可辨识范围（待独立复审）

单 Hopper GPU、CPU≤8、GPU显存≤80GiB、host≤192GiB、wall≤5400s / outer≤5700s、输出≤4GiB；编译/模型获取不包含在该 scope，必须有各自独立闭合证明。实际可用硬件、许可与适用并发上限未确认；这些是草案拟定上限，不是授权。固定作者给出的 BF16 FLUX-dev、1280×768、50步、同一开发 prompt、seed42；三臂依次在相同分辨率/模型/attention SDPA/patchify/初始 noise 下运行：A dense `mlp.is_enabled=false`; B 修补版启用 MLP且 `full_step_every=1`，每步 dense cache refresh neutral；C 修补版作者参数 `full_step_every=10, top_keys=.3, random_keys=.05, block_mask_cache=2, first_n_dense_layers=2`。三臂 attention false，step-cache false（FLUX 路径实际未检出调用，不能仅从配置断言停用），no fit。每臂同进程 2 次 warmup + 1 次记录、共9真实图片，初始 noise payload SHA、prompt、schedule/hash、weights、数值 recipe逐次记录；计数 trace 单独插桩运行可能影响速度，故独立 trace/speed 两子次序须冻结/计入并明确总次数，缺精确总预算不得登记。seed 相同不足以证明 noise 相同；记录原始张量 SHA，必要时显式复用相同 noise bytes，若需要修改作者采样代码须明确标注并经 review。计数完整证明每层位置、dense首步/周期/首两层、sparse path，时钟计时另读不侵入测量的 warmup/actual 工作量；取消/失败保留所有已生成记录。

**两竞争预测：** H1 稀疏跨步变化集中，实际 sparse counts 显著小于 dense、W1与W2的物理 kernel均调用且包含代理/cache的端到端MLP有净节省，初始输出/视频或图像语义与 Dense 差异有限；H2 小的块均值掩盖互相抵消或遗漏大逐token改变、或选择/缓存/稀疏填充吞噬收益：相似 counts 但差值大，或真实 kernel速度/完整 FFN 不优。中间结果只支持局部算子或个别输入，不推广视频。

**硬性 neutral 与可观测：** B 与 A 的同输入每 block 全量 MLP output BF16 定义 `atol=0.01, rtol=0.01`，同时记录 max/relative误差和非有限数；不满足则停止，不能以图像看似相似代替。实测并核 `csp_mlp_mm1`、`csp_mlp_mm2_and_scatter_add` 或作者准确实际分支、topk/copy/scatter，区分 launch/调用/物理 tile 计数，记录被选择的真实 counts、cache full/refresh、逐step/block；CUDA event同步统计完整 MLP（含代理、压缩打包、cache和分配）与整体 wall，warm/cold、编译和 setup 分开。one prompt 无法估计 source-level 泛化，视频运动/语义及论文声称的质量均 NOT_MEASURED；需后续单独已审多来源 scope。

**退出决策：** 若无 Hopper、缺权重或 kernel 编译失败，则阻断并保留收据；没有真实 sparse kernel 或 counter 有错即 FAIL/停，不能标复现；neutral失败停并调查 BF16、cache/bias 和重置；kernel存在但无净省算或误差大保留负例，不能称成功；两者满足也只支持单输入作者机制的有限 smoke，不直接声称论文质量、论文 speedup 或 Wan2.2 成功。新 Wan2.2 迁移需另立经review范围，跨 expert 禁止共享缓存。

## 关键 source SHA256

- `setup.py`: `3d74b0510d9063b3f8668250660d6059706224d3ab07448dea8694603af13f5e`
- `src/chipmunk/modules/mlp.py`: `28bb218e7cb474ee5da0d07d9ffa773c94fbc48f86203baf202d53e191759944`
- `src/chipmunk/modules/attn.py`: `a7d0d22be9d5506b63b03ba8d72527cb4948c2daa1bdb612ecd78793b2905abc`
- `src/chipmunk/util/layer_counter.py`: `478ed1ee821944738ab25dc15b9430f521ed75e4824310d9dfe041952767360b`
- `examples/flux/src/flux/modules/layers.py`: `608204c2bc1fe23588c1b4d6fa52695be5314fd9afe925dba43f091eb7ef8b2b`
- `examples/flux/chipmunk-config.yml`: `c2d44c79fb1d27581249e9a3650989564666954c35348fddef087a354205577a`
- `examples/wan/wan/modules/model.py`: `47b78cbff99b5773974ded33a92ab184a328ebb011ffbe211aa17d17b4d66710`
