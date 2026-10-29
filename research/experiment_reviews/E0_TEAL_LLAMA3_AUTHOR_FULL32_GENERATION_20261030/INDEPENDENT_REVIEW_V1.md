# 独立初审：TEAL 作者 Llama3 全32层生成 — REVISE

2026-10-30，独立 reviewer `/root/teal_full_llm_calibration_build_terra`，作者 `/root/sm120_full_quality_build_terra`。审阅当时 producer SHA `39e36d68103e486b8b790a983d23e858080ee84e3b10a133c72ab1cd2da5d2b9`、tests SHA `f3ae37e1d9c527d8807166e52352290ca6b12a12af3def5f058421fd8d4cce07`、无未来hist的草案 SHA `b084c76febc1082ffc00ee94b2b5210e3e1ef362b3cd33bd96469ed6a238f525`、作者原 gpt-fast model/generate/kernel/distribution 及私有compile wrapper。CPU-only、CUDA_VISIBLE_DEVICES='' 实际运行 author_import 模块导入成功；未加载checkpoint、数据、模型或GPU。当前校准父scope尚未闭合，不能批准最终资产身份或执行。

## 源码确认与边界

Dense调用作者 `_load_model(...hist_path=None)`，保留作者FP16 dense投影；作者 sparse 路径读取同一层 attention/MLP各 h1/h2 histogram，q/k/v分别使用同attention h1、o用h2、gate/up用MLP h1、down用h2，重排 W1/W3/W2、QKV、O 为作者列主序后 patch。truezero在相同patch后显式将224个threshold设0，和uniform50实际读CDF阈值可区分；若 NaN/零阈值或 split-K FP16原子加法造成偏差，neutral必须实际测量。`SparseQKVGEMV` 对单token按输出区段选各阈值，Llama3 Q/K/V边界4096/1024与作者autotune的 BLOCK_N候选都对齐；uniform阈值同源仍应分别记录真实值。多token sparse wrapper走未mask dense matmul，单token才走sparse；不能把结果称仅FFN。CPU source import通过不证明 CUDA/SM80/compile或kernel可用。要明确核实实际 `compile_wrapper.__file__` 确为已绑定私有副本，不能只检验 `kernels.sparse_gemv` 来源。

原 noninteractive `generate`/`decode_n_tokens`没有EOS提前break：固定64新token=1 prefill采样+63 decode；9次合计9 prefill/567 decode；EOS token被采样也继续生成，必须明示这一语义，输出实际EOS位置。计数是Python逻辑调用，不代表Triton autotune、CUDA graph或实际物理kernel数。三模型同卡驻留的资源/系统结果限定在这个工作负载，不视为单模型部署峰值。

## 可推翻反例与必要修订

1. **三臂KV容量不一致。** neutral比较先 `setup_caches(T+68)` 于dense/truezero；作者 `setup_caches` 容量足够时直接return，后续64 token generation 保留较大缓存；uniform50仅在首次 `generate` 时按T+64分配。两者按8向上取整可能不同，attention实际KV尺寸和内存/延迟也可能不同。应先对三臂同一计划统一实际cache容量，记录每次真实shape、有效mask/position和复用；在配对前核对一致。不能假定差异总因8对齐消失。
2. **neutral尚未覆盖生成decode数值路径。** 当前teacher-forced每步直接调用`model(x,pos)`，作者真实 `decode_n_tokens` 在 math-only SDPA context中调用并通过 `torch.compile` 包装；默认SDPA或Dynamo可能选择不同算子、数值和KV布局。相同teacher tokens和position是合理设计，但须同实际 math-only与compiled decode路径核验模型logits、top1、位置及cache语义，或将两条分别标明并限制结论。教师token来自prompt片段作为外部固定序列是合法teacher forcing，不可假称真实自回归轨迹。
3. **端到端/计时测量的身份。** 每token创建并记录两个CUDA events，逻辑wrapper与测量本身影响调度；事件时间覆盖该包裹路径，不是无侵入论文原始速度。首次 `torch.compile` 仅构造包装（通常lazy），编译/autotune发生于每臂warmup；须将CPU wrapper、实际首warmup总墙钟、稳定期、布局/阈值/模型加载、prefill与decode等分开，明确event插桩、图重放、多stream可能与实际物理launch未知。两次稳态测量是有限开发对照，非精确论文加速和系统验收。对同配置三臂按相同持久缓存状态记录，包括所有三模型同时驻留产生的代价。
4. **资产与实际执行闭合。** 未来校准实际EXIT0/64 histogram产物必须真实到位后再生成和freeze protocol，保持64文件自身重新预扣hash及作者256次hist load计数。建议消费者也要求父32 forward前后、token SHA匹配、observer与监控结果，而不只凭保存的32 act和sort数。登记前真实review recheck必须逐项绑定 exact producer、CPU/GPU私有spec、runtime、parent RUN/EXIT 和 histogram hashes。产生scope依赖的CPU preflight也须清楚登记其范围与实际EXIT0。模型密钥和严格source导入不应依赖可变非锁定文件。
5. **覆盖和控制。** 目前 neutral 比较只检查输出logit相对L2和top1，未直接检查每层KV写入位置或真实逻辑稀疏dispatch；应在预先预算的teacher/observer中检查cache更新与实际32层/seven投影覆盖，并将observer的224×两臂 mask/zero同步和4额外前向按单独开销报告。观察器不计入计时段，不能替代物理launch trace。source raw mask `abs(x)<=threshold` 对应作者 `abs(x)>threshold`；原本零值单列，均匀50是目标而非实测零率。

初审结论 REVISE。请作者响应和提交精确变更版，独立复核仅在真实父校准闭合、完整协议/spec和上述实验身份最终固定后进行；不会用此初审冒充对未来hist的批准。

补充工程检查：隔离转换环境、`CUDA_VISIBLE_DEVICES=''`、CPU12/13、Python `-I -B` 下实际调用 `author_import()` 后，`compile_wrapper.__file__` 实际指向已绑定的 `.runtime/teal_fb7373c93ac3594817c9ee64d4e08b47430a1822/kernels/compile_wrapper.py`。这是当前环境导入证据；仍应由受审执行时校验模块来源，不能据此宣称GPU兼容。

## Subsequent source analysis, before parent closure

For Llama3, QKV output has N=6144, segment boundaries Nq=4096 and Nkv=1024; W1/W3 input M=4096 and W2 input M=14336. These values are divisible by all author autotune BLOCK_N≤512 and BLOCK_M≤128 candidates. Thus the Q/K/V block threshold choice aligns with logical segment boundaries, and the unmasked out-of-range read concern does not arise for these specific full shapes. In the author code the `weight.T.contiguous().T` transformation preserves mathematical matrix entries while `rm*N+rn` indexes column-major storage; sparse zero threshold differs numerically from dense F.linear through FP32 partial sums and FP16 atomics, which still requires actual full-model neutral measurement. The pinned Triton autotuner source invokes the chosen configuration's `pre_hook` before every kernel launch; its author hook zeros Y. These are source/ALG facts only, not actual GPU validation.

Count the whole proposed model workload: 9×(1 prefill+63 decode)=576 calls, 10 teacher-forced and 4 observer calls, total 590; within that are 13 prefills and 577 single-token calls. Physical launches include unbounded-in-source autotuning, graph captures/replays and reductions, not those logical numbers. The CPU preflight's histogram byte limit does not include the separate actual tokenizer SHA read (2,183,982 B), parent/source/runtime package metadata and implementation reads; these should be explicitly accounted or described as separately bounded/unknown. Three mmap model loads and associated physical I/O, as well as runtime caches and per-token event instrumentation, retain their actual cost identities.

The author has revised common cache sizing, math-only eager neutral, EOS reporting, observer coverage and module checks. The updated draft remains pending; its math-only eager logits are correctly identified as not a compiled-logits equivalence proof and cache layout is not a per-element KV-write proof. Final audit requires a real V2 parent RUN/EXIT and exact complete final protocol/spec, with no approval implied by the analysis above.
