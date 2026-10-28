# B0末层状态采集：独立冻结前代码评议

**SRC/CPU测试结论**：先前发现的两个真实阻断已修复；审阅版本没有新增静态阻断，可在资产最终冻结和精确授权后进入受限原生采集。11项合成CPU测试独立运行通过，exit 0；另注意首次误用不存在的测试文件名得到0 tests，不计为验证。此评议未执行模型、GPU或真实flow。

版本：tap `7051af16de7fb32e879829047baefbbc089e83d385674ce80d0fe5da7acb85d3`，driver `a81466c2852f6375d0a9d243f61fd54c750d45e8a27c2114cc34503380215e42`，tests `d3a0d77c23de08cec8bd4205e07d133a6972e166a3cd8f77f4219db4e6959cab`，配置草案 `6785c1a97eb9fc148d5750173b9520d5882b549e5af746302dfe79245b7f3b54`；最终协议因冻结字段修改将另有SHA，需重新登记。作者提供的目标worker真实CPU preflight返回exit0、原协议SHA匹配、无Torch导入/CUDA/native调用；这是资产字节就绪，不是原生计算通过。

已修：原 `WanPostInfer` 返回FP32 flow，driver与tap现要求并原样保存FP32，其他指定原生BF16边界分开检查；测试调用真实原post类验证该dtype。monitor关闭、查询、违规与adapter cleanup异常被记录后传播，避免失败状态对应成功退出。原 `prepare_latents` 实际调用被包装并严格预扣/计数一次FP32噪声，而非由prepare次数推断。

原 `step_pre` 前导入step2保存FP32，先校验BF16精确扩展位模式，再原位恢复BF16并核raw SHA；step0源真实initial、step3源真实post2均核对pre-state，原step_post结果比对历史post-state SHA。观察器在最终block原e5写回后、原final norm前保存实际u；分别保留r/Y/gate行并用BF16原顺序做逐字节写回核验。head记录实际runtime权重 `[5120,64]` 的shape/stride及bias，转置后与原 `[64,5120]` checkpoint tensor SHA相符；原对象head结果经原post传递，旧历史raw head/flow未保存，不能把post-state一致改写为旧flow字节相等。

两source×step0/3为4 flow，实际每flow40 block、80 FFN linear/QDQ及320 non-FFN linear/QDQ；预扣和返回后观察分离，setup 800权重传输单列，失败不退款、不自动重试。CLIP、VAE decode、训练、分解、敏感探针与视频不在scope。实际u保存、历史匹配、监控及结果受真实RUN和launcher结果约束；2来源数据只生产B1所需状态，不判来源泛化或运动质量。

## scheduler instrumentation 修订追审

作者在冻结前发现原版本会把临时挂载的 `prepare`、`prepare_latents` callable 当成未知scheduler对象拒绝；先前结论适用于旧版本，不能用于运行旧版本。修订tap SHA `7b266df5657ca7bfc24a307334e19d4b55e6223cb0f81fc1fd8e6f79dbc84c10`，driver仍为上述 `a81466…`，tests `d9a4ac85ac372d08e51176e4d980e935cc341f2b7f23dffa2c2565a77af9fedf`，草案config `d1ffba115f9a0e293cf0ee05cf302b4d586a10d63bb247080b9afbd9a3821a76`。仅对这两个命名instrumentation允许callable并记录qualname及其源码SHA，未知普通字段继续拒绝；既有 `transformer_infer` 是明确标识的非数值引用。独立12项合成测试 exit 0，含未知callable拒绝；目标worker修订版CPU预检exit0，0 Torch/CUDA/native。未发现新增静态阻断，可按最终配置SHA冻结登记；真实4 flow仍未执行。
