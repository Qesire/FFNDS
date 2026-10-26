# 固定四输入坐标的非线性功能：同rank比较仍有较大残差

**新的结构是：四个实际W1投影的低阶非线性特征携带了超出共同仿射基与同维随机空间的功能信息，但这两个固定小函数族仍留下约43%–46%的原簇输出能量误差。** 相同15维预算下，分离幂基优于含交互的二次基，当前数据不支持“加入二次交互就更好”。竞争解释是收益主要来自已观察到的四个anchor自身非线性，而其余12个神经元仍有独立信息；另一个解释是坐标选择遗漏了输入流形的关键方向。便宜的下一判别应先区分这两点，再考虑持出函数拟合或扩大原生算子。

## DISCOVERY：用有明确省算预算的函数空间取代任意一维插值

固定原16个neuron，来源0/12×step0/1各32原始行合并为128行，取四个原anchor投影`T`。比较共同常数、`[1,T]`、15列joint quadratic、15列separable多项式（各坐标1–3次幂及前两坐标4次幂），另有包含常数的同rank预seed随机行空间。没有来源/step指示，没有逐case拟合。

令原16系数为`C`、原W2列为`V`，`K=VᵀV`，则共同raw输出空间的样本内数值投影误差为`tr[K Cᵀ(I−PΨ)C]`。该计算等价于样本内最小二乘能力检查，不能因未保存readout bank就称其无需拟合意义的优化。四个case的gate只对这个相同raw解重新评分，不是共同gated最优。

## EVIDENCE：共同目标、同rank及均值基线

| 特征空间 | 数值rank | continuous raw相对SSE | 保存pre-Q2 | 保存post-Q2 |
| --- | ---: | ---: | ---: | ---: |
| constant_only | 1 | 85.5012% | 85.5138% | 85.5278% |
| anchor_affine | 5 | 65.3436% | 65.3675% | 65.8716% |
| joint_quadratic | 15 | 45.8738% | 45.9061% | 46.2139% |
| seeded_random_rank15 | 15 | 75.3412% | 75.3538% | 75.2899% |
| separable15 | 15 | 42.6921% | 42.7256% | 43.0518% |

joint/separable的数值rank均稳定为15，条件数约6.29/13.56；共同affine条件数约1.81，均远离退化区间。同rank随机控制只取一个预seed，提供容量参考而不是统计显著性检验。continuous→preQ2差异很小，postQ2略有额外残差；不能由这一观察宣布Q2是下游失败主因。原全FFN global bias排除，此表只涉及这16个neuron，不是整个FFN。

在各case当前gate下，joint对continuous的相对SSE为39.48%–54.08%，separable为33.63%–50.29%。相对共同128行均值的创新能量，joint还剩41.20%–66.21%，separable35.09%–64.69%；收益并非只来自任意非零均值。共同constant raw SSE85.50%，共同affine65.34%，随机15维75.34%，joint45.87%，separable42.69%。部分改善可由anchor自身非线性解释，尚未做该因果消融或持出预测。

### 数值rank反例与修复保持可审计

独立review构造四个完全相同anchor：真实joint/separable多项式span为3/5维。若把SVD所有正浮点奇异值当真实模式再QR，可补入数值零空间，把原族正交target错误拟合到近零残差。当前实现以standard-eps可识别span为主，匹配对应随机rank；所有正浮点模式仅标为`EXPANDED_NUMERICAL_SPAN_SENSITIVITY`。退化/病态时家族比较保持OPEN。11项CPU软件测试含此反例和微小真实方向敏感性；本次真实小面板恰好rank稳定，允许有范围的数值比较，仍无严格家族排除或任意非线性不可行定理。

## EXECUTION：一次固定CPU研究完成

已登记[固定协议](../method_discussion/discoveries/E0/independent-fixed-coordinate-basis-oracle-plan-20261026_v1.json)，按临时持续授权执行一次。实际19,310,208个显式矩乘MAC、177次显式矩乘，数值阶段2.752秒（含imports/mmap，预检hash除外），RSS693,153,792字节；小特征SVD/QR另在RUN逐项记账。0模型/GPU/nativeQDQ/新权重分解/部署bank/视频/重试。原分片/操作数和所有旧协议字节保持。

[公开JSON](E0_FIXED_COORDINATE_BASIS_ORACLE.json)为原RUN精确副本，SHA256 `09e3acd05bf782405ebc39dddadb662cf8530677d24690d319520da1ca914c6c`。输入/行号、全部投影/数值敏感性/门控与创新分母、成本及环境都在原记录。独立[最终源码review v7](../method_discussion/discoveries/E0/independent-fixed-coordinate-basis-source-review-20261026_v7.md)保留反例、修复和11项软件测试的真实范围。

若一类共享坐标最终有效，16原神经元的两侧投影成本`32MD`可以变为四个输入坐标加14个非常数输出基的`18MD`，另计特征构造、量化、packing与必要余项；目前并无相应部署bank、原生速度或视频质量结果。这次较大残差优先推动信息坐标与功能归因，而不是直接生成全160位置的多项式替代。图锚点F04/F06/F08R/F08/F09为posthoc CPU能力证据；通向真正省算和完整视频的边保持conditional/OPEN。
