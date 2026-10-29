# 独立最终身份复核 V2：APPROVE（精确待登记身份）

Reviewer `/root/s2_theory_design_independent`；实现者 `/root/sm120_full_quality_build_terra`。此前科学/实现修订与限制见 [V1 recheck](INDEPENDENT_RECHECK_V1.md)。本次只复核下列明确身份，批准提交登记；**不等于 ACTIVE 登记、调度许可或实际执行**。

- 待登记协议 `config/e0_chipmunk_a100_delta_probe_20261029_pending_review_v1.json` SHA256 `27ccc2a74aa5efd0610f2c8113002d7dc9a8f6d4ba7a75be4482c92fba33da16`；静态 `validate(...,True)` PASS，重算 fingerprint `75bcb4302c617f4f4167a1f0374d218679225967d719a74f33dadad92e786ea2`。
- 与经V1审查草案逐字段比较，只有 `status` 与 private worker SHA 改变。producer `bdc1068d37d23fff0a50aadac222969f996117d467d8deb9a403217b33955759`、kernel `ec2994a7b446d31b7aa967b85a13a528791678c72eb0e93df35e356e2cbfc0d0`、tests `a3efacb5bdff652ab21d13f3a7ec6e591dc2b77bf09b3d0cd284eb90ede6050f`；本地重新核对协议绑定全部源码、parent RUN/EXIT/GEN、两packet metadata和作者源码SHA均通过，未读checkpoint或packet张量。
- 私有 supervisor spec SHA `40135b602ccbec044628d1c0ab2851e07990c839ec36ed952808728b82477b4a`，17项 argv 符合producer精确要求：固定 UUID 分配 GPU0 脚本、taskset 0–3、协议指定 venv Python、`-B`、该producer与确切协议及该spec；scope ID、协议 SHA、cwd、HF/output/RUN路径和1020秒 outer 一致。新输出路径当前本地不存在。私有worker profile 本地SHA与协议 `5a4e5bf320c525deb8f952e22972fd2f8bc8c35fe8f66dcea1db1e64ef8a90d2` 一致；本地 Python/包入口+RECORD 的环境快照及monitor二进制与协议一致。快照并非全部二进制依赖完整哈希证明。
- 协调者报告实际目标四张 A100 当前无 compute 应用、CPU0–15可用；这属于其时点的资源观察，本review未独立访问目标执行节点或核验未来状态。执行前仍必须在目标端依照 `worker_identity`、profile SHA、目标本机环境/monitor、GPU0 UUID、CPU 0–3、空卡锁及新输出检查，变化即不运行。用户最新全局并发上限4，scope GPU1；此单卡锁只管本卡，协调者须继续保证全局上限和与其他任务不冲突。
- 尚无此 scope 的 ACTIVE 登记；登记时绑定本确切协议SHA、该spec SHA、独立review及本V2 recheck，使用与协议一致资源/限额。producer会再次核验ACTIVE和scope fingerprint。任何源码、协议、数据、环境、worker、spec或限制变更均需重新审查。已审scope仅一source/单函数/非raster128行/2步、固定mask，无作者选择器和可外推系统速度。

本复核未运行模型/GPU/科学实验；V1 记录的10项非GPU静态测试通过。实际编译、数值、性能和最终资源占用均待授权执行后取得，不预设成功。
