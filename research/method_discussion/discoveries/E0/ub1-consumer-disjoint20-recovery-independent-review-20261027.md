# disjoint20 part0 独立补全：冻结前审阅

**结论（STATIC/CPU-MOCK）**：交审版本只允许 source0–4 的十条新 Dense/candidate run，旧40范围仅供资产预检；独立 ACTIVE 授权、隔离输出、先扣预算及原生计算复用的路径一致，未发现确定的代码阻断。该结论不证明目标worker资产可用或视频生成通过。最终协议须由协调者冻结、按其精确SHA登记，并在 B worker 完成实际CPU资产预检后才能执行。

当前 driver SHA256 `fb58a94798a23f50b9878dd7115b4b9011c379af69456ccf3f503ac7ea84db9d`，tests SHA256 `75973e27a60cab8603a6a1ce4d0fa5c9fd396a40e71e3593f81362d338c1e2b4`，草案协议 SHA256 `bfd39b2c86a8c7a35740d33526b3e9646a8a94e231bb186797fe55e33114c4ae`；独立执行10项CPU测试通过。作者仍在最终短读时审阅，若任一文件变化应重核版本。

旧40协议和原生 engine 分别按固定文件SHA/大小只读验证；engine以独立 module 对象载入，仅在该对象替换 scope ID 与 ledger 类，原 generation、160 FFN计数、模型数学、初始噪声与10秒 GPU monitor 实现保持原文件身份。新 cohort 由旧source0–4派生但赋予新scope的十个 run/asset/noise ID；固定一GPU、B worker profile 和单独 output root，拒绝原part目录子路径。原part0 FAILED监测收据与退出码1经SHA绑定验证并标为quarantined，不能被重写或重新标为成功。

原 preflight 收到真实 ACTIVE，只进行旧资产校验；其 `authorized` 状态被明确记录为非执行权限。新 scope 的执行入口检查真实 ACTIVE 的当前用户授权、精确协议SHA、十个完整run ID、总预算与一GPU限制；每次视频预扣前重新读取实际 ACTIVE。ledger限定part0、5来源、两臂与独立scope，异常记录沿用旧驱动的预扣不退款和 no-retry。完整每part预算为10视频、40 flow、10原生初始噪声文件、40后步 latent、10 core报告，768MiB HF上限；存在每源已暴露 DEVELOPMENT 身份，旧part1–3加新补全也仍是20来源，不因重跑增加独立 n。

测试覆盖范围授权、旧授权不能冒充新授权、超额和错来源、隔离、冻结资产及旧运行数学复用；CPU模拟不能证明真实模型、设备内存、视频帧、完整后续质量或成本。正式执行还须依实际RUN/退出码、每条noise/视频/core与监控样本判定；旧part0的超时期间隔离状态仍未知。
