# 启动前 NFS 可变目录扫描修订

协调者发现后台 worker 输出扫描对 `.nfs` 等可变条目执行 `is_symlink` 后 `stat` 存在消失竞态。已仅修改本次新转换 producer 的 `worker_usage`：`lstat` 或 `os.walk` 下降遇 `FileNotFoundError` 时跳过并计 `missing_entries`；所有其他 I/O 异常继续失败。每次持久检查记录本次缺失数，后台 monitor 累加缺失数。输入绑定、固定 fd、输入/输出哈希和 header 的缺失仍为硬失败；预算是采样软上限，消失条目的大小不可观测，不当作被测到的零成本。合成测试覆盖真实扫描中条目在 lstat 前消失、目录遍历 ENOENT 及 PermissionError 硬失败；原作者微型转换测试仍通过。此版本尚未执行转换，须原独立 reviewer 对确切 source/config/job 重新核定。

独立快速复核进一步指出 worker 根目录缺失不能被视作可变后代：现限定 `os.walk` 的 ENOENT 必须有 filename 且为严格后代；根本身前后 `lstat` 硬检查并比较设备/inode，根消失或替换均硬失败；增加根消失合成反例。
