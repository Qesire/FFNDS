# 完整归档

[下载Release](https://github.com/Qesire/FFNDS/releases/tag/handoff-20261020)。主分支保留浏览入口，完整服务器快照、来源清单、证据和核验工具在Release ZIP中。

原配置为A100调整，H20尚未验证。

2026-10-20 的 [上游校对补充](upstream_audit/README.md) 是源码恢复的必需补充：首版相对本地ZIP导出，遗漏15个W4A4文件；主分支现提供直接相对固定公开提交的92个差异文件、5份补丁、哈希和恢复工具。原Release ZIP及其SHA256保持原值，未包含此补充。先恢复固定上游，再使用新版差异；不要叠加首版旧ZIP补丁。
