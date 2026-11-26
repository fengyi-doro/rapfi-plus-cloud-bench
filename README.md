# Rapfi Plus 云端对局

GitHub Actions 从同一提交构建 `src_orig` 和 `src_plus`，用随仓库冻结的开局测试 Standard 与 Renju。默认赛程有 500 个不同开局，每个开局交换黑白各下一局。每台 Ubuntu Runner 同时运行四局，每局的两个单线程引擎轮流使用同一 vCPU，关闭后台思考。

在仓库的 Actions 页面选择 **Distributed Rapfi SPRT**，或从本地使用 GitHub CLI。

```powershell
gh workflow run distributed-sprt.yml -R fengyi-doro/rapfi-plus-cloud-bench --ref main -f pairs=500 -f candidate_label=v4 -f candidate_profile=v4 -f time_control=1500+40
```

| 输入 | 含义 |
|---|---|
| `pairs` | 1 至 500 对。分片最多 20 个，对数有余数时依次分给前面的分片。 |
| `candidate_label` | 写入棋谱与统计的候选名称。 |
| `candidate_profile` | 构建当前提交的 V3 或 V4 功能配置。V4 启用 F10。相同的编译参数也传给原版引擎。 |
| `time_control` | Fischer 时钟，单位为毫秒。`1500+40` 表示每方初始 1,500 毫秒，每手加 40 毫秒。 |

测试后续引擎时，先提交候选源码，再选择对应 Git 引用、构建配置和报告名称。构建任务在 QEMU Haswell 上加载双方的 Standard 与 Renju 权重并运行评估。分片核验运行资源和二进制的 SHA-256，再按外部单调时钟记录每手。每个分片独立复盘，汇总任务再次核对全部棋谱、计时、对换色和资源分配。

用运行编号下载战报。报告包含各规则的胜和负、按开局对计算的五项分布、Elo 与 95% 区间、LOS、SPRT 和吞吐。各分片的逐手棋谱保存在同一运行下的 `shard-*` 工件中。

```powershell
gh run download RUN_ID -R fengyi-doro/rapfi-plus-cloud-bench -n pentanomial-elo-los-sprt -D cloud_report
```

比赛使用已冻结的 500 对开局。61,490,563 字节的 Gomocup 2025 历史原始归档仅在重新生成历史数据时使用，其官方来源与校验值见[开局来源记录](data/v2/opening_design.json)。

引擎源码采用 GPL，官方权重采用 CC0。详见[引擎许可证](Copying.txt)与[权重许可证](Networks-LICENSE.txt)。
