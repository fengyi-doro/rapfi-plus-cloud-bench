# Rapfi Plus v5 · F23 云端千局实证

**本轮保留 V4＋F10，F23 新增模块为 0。** 五个单模块和两个 B 组合均未通过对 V4＋F10 的晋级条件。最终对原版的认证成绩为 **1T +18.78 Elo，4T +62.87 Elo**。同一可执行文件完成两场认证，SHA-256 为 `299e3ba3daaed9d3d317285e0c349b4543938b707829729120ecc651841b60d4`。[冠军源码](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/tree/release/rapfi-plus-v5) · [Linux AVX2 认证包](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45852450242/artifacts/15132724884)

## 冠军认证

每场 1,000 局，原版对手固定 1T。最后一列比较同一组开局上本轮与历史纪录的 Elo 差，括号内为配对 95% 区间。

| 配置 | 胜 / 和 / 负 | 得分率 | 全场 Elo [95% CI] | Standard Elo [95% CI] | Renju Elo [95% CI] | 相对历史纪录 ΔElo [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| 1T vs 1T | 436 / 182 / 382 | 52.70% | +18.78 [+6.25, +31.35] | +25.06 [+7.64, +42.60] | +12.51 [-5.56, +29.95] | -10.12 [-27.28, +6.98] |
| 4T vs 1T | 490 / 199 / 311 | 58.95% | +62.87 [+49.32, +76.61] | +53.93 [+34.86, +72.61] | +71.89 [+52.51, +91.71] | -10.47 [-28.80, +7.94] |

历史纪录为 1T +28.90、4T +73.34 Elo。本轮复用了历史冠军二进制，两次历史差值的区间均覆盖零。1T 复测在单项筛选后派发，补充组合的选择只依据各自的直接对照。[历史 1T](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45832263308) · [历史 4T](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45833773520) · [本轮 1T](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45849673591) · [本轮 4T](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45851428940)

## 千局消融

每一行均单独完成 500 对换色开局，Standard 与 Renju 各 500 局。实胜率为 W/N，净胜率为 (W−L)/N，得分率为 (W＋D/2)/N。Elo 按 400 log₁₀(S/(1−S)) 换算，95% 区间按规则分层重抽整对开局 100,000 次。

| 模块 | 对手 | 胜 / 和 / 负 | 实胜率 | 净胜率 | 得分率 | 全场 Elo [95% CI] | Standard / Renju Elo |
|---|---|---:|---:|---:|---:|---:|---:|
| A · F15 | orig | 452 / 155 / 393 | 45.20% | +5.90% | 52.95% | +20.52 [+8.34, +33.11] | +18.08 / +22.96 |
| A · F15 | V4＋F10 | 401 / 176 / 423 | 40.10% | -2.20% | 48.90% | -7.64 [-19.48, +4.17] | +1.39 / -16.69 |
| B1 · 奇异应手 | orig | 455 / 177 / 368 | 45.50% | +8.70% | 54.35% | +30.30 [+15.65, +45.07] | +40.48 / +20.17 |
| B1 · 奇异应手 | V4＋F10 | 428 / 169 / 403 | 42.80% | +2.50% | 51.25% | +8.69 [-5.91, +23.31] | +12.51 / +4.86 |
| B2 · 负向历史 | orig | 446 / 162 / 392 | 44.60% | +5.40% | 52.70% | +18.78 [+4.52, +33.11] | +25.76 / +11.82 |
| B2 · 负向历史 | V4＋F10 | 421 / 163 / 416 | 42.10% | +0.50% | 50.25% | +1.74 [-12.86, +15.99] | -0.69 / +4.17 |
| B3 · 残差修正 | orig | 436 / 149 / 415 | 43.60% | +2.10% | 51.05% | +7.30 [-6.25, +20.87] | +8.34 / +6.25 |
| B3 · 残差修正 | V4＋F10 | 415 / 171 / 414 | 41.50% | +0.10% | 50.05% | +0.35 [-13.56, +14.60] | +6.95 / -6.25 |
| B4 · 稳定性 LMR | orig | 455 / 166 / 379 | 45.50% | +7.60% | 53.80% | +26.46 [+12.86, +40.13] | +43.30 / +9.73 |
| B4 · 稳定性 LMR | V4＋F10 | 426 / 175 / 399 | 42.60% | +2.70% | 51.35% | +9.38 [-4.52, +23.31] | +13.90 / +4.86 |
| B1×B4 | orig | 443 / 170 / 387 | 44.30% | +5.60% | 52.80% | +19.48 [+4.52, +34.16] | +29.25 / +9.73 |
| B1×B4 | V4＋F10 | 402 / 190 / 408 | 40.20% | -0.60% | 49.70% | -2.08 [-15.99, +11.82] | -2.08 / -2.08 |
| B1+B2+B3+B4 | orig | 445 / 161 / 394 | 44.50% | +5.10% | 52.55% | +17.73 [+3.47, +32.05] | +9.73 / +25.76 |
| B1+B2+B3+B4 | V4＋F10 | 415 / 171 / 414 | 41.50% | +0.10% | 50.05% | +0.35 [-13.21, +13.90] | +8.34 / -7.64 |

赛前指定 A 以直接对 V4＋F10 的 Renju Elo 为主终点，B 以全场 Elo 为主终点。晋级要求主终点 95% 区间下限高于零，且对两个对手的全场 Elo 均为正。B1 与 B4 是单项点估计最高的两项，随后冻结其组合及完整 B 组，各对两个对手重新打满千局。逐项区间未经多重比较校正。每种规则的 250 对开局来自 12 个母局面。逐规则胜和负、置信区间、LOS、五项分布和每对开局得分已归档。[单项 10,000 局](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45847727373) · [组合 4,000 局](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45850420120) · [逐场数据](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/tree/experiments/f23-search-v5/data/f23/results)

## F15 的战术收益与实战代价

9,069 个 Renju 局面复现了 **172→252** 个白方强胜证明，新增 80 个，其中 41 个轮到白走、39 个轮到黑应四。252 份证书全部通过原生回放和独立规则复核。在新增的 41 个白走局面上，深度 4 的原版和 V4 均选中 **31/41** 个可验证胜着、报告 **29/41** 个杀棋。接入 F15 后，两项都达到 **41/41**。B1 的深度 8 对照与 V4 均选中 32/41 个证书胜着。[逐局面诊断](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45847727373/artifacts/15131308945)

实际 Renju 对局中，A 对原版为 **+22.96 Elo [+5.56, +40.48]**，对 V4＋F10 为 **−16.69 Elo [−34.86, +1.39]**。本轮未达到 +45 至 +60 Elo。对 V4 的 500 局 Renju 共触发 1,609,745 次 F15 查询，访问 10,193,387 个求证器白方节点。求证和证书回放合计 19.78 秒，平均 12.29 µs/次。原研究的离线全根查询中位数为 0.90 µs，P95 为 100.50 µs。

白方根节点只采用属于原生合法候选表的证书着。主搜和 VCF 先经过原有精确门禁，仅在经回放的杀棋分数下界达到 β 时截断。回放检查全盘黑方精确五反击、递归禁手和最大手数。无赛钟根分析完整求证，计时根查询最多 65,536 个白方节点，分配本手最优预算的八分之一，限制在 1–8 ms。预算耗尽后继续原生搜索。[F15 宿主实现](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/blob/0b10bec/src_plus/search/ab/f15_hook.h)

## 搜索改动与节点平衡

B1 在原有奇异延伸门槛前一层，对可信战术 TT 着追加排除搜索，次优间隙额外提高 20 分，验证通过后延长 1 ply。B2 在攻击着产生 β 截断时，对先前已搜索的安静失败着施加半额引力负奖。B3 用每线程、每方 2,048 槽的棋形残差表纠正静态评估，修正上限为 ±24，TT 保存原始评估。B4 用 Beta(3,3) 先验估计根着和评分的稳定性，对晚出的安静非 PV 着调整最多 0.20 ply。四项均保留原有 Razoring 和 Futility 路径。[Rapfi 实验源码](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/blob/0b10bec/src_plus/search/ab/search.cpp) · [Stockfish 17 残差与历史实现](https://github.com/official-stockfish/Stockfish/blob/sf_17/src/search.cpp)

同一批 32 个根局面按两种执行顺序各测一次，深度上限为 7。下表以各自云端批次的 V4 为对照。原生节点列使用 Rapfi 的协议计数，A 另访问了 12,710 个求证器白方节点，耗时列包含求证工作。

| 候选 | 原生节点变化 | 平均 ms/根 | 墙钟变化 |
|---|---:|---:|---:|
| A | -6.99% | 4.444 | +9.60% |
| B1 | -0.14% | 4.048 | -0.17% |
| B2 | +0.04% | 4.082 | +0.67% |
| B3 | 0.00% | 4.025 | -0.75% |
| B4 | +0.01% | 4.026 | -0.71% |
| B1×B4 | -0.14% | 4.155 | -0.42% |
| 完整 B 组 | +0.09% | 4.152 | -0.51% |

## 裁判、资源与复现

20 台四核 GitHub runner 提供 80 vCPU。消融赛每台开四条单核赛道，认证赛每台独占一局。双方使用冻结的 Mix9SVQ 权重、8 MiB 置换表和 1,500 ms＋每手 40 ms 的外部时钟。A 对 V4 的千局开赛至完赛为 59.52 秒，峰值 80 局同时进行。各场实际墙钟和并发数均保存在云端汇总中。[历史 63.69 秒高速场](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45827645286)

**16 场共 16,000 局，8,000 对换色记录，独立裁判逐手复盘 1,183,502 手。裁决分歧 0，JSON 与 PSQ 棋谱分歧 0，CPU 分配重叠 0。** 共 17 次超时判负、34 次禁着或非法落子判负。其中 34 次经独立裁判枚举全盘，确认黑方已没有合法应手。全部判负均计入 Elo。历史纪录差值按同开局配对重抽 30,000 次。结论适用于这批冻结开局、赛钟和线程配置。[全部换色账本与分片裁判记录](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/tree/experiments/f23-search-v5/data/f23/results)

首次云编译发现旧头文件重复引用。修正后的首次证书回归发现 1,024 节点上限漏掉两条长线，分别需要 16,635 和 56,259 个白方节点。两次失败均发生在开赛前。修订后才启动正式筛选。[首次构建](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45846476702) · [首次构建修正](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45846805357) · [预算回归](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45847129841) · [正式单项构建](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45847415698) · [组合构建](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45850203845)

初版认证包的 SHA256SUMS 行分隔有误。最终交付包合并两场认证记录，云端解包逐文件校验，整包 SHA-256 为 `81bfc788c05aa0ee61f87fd51337bca8903c6a80ed7b631174fa53bdeeb0de62`。[最终打包与验包](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45852450242)

本机子进程实测为 Idle，亲和性仅含逻辑核 15，Windows 效率类别为 0，OMP_NUM_THREADS=1，GPU 后端已屏蔽。编译、求证、对局及独立回放均在 GitHub Actions 执行。[实验分支](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/tree/experiments/f23-search-v5) · [云端认证包](https://github.com/fengyi-doro/rapfi-plus-cloud-bench/actions/runs/45852450242/artifacts/15132724884)
