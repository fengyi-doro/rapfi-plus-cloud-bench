"""Publish the two separate local campaigns, the frozen binaries and verified cloud smoke."""
from collections import Counter
from datetime import datetime
import json
from datasets import ROOT, DATA, dump, sha


def get(name):
    return json.loads((ROOT/name).read_bytes())


def ci(x):
    return f"[{x[0]:+.2f}, {x[1]:+.2f}]"


def los(x):
    return f"{100*x:.4f}%" if .99995<=x<1 else f"{100*x:.2f}%"


def checkpoint(directory):
    d=get(directory+"/design.json")
    ids=set(d["opening_ids"][:200])
    games=[json.loads(p.read_bytes()) for p in (ROOT/directory/"games").glob("*.json")]
    first=[g for g in games if g["opening"] in ids]
    assert Counter(g["rule"] for g in first)=={1:200,2:200}
    start=datetime.fromisoformat(get(directory+"/timing.json")["start"])
    return (max(datetime.fromisoformat(g["ended"]) for g in first)-start).total_seconds(),games


def standings(scores):
    rows=["| 组别 | 局数 | v3 胜 / 和 / 负 | 得分 | Elo | 95% 配对区间 | LOS |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("All","Standard","Renju"):
        s=scores[name]
        rows.append(f"| {'合计' if name=='All' else name} | {s['games']} | {s['W']} / {s['D']} / {s['L']} | "
                    f"{s['W']+.5*s['D']:g} | {s['elo']:+.2f} | {ci(s['ci95_elo'])} | {los(s['los'])} |")
    return "\n".join(rows)


def run():
    files={"orig":"pbrain-rapfi-orig.exe","v2":"pbrain-rapfi-plus-v2.exe","v3":"pbrain-rapfi-plus-v3.exe"}
    hashes={k:sha(ROOT/v) for k,v in files.items()}
    original_sources=get("data/orig_files.json")
    assert all(sha(ROOT/"src_orig"/name)==digest for name,digest in original_sources.items())
    assert sha(ROOT/"pbrain-rapfi-plus-v2.pdb")==sha(ROOT/"pbrain-rapfi-plus.pdb")
    main=get("match_v3_fast/summary.json")
    basic=get("match_v3/summary.json")
    direct=get("match_v3_long/summary.json")
    s,first=main["scores"]["All"],basic["scores"]["All"]
    audit=get("match_v3_fast/replay.json")
    four,games=checkpoint("match_v3_fast")
    old_four,_=checkpoint("match_v3")
    final_sec=get("match_v3_fast/timing.json")["wall_seconds"]
    old_sec=get("match_v3/timing.json")["wall_seconds"]
    v2_sec=get("match_v2/timing.json")["wall_qpc_ns"]/1e9
    p=Counter("P" if max(g["cores"].values())<4 else "E" for g in games)
    fixed=get("bench/v3/interleaved/summary.json")
    fd=fixed["subsets"]["all"]
    a,b=fd["orig_v3"],fd["v2_v3"]
    preliminary=get("bench/v3/release/summary.json")["comparisons"]
    f9=get("bench/v3/f9/summary.json")
    f=f9["comparisons"]["v2_f9"]
    late=get("bench/v3/endgame/summary.json")
    h,t,proof=[get("validation/v3/"+x+".json") for x in ("heads","tactics","pairing")]
    c=proof["counts"]
    timing=get("bench/v3/proof/summary.json")["classes"]
    tight,wide=timing["both_at_most_24"],timing["either_at_most_64_other_over_24"]
    workflow=get("validation/v3/workflow.json")
    smoke=get("validation/v3/cloud_smoke/report/summary.json")
    shard_path="validation/v3/cloud_full_shard"
    shard=get(shard_path+"/design.json")
    shard_timing=get(shard_path+"/timing.json")
    shard_audit=get(shard_path+"/replay.json")
    assert hashes["v2"]==get("data/v2/release.json")["exe"]["plus"]==sha(ROOT/"pbrain-rapfi-plus.exe")
    assert hashes==fixed["exe"] and f9["admitted"] and all(x["root_mismatches"]==x["node_mismatches"]==0 for x in fd.values())
    assert h["mismatches"]==t["errors"]==proof["false_claims"]==get("validation/v3/line_gate.json")["mismatches"]==0
    assert h["exe"]["plus-v3"]==t["exe"]["v3"]==proof["playing_sha256"]==hashes["v3"]
    assert all(m["binary_hashes"]["Windows"]["v3"]==hashes["v3"] for m in (main,basic,direct))
    assert workflow["workflow_sha256"]==sha(ROOT/".github/workflows/distributed-sprt.yml") and smoke["scores"]["All"]["games"]==8
    assert shard["shard"]==11 and shard["shards"]==20 and shard_timing["count"]==50
    assert shard["exe"]==smoke["binary_hashes"]["Windows"] and len(shard["lanes"])==2 and shard_audit["referee_mismatches"]==0
    if b["ci_wall_pct"][0]>0:
        v2_precision="此次相邻交替测速确认 v3 相对 v2 净提速。"
    elif b["ci_wall_pct"][1]<0:
        v2_precision="此次相邻交替测速确认 v3 相对 v2 出现净回退。"
    else:
        v2_precision="v3 对 v2 的墙钟区间跨零，当前样本无法确定整机净增速。"
    pts=s["W"]+.5*s["D"]
    why="，".join(f"{k} {v}" for k,v in sorted(main["reasons"].items()))
    late_rows=["| 同深度残局重放 | v2 节点 | v3 节点 | 节点减少 | v2 / v3 思考时间 | 思考墙钟减少 |","|---|---:|---:|---:|---:|---:|"]
    for group,cmps in late["subsets"].items():
        x=cmps["v2_v3"]
        n0,n1=(x["totals"][v]["nodes"] for v in ("v2","v3"))
        t0,t1=(x["totals"][v]["wall_ns"]/1e9 for v in ("v2","v3"))
        late_rows.append(f"| {'合计' if group=='all' else group} | {n0:,} | {n1:,} | {100*(1-n1/n0):+.2f}% | {t0:.3f} / {t1:.3f} 秒 | {x['wall_saved_pct']:+.2f}% |")
    lines=[
    "# Rapfi Plus v3 实测报告",
    f"**最终赛场用 {final_sec/60:.2f} 分钟完成 600 局。v3 对 orig 得分 {pts:g}∶{600-pts:g}，{s['elo']:+.2f} Elo，"
    f"95% 配对区间 {ci(s['ci95_elo'])}，LOS {los(s['los'])}。** 均衡的前 400 局用时 {four/60:.2f} 分钟。"
    f"同核深度 12 的 v3 对 orig NPS 为 {a['wall_nps_pct']:+.2f}%，对 v2 为 {b['wall_nps_pct']:+.2f}%（95% 区间 {ci(b['ci_wall_pct'])}%）。"
    " [v3 程序](pbrain-rapfi-plus-v3.exe) · [v2 备份](pbrain-rapfi-plus-v2.exe)",
    "## 常规赛与并行吞吐",standings(main["scores"]),
    f"300 个同开局换色对的五项分布（LL、LD、DD/WL、WD、WW）为 **{s['penta']}**。95% 区间用 100,000 次按规则分层的整对重抽，"
    f"LOS 用配对标准误的正态近似。按母开局聚类重抽得到 {ci(main['mother_cluster_ci95_elo'])} Elo。"
    f"SPRT 的 H0 为 0、H1 为 +5 Elo，α＝β＝0.05，最终 LLR {main['sprt']['llr']:+.3f}，边界 ±2.944，"
    f"按开局书顺序的判定为 {main['sprt']['decision']}。 [逐对得分](match_v3_fast/summary.json) · [棋谱](match_v3_fast/psq)",
    "\n".join(["| 同一时限，每方 1.5 秒加每手 40 毫秒 | 200 Standard + 200 Renju | 300 Standard + 300 Renju |",
    "|---|---:|---:|",f"| v2 历史赛场，两对性能核 | — | {v2_sec/60:.2f} 分 |",
    f"| v3 初版，五对静态分配 | {old_four/60:.2f} 分 | {old_sec/60:.2f} 分 |",
    f"| v3 最终版，五对领取下一完整换色对 | {four/60:.2f} 分 | {final_sec/60:.2f} 分 |"]),
    f"最终 600 局吞吐为 v2 历史赛场的 {v2_sec/final_sec:.2f} 倍，计入引擎启动、复位和写谱。初版也实跑了同一套 600 局，"
    f"{first['W']} 胜、{first['D']} 和、{first['L']} 负，{first['elo']:+.2f} Elo。两轮复用同一批母局，各自统计。"
    " [初版结果](match_v3/summary.json) · [最终计时](match_v3_fast/timing.json)",
    f"本机 Intel Core Ultra X7 358H 的 P 核组为 0/1、2/3，E 核组为 4/5、6/7、8/9，Mix11 使用 14/15。"
    f"10 个引擎各用一个搜索线程和 8 MiB 哈希。两腿同步换色换核，双方并行复位，裁判确认时钟参数后才发出着法命令。"
    f"最终 P/P 对战 {p['P']} 局、E/E 对战 {p['E']} 局。逐手复盘 {audit['board_moves_replayed']:,} 手定时着法，"
    f"超时 {audit['hard_flags']} 局，物理核重叠 {audit['overlapping_physical_core_assignments']} 次，棋谱及裁判差异 {audit['referee_mismatches']}。"
    " [赛场设计](match_v3_fast/design.json) · [独立复盘](match_v3_fast/replay.json)",
    "强制胜须有合法成五点、双成五点或可重放的连续冲四证书。连续冲四对所有合法防守分支复核。显式协议和棋须双方连续发出 DRAW_OFFER。"
    "双方各连续两次报零分时，还须已走至少 120 手且每个五格窗口都有双色棋子，或在 Standard 提交覆盖双方全部存活窗口的配对证书。"
    f"最终终局计数为 {why}。初版 R027_0 第 225 手的黑方长连发生在唯一空点，独立枚举确认黑方已无合法着。"
    f"最终记录 {audit['illegal_moves']} 次非法坐标或禁手，其中 {len(audit['no_legal_black_reply'])} 次黑方无合法选择。"
    "裁判控制用例复核了单边提议、过期提议、连续双方提议、可立即成五，以及第 120 手双方报零仍有存活窗口的情况。"
    " [赛场](scripts/arena_fast.py) · [裁判证书](scripts/arena_certificates.py) · [协议控制用例](validation/v3/adjudication_controls.json) · [原始禁手棋谱](match_v3/games/R027_0.json)",
    "## 两个模块与正确性门禁",
    "**M-F9 已合入。** 原始半线键直接映射到商类行偏移或列号。含映射的总有效载荷从 846,067 B 降至 41,658 B。"
    "Board::move 更新四个方向的棋型，防守着生成器也通过相同索引读取双边防守掩码。"
    "Freestyle、Standard、Renju 分别为 10,515、16,539、14,604 B，数组另作 64 字节对齐。"
    "本机 P 核 L1D 为 48 KiB、E 核为 32 KiB，每种规则的工作集都能容纳。实际缓存驻留率未测。"
    " [战术线表](src_plus/game/pattern.h) · [缓存实测](validation/v3/cache.json)",
    f"只给 v2 加 F9，同一性能核 2、24 局面、深度 12、8 轮交替。v2 为 {f['nps']['v2']:,.0f}，F9 为 {f['nps']['f9']:,.0f} NPS，"
    f"净升 **{f['wall_nps_pct']:.2f}%**，95% 配对区间 {ci(f['ci_wall_pct'])}%。每工作线程周期节点数变化 {f['cycle_nps_pct']:+.2f}%，"
    f"区间 {ci(f['ci_cycle_pct'])}%。双方各 {f['totals']['v2']['nodes']:,} 节点，根着、根分和节点计数相同。"
    "复核编译后的全部 279,633 个线键，双色标签与两种防守掩码零差异。保留条件是墙钟区间下界大于零且周期点估计为正。"
    " [预设条件](bench/v3/f9/design.json) · [单模块原始结果](bench/v3/f9/summary.json) · [全键校验](validation/v3/line_gate.json)",
    "**M-F8 已合入。** 15×15 棋盘的根局面至少 120 手、至少一色存活窗口 ≤64，才切入残局搜索。非根节点剩余深度至少 3，"
    "双方各 ≤24 个存活窗口才试低秩后备。Standard 接受双证精确和棋，或以行棋方不败证书在 beta ≤ VALUE_DRAW 截断。"
    "Renju 只在白方行棋时使用后一种下界。每个根局面初始化资格一次，各轮根搜索据此择路。中盘递归实例在编译时消除证明调用。"
    "遇到空着、排除着、特殊和棋计分、有限局长、换手、多 PV、受限着法和数据库模式另有保护。论文提供双色秩二博弈的数学背景。"
    " [论文](https://arxiv.org/abs/2503.18163v2) · [搜索接入](src_plus/search/ab/search.cpp) · [边界与路径缓存](src_plus/search/ab/pairing/search_hook.h)",
    f"在效率核 12 对 480 个历史局面热测，每个局面与调用方向重复 64 次。双方均 ≤24 时，批均值的 P50/P95 为 "
    f"{tight['p50_ns']:.0f}/{tight['p95_ns']:.0f} ns。至少一方 ≤64 且另一方 >24 时为 {wide['p50_ns']:.0f}/{wide['p95_ns']:.0f} ns。"
    "计时从已构造的局面开始。 [分类器计时](bench/v3/proof/summary.json)",
    "\n".join(["| 门禁 | 样本 | 结果 |","|---|---|---|",
      f"| 发布程序原始网络头 | {h['positions']} 个三规则局面，{h['wdl_words']} 个 W/L/D、{h['both_color_empty_scores']:,} 个双方空点 Policy 位型 | 逐位零差异 |",
      f"| VCF | {t['cases']} 题，{t['direct_vcf_entries_equal']} 个直接入口比对，{t['legal_defenses']:,} 个合法防点 | 根着、分值、防守图零差异 |",
      f"| 残局证明 | {proof['games']} 盘真实长局，{c['nodes_1']+c['nodes_2']:,} 个后期非终局，{c['coordinate_witnesses']:,} 份配对证书 | 逐窗核验零错误 |"]),
    f"残局门禁另含 {c['minimax_positions']} 个局面的完整规则穷举（{c['minimax_states']:,} 个状态）、{c['decisive_tactics']} 个独立可定胜负的历史战术局面、"
    f"{proof['rank_graphs']:,} 个秩二博弈和 8 个必胜／必败的合法控制局面。其中 3 个只能靠低秩后备取得下界，抽检胜负零误判。"
    " [网络位型](validation/v3/heads.json) · [VCF](validation/v3/tactics.json) · [残局证明](validation/v3/pairing.json)",
    f"用宽窗口和 fail-high 窗口分别检查历史节点。Standard 的 {c['nodes_1']:,} 个节点中有 {c['exact_1']:,} 个取得精确和棋，"
    f"{c['lower_1']:,} 个支持单边下界。两组可重叠。Renju 的 {c['nodes_2']:,} 个节点中有 {c['lower_2']:,} 个取得白方下界，"
    f"精确和棋截断 {c['exact_2']} 个。历史样本中有 {c['rank_only_lower']} 个下界仅靠低秩后备取得。"
    " [历史节点判定](validation/v3/pairing.json) · [必胜和必败控制局面](validation/v3/rank_controls.json)",
    "## 同核深度 12 与残局直接试验",
    "\n".join(["| 发布程序 | 精确节点 / 思考墙钟 NPS | 相对 orig | 相对 v2 |","|---|---:|---:|---:|",
       f"| orig | {a['nps']['orig']:,.0f} | 基准 | — |",f"| v2 | {b['nps']['v2']:,.0f} | {fd['orig_v2']['wall_nps_pct']:+.2f}% | 基准 |",
       f"| v3 | {a['nps']['v3']:,.0f} | {a['wall_nps_pct']:+.2f}% | {b['wall_nps_pct']:+.2f}% |"]),
    f"24 个局面在性能核 2 各重复 12 轮。三个程序按局面相邻交替，每次清哈希，从发布程序及对应 PDB 读取精确节点。"
    f"每版本累计 {a['totals']['v3']['nodes']:,} 节点，根着、根分及节点数全等。v3 对 orig 和 v2 的配对墙钟区间分别是 "
    f"{ci(a['ci_wall_pct'])}% 和 {ci(b['ci_wall_pct'])}%。{v2_precision} 首轮按整组轮换的 6 轮实测分别是 "
    f"{preliminary['orig_v3']['nps']['orig']:,.0f}、{preliminary['v2_v3']['nps']['v2']:,.0f} 和 {preliminary['orig_v3']['nps']['v3']:,.0f} NPS，"
    f"其中 v3 对 v2 的区间为 {ci(preliminary['v2_v3']['ci_wall_pct'])}%。"
    " [相邻交替原始数据](bench/v3/interleaved/trials.jsonl) · [最终测速](bench/v3/interleaved/summary.json) · [首轮测速](bench/v3/release/summary.json)",
    "**v3 对 v2** 从 v2 实战预选 32 个不同的 120–170 手开局，其中 29 个母局历史上和棋。每个开局双向换色，关闭全部早判。"
    f"64 局为 **1 胜、62 和、1 负**，双方各得 32 分，LOS 50%，未测出 v3 的棋力优势。"
    f"全部换色对打平，按非中性配对出现频率的精确上界给出 {ci(direct['scores']['All']['ci95_elo'])} Elo 区间。"
    " [残局开局及来源](data/v3/long_openings.json) · [直接赛完整统计](match_v3_long/summary.json)",
    "\n".join(late_rows),
    f"32 个残局另做 4 轮同核深度 12 重放。128 次配对观察有 {late['excluded_observations'].get('all',0)} 次未满足双方均达目标深度且节点为正的条件。"
    f"纳入观察中有 {late['subsets']['all']['v2_v3']['root_mismatches']} 次根着或根分变化。表中节点量及墙钟分别比较。"
    " [逐局面记录](bench/v3/endgame/trials.jsonl) · [残局成本汇总](bench/v3/endgame/summary.json)",
    "## 云端千局工作流",
    "将项目置于公开 GitHub 仓库，在 Actions 页手动运行 **Distributed Rapfi SPRT**。Ubuntu 构建作业使用 Zig 0.14.1 为 x86-64 Linux 和 Windows "
    "交叉编译 orig 与 v3，按源代码和参数缓存二进制，并校验 SHA-256。20 个分片分配 500 个不同开局，每片 25 个换色对。"
    "10 台 Linux、10 台 Windows 合计 500 局 Standard、500 局 Renju。汇总作业核验双方颜色与物理核、时钟、裁决、开局唯一性、"
    "平台内二进制身份及完整分片，上传逐局棋谱、五项分布、Elo、LOS、首次 SPRT 边界及 Markdown 报告。"
    " [工作流](.github/workflows/distributed-sprt.yml) · [分片开局](data/v3/cloud_openings.json) · [汇总器](scripts/distributed_sprt.py)",
    f"本地通过 actionlint {workflow['actionlint']}、YAML 解析、20 分片唯一性与三种 SPRT 决策控制。交叉编译的 ELF 和 PE 在 WSL Linux 与原生 Windows "
    "各实跑、复盘 4 局，8 局跨平台汇总通过。烟测采用 750 ms 加每手 20 ms。公开 Runner 的 1,000 局与 5 分钟目标尚待实跑。"
    "工作流同时报告首片开赛至末片完赛、工作流启动至末局完赛、启动至生成已核验报告的墙钟，分别核对两个 300 秒对战目标。"
    " [本地语法与分片校验](validation/v3/workflow.json) · [跨平台实跑报告](validation/v3/cloud_smoke/report/summary.json)",
    f"另在本机为 Windows 第 11 分片分配 4 个物理效率核（4/5、6/7），使用工作流的正式开局、交叉编译产物与 1.5 秒加每手 40 毫秒时限。"
    f"25 个换色对全部完成，50 局用时 {shard_timing['wall_seconds']:.2f} 秒，复盘 {shard_audit['board_moves_replayed']:,} 手，"
    f"超时 {shard_audit['hard_flags']} 局。公开 VM 若只暴露一对不同物理核，赛场将只开一对引擎。"
    " [正式分片计时](validation/v3/cloud_full_shard/timing.json) · [正式分片复盘](validation/v3/cloud_full_shard/replay.json)",
    "GitHub 公共仓库的标准 x64 Linux 与 Windows Runner 各提供 4 vCPU，标准 Runner 计算免费。GitHub Free 的标准并发作业上限为 20。"
    "80 vCPU 是 20 片同时获调度时的容量，所需时间受队列、VM 核拓扑、冷编译和传输影响。"
    " [Runner 规格](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) · [并发限制](https://docs.github.com/en/actions/reference/limits)",
    "## 发布与复跑",
    "\n".join(["| 发布文件 | SHA-256 |","|---|---|",*[f"| {v} | {hashes[k]} |" for k,v in files.items()]]),
    f"核对原版源文件清单，{len(original_sources)} 个文件的 SHA-256 全部相符。v2 备份的 EXE 和 PDB 与原发布件逐字节一致。"
    "本地发版沿用 Zig 0.14.1、AVX2/FMA/BMI2、O3、关闭浮点收缩。位型和精确节点门禁使用 Windows SDK 的 CDB 及发布 PDB。"
    " [构建清单](build/v3/manifest.json) · [原始 v2 发布哈希](data/v2/release.json)",
    chr(96)*3+"powershell\n"+"\n".join([
       "python scripts/build_v3.py f9","python scripts/build_v3.py v3","python scripts/node_symbols.py f9 v3",
       "python scripts/v3_line_gate.py","python scripts/probes.py v3","python scripts/bit_gate.py --candidate plus-v3 --out validation/v3",
       "python scripts/tactics.py --candidate v3 --out validation/v3 --core 0","python scripts/v3_pairing_gate.py",
       "python scripts/v3_benchmark.py f9 --core 2","python scripts/v3_interleaved.py interleaved --core 2","python scripts/v3_interleaved.py endgame --core 2",
       "python scripts/arena_fast.py --out rerun_v3 --pairs 300 --exclude 14 15",
       "python scripts/distributed_sprt.py rerun_v3 --book data/v2/openings.json --pairs 300 --shards 1 --out rerun_v3"])+"\n"+chr(96)*3]
    dump(DATA/"v3/release.json",dict(exe=hashes,f9_admitted=True,f9_gate="bench/v3/f9/summary.json",
         k_min_ply=120,pair_window_cap=64,rank_window_cap=24,formal_match="match_v3_fast",direct_match="match_v3_long",
         thousand_game_cloud_timing_measured=False))
    path=ROOT/"MATCH_REPORT_V3.md"
    tmp=path.with_suffix(".tmp")
    content=("\n\n".join(lines)+"\n").encode("utf8")
    tmp.write_bytes(content)
    try:
        tmp.replace(path)
    except PermissionError:
        # An open Windows preview can permit writes while denying rename.
        # Keep the staged report until the named document has been verified.
        if path.read_bytes()!=content:
            with path.open("r+b") as output:
                output.write(content)
                output.truncate()
            if path.read_bytes()!=content:
                raise OSError("report write verification failed")
        tmp.unlink()
    print(path)


if __name__=="__main__":
    run()

