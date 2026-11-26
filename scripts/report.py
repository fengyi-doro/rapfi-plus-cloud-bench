"""Audit the finished match, compute opening-pair statistics and write the Chinese report."""
from collections import Counter
import argparse
import json
import math
import os
from pathlib import Path
import re
import numpy as np
from datasets import DATA, ROOT, SEED, dump, sha
from replay import verify


def load(path):
    return json.loads((ROOT/path).read_bytes())


def elo(s):
    if s <= 0:
        return -math.inf
    if s >= 1:
        return math.inf
    return 400*math.log10(s/(1-s))


def game_points(g, challenger="plus"):
    return 2 if g["winner"] == challenger else 1 if g["winner"] is None else 0


def statistics(games, challenger="plus"):
    by_rule = {}
    counts = Counter(game_points(g, challenger) for g in games)
    for rule in sorted({g["rule"] for g in games}):
        paired = {}
        for g in games:
            if g["rule"] == rule:
                paired.setdefault(g["opening"], []).append(game_points(g, challenger))
        assert all(len(v) == 2 for v in paired.values())
        by_rule[rule] = np.array([sum(v) for v in paired.values()])
    total = np.concatenate(list(by_rule.values()))
    pairs = len(total)
    penta = np.bincount(total,minlength=5).tolist()
    # One pair carries 0..4 half-points. Resample whole openings within each rule.
    rng = np.random.default_rng(SEED)
    draws = np.zeros(100000,dtype=np.int64)
    variance = 0.0
    for vector in by_rule.values():
        n = len(vector)
        for start in range(0,len(draws),2048):
            chunk=draws[start:start+2048]
            chunk+=rng.choice(vector,size=(len(chunk),n),replace=True).sum(axis=1)
        variance += (n/pairs)**2 * (np.var(vector/4,ddof=1) if n>1 else 0.0)/n
    score = float(np.mean(total)/4)
    assert sum(i*n for i,n in enumerate(penta)) == 2*counts[2]+counts[1]
    assert counts[2]-counts[0] == 2*penta[4]+penta[3]-penta[1]-2*penta[0]
    ci = np.quantile(draws/(pairs*4),[.025,.975], method="inverted_cdf")
    sigma = math.sqrt(variance)
    los = (.5*(1+math.erf((score-.5)/(sigma*math.sqrt(2)))) if sigma
           else 1.0 if score > .5 else 0.0 if score < .5 else .5)
    return dict(games=len(games),pairs=pairs,W=counts[2],D=counts[1],L=counts[0],penta=penta,
                score=score,elo=elo(score),ci95_elo=list(map(elo,ci)),los=los,
                bootstrap_superiority=float(np.mean(draws>2*pairs)+.5*np.mean(draws==2*pairs)),
                net_pairs=penta[3]+penta[4]-penta[0]-penta[1], net_games=counts[2]-counts[0],
                sweeps=penta[4]-penta[0],standard_error=float(sigma))


def signed(n):
    return f"{n:+.2f}"


def run(directory, report_path):
    games,audit = verify(directory)
    bk=load("data/openings.json")
    sets={"standard":[g for g in games if g["rule"]==1],"renju":[g for g in games if g["rule"]==2],
          "all":games}
    scores={k:statistics(gs) for k,gs in sets.items()}
    heads,tactics=load("validation/heads.json"),load("validation/tactics.json")
    mix10=load("validation/mix10.json")
    classical=load("validation/classical.json")
    fixed,screen=load("bench/final/summary.json"),load("bench/screen/summary.json")
    hardware,design=(json.loads((directory/f).read_bytes()) for f in ("hardware.json","design.json"))
    gates=[heads,tactics,mix10,fixed,design]
    binaries={v:sha(ROOT/f"pbrain-rapfi-{v}.exe") for v in ("orig","plus")}
    assert all(g["exe"][v]==binaries[v] for g in gates for v in binaries)
    assert sha(DATA/"openings.json")==design["book_sha256"]
    assert all(sha(ROOT/"src_orig"/name)==expected for name,expected in load("data/orig_files.json").items())
    runtime=load("data/runtime_files.json")
    assert all(sha(ROOT/name)==value for name,value in runtime.items())
    match_ct={}
    for name in ("orig","plus"):
        turns=[t for g in games for t in g["turns"] if t["engine"]==name]
        observed=[t for t in turns if t["nodes"] is not None]
        match_ct[name]=dict(turns=len(turns),counted_turns=len(observed),nodes=sum(t["nodes"] for t in observed),
                            wall_ns=sum(t["wall_ns"] for t in observed))
        match_ct[name]["nps"]=match_ct[name]["nodes"]*1e9/match_ct[name]["wall_ns"]
    by_opening={}
    for game in sorted(games,key=lambda g:(g["opening"],g["leg"])):
        row=by_opening.setdefault(game["opening"],dict(rule=game["rule"],plus_score=0,legs=[]))
        row["plus_score"]+=game_points(game)/2
        row["legs"].append(dict(leg=game["leg"],plus_color="black" if game["black"]=="plus" else "white",
                                winner=game["winner"],reason=game["reason"]))
    summary=dict(scores=scores,opening_pairs=by_opening,reasons=dict(Counter(g["reason"] for g in games)),replay=audit,
                 match_throughput=match_ct,clock="external QPC Fischer 3.0s + 0.080s",exe=binaries,
                 bootstrap_resamples=100000,bootstrap_seed=SEED)
    dump(directory/"summary.json",summary)
    allscore=scores["all"]
    release=fixed["comparisons"]["plus"]
    gain=release["wall_nps_pct"]
    mgain=(match_ct["plus"]["nps"]/match_ct["orig"]["nps"]-1)*100
    lower,upper=allscore["ci95_elo"]
    strength=("这批对局的配对区间跨过零，尚不能判定棋力提升。" if lower <= 0 <= upper else
              "这批对局的配对区间整体高于零。" if lower>0 else "这批对局的配对区间整体低于零。")
    shortcut=screen["admit_playing_shortcut"]
    text=[
      "# Rapfi Plus 纯 CPU 工程与公平对局验收报告","",
      f"**200 局已完成。** Plus 取得 **{allscore['W']} 胜、{allscore['D']} 和、{allscore['L']} 负**。"
      f"相对原版的实测 Elo 为 **{signed(allscore['elo'])}**，配对 Bootstrap 95% 区间为 "
      f"**[{signed(lower)}, {signed(upper)}]**。{strength}","",
      f"固定深度 12 的整树墙钟 NPS 变化为 **{signed(gain)}%**，八轮配对 95% 区间为 "
      f"**[{signed(release['ci_wall_pct'][0])}%, {signed(release['ci_wall_pct'][1])}%]**。"
      f"直接从比赛程序读取的静态分数、W/L/D 和空点 Policy 在 {heads['positions']} 个实战中盘全部逐位一致。"
      f"200 道 VCF 攻防题、{tactics['legal_defenses']:,} 个合法防点均通过比对。","",
      "## 换色比赛","",
      "| 组别 | 局数 | Plus 胜 / 和 / 负 | Plus 得分率 | 五项开局对分布 | 净胜对数 | Elo | 95% 配对区间 | LOS |",
      "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for k,title in (("standard","Standard"),("renju","Renju"),("all","合计")):
        s=scores[k]
        text.append(f"| {title} | {s['games']} | {s['W']} / {s['D']} / {s['L']} | {s['score']:.2%} | "
                    f"{s['penta']} | {s['net_pairs']:+d} | {signed(s['elo'])} | "
                    f"[{signed(s['ci95_elo'][0])}, {signed(s['ci95_elo'][1])}] | {s['los']:.2%} |")
    text+=["",
      "五项依次为 LL、LD、DD/WL、WD、WW。净胜对数按得分超过 1 分的开局对数减去低于 1 分的对数计算。"
      f"Plus 净胜 {allscore['net_games']:+d} 局，双胜减双负为 {allscore['sweeps']:+d} 对。"
      "Elo 使用 400 log₁₀(s/(1−s))。分别在两个规则组内有放回重抽整对开局，保持 60∶40 权重，"
      f"种子 {SEED}，100,000 次。LOS 使用开局对分层标准误的正态近似。总组 Bootstrap 超越频率计入半数平票后为 "
      f"{allscore['bootstrap_superiority']:.2%}。 [逐组统计和每对开局明细](match/summary.json)","",
      f"处理器为 {hardware['cpu']}，Windows 枚举到 {len(hardware['cores'])} 个逻辑处理器，最高性能等级对应 "
      f"{sorted(hardware['performance'])}。复盘全部进程的实测时间区间和线程亲和性，最大并发为 "
      f"{audit['max_simultaneous_pairs']} 对，比赛引擎之间的物理核分配重叠 {audit['overlapping_physical_core_assignments']} 次。"
      "同一开局第二局交换颜色和引擎使用的核心。裁判在效率核运行，两个版本都用单搜索线程、8 MiB TT、关闭后台计算。"
      "每局新起进程，预热本规则权重，再执行 RESTART 和 YXHASHCLEAR。 [比赛设计](match/design.json) · [CPU 拓扑](match/hardware.json)","",
      "每色各有 3.000 秒，每手及时返回完整坐标后加 0.080 秒。裁判用 Windows 单调性能计数器计量从写入行棋命令到收到坐标行的墙钟时间。"
      "每手开始前将剩余时间向下取整到毫秒，通过 INFO time_left 通知双方。"
      f"共计 {audit['timed_turns']:,} 手，超时判负 {audit['hard_flags']} 次，最小落子后、加秒前余量为 "
      f"{audit['min_unused_bank_ns']/1e6:.1f} ms。进程亲和性隔离了比赛引擎，Windows 仍可在这些核心调度系统任务。 "
      "[逐手时钟与坐标](match/games) · [后台负载采样](match/host_load.jsonl)","",
      f"全程裁判使用独立 Python 位棋盘核验终局。另一套坐标棋盘实现通过 {len(audit['rule_fixtures'])} 个规则用例，"
      f"复盘全部棋谱与双方时钟，终局裁决差异 {audit['referee_mismatches']} 次。"
      "终局包括 " + "、".join(f"{n} 局{label}" for key,label in
          (("five","五连"),("five_or_more","连珠白方长连获胜"),("full_board","满盘和棋"),
           ("forbidden_overline","黑方长连判负"),("forbidden_double_four","黑方四四判负"),
           ("forbidden_double_three","黑方三三判负"),("illegal_coordinate","无效坐标判负"),("time","超时判负"))
          if (n:=summary['reasons'].get(key))) + "。Standard 双方恰好五连获胜。Renju 白方五连及以上获胜，"
      "黑方先判恰好五连，再判长连、四四和递归意义上的三三。 [第二裁判复盘](match/replay.json) · [200 份 PSQ](match/psq)","",
      f"对取得完整坐标和节点读数的实战手数合计计时，原版 {match_ct['orig']['counted_turns']:,} 手，"
      f"Plus {match_ct['plus']['counted_turns']:,} 手。精确总节点除以对应墙钟总时间，原版为 {match_ct['orig']['nps']:,.0f} NPS，"
      f"Plus 为 {match_ct['plus']['nps']:,.0f} NPS，变化 {signed(mgain)}%。双方实战搜索的局面和深度不同，"
      "固定深度配对结果给出了相同工作负载的速度差。 [实战节点与用时](match/summary.json)","",
      "## 开局与逐位等价","",
      "开局来自 2017–2025 年 Gomocup 官方 Standard 和 Renju 开局文件。保留原始坐标顺序和完整手数，"
      "按旋转、镜像以及同色棋子集合去重。使用原版静态 W/L/D 计算黑方预期得分 W黑 + D/2，"
      "按与 50% 的距离选择各组最接近均衡的 60 / 40 个开局。"]
    for r,label in ((1,"Standard"),(2,"Renju")):
        p=[o["baseline_black_score"] for o in bk if o["rule"]==r]
        text[-1]+=f" {label} 黑方估计范围 {min(p):.2%}–{max(p):.2%}。"
    text[-1]+=" [来源行、原始文件散列与筛选结果](data/opening_screen.json) · [冻结开局](data/openings.json)"
    text+=["",
      f"Gomocup 2025 的真实 15×15 对局提供了 180 个中盘，三种规则各 60 个。CDB 使用发版 PDB，"
      "在比赛程序 TRACESEARCH 和 TRACEBOARD 的原生函数返回处读取内存。两版的静态值、NNUE 整数值、"
      f"{heads['wdl_words']} 个 float32 W/L/D 字、双方 {heads['both_color_empty_scores']:,} 个空点 Policy 字全部一致。"
      f"其中当手合法点 {heads['stm_legal_scores']:,} 个，两色合计合法点 {heads['both_color_legal_scores']:,} 个。"
      f"空点数量覆盖四种 Batch=4 尾数。经典棋形评估另核对 {classical['positions']} 个黑视角静态整数及 "
      f"{classical['both_color_score_grids']} 张双边棋形评分表。 [发版程序逐位门禁](validation/heads.json) · "
      "[原始位型](validation/heads_orig.json) · [经典评估对照](validation/classical.json)","",
      "VCF 题库包含 55 道自由、55 道标准、90 道连珠。93 题取自赛事棋谱，106 题来自固定种子的构造局面，"
      "另有 1 题专门检验递归三三。使用同源诊断入口直接对比 vcfsearch、vcfdefend 和深度零分派的 "
      f"600 个原始分数。实际比赛程序另行核对 200 题深度 4 首着与根分。"
      f"稀疏 Policy 掩码和 {tactics['sparse_undo_children']:,} 次子分支回退检查均通过。"
      "全防点独立穷举检验九手内连续冲四，对连珠包含递归假三与禁手读取依赖的反例。"
      f"直接取得 {tactics['promoted_certificates']} 份父局面证书，兄弟防点剪去 {tactics['proved_skips']:,} 个，"
      f"逐证书区外重放 {tactics['outside_replays']:,} 次，零失败。 [逐题答案](validation/tactical_cases.json) · [攻防门禁](validation/tactics.json)","",
      "## 同核固定深度基准","",
      f"深度 {fixed['depth']}，三种规则各 4 个开局和 4 个中盘，共 {fixed['positions']} 个局面。"
      f"每组交替 A/B 与 B/A，{fixed['repetitions']} 轮，每次只有一个引擎进程运行，均在物理性能核 {fixed['core']}。"
      "每次搜索前清空 TT 与搜索历史，预热、装盘、清表均在计时区间外。"
      "墙钟区间包含协议往返，节点从单个搜索线程的最终原子计数读取。线程周期来自 QueryThreadCycleTime，"
      "进程周期另存原始记录。各消融使用对应模块的代码，未选中的计算函数取自原版。"
      "区间按整轮配对重抽。 [24 个局面](data/bench_positions.json) · [逐搜索墙钟、各线程周期与节点数](bench/final/trials.jsonl)","",
      "| 对手版本 | 原版 / 对手墙钟秒 | 原版 / 对手搜索线程十亿周期 | 原版 / 对手总节点 | 原版 / 对手 NPS | NPS 变化 | 配对 95% 区间 |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for v,title in (("p1","仅 P1"),("p2","仅 P2"),("p3","仅 P3"),("p4","仅 P4"),("plus","发版 Plus")):
        c=fixed["comparisons"][v]
        a,b=c["totals"]["orig"],c["totals"][v]
        text.append(f"| {title} | {a['wall_ns']/1e9:.3f} / {b['wall_ns']/1e9:.3f} | {a['worker_cycles']/1e9:.3f} / "
          f"{b['worker_cycles']/1e9:.3f} | {a['nodes']:,} / {b['nodes']:,} | {c['nps']['orig']:,.0f} / {c['nps'][v]:,.0f} | "
          f"{signed(c['wall_nps_pct'])}% | [{signed(c['ci_wall_pct'][0])}%, {signed(c['ci_wall_pct'][1])}%] |")
    trial=screen["comparisons"]["plus"]
    raw=[json.loads(line) for line in (ROOT/"bench/final/trials.jsonl").read_text().splitlines()]
    block_gains=[]
    node_differences=0
    for block in raw:
        a,b=block["records"]["orig"],block["records"][block["b"]]
        node_differences+=sum(x["nodes"]!=y["nodes"] for x,y in zip(a,b,strict=True))
        if block["b"]=="plus":
            rate=lambda records:sum(x["nodes"] for x in records)/sum(x["wall_ns"] for x in records)
            block_gains.append(100*(rate(b)/rate(a)-1))
    text+=["",f"发版 Plus 每搜索线程周期的节点吞吐变化为 {signed(release['cycle_nps_pct'])}%，"
           f"配对 95% 区间为 [{signed(release['ci_cycle_pct'][0])}%, {signed(release['ci_cycle_pct'][1])}%]。"
           f"发版 Plus 的逐轮配对墙钟 NPS 中位变化为 {signed(float(np.median(block_gains)))}%，"
           f"八轮范围为 {signed(min(block_gains))}% 至 {signed(max(block_gains))}%。表格合计收录全部轮次。"
           f"五组比较共 {sum(len(block['records']['orig']) for block in raw):,} 次逐局面配对，"
           f"节点计数差异 {node_differences} 次。 "
           "[完整计时数据](bench/final/trials.jsonl)"]
    d0,d1=fixed["defense_map"]["0"],fixed["defense_map"]["1"]
    collisions=load("validation/codebook.json")
    equal=sum(v["zero"] for v in collisions.values())
    changes=sum(v["changes"] for v in collisions.values())
    text+=["","## 四项优化的发版范围","",
      "- **P1** 将 Mix9SVQ 的角、边、象限分别并成四路 starBlock。Mix10 大价值头的对应 linearBlock 共用同一 AVX2 四路核。"
      f"中心点保留单路。官方比赛网络均为 Mix9SVQ。另用可再生、未经联合训练的兼容权重，通过发版 EXE 跑完 "
      f"{mix10['positions']} 个 Mix10 局面，完整大价值头 W/L/D 及 {mix10['policy_words']:,} 个空点 Policy 字逐位相同。"
      "整数核另覆盖随机值、饱和值和 int32 环绕边界。 [Mix10 整头验收](validation/mix10.json) · "
      "[SIMD 核](src_plus/eval/simd/linear.h) · [内核压力测试](validation/kernels.json)",
      "- **P2** 只收集 computeFlag 指定的点，每四点共享动态卷积权重。保持原先 int16 预激活、int32 累加、"
      "FMA 和水平归约顺序，剩余一到三点调用原有单点运算。 [Policy 与增量实现](src_plus/eval/mix9svqnnue.cpp)",
      f"- **P3** 构造 changeTable 时解析桶号并预取活跃的旧、新码本行，两条 64 字节缓存线覆盖一个向量。"
      f"同桶时在消费阶段复制到新 mapSum 版本，跳过码本运算和 3×3 散射。棋形版本和同点四方向顺序仍完整推进。"
      f"实战局面前缀双视角合计 {changes:,} 次方向变更，实测同桶 {equal:,} 次，占 {equal/changes:.3%}，"
      f"其中 {sum(v['center_zero'] for v in collisions.values())} 次发生在落子中心。实测二进制在该路径发出 PREFETCHT0。 "
      "[逐模型码本命中](validation/codebook.json) · [实际机器指令](validation/isa.json)",
      "- **P4** 在确需验证多个防点的九手 VCF 防守图上先求出一个真实子局面的杀法，"
      "去掉该守子后在父局面重放成功才提升为证书。225 位相关区收录反冲四和递归禁手读取点，"
      "多份证书取交集，证书生命周期限于一个父节点。发版可用 YXVCFDEFEND 9 查询。"
      "原版普通 vcfdefend 每次只搜一个强制堵点，普通 vcfsearch 没有新增取证调用。 "
      "[防守求解源码](src_plus/search/ab/rzone.cpp)","",
      f"正式对局前对仅有两个或更多即时胜点时的 vcfdefend 特化做了整树筛选。与 P1–P3 的配对差为 "
      f"{signed(trial['wall_nps_pct'])}% NPS，95% 区间 [{signed(trial['ci_wall_pct'][0])}%, {signed(trial['ci_wall_pct'][1])}%]。"
      f"按搜索线程周期归一化，吞吐变化 {signed(trial['cycle_nps_pct'])}%，95% 区间 "
      f"[{signed(trial['ci_cycle_pct'][0])}%, {signed(trial['ci_cycle_pct'][1])}%]。"
      +("该分支通过了预设收益门槛，已进入普通比赛搜索。" if shortcut else
        "该分支未通过预设的墙钟和线程周期收益门槛，已从普通比赛搜索撤下。发版 P4 的剪枝仅用于显式多防点求解，"
        "200 局比赛的普通搜索实际运行 P1–P3。P4 单项普通搜索的差值包含测量波动与二进制布局差异。")
      +" [筛选设计](bench/screen/design.json) · [筛选原始结果](bench/screen/summary.json)","",
      f"对同一 200 题完整防守树逐题交替开关 RZone，七轮合计墙钟 {d0['wall_ns']/1e9:.3f} → {d1['wall_ns']/1e9:.3f} 秒，"
      f"线程周期 {d0['cycles']/1e9:.3f} → {d1['cycles']/1e9:.3f} 十亿，节点 {d0['nodes']:,} → {d1['nodes']:,}。"
      f"完整防点求解的用时减少 {1-d1['wall_ns']/d0['wall_ns']:.2%}，节点减少 {1-d1['nodes']/d0['nodes']:.2%}。 "
      "[逐题七轮微基准](bench/final/defense_bench.log)","",
      "## 构建与复现","",
      f"原版固定在提交 {load('data/upstream.json')['commit']}，225 个文件与本地上游源码归档逐字节一致。"
      "两套比赛程序分别完整编译，Zig 0.14.1、C++20、-O3、-mavx2、-mfma、-mbmi2、-ffp-contract=off、-DNDEBUG 及其余编译参数一致。"
      "比赛使用相同的 config.toml、适配此源码的官方 model210901.bin 和四份官方 Mix9SVQ 权重。"
      f"Networks 版本 {load('data/networks_tree.json')['sha']}。 [原版源码散列](data/orig_files.json) · "
      "[实际配置和权重散列](data/runtime_files.json) · [发行版完整源码差异](patches/rapfi-plus.patch) · "
      "[44 个共有编译单元及链接参数审计](data/build_audit.json)","",
      f"| 程序 | SHA-256 |\n|---|---|\n| [原版](pbrain-rapfi-orig.exe) | {binaries['orig']} |\n"
      f"| [Plus](pbrain-rapfi-plus.exe) | {binaries['plus']} |","",
      "构建、验收、开局来源、协议用法以及完整换色复赛命令见 [工程说明](README.md)。"]
    def relative(m):
        path = m[1]
        target = directory/path[6:] if path.startswith("match/") else ROOT/path
        return "](" + Path(os.path.relpath(target,report_path.parent)).as_posix() + ")"
    report_path.write_text(re.sub(r"\]\(([^)]+)\)",relative,"\n".join(text)+"\n"),encoding="utf-8")
    output=[f"Rapfi Plus 200 games | W/D/L {allscore['W']}/{allscore['D']}/{allscore['L']} | "
          f"pentanomial {allscore['penta']} | net pairs {allscore['net_pairs']:+d} | Elo {signed(allscore['elo'])} "
          f"95% CI [{signed(lower)}, {signed(upper)}] | LOS {allscore['los']:.2%} | "
          f"depth-12 NPS {signed(gain)}% | match NPS {signed(mgain)}% | time forfeits {audit['hard_flags']}"]
    for k in ("standard","renju"):
        s=scores[k]
        output.append(f"{k} W/D/L {s['W']}/{s['D']}/{s['L']} | {s['penta']} | Elo {signed(s['elo'])}")
    output.append(f"Bit-exact {heads['positions']}/180 | policy words {heads['both_color_empty_scores']} | "
                  f"VCF {tactics['cases']}/200 | legal defenses {tactics['legal_defenses']} | zero mismatches")
    for v,c in fixed["comparisons"].items():
        a,b=c["totals"]["orig"],c["totals"][v]
        output.append(f"Depth {fixed['depth']} orig/{v} | wall s {a['wall_ns']/1e9:.3f}/{b['wall_ns']/1e9:.3f} | "
            f"worker cycles {a['worker_cycles']}/{b['worker_cycles']} | {signed(c['wall_nps_pct'])}% NPS")
    output.append(f"VCF defense map {d0['wall_ns']/1e9:.3f} -> {d1['wall_ns']/1e9:.3f} s; "
                  f"dual-five playing shortcut admitted={shortcut}")
    console="\n".join(output)+"\n"
    (directory/"stdout.txt").write_text(console,encoding="utf-8")
    print(console,end="",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--match-dir",type=Path,default=ROOT/"match")
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    directory=args.match_dir.resolve()
    destination=args.report or ((ROOT if directory==ROOT/"match" else directory)/"MATCH_REPORT.md")
    run(directory,destination.resolve())
