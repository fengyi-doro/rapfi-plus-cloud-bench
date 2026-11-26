"""Independently replay the fixed 600-game season and publish score, cycle, throughput and provenance."""
import collections
import argparse
import os
from datetime import datetime
import json
import re
import zipfile
from pathlib import Path
import numpy as np
from datasets import DATA,ROOT,SEED,canonical,dump,sha
from replay import verify
from report import statistics,elo,game_points,signed

M=ROOT/"match_v2"
B=ROOT/"bench/v2"
V=ROOT/"validation"

def load(p):return json.loads(p.read_bytes())

def audit_book(rows):
    design=load(DATA/"v2/opening_design.json")
    archive=DATA/"v2/gomocup2025results.zip"
    assert sha(archive)==design["archive_sha256"]
    with zipfile.ZipFile(archive) as z:
        for rule,label in ((1,"standard"),(2,"renju")):
            group=[r for r in rows if r["rule"]==rule]
            assert len(group)==len({canonical(r["moves"]) for r in group})==150
            opening_file="openings/openings_"+label+".txt"
            source=z.read(opening_file).splitlines()
            counts=collections.Counter(r["mother"] for r in group)
            assert sorted(counts.values())==[12]*6+[13]*6
            for r in group:
                xy=list(map(int,re.findall(rb"-?\d+",source[r["mother"]-1])))
                mother=[x+7+15*(y+7) for x,y in zip(xy[::2],xy[1::2],strict=True)]
                assert canonical(mother)==canonical(r["moves"][:len(mother)])
                if r["continuation_plies"]:
                    prefix=z.read(r["source"]).splitlines()[1:1+len(r["moves"])]
                    clean=[tuple(map(int,l.split(b","))) for l in prefix]
                    assert [x-1+15*(y-1) for x,y,t in clean]==r["moves"]
                else:
                    assert r["moves"]==mother and r["source"]==opening_file
    return design


def mother_interval(games,book,challenger="plus"):
    """Bootstrap the 12 published mother openings within each rule, retaining their descendants."""
    rng=np.random.default_rng(SEED+1)
    draws=np.zeros(100000)
    index={r["id"]:r for r in book}
    for rule in (1,2):
        scores=np.zeros(12)
        lengths=np.zeros(12)
        for g in games:
            if g["rule"]!=rule:continue
            n=index[g["opening"]]["mother"]-1
            scores[n]+=game_points(g,challenger)/2
            lengths[n]+=1
        assert np.all(lengths>0)
        clusters=rng.integers(0,12,(len(draws),12))
        draws+=.5*scores[clusters].sum(axis=1)/lengths[clusters].sum(axis=1)
    return list(map(elo,np.quantile(draws,[.025,.975],method="inverted_cdf")))


def rate_of(a,b):return (b/a-1)*100
def reversed_gain(x):return (1/(1+x/100)-1)*100

def run(directory=M):
    M=directory.resolve()
    games,audit=verify(M)
    assert len(games)==600
    book=load(DATA/"v2/openings.json")
    book_design=audit_book(book)
    hardware,design,timing=map(load,[M/"hardware.json",M/"design.json",M/"timing.json"])
    perf=sorted(hardware["performance"][:design["lanes"]*2])
    before_load={cpu:float(np.mean([sample[cpu] for sample in hardware["load_pct_before"]])) for cpu in perf}
    load_samples=[json.loads(line)["cpu_pct"] for line in (M/"host_load.jsonl").read_text(encoding="utf8").splitlines() if line]
    live_load={cpu:float(np.mean([sample[cpu] for sample in load_samples])) for cpu in perf}
    load_text=lambda readings:"、".join(f"{value:.1f}%（核 {cpu}）" for cpu,value in readings.items())
    wall_minutes=(timing["wall_qpc_ns"]/60e9 if timing.get("wall_qpc_ns") else
                  (datetime.fromisoformat(timing["end"])-datetime.fromisoformat(timing["start"])).total_seconds()/60)
    s={name:statistics(gs) for name,gs in (("Standard",[g for g in games if g["rule"]==1]),
                ("Renju",[g for g in games if g["rule"]==2]),("total",games))}
    cluster_ci=mother_interval(games,book)
    head,tactical,m10=map(load,[V/"heads.json",V/"tactics.json",V/"mix10.json"])
    benchmark_path=ROOT/design.get("benchmark_path","bench/v2/final")
    fixed=load(benchmark_path/"summary.json")
    build=load(DATA/"build_audit.json")
    release=load(DATA/"v2/release.json")
    for name in ("orig","plus"):
        h=sha(ROOT/f"pbrain-rapfi-{name}.exe")
        assert all(r["exe"][name]==h for r in [design,head,tactical,m10,fixed,release])
    assert sha(ROOT/"pbrain-rapfi-plus-v1.exe")==fixed["exe"]["v1"]==release["exe"]["v1"]
    assert sha(ROOT/"patches/rapfi-plus.patch")==build["source_patch_sha256"]
    assert all(sha(ROOT/"src_plus"/p)==h for p,h in release["source"].items())
    assert all(sha(ROOT/"src_orig"/p)==h for p,h in load(DATA/"orig_files.json").items())
    assert all(sha(ROOT/p)==h for p,h in load(DATA/"runtime_files.json").items())
    assert build["identical_flags"] and sha(DATA/"v2/openings.json")==design["book_sha256"]==book_design["match_book_sha256"]
    assert (ROOT/"src_orig/search/ab/search.cpp").read_bytes()==(ROOT/"src_plus/search/ab/search.cpp").read_bytes()
    orig_policy=(ROOT/"src_orig/eval/mix9svqnnue.cpp").read_text(encoding="utf8")
    plus_policy=(ROOT/"src_plus/eval/mix9svqnnue.cpp").read_text(encoding="utf8")
    policy=lambda x:x[x.index("void Accumulator::evaluatePolicy("):x.index("Evaluator::Evaluator(")]
    assert policy(orig_policy)==policy(plus_policy)
    def throughput(group,v):
        turns=[t for g in group for t in g["turns"] if t["engine"]==v]
        measured=[t for t in turns if t["nodes"] is not None]
        ns=sum(t["wall_ns"] for t in measured)
        nodes=sum(t["nodes"] for t in measured)
        return dict(turns=len(turns),measured=len(measured),nodes=nodes,wall_ns=ns,nps=nodes*1e9/ns)
    through={v:throughput(games,v) for v in ("orig","plus")}
    by_rule={label:{v:throughput([g for g in games if g["rule"]==rule],v) for v in ("orig","plus")}
             for rule,label in ((1,"Standard"),(2,"Renju"))}
    live_gain=rate_of(through["orig"]["nps"],through["plus"]["nps"])
    pent=s["total"]
    outcomes=collections.Counter(g["reason"] for g in games)
    reason_names={"five":"五连","five_or_more":"连珠白方长连获胜","full_board":"满盘和棋","time":"超时",
                  "illegal_coordinate":"坐标非法","forbidden_overline":"黑方长连禁手",
                  "forbidden_double_four":"黑方四四禁手","forbidden_double_three":"黑方三三禁手"}
    reasons="、".join(f"{n} 局{reason_names[k]}" for k,n in outcomes.items())
    ledger={}
    for g in sorted(games,key=lambda g:(g["opening"],g["leg"])):
        row=ledger.setdefault(g["opening"],dict(legs=[],score=0))
        row["legs"].append(dict(leg=g["leg"],black=g["black"],winner=g["winner"],reason=g["reason"]))
        row["score"]+=game_points(g)/2
    passed=bool(pent["ci95_elo"][0]>0 and pent["los"]>.975)
    summary=dict(scores=s,opening_pairs=ledger,replay=audit,clock=design,wall_minutes=wall_minutes,
                 match_nps=through,match_nps_by_rule=by_rule,match_nps_change=live_gain,outcomes=outcomes,cluster_ci=cluster_ci,
                 criteria_met=passed,bootstrap_resamples=100000,bootstrap_seed=SEED,
                 cluster_seed=SEED+1,exe=release["exe"],performance_core_load_pct=dict(before=before_load,during=live_load,samples=len(load_samples)))
    dump(M/"summary.json",summary)
    old=load(ROOT/"archive/v1/match/summary.json")["scores"]["all"]
    old_speed=load(ROOT/"archive/v1/bench/final/summary.json")["comparisons"]["plus"]["wall_nps_pct"]
    primary=fixed["comparisons"]["plus"]
    defense_full,defense_zone=(fixed["defense_map"][variant] for variant in ("0","1"))
    v1_pair=fixed["comparisons"]["v1"]
    v1gain=reversed_gain(v1_pair["wall_nps_pct"])
    v1ci=sorted(reversed_gain(k) for k in v1_pair["ci_wall_pct"])
    labels={"geometry":"增量卷积的有效格点与分组累加","orbit":"价值头四路整数点积",
            "pattern":"棋形查表的 BMI2 位提取","counts":"相同棋形分类的计数短路"}
    ip=load(B/"profile/ranking.json")
    direct=load(B/"profile/selected_ranking.json")
    global_probe=load(B/"profile/call_ranking.json")
    hotspot=direct["ranking"][1:]
    ip_names=["Evaluation::mix9svq::Accumulator::move","\x60anonymous namespace'::starBlock4",
              "Board::move","Evaluation::mix9svq::Accumulator::evaluatePolicy"]
    sampled={r["function"]:r for r in ip["ranking"]}
    individual={}
    for stage in ("gates","counts_gate"):
        individual.update(load(B/stage/"summary.json")["comparisons"])
    lo,hi=pent["ci95_elo"]
    book_ranges=[]
    for rule,label in ((1,"Standard"),(2,"Renju")):
        all_scores=[r["baseline_black_score"] for r in book if r["rule"]==rule]
        continuations=[r["baseline_black_score"] for r in book if r["rule"]==rule and r["continuation_plies"]]
        book_ranges.append(f"{label} 续着的原版黑方预期得分范围 {min(continuations):.2%}–{max(continuations):.2%}，"
                           f"计入保留的原始行后为 {min(all_scores):.2%}–{max(all_scores):.2%}。")
    conclusion=("总组达到预设的两个棋力门槛。" if passed else "总组未同时达到 95% 区间下界大于零和 LOS 大于 97.5% 的目标。")
    lines=["# Rapfi Plus v2 工程与 600 局确权赛","",
     f"**v2 取得 {pent['W']} 胜、{pent['D']} 和、{pent['L']} 负，得分 {pent['score']:.2%}。** "
     f"相对未修改原版的 Elo 为 **{signed(pent['elo'])}**，95% 配对 Bootstrap 区间 "
     f"**[{signed(lo)}, {signed(hi)}]**，LOS **{pent['los']:.2%}**。{conclusion}","",
     f"固定深度 12 的墙钟 NPS 相对原版 **{signed(primary['wall_nps_pct'])}%**，95% 区间 "
     f"**[{signed(primary['ci_wall_pct'][0])}%, {signed(primary['ci_wall_pct'][1])}%]**。"
     f"相同局面同场与 v1 配对比较，v2 为 **{signed(v1gain)}%**。真实时钟比赛的 NPS 差为 **{signed(live_gain)}%**。",
     "","## 比分与统计","",
     "| 规则 | 局数 | v2 胜 / 和 / 负 | 得分率 | 五项分布 | 净胜对 | Elo | 95% 区间 | LOS |",
     "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for title,x in s.items():
        lines.append(f"| {'合计' if title=='total' else title} | {x['games']} | {x['W']} / {x['D']} / {x['L']} | "
          f"{x['score']:.2%} | {x['penta']} | {x['net_pairs']:+} | {signed(x['elo'])} | "
          f"[{signed(x['ci95_elo'][0])}, {signed(x['ci95_elo'][1])}] | {x['los']:.2%} |")
    lines+=["",f"五项为 LL、LD、DD/WL、WD、WW。v2 总得分 {pent['score']*600:.1f}∶{(1-pent['score'])*600:.1f}，"
      f"净胜 {pent['net_games']:+} 局，净胜 {pent['net_pairs']:+} 对。Elo 使用 400 log₁₀(s/(1−s))。"
      "按规则等权分层，在各组有放回重抽完整换色对，100,000 次。LOS 使用换色对的分层标准误和单侧正态近似。"
      f"Bootstrap 超越频率在平局计半后为 {pent['bootstrap_superiority']:.2%}。"
      f"按每组 12 个正式母开局整体聚类重抽，同一母开局的续着始终一起进入样本，Elo 敏感性区间为 "
      f"[{signed(cluster_ci[0])}, {signed(cluster_ci[1])}]。全部 600 局纳入主分析。 [逐开局两局明细](match_v2/summary.json)","",
      f"v1 的独立 200 局结果为 {old['W']} 胜、{old['D']} 和、{old['L']} 负，Elo {signed(old['elo'])}，当时的固定深度 NPS 差为 {signed(old_speed)}%。"
      "两场使用不同开局、时钟和样本量。v2 与 v1 的直接速度比较来自本次同场固定深度基准。 [v1 原报告与数据](archive/v1/MATCH_REPORT.md)",
      "","## 开局、实时时钟与双裁判","",
      "官方 Gomocup 2025 压缩包的 Standard 与 Renju 正式开局清单各有 12 行。赛季保留全部 24 行，另外 "
      "276 个起始局面由同届官方 PSQ 在所属母开局之后的 2–8 手真实续着组成。按旋转、镜像和同色棋子集合去重。"
      "每个母开局均进入 12 或 13 对，按原版静态黑方 W + D/2 与 50% 的距离筛选续着。"
      f"共筛查 {book_design['shortlisted']:,} 个候选，正式棋谱逐一回对源文件。"+"".join(book_ranges)
      +f"冻结开局 SHA-256 为 {design['book_sha256']}。 "
      "[原始官方归档](data/v2/gomocup2025results.zip) · [开局来源和逐局面评估](data/v2/openings.json)","",
      f"{hardware['cpu']} 的最高性能物理核为 {sorted(hardware['performance'])}。双方各用单线程、8 MiB TT，禁用后台计算和节点上限。"
      f"最大并发 {audit['max_simultaneous_pairs']} 对，每局两个引擎各占一核，同一开局第二局同步换色换核。"
      f"物理核分配重叠 {audit['overlapping_physical_core_assignments']} 次。双方设置相同的 Windows 高于正常优先级，裁判及管道读取线程绑定能效核。"
      "双方进程常驻，开局前执行 RESTART、YXHASHCLEAR 并重设时钟。"
      "对手思考期间通知下一手剩余时间，等待 ABOUT 确认后才开始该手计时。"
      f"每方 1.500 秒，每手加 40 毫秒，发出落子命令至收到完整坐标的单调墙钟计入该方。"
      f"本次赛程实际 {wall_minutes:.2f} 分钟，{'达到' if 12<=wall_minutes<=18 else '超出' if wall_minutes>18 else '短于'}预估的 12–18 分钟。 "
      "[赛制与二进制散列](match_v2/design.json) · [常驻与冷启根答案、节点对照](validation/v2_restart.json) · [起止时间](match_v2/timing.json) · [处理器负载记录](match_v2/host_load.jsonl)","",
      f"开赛前四次各 0.4 秒采样的平均逻辑核占用为{load_text(before_load)}。赛中每两秒采样共 {len(load_samples)} 次，"
      f"含引擎在内的平均占用为{load_text(live_load)}。亲和性与复盘记录验证参赛进程之间的核隔离。Windows 仍可在这些核上调度系统任务。 "
      "[物理核拓扑与开赛前采样](match_v2/hardware.json)","",
      f"独立位棋盘裁判逐手检查五连、长连、四四和递归三三。另一套坐标裁判通过 {len(audit['rule_fixtures'])} 个规则用例，"
      f"复盘 {audit['timed_turns']:,} 个限时回合。非法着 {audit['illegal_moves']}，超时判负 {audit['hard_flags']}，"
      f"裁决差异 {audit['referee_mismatches']}，逐手 JSON 与 PSQ 差异 {audit['json_psq_mismatches']}。"
      f"及时落子的最低加秒前余量为 {audit['min_unused_bank_ns']/1e6:.1f} ms。"
      f"终局包括 {reasons}。 "
      "[600 份 PSQ](match_v2/psq) · [逐手纳秒时钟与节点](match_v2/games) · [独立复盘](match_v2/replay.json)","",
      f"实战原版 {through['orig']['nodes']:,} 节点 / {through['orig']['wall_ns']/1e9:,.3f} 秒 = {through['orig']['nps']:,.0f} NPS，"
      f"v2 {through['plus']['nodes']:,} 节点 / {through['plus']['wall_ns']/1e9:,.3f} 秒 = {through['plus']['nps']:,.0f} NPS。"
      f"取得节点数的限时回合覆盖原版 {through['orig']['measured']:,}/{through['orig']['turns']:,} 手，"
      f"v2 {through['plus']['measured']:,}/{through['plus']['turns']:,} 手。" + "、".join(f"{label} 实战 NPS {signed(rate_of(x['orig']['nps'],x['plus']['nps']))}%" for label,x in by_rule.items())
      +"。实战局面、搜索深度可能不同。 [实战吞吐](match_v2/summary.json)","",
      "## P1+P3 完整搜索树的 CPU 热点","",
      f"回滚后基线仍提供 YXVCFDEFEND。首先在发版优化参数的 P1+P3 程序上，对 24 个深度 12 局面重复 {ip['reps']} 轮。"
      f"按相邻 QueryThreadCycleTime 差值对随机指令指针采样加权，共 {ip['samples']:,} 个样本。"
      f"引擎映像外地址占加权周期 {sampled['outside engine image']['pct']:.2f}%，计入下表采样百分比的分母。"
      "另在同一套局面上做四轮窄范围函数出入栈插桩，记录搜索线程的直接周期，"
      "父区间扣除所测子函数。百分比以插桩版根区间为分母。系统线程计数对整个区间准确，零扰动程序的单函数占比为采样估计。","",
      "| 函数 | 基线采样加权占比 | 窄插桩直接周期 | 调用数 | 窄插桩占比 |","|---|---:|---:|---:|---:|"]
    for name,row in zip(ip_names,hotspot,strict=True):
        lines.append(f"| {row['function']} | {sampled[name]['pct']:.2f}% | {row['cycles']:,} | {row['calls']:,} | {row['pct']:.2f}% |")
    lines+=["",f"全量出入栈插桩本身占 {global_probe['instrumentation_cycles']/global_probe['scope_cycles']:.1%} 的根区间，"
      "并改变了高频小函数之间的排名。优化定位依据原始程序的采样，并用四函数窄插桩复核。"
      f"窄插桩的相邻计数读取中位开销为 {direct['timer_floor_cycles']} 周期。"
      "每个探测版本在固定深度返回同一节点数和根答案。 "
      "[原始指令热度](bench/v2/profile/instructions.json) · [采样排名](bench/v2/profile/ranking.json) · "
      "[窄插桩直接周期](bench/v2/profile/selected_ranking.json) · [全边界插桩结果](bench/v2/profile/call_ranking.json)",
      "","## 微优化筛选与正式速度","",
      "各项单独对 P1+P3 的 24 个同深度局面交替 A/B 与 B/A，六轮，只有一个搜索进程使用指定性能核。"
      "预设保留条件为根着、根分和节点完全相同，合计墙钟 NPS 与每线程周期吞吐均为正，且过半轮次墙钟更快。"
      "独立过门的候选按单项收益排序，叠加到当前最快程序后再用同一标准验收。","",
      "| 单项 | 整树 NPS | 95% 配对区间 | 每周期吞吐 | 更快轮次 | 单项结果 |","|---|---:|---:|---:|---:|---|"]
    for name,c in individual.items():
        lines.append(f"| {labels[name]} | {signed(c['wall_nps_pct'])}% | [{signed(c['ci_wall_pct'][0])}%, {signed(c['ci_wall_pct'][1])}%] | "
         f"{signed(c['cycle_nps_pct'])}% | {c['positive_blocks']}/6 | {'通过' if c['admitted'] else '淘汰'} |")
    lines+=["","正式保留"+("、".join(labels[n] for n in release["selected"]) or "P1+P3 基线")+"。"]
    for step in release["incremental"]:
        a=load(B/step["gate"]/"summary.json")["comparisons"][step["candidate"]]
        lines.append(f"在当前最快版本叠加{labels[step['added']]}后，墙钟 NPS {signed(a['wall_nps_pct'])}%，"
                     f"每周期吞吐 {signed(a['cycle_nps_pct'])}%，{'保留' if step['admitted'] else '淘汰'}。 "
                     f"[本次叠加原始数据](bench/v2/{step['gate']}/trials.jsonl)")
    lines+=["","有效格点方案按棋盘尺寸预计算四方向变更和 3×3 卷积可影响的真实格点，仍按原顺序更新中心的四个版本。"
      "棋形位提取方案利用未变化三方向的排列对称性。棋形分类短路只在双方聚合型完全一致时省去抵消的计数更新，"
      "保留战术位置刷新。价值头方案按 CPUID 分派到 VEX 编码的有符号及无符号 VNNI 点积，每次重用四个输入和相邻两组输出。"
      "AVX2 机器可走原四路核。各候选的发版状态以上表与叠加试验为准。 "
      "[单项原始计时](bench/v2/gates/trials.jsonl) · [棋形计数试验](bench/v2/counts_gate/trials.jsonl) · [发版选择](data/v2/release.json)","",
      f"正式基准在物理性能核 {fixed['core']} 上完成八轮。沿用 v1 冻结的 24 个局面，每种规则各四个开局和四个中盘。"
      "局面装载、预热、清表在计时区间外，精确节点从发版搜索线程读取。"
      "表内的速度百分比分别按该行右侧程序相对左侧程序计算。 [固定深度局面集](data/bench_positions.json)","",
      "| 左 / 右 | 墙钟秒 | 搜索线程十亿周期 | 总节点 | NPS | 右侧 NPS 变化 | 95% 配对区间 |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for k,c in fixed["comparisons"].items():
        a,b=c["a"],c["b"]
        x,y=c["totals"][a],c["totals"][b]
        assert c["root_mismatches"]==c["node_mismatches"]==0
        title={"plus":"原版 / v2","p13":"原版 / P1+P3","v1":"v2 / v1"}[k]
        lines.append(f"| {title} | {x['wall_ns']/1e9:.3f} / {y['wall_ns']/1e9:.3f} | {x['worker_cycles']/1e9:.3f} / "
          f"{y['worker_cycles']/1e9:.3f} | {x['nodes']:,} / {y['nodes']:,} | {c['nps'][a]:,.0f} / {c['nps'][b]:,.0f} | "
          f"{signed(c['wall_nps_pct'])}% | [{signed(c['ci_wall_pct'][0])}%, {signed(c['ci_wall_pct'][1])}%] |")
    comparisons=len(fixed["comparisons"])*fixed["positions"]*fixed["repetitions"]
    lines+=["",f"共 {comparisons} 次固定深度局面配对，逐局面根着、根分和精确节点全部相同。"
      f"按 v2 相对 v1 的方向换算为 {signed(v1gain)}%，95% 区间 [{signed(v1ci[0])}%, {signed(v1ci[1])}%]。 "
      "[正式固定深度逐搜索记录](bench/v2/final/trials.jsonl) · [合计和周期区间](bench/v2/final/summary.json)","",
      f"显式九手防守查询另用 200 题重复七次，并交替运行穷举与 RZone。二者分别展开 {defense_full['nodes']:,} 和 "
      f"{defense_zone['nodes']:,} 节点，墙钟合计 {defense_full['wall_ns']/1e9:.3f} 和 {defense_zone['wall_ns']/1e9:.3f} 秒。"
      f"区外跳过 {defense_zone['skips']:,} 次。这项计时只覆盖 YXVCFDEFEND 的多防点计算。 [防守查询逐题计时](bench/v2/final/defense_bench.log)","",
      "## 正确性与构建","",
      f"180 个官方实战中盘，每规则 60 个。两个正式 EXE 的静态评估、{head['wdl_words']} 个 float32 W/L/D 位型、"
      f"{head['both_color_empty_scores']:,} 个双方空点 Policy 位型全部相同，其中 {head['both_color_legal_scores']:,} 个属于合法落点。"
      f"{tactical['cases']} 道战术题的 {tactical['direct_vcf_entries_equal']} 个同源 VCF 原始分以及正式程序的根答案一致。"
      f"九手内的 {tactical['legal_defenses']:,} 个合法防点和 {tactical['outside_replays']:,} 次区外重放零差异。"
      f"附测 {m10['positions']} 个 Mix10 局面，使用可复现的兼容测试权重。普通比赛搜索使用官方 Mix9SVQ 权重，P4 继续提供显式多防点查询。 "
      "[逐位门禁](validation/heads.json) · [战术门禁](validation/tactics.json) · [SIMD 边界压力测试](validation/kernels.json)","",
      f"原版源码与给定的 src_base 目录具有相同的 225 个文件，逐字节相同。两版均用 Zig {build['compiler']}。"
      f"{build['common_translation_units']} 个共有编译单元的参数完全相同，包括 C++20、-O3、-mavx2、-mfma、-mbmi2、-ffp-contract=off 和 -DNDEBUG。"
      "链接参数相同，v2 额外链接 RZone 对象文件。"
      "P2 的策略函数体与原版相同。普通 alpha-beta 搜索源文件与原版相同。 "
      "[构建参数审计](data/build_audit.json) · [完整源码差异](patches/rapfi-plus.patch) · [复现说明](README.md)","",
      "| 程序 | SHA-256 |","|---|---|"]
    for name,path in (("orig","pbrain-rapfi-orig.exe"),("v1","pbrain-rapfi-plus-v1.exe"),("plus","pbrain-rapfi-plus.exe")):
        lines.append(f"| [{name}]({path}) | {release['exe'][name]} |")
    report="\n".join(lines)+"\n"
    supplementary=[]
    if "geometry" in release["selected"]:
        check=load(V/"v2_geometry.json")
        supplementary.append(f"对 5×5 至 22×22 的全部 {check['move_origins']:,} 个落子起点，"
            f"逐一核验 {check['destination_checks']:,} 个卷积目标格点的可达性，差异 {check['mismatches']}。 "
            "[棋盘几何穷举](validation/v2_geometry.json)")
    if "orbit" in release["selected"]:
        dispatch=load(V/"v2_dispatch.json")
        assert dispatch["enabled"] and dispatch["exe"]==release["exe"]["plus"]
        supplementary.append(f"在正式 v2 程序观察到 VNNI 点积分支进入 {sum(dispatch['native_routine_entries'].values())} 次。"
            f"独立极值与随机压力测试比较 {load(V/'kernels.json')['int32_words']:,} 个整数结果，逐位相同。 "
            "[正式 EXE 的分派观测](validation/v2_dispatch.json)")
    if supplementary:
        report=report.replace("| 程序 | SHA-256 |","\n\n".join(supplementary)+"\n\n| 程序 | SHA-256 |",1)
    official=(M==ROOT/"match_v2")
    destination=(ROOT if official else M)/"MATCH_REPORT_V2.md"
    if not official:
        def relative(match):
            path=match[1]
            if path.startswith("match_v2/"):
                target=M/path[len("match_v2/"):]
            elif path.startswith("bench/v2/final/"):
                target=benchmark_path/path[len("bench/v2/final/"):]
            else:
                target=ROOT/path
            try:
                target=Path(os.path.relpath(target,destination.parent))
            except ValueError:  # Distinct Windows volumes have no common relative path.
                target=target.resolve()
            return "]("+target.as_posix()+")"
        report=re.sub(r"\]\(([^)]+)\)",relative,report)
    destination.write_text(report,encoding="utf8")
    v1=(ROOT/"archive/v1/MATCH_REPORT.md").read_text(encoding="utf8")
    old_self=re.search(r"\]\(([A-Za-z]:[/\\][^)]+[/\\]MATCH_REPORT\.md)\)",v1,re.I)
    historical_root=Path(old_self[1]).parent if old_self else ROOT
    def archived_link(match):
        raw=match[1]
        if raw.startswith(("https://","http://","#")):
            return match[0]
        path=Path(raw)
        if path.is_absolute():
            if not path.is_relative_to(historical_root):
                return match[0]
            path=path.relative_to(historical_root)
        return "](archive/v1/"+path.as_posix()+")"
    v1=re.sub(r"\]\(([^)]+)\)",archived_link,v1)
    if v1.startswith("# Rapfi Plus 纯 CPU 工程与公平对局验收报告"):
        v1=v1.replace("# Rapfi Plus 纯 CPU 工程与公平对局验收报告","## v1 的 200 局原始报告",1)
    else:
        v1="## v1 的 200 局原始摘要\n\n"+v1
    intro=("\n".join(lines[:5]).replace("# Rapfi Plus v2 工程与 600 局确权赛","# Rapfi Plus 验收摘要")
           +"\n\n[600 局完整报告](MATCH_REPORT_V2.md) · [600 局棋谱](match_v2/psq) · [标准输出](match_v2/stdout.txt)\n\n---\n\n")
    if official:
        (ROOT/"MATCH_REPORT.md").write_text(intro+v1,encoding="utf8")
    output=[f"v2 600 games | W/D/L {pent['W']}/{pent['D']}/{pent['L']} | penta {pent['penta']} | net pairs {pent['net_pairs']:+} | "
      f"Elo {signed(pent['elo'])} CI95 [{signed(lo)}, {signed(hi)}] | LOS {pent['los']:.2%} | targets met {passed}",
      f"Mother-cluster CI [{signed(cluster_ci[0])}, {signed(cluster_ci[1])}] | paired points {pent['score']*600:.1f}/{(1-pent['score'])*600:.1f}"]
    for label in ("Standard","Renju"):
        x=s[label]; output.append(f"{label} W/D/L {x['W']}/{x['D']}/{x['L']} | penta {x['penta']} | Elo {signed(x['elo'])} "
            f"CI95 [{signed(x['ci95_elo'][0])}, {signed(x['ci95_elo'][1])}] | LOS {x['los']:.2%}")
    output +=[f"Depth12 identical nodes: v2 vs orig {signed(primary['wall_nps_pct'])}% | v2 vs v1 {signed(v1gain)}% | live NPS {signed(live_gain)}%",
     f"180/180 raw heads; 68020 Policy; 200/200 VCF; {tactical['legal_defenses']} legal defenses; 600/600 independently replayed",
     f"Fischer 1.500 + 0.040 s | {wall_minutes:.2f} actual minutes | {audit['timed_turns']} turns | "
     f"time flags {audit['hard_flags']}, illegal moves {audit['illegal_moves']}, core overlaps {audit['overlapping_physical_core_assignments']}"]
    console="\n".join(output)+"\n"
    (M/"stdout.txt").write_text(console,encoding="utf8")
    print(console,end="",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--match-dir",type=Path,default=M)
    run(parser.parse_args().match_dir)

