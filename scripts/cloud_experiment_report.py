"""Download verified summaries and refresh the local experiment tables without replaying games."""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "cloud_report" / "experiments"
REPORT = ROOT / "CLOUD_EXPERIMENTS_REPORT.md"
MARK = "<!-- verified-cloud-results -->"


def load(path):
    return json.loads(path.read_bytes())


def collect(run):
    destination = ARCHIVE / str(run)
    if (destination / "summary.json").exists() and (destination / "run.json").exists():
        return
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    data = subprocess.check_output(["gh", "run", "view", str(run), "--json",
                                    "databaseId,url,headSha,createdAt,updatedAt,status,conclusion"], cwd=ROOT)
    metadata = json.loads(data)
    if metadata["status"] != "completed" or metadata["conclusion"] != "success":
        raise ValueError(f"run {run} has no verified final result")
    with TemporaryDirectory(dir=ARCHIVE) as temp:
        subprocess.run(["gh", "run", "download", str(run), "-n", "pentanomial-elo-los-sprt", "-D", temp],
                       cwd=ROOT, check=True)
        summary = load(Path(temp) / "summary.json")
        if summary["protocol"]["commit"] != metadata["headSha"]:
            raise ValueError("artifact revision differs from the Actions run")
        destination.mkdir(parents=True, exist_ok=True)
        for path in Path(temp).iterdir():
            path.replace(destination / path.name)
        (destination / "run.json").write_bytes(data)


def delta(first, second):
    # Scores share the same openings. Estimate the uncertainty in their difference
    # from paired changes, treating each rule as its own stratum.
    def ledger(s):
        return {r["opening"]:r["half_points"] / 4 for r in s["pair_ledger"]}
    a, b = ledger(first), ledger(second)
    if a.keys() != b.keys():
        return None
    strata = [[b[k] - a[k] for k in a if k.startswith(prefix)] for prefix in ("S", "R")]
    n = len(a)
    change = sum(map(sum, strata)) / n
    se = math.sqrt(sum(len(group) * statistics.variance(group) for group in strata) / n**2)
    return [100 * x for x in (change, change-1.96*se, change+1.96*se)]


def win_interval(scores, rule):
    groups = [scores[r] for r in (("Standard", "Renju") if rule == "All" else (rule,))]
    pairs = sum(g["pairs"] for g in groups)
    variance = 0.
    for g in groups:
        # The WW bin holds two wins. Every remaining win belongs to a separate pair.
        sweeps = g["penta"][4]
        singles = g["W"] - 2 * sweeps
        n = g["pairs"]
        mean = g["W"] / (2 * n)
        variance += n * max(0., (sweeps + .25 * singles - n * mean**2) / (n - 1))
    center = scores[rule]["W"] / scores[rule]["games"]
    margin = 1.96 * math.sqrt(variance) / pairs
    return max(0., center-margin), min(1., center+margin)


def render():
    baseline = load(ROOT / "cloud_report/summary.json")
    reference_hash = baseline["binary_hashes"]["Linux"]["orig"]
    rows = []
    extra = []
    scaling = []
    previous = []
    top = {"纯 Mix9SVQ、1T":[("历史 V4", baseline["scores"]["All"])],
           "外部权重、1T":[], "4T 对 1T":[]}
    for folder in sorted(ARCHIVE.iterdir()) if ARCHIVE.is_dir() else []:
        if not (folder / "run.json").is_file():
            continue
        run, s = load(folder / "run.json"), load(folder / "summary.json")
        p, scores = s["protocol"], s["scores"]
        if (p["commit"] != run["headSha"] or run["conclusion"] != "success"
                or s["binary_hashes"]["Linux"]["orig"] != reference_hash
                or p["runtime"] != baseline["protocol"]["runtime"]
                or p["book_sha256"] != baseline["protocol"]["book_sha256"]):
            raise ValueError(f"baseline or run identity differs in {folder}")
        name = s["players"][0]
        layout = p.get("lane_layout") or "packed-1t"
        threads = p.get("threads_by_player") or {name:1, "orig":1}
        label = "4T / 1T 独占" if threads[name] == 4 else ("1T / 1T 独占" if layout == "isolated-1t" else "1T / 1T")
        category = "4T 对 1T" if threads[name] == 4 else ("外部权重、1T" if p.get("oracle_sha256") else "纯 Mix9SVQ、1T")
        top[category].append((name, scores["All"]))
        link = f"[{folder.name}]({run['url']})"
        local = f"[JSON]({(folder/'summary.json').relative_to(ROOT).as_posix()})"
        for rule in ("All", "Standard", "Renju"):
            r = scores[rule]
            lo, hi = r["ci95_elo"]
            win_lo, win_hi = win_interval(scores, rule)
            rows.append(f"| {name} | {category} | {label} | {rule} | {r['W']} / {r['D']} / {r['L']} | "
                        f"{r['W']/r['games']:.2%} [{win_lo:.2%}, {win_hi:.2%}] | {r['score']:.2%} | {r['elo']:+.2f} "
                        f"[{lo:+.2f}, {hi:+.2f}] | {r['los']:.4%} | {link+' · '+local if rule=='All' else ''} |")
        audit = Counter()
        for shard in s["shards"]:
            for k in ("games", "board_moves_replayed", "hard_flags", "illegal_moves", "referee_mismatches", "json_psq_mismatches"):
                audit[k] += shard["audit"][k]
        if audit["games"] != scores["All"]["games"] or audit["referee_mismatches"] or audit["json_psq_mismatches"]:
            raise ValueError(f"incomplete replay in {folder}")
        timing = s["move_timing"][name]
        a, b = scores["All"], baseline["scores"]["All"]
        diff = delta(baseline, s)
        metrics = (f"同一组开局比历史 V4 的实胜率变化 {100*(a['W']/a['games']-b['W']/b['games']):+.2f} 个百分点，"
                   f"得分率变化 {diff[0]:+.2f} 个百分点，配对差值 95% 区间 [{diff[1]:+.2f}, {diff[2]:+.2f}]。") if diff else ""
        detail = (f"挑战方 {timing['turns']:,} 手，整手平均 {timing['mean_ms']:.3f} ms、P95 {timing['p95_ms']:.3f} ms。"
                  if timing["turns"] else "")
        if layout == "isolated-1t" and p.get("root_mode") != "control":
            for _, peer in reversed(previous):
                if peer["protocol"].get("lane_layout") == layout and peer["protocol"].get("root_mode") == "control":
                    q = peer["scores"]["All"]
                    d = delta(peer, s)
                    metrics += (f"与同为独占 1T 的 V4＋F10 对照相比，实胜率相差 "
                                f"{100*(a['W']/a['games']-q['W']/q['games']):+.2f} 个百分点，"
                                f"得分率相差 {d[0]:+.2f} 个百分点，配对 95% 区间 [{d[1]:+.2f}, {d[2]:+.2f}]。")
                    break
        if timing.get("threat_extensions"):
            detail += f"实际增加搜索深度 {timing['threat_extensions']:,} 次。"
        if timing.get("threat_history_updates"):
            detail += f"威胁条件历史在发生 β 截断后更新 {timing['threat_history_updates']:,} 次。"
        if timing.get("child_positions"):
            detail += f"根部批量评估共覆盖 {timing['child_positions']:,} 个实际候选子局面。"
        if three := timing.get("three"):
            detail += (f"根部活三起手尝试 {three['three_attempts']:,} 着，"
                       f"覆盖所有合法应手的胜着 {three['three_wins']:,} 次，"
                       f"在实际子局面验证这些胜着的应手 {three['three_verified_replies']:,} 个。")
        if proof := timing.get("proof"):
            detail += f"根部证明直接选出胜着 {proof['root_proven_win']:,} 手，排除已证败着 {proof['root_proven_losses']:,} 个。"
            if "root_proof_ns" in proof:
                detail += f"胜证计算摊到所有挑战方落子为 {proof['root_proof_ns']/timing['turns']/1e6:.3f} ms/手。"
        if "gate_skip_ns" in timing:
            detail += f"决定跳过全局网络的局面检查摊到所有手为 {timing['gate_skip_ns']/timing['turns']/1e6:.3f} ms/手。"
        if prior := timing.get("root_prior"):
            detail += (f"根评估在 {prior['calls']:,} 手触发，单次均值 {prior['per_call_ms']:.3f} ms，"
                       f"P50 / P95 为 {prior['p50_call_ms']:.3f} / {prior['p95_call_ms']:.3f} ms。"
                       f"摊到挑战方所有手为 {prior['per_turn_ms']:.3f} ms/手。")
        if (folder / "oracle.json").exists():
            model = load(folder / "oracle.json")
            comparison = "学生浮点与比赛格式" if model.get("family") == "distilled" else "上游权重与比赛格式"
            detail += (f"CPU 模型合计 {model['bytes']:,} 字节，MatMulInteger / DequantizeLinear 节点分别为 "
                       f"{model['operators'].get('MatMulInteger', 0)} / {model['operators'].get('DequantizeLinear', 0)}。"
                       f"{len(model['probes'])} 个{comparison}对照局面的最大 WDL 得分差 {model['max_wdl_score_delta']:.4f}，"
                       f"空点策略第一选择一致 {sum(p['top1_agree'] for p in model['probes'])}/{len(model['probes'])}。"
                       f"[模型清单]({(folder/'oracle.json').relative_to(ROOT).as_posix()})")
            if model.get("family") == "distilled":
                detail += (f"比赛使用 {model['precision']} 学生。在独立源棋谱测试集上与 B24 教师"
                           f"首选着一致率 {model['test']['teacher_top1']:.1%}，WDL 得分平均绝对差 "
                           f"{model['test']['mean_abs_score_error']:.4f}。[精度选择 Run]({model['training_run']})")
        probe_files = sorted((ROOT/"cloud_report/root_costs").glob(f"*/root-cost-{folder.name}/root-cost.json"))
        if probe_files:
            path = probe_files[-1]
            probe = load(path)
            if probe["match_commit"] != p["commit"] or probe["binary_sha256"] != s["binary_hashes"]["Linux"][name]:
                raise ValueError(f"mismatched fixed-depth probe for {name}")
            lo, hi = probe["ci95_delta_ms"]
            detail += (f" 云端在相同 {probe['positions']} 个局面、深度上限 {probe['depth_cap']} 下交叉测量。"
                       f"同一个 V4 可执行文件关闭 / 开启该实验时平均 {probe['mean_ms']['off']:.3f} / "
                       f"{probe['mean_ms']['on']:.3f} ms/手，整手净差 {probe['mean_delta_ms']:+.3f} ms，"
                       f"配对 95% 区间 [{lo:+.3f}, {hi:+.3f}] ms。"
                       f"[固定深度 Run]({probe['run']}) · [逐手 JSON]({path.relative_to(ROOT).as_posix()})")
        extra += [f"**{name}**　{metrics}{detail}",
                  f"复盘 {audit['games']:,} 局、{audit['board_moves_replayed']:,} 手，裁决分歧 {audit['referee_mismatches']}，"
                  f"JSON/PSQ 分歧 {audit['json_psq_mismatches']}，超时 {audit['hard_flags']}，非法着或禁手 {audit['illegal_moves']}。"
                  f"20 个分片比赛跨度 {s['playing_span_seconds']:.2f} 秒，同时在赛峰值 {s['peak_simultaneous_games']} 局。{link}"]
        if s.get("flags_by_player"):
            extra[-1] += f" 超时判负分别为挑战方 {s['flags_by_player'].get(name,0)} 局、原版 {s['flags_by_player'].get('orig',0)} 局。"
        if layout == "isolated-4t":
            for _, peer in reversed(previous):
                if (peer["protocol"].get("lane_layout") == "isolated-1t"
                        and all(peer["protocol"].get(k) == p.get(k) for k in ("root_mode","clock_policy","oracle_sha256","oracle_usage","proof_first"))
                        and peer["binary_hashes"]["Linux"][peer["players"][0]] == s["binary_hashes"]["Linux"][name]):
                    peer_all = peer["scores"]["All"]
                    d = delta(peer, s)
                    scaling.append(f"同架构同为每台 runner 独占一局，4T 相对 1T 的实胜率变化 "
                                   f"{100*(a['W']/a['games']-peer_all['W']/peer_all['games']):+.2f} 个百分点，"
                                   f"得分率变化 {d[0]:+.2f} 个百分点，配对 95% 区间 [{d[1]:+.2f}, {d[2]:+.2f}]。"
                                   f"对同一原版的 Elo 点估计差为 {a['elo']-peer_all['elo']:+.2f}。{link}")
                    break
        previous.append((name, s))
    header = REPORT.read_text(encoding="utf-8-sig").split(MARK, 1)[0].rstrip()
    best_rows = []
    for category, group in top.items():
        if group:
            name, r = max(group, key=lambda entry:(entry[1]["W"]/entry[1]["games"], entry[1]["score"]))
            best_rows.append(f"| {category} | {name} | {r['W']}/{r['games']} | "
                             f"{r['W']/r['games']:.2%} | {90-100*r['W']/r['games']:.2f} 个百分点 |")
    table = ["## 90% 实胜率进度", "", "下表按每组实际赢局占比选取已观测到的最高点估计。各组使用相同的原版对手。", "",
             "| 实验组 | 挑战方 | 胜 / 总局数 | 实胜率 | 距 90% |", "|---|---|---:|---:|---:|", *best_rows, "",
             "## 已完成的云端对局", "", "实胜率的 95% 区间按 Standard 和 Renju 分层，将同一开局两局的胜局比例作为一个观测计算标准误。Elo 区间采用云端按开局配对的自助法。", "",
             "| 挑战方 | 实验组 | 线程（挑战方 / 原版） | 规则 | 胜 / 和 / 负 | 实胜率 [95% CI] | 得分率 | Elo [95% CI] | LOS | Run 与汇总 |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---|", *rows, "",
             "## 逐手开销与独立复盘", "", "\n\n".join(extra)]
    if scaling:
        table += ["", "## Lazy SMP 增量", "", "\n\n".join(scaling)]
    tmp = REPORT.with_suffix(".tmp")
    content = header + "\n\n" + MARK + "\n\n" + "\n".join(table) + "\n"
    tmp.write_text(content, encoding="utf8")
    try:
        tmp.replace(REPORT)
    except PermissionError:
        # Some Windows viewers allow writes to an open document but deny a rename.
        REPORT.write_text(content, encoding="utf8")
        tmp.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=int, nargs="*")
    for run_id in parser.parse_args().runs:
        collect(run_id)
    render()
