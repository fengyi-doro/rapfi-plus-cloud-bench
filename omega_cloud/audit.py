"""Replay every move and Fischer bank with the independent directional referee, then score whole pairs."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from matches import BOOK, FULL, OPPONENT, schedule
from datasets import SEED, dump
from replay import verify
from report import statistics

def summary_stats(games):
    pairs = defaultdict(list)
    for g in games:
        pairs[g["rule"],g["opening"]].append(g)
    assert all(len(v)==2 for v in pairs.values())
    stats = statistics(games,FULL)
    by_rule = defaultdict(list)
    for (rule,_),group in pairs.items():
        by_rule[rule].append(sum(g["winner"]==FULL for g in group))
    # Each observation is the number of wins in the complete two-colour pair.
    rng = np.random.default_rng(SEED)
    total = np.zeros(100_000,dtype=np.int64)
    for wins in by_rule.values():
        frequencies = np.bincount(wins,minlength=3) / len(wins)
        draws = rng.multinomial(len(wins),frequencies,size=len(total))
        total += draws[:,1]+2*draws[:,2]
    stats["win_rate"] = stats["W"]/len(games)
    stats["ci95_win"] = np.quantile(total/len(games),[.025,.975],method="inverted_cdf").tolist()
    stats["ci95_score"] = [1/(1+10**(-e/400)) for e in stats["ci95_elo"]]
    return stats

def make_scores(games):
    result = {"all":summary_stats(games)}
    for rule in sorted({g["rule"] for g in games}):
        result[str(rule)] = summary_stats([g for g in games if g["rule"]==rule])
    return result

def move_penalties(games):
    offenders=Counter()
    for game in games:
        if game["reason"].startswith("forbidden") or game["reason"]=="illegal_coordinate":
            loser=game["black"] if game["winner"]==game["white"] else game["white"]
            offenders[loser]+=1
    return dict(offenders)

def diagram_key(row):
    images = []
    for d in range(8):
        colors = [[],[]]
        for i,p in enumerate(row["moves"]):
            x,y = p%15,p//15
            if d&4: x,y=y,x
            if d&1: x=14-x
            if d&2: y=14-y
            colors[i%2].append(x+15*y)
        images.append(tuple(tuple(sorted(c)) for c in colors))
    return row["rule"],min(images)

def run(stage, root, out, previous):
    shards = sorted(root.glob("shard-*"))
    assert len(shards)==20
    plan = schedule(stage)
    keys = {(r["id"],leg) for r in plan for leg in (0,1)}
    counters, overlap, fixtures = Counter(), [], set()
    telemetry = defaultdict(Counter)
    accepted = []
    forced_fouls = []
    identity = None
    for folder in shards:
        design = json.loads((folder/"design.json").read_bytes())
        assert design["stage"]==stage and design["opening_ids"]==[r["id"] for r in plan[design["shard"]::20]]
        assert design["players"]==[FULL,OPPONENT[stage]] and design["clock"]=="3000+100"
        provenance = {k:design[k] for k in ("binaries","models","build_commit","book_sha256","clock","batch","intra_op")}
        if identity is None: identity=provenance
        assert identity==provenance
        records, result = verify(folder,BOOK)
        fixtures.update(result["rule_fixtures"])
        forced_fouls.extend(result["no_legal_black_reply"])
        for key in ("games","timed_turns","board_moves_replayed","hard_flags","json_psq_mismatches",
                    "referee_mismatches","illegal_moves","overlapping_vcpu_assignments"):
            counters[key] += result[key]
        for g in records:
            accepted.append({k:g[k] for k in ("opening","leg","rule","black","white","winner","reason")})
            overlap.extend([(g["started"],1),(g["ended"],-1)])
            for turn in g["turns"]:
                if "omega" not in turn:
                    continue
                name=turn["engine"]
                meta=turn["omega"]
                assert meta["move"]==turn["x"]+15*turn["y"] and meta["discarded"]<=meta["eval_rows"]
                assert not (meta["f15_calls"] and (name=="Omega-F15-off" or g["rule"]!=2))
                assert not (meta["f17_calls"] and name=="Omega-F17-off")
                bucket=telemetry[name]
                bucket["turns"]+=1
                bucket["wall_s"]+=turn["wall_ns"]/1e9
                bucket["shield_advice"]+=int(meta["cpu_move"]>=0)
                for key in ("eval_rows","slots","discarded","visits","reused","hits","probes","vcf_work",
                            "vct_work","f15_calls","f15_decisive","f17_calls","f17_nodes","f17_wins",
                            "takeover","veto","correction","shield_lost","shield_deferred"):
                    bucket[key]+=meta[key]
    assert len(accepted)==len(keys) and {(g["opening"],g["leg"]) for g in accepted}==keys
    for name, counts in telemetry.items():
        assert counts["eval_rows"]>counts["discarded"] and counts["visits"]>0 and counts["shield_advice"]>0
        if name != "Omega-F15-off": assert counts["f15_calls"]>0
        if name != "Omega-F17-off": assert counts["f17_calls"]>0
    assert set(telemetry)==set(design["omega_players"])
    active=peak=0
    for _,delta in sorted(overlap):
        active+=delta;peak=max(peak,active)
    assert active==0 and peak<=20
    span=(datetime.fromisoformat(max(t for t,_ in overlap))-datetime.fromisoformat(min(t for t,_ in overlap))).total_seconds()
    summary=dict(stage=stage,opponent=OPPONENT[stage],scores=make_scores(accepted),audit=dict(counters),
                 fixtures=sorted(fixtures),identity=identity,telemetry=telemetry,peak_games=peak,peak_vcpus=4*peak,
                 play_span_s=span,reasons=dict(Counter(g["reason"] for g in accepted)),
                 move_penalties=move_penalties(accepted),forced_fouls=forced_fouls,
                 losses_on_time=dict(Counter(g["black"] if g["winner"]==g["white"] else g["white"]
                                             for g in accepted if g["reason"]=="time")))
    out.mkdir(exist_ok=True)
    if previous:
        assert stage=="extension"
        prev=json.loads((previous/"summary.json").read_bytes())
        prior=json.loads((previous/"verified-games.json").read_bytes())
        assert prev["stage"]=="screen" and prev["identity"]==identity
        assert {g["opening"] for g in prior}=={r["id"] for r in schedule("screen")}
        assert len(prior)==prev["audit"]["games"]==200
        assert not {g["opening"] for g in accepted}.intersection(g["opening"] for g in prior)
        rows_screen,rows_final=schedule("screen"),schedule("extension")
        def overlap_count(fn):
            return len({fn(r) for r in rows_screen}&{fn(r) for r in rows_final})
        summary["split_overlap"]={"ids":overlap_count(lambda r:r["id"]),"diagrams_d4":overlap_count(diagram_key),
                                  "sources":overlap_count(lambda r:r["source"]),"mothers":overlap_count(lambda r:(r["rule"],r["mother"]))}
        merged=prior+accepted
        assert len(merged)==1000
        summary["combined_scores"]=make_scores(merged)
        summary["combined_audit"]={k:v+prev["audit"][k] for k,v in counters.items()}
        summary["combined_reasons"]=dict(Counter(g["reason"] for g in merged))
        summary["combined_move_penalties"]=move_penalties(merged)
        summary["combined_losses_on_time"]=dict(Counter(g["black"] if g["winner"]==g["white"] else g["white"]
                                                    for g in merged if g["reason"]=="time"))
    dump(out/"summary.json",summary)
    dump(out/"verified-games.json",accepted)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("telemetry","fixtures")},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=OPPONENT)
    p.add_argument("shards",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--previous",type=Path)
    a=p.parse_args()
    run(a.stage,a.shards,a.out,a.previous)
