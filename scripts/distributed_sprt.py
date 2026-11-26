"""Validate complete colour pairs and aggregate them in a predeclared SPRT observation order."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from arena_fast import load_book
from datasets import DATA, dump, sha
from replay import verify
from report import elo, game_points, statistics
from v2_report import mother_interval


def log_profile(counts, elo):
    # Multinomial constrained MLE on pair scores 0,1/4,1/2,3/4,1.
    # Jeffreys half-count per cell keeps the likelihood defined in empty bins.
    mean = 1 / (1 + 10**(-elo/400))
    delta = [i/4-mean for i in range(5)]
    n = [x+.5 for x in counts]
    low, high = -1/(1-mean), 1/mean
    for _ in range(60):
        mid=(low+high)/2
        gradient=sum(w*d/(1+mid*d) for w,d in zip(n,delta))
        if gradient>0: low=mid
        else: high=mid
    return -sum(w*math.log1p((low+high)*.5*d) for w,d in zip(n,delta))


def sprt(values, elo0, elo1, alpha=.05, beta=.05):
    if not -100 < elo0 < elo1 < 100: raise ValueError("invalid SPRT Elo hypotheses")
    lower,upper=math.log(beta/(1-alpha)),math.log((1-beta)/alpha)
    counts=[0]*5
    crossed=None
    llr=0.0
    for i,value in enumerate(values,1):
        counts[value]+=1
        llr=log_profile(counts,elo1)-log_profile(counts,elo0)
        if crossed is None and (llr<=lower or llr>=upper):
            crossed=dict(pair=i, llr=llr, decision="H0" if llr<=lower else "H1")
    return dict(elo0=elo0,elo1=elo1,alpha=alpha,beta=beta,lower=lower,upper=upper,
                llr=llr,first_crossing=crossed,decision=crossed["decision"] if crossed else "inconclusive",
                method="pentanomial constrained multinomial profile LLR; half-count per bin; Wald boundaries")


def finite(obj):
    if isinstance(obj,float) and not math.isfinite(obj): return "+inf" if obj>0 else "-inf"
    if isinstance(obj,dict): return {k:finite(v) for k,v in obj.items()}
    if isinstance(obj,list): return [finite(v) for v in obj]
    return obj


def paired_stats(games, challenger):
    row=statistics(games,challenger)
    row["ci_method"]="rule-stratified opening-pair bootstrap"
    if row["penta"][2]==row["pairs"]:
        # No off-centre pairs in n observations. The one-sided exact 95%
        # upper bound on their frequency bounds both sides of the mean score.
        upper=1-.05**(1/row["pairs"])
        row["bootstrap_ci95_elo"]=row["ci95_elo"]
        row["ci95_elo"]=[elo(.5*(1-upper)),elo(.5*(1+upper))]
        row["ci_method"]="exact zero-event bound on non-neutral pair frequency"
    return row


def aggregate(directories, book, pairs, shards, out, elo0, elo1, workflow_start=None, build_info=None):
    out.mkdir(parents=True,exist_ok=True)
    designs={}
    games=[]
    shard_rows=[]
    identities={}
    scores_by_platform={}
    reference=None
    for directory in directories:
        d=json.loads((directory/"design.json").read_bytes())
        ix=d["shard"]
        if ix in designs or d["shards"]!=shards or d["total_pairs"]!=pairs or d["book_sha256"]!=sha(book):
            raise ValueError(("duplicate or incompatible shard",directory))
        planned=[o["id"] for o in load_book(book,pairs,ix,shards)]
        if d["opening_ids"]!=planned: raise ValueError(("shard schedule mismatch",ix))
        parameters={key:d.get(key) for key in ("players","initial_ns","increment_ns","runtime","book_sha256",
            "adjudication","thread_num","hash_kib","pondering","commit","scheduler","parallel_reset",
            "one_vcpu_per_game","build_profile","per_rule_engines","root_mode","lane_layout","threads_by_player","oracle_sha256","clock_policy",
            "oracle_family","oracle_by_rule","oracle_usage","proof_first","readout_sha256")}
        if reference is not None and parameters!=reference: raise ValueError("shard protocol or revision mismatch")
        reference=parameters
        hardware=json.loads((directory/"hardware.json").read_bytes())
        os_name=hardware["os"]
        platform="Windows" if os_name.startswith("Windows") else "Linux"
        if platform in identities and identities[platform]!=d["exe"]:
            raise ValueError(("mixed binary hashes for platform",platform))
        identities[platform]=d["exe"]
        checked,audit=verify(directory, book)
        timing=json.loads((directory/"timing.json").read_bytes())
        if timing["count"] != len(checked):
            raise ValueError(("timing ledger and replay game counts differ", ix))
        designs[ix]=d
        shard_rows.append(dict(shard=ix,platform=platform,timing=timing,lanes=d["lanes"],audit=audit,
                               vcpus=len(hardware["cores"]),cpu=hardware["cpu"],cpu_flags=hardware.get("cpu_flags", [])))
        games.extend(checked)
        scores_by_platform.setdefault(platform,[]).extend(checked)
    if sorted(designs)!=list(range(1,shards+1)) or len(games)!=2*pairs: raise ValueError("incomplete tournament")
    challenger,opponent=reference["players"]
    if build_info is not None:
        compiled = json.loads(build_info.read_bytes())
        if (reference["build_profile"] != compiled["profile"] or reference["runtime"] != compiled["runtime"]
                or identities != {"Linux": {challenger: compiled["binaries"][compiled["profile"]],
                                            opponent: compiled["binaries"]["orig"]}}):
            raise ValueError("shard engines or weights differ from the verified build")
        dump(out/"build-info.json", compiled)
        if reference["root_mode"] == "readout":
            readout = json.loads((build_info.parent/"readout.json").read_bytes())
            if (readout["weights_sha256"] != reference["readout_sha256"] or
                    readout["binary_sha256"] != identities["Linux"][challenger]):
                raise ValueError("fitted head or checked playing binary differs across shards")
            dump(out/"readout.json", readout)
        if reference["root_mode"] == "kata":
            manifest = json.loads((build_info.parent/"oracle.json").read_bytes())
            if reference["oracle_sha256"] != manifest["model_sha256"] or manifest["providers"] != ["CPUExecutionProvider"]:
                raise ValueError("playing oracle differs from the verified CPU model")
            if reference["oracle_by_rule"]:
                expected = {str(r):manifest["models"].get(str(r), manifest["models"].get("both"))["sha256"] for r in (1, 2)}
                if reference["oracle_by_rule"] != expected or reference["oracle_family"] != manifest["family"]:
                    raise ValueError("rule-specific model differs from the build")
            dump(out/"oracle.json", manifest)
    by_id={}
    for g in games: by_id.setdefault(g["opening"],[]).append(g)
    ordered=[o["id"] for o in load_book(book,pairs,1,1)]
    if set(ordered)!=set(by_id) or any(len(v)!=2 for v in by_id.values()): raise ValueError("invalid pair ledger")
    tally=[sum(game_points(g,challenger) for g in by_id[o]) for o in ordered]
    groups={"All":games,"Standard":[g for g in games if g["rule"]==1],
            "Renju":[g for g in games if g["rule"]==2]}
    scores={k:paired_stats(gs,challenger) for k,gs in groups.items() if gs}
    check=sprt(tally,elo0,elo1)
    timing_start=min(datetime.fromisoformat(s["timing"]["start"]) for s in shard_rows)
    timing_end=max(datetime.fromisoformat(s["timing"]["end"]) for s in shard_rows)
    result=dict(players=reference["players"],scores=scores,sprt=check,
                per_platform={k:paired_stats(gs,challenger) for k,gs in scores_by_platform.items()},
                pair_ledger=[dict(opening=k, half_points=v, wins=sum(g["winner"] == challenger for g in by_id[k]))
                             for k, v in zip(ordered,tally)],
                reasons=dict(Counter(g["reason"] for g in games)),shards=sorted(shard_rows,key=lambda s:s["shard"]),
                playing_span_seconds=(timing_end-timing_start).total_seconds(),binary_hashes=identities,
                protocol=reference)
    events = [(datetime.fromisoformat(g[key]), delta) for g in games for key, delta in (("started", 1), ("ended", -1))]
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    result["peak_simultaneous_games"] = peak
    result["flags_by_player"] = dict(Counter(g["turns"][-1]["engine"] for g in games if g["reason"] == "time"))
    result["illegal_by_player"] = dict(Counter(g["turns"][-1]["engine"] for g in games
        if g["reason"] == "illegal_coordinate" or g["reason"].startswith("forbidden")))
    result["games_per_minute_playing"] = len(games) * 60 / result["playing_span_seconds"]
    if pairs==500 and shards==20:
        result["five_minute_play_target_met"]=result["playing_span_seconds"]<=300
    source=json.loads(book.read_bytes())
    if all(len({o["mother"] for o in source if o["rule"]==r and o["id"] in by_id})==12 for r in (1,2)):
        result["mother_cluster_ci95_elo"]=mother_interval(games,source,challenger)
    for name in reference["players"]:
        all_turns = [t for g in games for t in g["turns"] if t["engine"] == name]
        wall = sorted(t["wall_ns"] for t in all_turns)
        result.setdefault("move_timing", {})[name] = dict(
            turns=len(wall), mean_ms=sum(wall)/len(wall)/1e6 if wall else None,
            p95_ms=wall[math.ceil(.95*len(wall))-1]/1e6 if wall else None,
            threat_extensions=sum(t.get("root_threat_extensions", 0) for t in all_turns))
        prior = sorted(t["root_prior_ns"] for t in all_turns if "root_prior_ns" in t)
        if name == challenger and reference["root_mode"] in ("d4", "kata", "proof", "vct") and not prior:
            raise ValueError("no timed root-oracle calls in the match")
        if prior:
            if name != challenger or any(t["root_prior"] != reference["root_mode"] for t in all_turns if "root_prior_ns" in t):
                raise ValueError("oracle telemetry belongs to the wrong player or experiment")
            result["move_timing"][name]["root_prior"] = dict(
                calls=len(prior), total_ns=sum(prior), per_turn_ms=sum(prior)/len(wall)/1e6,
                per_call_ms=sum(prior)/len(prior)/1e6, p50_call_ms=prior[(len(prior)-1)//2]/1e6,
                p95_call_ms=prior[math.ceil(.95*len(prior))-1]/1e6)
            for group, used in (("with_prior", True), ("without_prior", False)):
                depths = sorted(t["depth"] for t in all_turns if ("root_prior_ns" in t) == used and t["depth"] is not None)
                result["move_timing"][name]["root_prior"][group] = dict(
                    recorded_depths=len(depths), median_depth=depths[(len(depths)-1)//2] if depths else None,
                    zero_depth=sum(v == 0 for v in depths))
        if reference["root_mode"] == "history" and name == challenger:
            total = sum(t.get("threat_history_updates", 0) for t in all_turns)
            if not total:
                raise ValueError("no threat-conditioned history updates in the match")
            result["move_timing"][name]["threat_history_updates"] = total
        if reference["root_mode"] == "rotate-smp" and name == challenger:
            observed = [t["rotation_worker_ns"] for t in all_turns if "rotation_worker_ns" in t]
            if reference["lane_layout"] != "isolated-4t" or not observed:
                raise ValueError("rotated SMP requires verified four-CPU matches")
            result["move_timing"][name]["rotations"] = dict(
                turns=len(observed), max_worker_ms=sum(max(d) for d in observed)/len(observed)/1e6,
                all_worker_ms=sum(sum(d) for d in observed)/len(observed)/1e6)
        if reference["root_mode"] == "vct" and name == challenger:
            result["move_timing"][name]["three"] = {
                k:sum(t.get(k, 0) for t in all_turns) for k in ("three_attempts", "three_wins", "three_verified_replies", "three_nodes", "three_ns")}
        readout_turns = [(g["rule"], t["color"], t["fitted_readout_sha256"]) for g in games for t in g["turns"]
                        if t["engine"] == name and "fitted_readout_sha256" in t]
        if reference["root_mode"] == "readout" and name == challenger:
            if ({(rule, side) for rule,side,_ in readout_turns} != {(1,0),(1,1),(2,0),(2,1)}
                    or any(digest != reference["readout_sha256"] for _,_,digest in readout_turns)):
                raise ValueError("playing heads did not match the verified readout for both rules and colours")
            result["move_timing"][name]["readout_turns"] = len(readout_turns)
        elif readout_turns:
            raise ValueError("fitted readout activated for the wrong engine or experiment")
        if name == challenger and reference["root_mode"] == "outcome":
            adjusted = [t for t in all_turns if "draw_utility_permille" in t]
            if not adjusted or any(t["utility_color"] != t["color"] or t["draw_utility_permille"] != 255 for t in adjusted):
                raise ValueError("unexpected NNUE draw utility or colour")
            result["move_timing"][name]["utility_turns"] = len(adjusted)
        if name == challenger and reference["root_mode"] == "puct":
            trees = [t for t in all_turns if "mcts_new_visits" in t]
            if not trees or any(t["mcts_memory_kib"] != 8192 for t in trees):
                raise ValueError("native MCTS did not use the fixed 8192 KiB search budget")
            result["move_timing"][name]["mcts"] = dict(turns=len(trees), visits=sum(t["mcts_new_visits"] for t in trees))
        if (reference["root_mode"] in ("proof", "vct") or reference["proof_first"]) and name == challenger:
            result["move_timing"][name]["proof"] = {
                k:sum(t.get(k, 0) for t in all_turns)
                for k in ("root_proven_win", "root_proven_losses", "root_proof_nodes", "root_proof_ns")}
        if reference["oracle_usage"] == "children" and name == challenger:
            batches = [t["root_children"] for t in all_turns if "root_children" in t]
            if len(batches) != len(prior) or any(not 4 <= n <= 6 for n in batches):
                raise ValueError("invalid root-child batch coverage")
            result["move_timing"][name]["child_positions"] = sum(batches)
        if reference["oracle_usage"] in ("selective", "children") and name == challenger:
            result["move_timing"][name]["gate_skip_ns"] = sum(t.get("root_gate_ns", 0) for t in all_turns)
        turns=[t for g in games for t in g["turns"] if t["engine"]==name and t["nodes"] is not None]
        counts=sum(t["nodes"] for t in turns)
        ns=sum(t["wall_ns"] for t in turns)
        result.setdefault("timed_nodes",{})[name]=dict(known_turns=len(turns),nodes=counts,nanoseconds=ns,
                                                     nps=counts*1e9/ns if ns else None)
    if workflow_start is not None:
        start=datetime.fromisoformat(workflow_start.replace("Z","+00:00"))
        if start.tzinfo is None or start>timing_start:
            raise ValueError("workflow start must be a zoned timestamp preceding all shards")
        result["workflow_started_at"]=start.isoformat()
        result["workflow_to_last_game_seconds"]=(timing_end-start).total_seconds()
        result["workflow_to_report_seconds"]=(datetime.now(timezone.utc)-start).total_seconds()
        if pairs==500 and shards==20:
            result["five_minute_dispatch_to_last_game_met"]=result["workflow_to_last_game_seconds"]<=300
    dump(out/"summary.json",finite(result))
    markdown=[f"# {challenger} vs {opponent}", "", "| Rule | Games | W / D / L | Pentanomial | Elo (95% paired CI) | LOS |",
              "|---|---:|---|---|---:|---:|"]
    for name,row in scores.items():
        lo,hi=row["ci95_elo"]
        los=f"{100*row['los']:.4f}" if .99995<=row['los']<1 else f"{100*row['los']:.2f}"
        markdown.append(f"| {name} | {row['games']} | {row['W']} / {row['D']} / {row['L']} | {row['penta']} | "
                        f"{row['elo']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {los}% |")
    markdown+=["",f"SPRT compares H0 {elo0:+g} Elo with H1 {elo1:+g} Elo at α = β = 0.05. "
                 f"Final LLR {check['llr']:+.4f} against [{check['lower']:.4f}, {check['upper']:.4f}]. "
                 f"First boundary decision **{check['decision']}**.",
                 "", "Opening pairs are observed in book order across shards. Both colours complete before a pair enters the test. "
                 "All scheduled games count in the fixed-sample Elo and LOS.",
                 "",f"Elapsed span from the first shard entering play to the last leaving play was {result['playing_span_seconds']:.2f} s "
                    f"({result['games_per_minute_playing']:.2f} games/min, peak {peak} simultaneous games)."]
    if "five_minute_play_target_met" in result:
        status="met" if result["five_minute_play_target_met"] else "missed"
        slowest=max(s["timing"]["wall_seconds"] for s in shard_rows)
        markdown.append(f"The 300 s playing target was **{status}**. The longest shard took {slowest:.2f} s.")
    if "workflow_started_at" in result:
        markdown.append(f"From workflow start, the final game finished in {result['workflow_to_last_game_seconds']:.2f} s "
                        f"and the verified report was generated in {result['workflow_to_report_seconds']:.2f} s.")
        if "five_minute_dispatch_to_last_game_met" in result:
            status="met" if result["five_minute_dispatch_to_last_game_met"] else "missed"
            markdown.append(f"The 300 s workflow-start-to-last-game target was **{status}**.")
    if check["first_crossing"]: markdown.append(f"The first boundary crossing occurred at pair {check['first_crossing']['pair']}.")
    if any("bootstrap_ci95_elo" in s for s in scores.values()):
        markdown.append("For groups with exclusively neutral pairs, the interval uses the exact 95% upper bound on the frequency of a non-neutral opening pair.")
    if "mother_cluster_ci95_elo" in result:
        lo,hi=result["mother_cluster_ci95_elo"]
        markdown.append(f"The mother-opening cluster interval was [{lo:+.2f}, {hi:+.2f}] Elo.")
    (out/"summary.md").write_text("\n".join(markdown)+"\n",encoding="utf8")
    print("\n".join(markdown),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("inputs",type=Path)
    p.add_argument("--book",type=Path,default=DATA/"v3/cloud_openings.json")
    p.add_argument("--pairs",type=int,default=500)
    p.add_argument("--shards",type=int,default=20)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--build-info", type=Path, help="check the played binaries and models against the build job")
    p.add_argument("--elo0",type=float,default=0)
    p.add_argument("--elo1",type=float,default=5)
    p.add_argument("--workflow-start",help="UTC timestamp of dispatch, or of the current run attempt on a rerun")
    a=p.parse_args()
    paths=[a.inputs] if (a.inputs/"design.json").is_file() else sorted(a.inputs.glob("shard-*"))
    aggregate(paths,a.book,a.pairs,a.shards,a.out,a.elo0,a.elo1,a.workflow_start,a.build_info)

