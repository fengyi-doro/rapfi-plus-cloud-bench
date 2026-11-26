"""Fixed-depth, serial A/B trials, one playing process on one physical P-core."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time
import numpy as np
import psutil
from datasets import DATA, ROOT, SEED, dump, sha
from engine import Engine, filename, pin_handle, topology, utc
from tactics import payload

DEPTH = 12
BENCH = ROOT / "bench"


def setup(engine, rule):
    engine.select(rule)
    engine.sync(f"INFO timeout_turn 0\nINFO timeout_match 0\nINFO max_depth {DEPTH}")
    engine.clean_position([112, 113, 100, 126, 98])
    engine.think()
    engine.sync("STOP")


def select_positions(core):
    mids = json.loads((DATA/"positions.json").read_bytes())
    book = json.loads((DATA/"openings.json").read_bytes())
    out = []
    with Engine("orig", core) as engine:
        for rule in range(3):
            setup(engine, rule)
            for kind in ("opening", "middlegame"):
                if kind == "opening" and rule == 0:
                    rows = [dict(moves=r["moves"][:8], id=r["id"]) for r in mids if r["rule"] == 0]
                else:
                    rows = [r for r in (book if kind == "opening" else mids) if r["rule"] == rule]
                seen = set()
                for row in rows:
                    if tuple(row["moves"]) in seen:
                        continue
                    engine.clean_position(row["moves"])
                    obs = engine.think()
                    engine.sync("STOP")
                    if obs["depth"] == DEPTH and obs["nodes"] >= 700:
                        seen.add(tuple(row["moves"]))
                        out.append(dict(id=len(out)+1, source=row["id"], kind=kind, rule=rule,
                                        moves=row["moves"], pilot_nodes=obs["nodes"]))
                    if len(seen) == 4:
                        break
                assert len(seen) == 4
    dump(DATA/"bench_positions.json", out)
    return out


def measure(variant, core, boards):
    records = []
    with Engine(variant, core) as engine:
        prev = None
        for board in boards:
            if board["rule"] != prev:
                setup(engine, board["rule"])
                prev = board["rule"]
            engine.clean_position(board["moves"])
            result = engine.think(measure_cycles=True)
            assert result["depth"] == DEPTH and result["nodes"] > 0, (variant, board["id"], result)
            records.append(dict(id=board["id"], rule=prev, **result))
            engine.sync("STOP")
        assert engine.ps.cpu_affinity() == [core]
    return records


def totals(rows):
    return {key:sum(r[key] for r in rows) for key in ("wall_ns", "nodes", "worker_cycles", "process_cycles")}


def compare(runs, a, b):
    agg = {v:totals([r for run in runs for r in run[v]]) for v in (a,b)}
    nps = {v: agg[v]["nodes"]*1e9 / agg[v]["wall_ns"] for v in (a,b)}
    per_rep = {v:np.array([[totals(run[v])[k] for k in ("nodes","wall_ns","worker_cycles")]
                           for run in runs], dtype=np.float64) for v in (a,b)}
    indices = np.random.default_rng(SEED).integers(0, len(runs), size=(20000, len(runs)))
    A, B = (per_rep[v][indices].sum(axis=1) for v in (a,b))
    wall_gain = (B[:,0]/B[:,1]) / (A[:,0]/A[:,1])-1
    cycle_gain = (B[:,0]/B[:,2]) / (A[:,0]/A[:,2])-1
    roots = sum((x["x"], x["y"], x["eval"], x["depth"]) != (y["x"], y["y"], y["eval"], y["depth"])
                for run in runs for x,y in zip(run[a],run[b],strict=True))
    node_mismatches = sum(x["nodes"] != y["nodes"] for run in runs for x,y in zip(run[a],run[b],strict=True))
    return dict(a=a,b=b,totals=agg,nps=nps,wall_nps_pct=(nps[b]/nps[a]-1)*100,
                wall_saved_pct=(1-agg[b]["wall_ns"]/agg[a]["wall_ns"])*100,
                cycle_nps_pct=(agg[b]["nodes"]/agg[b]["worker_cycles"]/
                               (agg[a]["nodes"]/agg[a]["worker_cycles"])-1)*100,
                ci_wall_pct=(np.quantile(wall_gain,[.025,.975])*100).tolist(),
                ci_cycle_pct=(np.quantile(cycle_gain,[.025,.975])*100).tolist(), root_mismatches=roots,
                node_mismatches=node_mismatches)


def defense_bench(core, directory):
    rows = json.loads((DATA/"tactics.json").read_bytes())
    exe = ROOT/"build/plus/defense_probe.exe"
    p = subprocess.Popen([str(exe),"7"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    pin_handle(p, core)
    stdout, stderr = p.communicate(payload(rows), timeout=150)
    assert not p.returncode, stderr
    (directory/"defense_bench.log").write_text(stdout)
    stats = {mode:dict(nodes=0,wall_ns=0,cycles=0,skips=0) for mode in ("0","1")}
    for line in stdout.splitlines():
        fields = line.split()
        if fields[0] == "MAP":
            item = stats[fields[3]]
            for key, col in (("nodes",4),("skips",5),("wall_ns",8),("cycles",9)):
                item[key] += int(fields[col])
    return stats


def run(stage, reps, out=None):
    v2 = stage=="final" and (DATA/"v2/release.json").exists()
    directory = out or BENCH/("v2/final" if v2 else stage)
    directory.mkdir(parents=True,exist_ok=True)
    if stage == "screen" and (directory/"binaries").is_dir():
        os.environ["RAPFI_SCREEN_BINARIES"] = str(directory/"binaries")
    topo = topology()
    core = topo["performance"][0]
    if topo["helper"]:
        psutil.Process().cpu_affinity(topo["helper"])
    dump(directory/"hardware.json", topo)
    path = DATA/"bench_positions.json"
    boards = json.loads(path.read_bytes()) if path.exists() else select_positions(core)
    comparisons = [("p123","plus"),("orig","p4")] if stage == "screen" else [
        ("orig","plus"),("orig","p1"),("orig","p2"),("orig","p3"),("orig","p4")]
    if v2:
        comparisons=[("orig","plus"),("orig","p13"),("plus","v1")]
    begin = utc()
    pairs = {b:[] for a,b in comparisons}
    hashes = {v:sha(filename(v)) for pair in comparisons for v in pair}
    dump(directory/"design.json", dict(depth=DEPTH,repetitions=reps,core=core,comparison=comparisons,
        boards_sha256=sha(path),exe=hashes,start=begin,
        admission=("frozen v2 release and rollback control; all 24 positions must match nodes and root answers" if v2 else
        "Dual-five shortcut requires zero root differences and positive lower 95% paired block endpoints for both wall NPS and nodes/worker cycle vs P123; P4-only wall and cycle gains must also be positive.")))
    with (directory/"trials.jsonl").open("w", encoding="utf-8") as stream:
        for rep in range(reps):
            rng = random.Random(SEED + rep)
            ordered = boards.copy()
            rng.shuffle(ordered)
            rules = list(range(3))
            rng.shuffle(rules)
            ordered.sort(key=lambda r:rules.index(r["rule"]))
            pair_order = comparisons.copy()
            rng.shuffle(pair_order)
            for a,b in pair_order:
                order = (a,b) if (rep + comparisons.index((a,b))) % 2 == 0 else (b,a)
                trial = {v:measure(v,core,ordered) for v in order}
                pairs[b].append(trial)
                stream.write(json.dumps(dict(rep=rep,a=a,b=b,order=order,records=trial))+"\n")
                stream.flush()
            print(stage, "completed round", rep+1, "of", reps, flush=True)
    results = {b:compare(pairs[b],a,b) for a,b in comparisons}
    summary = dict(stage=stage,core=core,depth=DEPTH,positions=len(boards),repetitions=reps,
                   start=begin,end=utc(),comparisons=results,exe=hashes)
    if stage == "screen":
        main, alone = results["plus"],results["p4"]
        summary["admit_playing_shortcut"] = (main["root_mismatches"] == alone["root_mismatches"] == 0
            and min(main["ci_wall_pct"]) > 0 and min(main["ci_cycle_pct"]) > 0
            and alone["wall_nps_pct"] > 0 and alone["cycle_nps_pct"] > 0)
    else:
        assert all(r["root_mismatches"] == r["node_mismatches"] == 0 for r in results.values()), results
        summary["defense_map"] = defense_bench(core,directory)
    dump(directory/"summary.json", summary)
    print(stage, {v:round(r["wall_nps_pct"],3) for v,r in results.items()},
          "shortcut", summary.get("admit_playing_shortcut"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("screen","final"))
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    run(args.stage,args.reps,args.out)
