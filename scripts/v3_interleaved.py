"""Compare idle resident binaries serially, interleaving variants separately for every position."""
from collections import Counter
from contextlib import ExitStack
from itertools import permutations
import json
from pathlib import Path
import random
import time

import psutil

from benchmark import compare, setup
from datasets import DATA, ROOT, SEED, dump, sha
from engine import Engine, filename, topology, utc


def run(names, path, stage, reps, core, fixed_tree, dest=None):
    dest=dest or ROOT/"bench/v3"/stage
    dest.mkdir(parents=True,exist_ok=True)
    board_list=json.loads(path.read_bytes())
    hw=topology()
    if core not in hw["performance"] or core in (14,15): raise ValueError("physical P core unavailable")
    psutil.Process().cpu_affinity([c for c in hw["helper"] if c not in (14,15)] or [core])
    hashes={v:sha(filename(v)) for v in names}
    orders=list(permutations(names))
    trials=[{v:[] for v in names} for _ in range(reps)]
    start=time.perf_counter()
    dump(dest/"design.json",dict(utc=utc(),hardware=hw,depth=12,exe=hashes,core=core,reps=reps,
        book_sha256=sha(path),method="one active engine, adjacent variants per position; TT cleared before every observation"))
    with ExitStack() as stack, (dest/"trials.jsonl").open("w") as log:
        engines={v:stack.enter_context(Engine(v,core)) for v in names}
        for rule in sorted({b["rule"] for b in board_list}):
            for engine in engines.values(): setup(engine,rule)
            for rep in range(reps):
                work=[(i,b) for i,b in enumerate(board_list) if b["rule"]==rule]
                random.Random(SEED+rep).shuffle(work)
                for i,b in work:
                    order=orders[(rep+i)%len(orders)]
                    row={}
                    for v in order:
                        e=engines[v]
                        e.clean_position(b["moves"])
                        record=dict(id=b["id"],rule=rule,**e.think(measure_cycles=True))
                        e.sync("STOP")
                        row[v]=record
                    log.write(json.dumps(dict(rep=rep,order=order,records=row))+"\n")
                    for v in names: trials[rep][v].append(row[v])
                log.flush()
            print(stage,"rule",rule,"complete",flush=True)
        if any(e.ps.cpu_affinity()!=[core] for e in engines.values()): raise ValueError("affinity changed")
    mismatched=Counter()
    subsets={}
    for label,rule in [("all",None),("Standard",1),("Renju",2)]:
        cleaned=[]
        for sample in trials:
            accepted={v:[] for v in names}
            for k in range(len(sample[names[0]])):
                row=[sample[v][k] for v in names]
                if rule is not None and row[0]["rule"]!=rule: continue
                if any(r["depth"]!=12 or r["nodes"]<=0 for r in row):
                    mismatched[label]+=1
                    continue
                for v,r in zip(names,row): accepted[v].append(r)
            cleaned.append(accepted)
        if all(cleaned[0].values()):
            subsets[label]={a+"_"+b:compare(cleaned,a,b) for i,a in enumerate(names) for b in names[i+1:]}
    if fixed_tree and (mismatched["all"] or any(c["node_mismatches"] or c["root_mismatches"]
                                              for c in subsets["all"].values())):
        raise RuntimeError("midgame fixed-depth parity failure")
    result=dict(core=core,depth=12,positions=len(board_list),reps=reps,seconds=time.perf_counter()-start,
                excluded_observations=dict(mismatched),subsets=subsets,exe=hashes)
    dump(dest/"summary.json",result)
    print(stage,{k:v["nps"] for k,v in subsets["all"].items()},flush=True)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["interleaved","endgame"])
    p.add_argument("--core",type=int,default=2)
    p.add_argument("--variants", nargs="+")
    p.add_argument("--out",type=Path)
    p.add_argument("--reps",type=int)
    a=p.parse_args()
    if a.stage=="interleaved":
        run(a.variants or ["orig","v2","v3"],DATA/"bench_positions.json",a.stage,a.reps or 12,a.core,True,a.out)
    else:
        run(a.variants or ["v2","v3"],DATA/"v3/long_openings.json",a.stage,a.reps or 4,a.core,False,a.out)

