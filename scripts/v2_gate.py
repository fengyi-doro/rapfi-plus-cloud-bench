"""Counterbalanced, serial fixed-depth trials; each proposed change faces the same P1+P3 control."""
import argparse
import json
import random
import statistics
from pathlib import Path
import psutil
from benchmark import compare,measure
from datasets import DATA,ROOT,SEED,dump,sha
from engine import filename,topology,utc


def run(names,reps,stage,control):
    dest=ROOT/"bench/v2"/stage
    dest.mkdir(parents=True,exist_ok=True)
    boards=json.loads((DATA/"bench_positions.json").read_bytes())
    hw=topology()
    core=hw["performance"][0]
    psutil.Process().cpu_affinity(hw["helper"])
    dump(dest/"design.json",dict(control=control,candidates=names,repetitions=reps,core=core,
        boards_sha256=sha(DATA/"bench_positions.json"),depth=12,started=utc(),criterion=
        "keep only if all root scores, coordinates and nodes match, pooled wall NPS and worker-cycle throughput are positive, and a majority of paired blocks are faster in wall time",
        exe={v:sha(filename(v)) for v in [control,*names]}))
    blocks={n:[] for n in names}
    invalid=set()
    with (dest/"trials.jsonl").open("w",encoding="utf8") as log:
        for rep in range(reps):
            rng=random.Random(SEED+rep)
            work=boards.copy()
            rng.shuffle(work)
            rule_order=[0,1,2]
            rng.shuffle(rule_order)
            work.sort(key=lambda b:rule_order.index(b["rule"]))
            order=names.copy()
            rng.shuffle(order)
            for n in order:
                if n in invalid:continue
                pair=[control,n] if (rep+names.index(n))%2==0 else [n,control]
                results={s:measure(s,core,work) for s in pair}
                blocks[n].append(results)
                log.write(json.dumps(dict(rep=rep,order=pair,b=n,records=results))+"\n")
                log.flush()
                mismatch=any((x["x"],x["y"],x["eval"],x["depth"],x["nodes"])!=
                             (y["x"],y["y"],y["eval"],y["depth"],y["nodes"])
                             for x,y in zip(results[control],results[n],strict=True))
                if mismatch:invalid.add(n)
            print(stage,"round",rep+1,"of",reps,"non-equivalent",list(invalid),flush=True)
    scores={}
    for n,rows in blocks.items():
        c=compare(rows,control,n)
        wins=sum(sum(q["wall_ns"] for q in r[n])<sum(q["wall_ns"] for q in r[control]) for r in rows)
        c["positive_blocks"]=wins
        c["admitted"]=n not in invalid and c["wall_nps_pct"]>0 and c["cycle_nps_pct"]>0 and wins>len(rows)/2
        scores[n]=c
    dump(dest/"summary.json",dict(ended=utc(),control=control,reps=reps,core=core,comparisons=scores))
    print(stage,{k:(round(v["wall_nps_pct"],3),v["admitted"]) for k,v in scores.items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("names",nargs="+")
    p.add_argument("--reps",type=int,default=6)
    p.add_argument("--stage",default="gates")
    p.add_argument("--control",default="p13")
    a=p.parse_args()
    run(a.names,a.reps,a.stage,a.control)

