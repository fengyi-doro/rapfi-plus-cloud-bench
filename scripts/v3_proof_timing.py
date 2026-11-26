"""Per-position median and p95 of the hot production classifier, outside the search path cache."""
import json
import random
import subprocess
import sys

import numpy as np
import psutil

from datasets import ROOT, SEED, dump
sys.path.insert(0,str(ROOT))
from build import compile_unit,link


def run(core):
    psutil.Process().cpu_affinity([core])
    tree=ROOT/"src_plus"
    exe=ROOT/"build/v3/pairing_probe.exe"
    obj=exe.with_suffix(".o")
    subprocess.run(compile_unit(ROOT/"scripts/v3_pairing_probe.cpp",tree,obj,["-I"+str(tree)]),check=True)
    subprocess.run(link([obj],exe),check=True)
    snapshots=[]
    for path in sorted((ROOT/"match_v2/games").glob("*.json")):
        g=json.loads(path.read_bytes())
        for ply in range(121,len(g["moves"]),12):
            snapshots.append((g["rule"],g["moves"][:ply]))
    random.Random(SEED).shuffle(snapshots)
    payload="".join(f"{i} {r} {len(m)} "+" ".join(map(str,m))+"\n" for i,(r,m) in enumerate(snapshots[:480]))
    data=subprocess.check_output([exe,"--timing"],input=payload,text=True)
    dest=ROOT/"bench/v3/proof"
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"timings.txt").write_text(data)
    values={"both_at_most_24":[],"either_at_most_64_other_over_24":[]}
    for row in data.splitlines():
        _,_,_,_,a,b,cost,_=row.split()
        key="both_at_most_24" if max(int(a),int(b))<=24 else "either_at_most_64_other_over_24"
        values[key].append(float(cost))
    result=dict(core=core,repeats_per_observation=64,sampled_positions=min(480,len(snapshots)),
                metric="wall ns per hot classification; history preparation excluded",
                classes={k:dict(n=len(v),p50_ns=float(np.median(v)),p95_ns=float(np.percentile(v,95)))
                         for k,v in values.items() if v})
    dump(dest/"summary.json",result)
    print(result)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--core",type=int,default=12)
    run(p.parse_args().core)

