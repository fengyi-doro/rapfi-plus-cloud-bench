"""Counterbalanced serial whole-search throughput with exact nodes on one physical P core."""
import argparse
import itertools
import json
import random
import time

import psutil

from benchmark import compare, measure
from datasets import DATA, ROOT, SEED, dump, sha
from engine import filename, topology, utc


def run(names, reps, core, stage):
    dest = ROOT / "bench/v3" / stage
    dest.mkdir(parents=True, exist_ok=True)
    hw = topology()
    if core not in hw["performance"]:
        raise ValueError(f"core {core} is not an eligible P core")
    psutil.Process().cpu_affinity([c for c in hw["helper"] if c not in (14, 15)] or [core])
    path = DATA / "bench_positions.json"
    boards = json.loads(path.read_bytes())
    hashes = {v:sha(filename(v)) for v in names}
    dump(dest/"design.json", dict(exe=hashes, depth=12, hardware=hw, core=core,
         repetitions=reps, boards_sha256=sha(path), started=utc(),
         f9_admission="identical roots and nodes, positive pooled wall and worker-cycle gains, lower 95% wall endpoint > 0"))
    trials = []
    start = time.perf_counter()
    orders = list(itertools.permutations(names))
    with (dest/"trials.jsonl").open("w") as log:
        for rep in range(reps):
            rng = random.Random(SEED+rep)
            positions = boards.copy()
            rng.shuffle(positions)
            rules = list(range(3))
            rng.shuffle(rules)
            positions.sort(key=lambda b:rules.index(b["rule"]))
            order = orders[rep % len(orders)]
            records = {v:measure(v, core, positions) for v in order}
            trials.append(records)
            log.write(json.dumps(dict(round=rep, order=order, records=records))+"\n")
            log.flush()
            print(stage, rep+1, "/", reps, flush=True)
    comparisons = {a+"_"+b:compare(trials,a,b) for a,b in itertools.combinations(names,2)}
    result = dict(depth=12, core=core, positions=len(boards), reps=reps, seconds=time.perf_counter()-start,
                  exe=hashes, comparisons=comparisons, completed=utc())
    if stage == "f9":
        c = comparisons["v2_f9"]
        result["admitted"] = (c["root_mismatches"] == c["node_mismatches"] == 0
                              and c["ci_wall_pct"][0] > 0 and c["cycle_nps_pct"] > 0)
    dump(dest/"summary.json", result)
    print({k:round(v["wall_nps_pct"], 2) for k,v in comparisons.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["f9", "release"])
    p.add_argument("--core", type=int, default=2)
    p.add_argument("--reps", type=int)
    a = p.parse_args()
    f9 = a.stage == "f9"
    run(["v2", "f9"] if f9 else ["orig", "v2", "v3"], a.reps or (8 if f9 else 6), a.core, a.stage)

