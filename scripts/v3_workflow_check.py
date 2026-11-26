"""Lint the actual workflow and check its 500 unique colour pairs and test boundaries."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess

import yaml

from arena_fast import load_book
from cloud_schedule import schedule
import json
from datasets import DATA, ROOT, canonical, dump, sha
from distributed_sprt import sprt


def check(executable):
    path=ROOT/".github/workflows/distributed-sprt.yml"
    subprocess.run([str(executable),str(path)],check=True)
    version=subprocess.check_output([str(executable),"--version"],text=True).splitlines()[0]
    document=yaml.load(path.read_text(encoding="utf8"),Loader=yaml.BaseLoader)
    inputs=document["on"]["workflow_dispatch"]["inputs"]
    plan=schedule(int(inputs["pairs"]["default"]), inputs["candidate_label"]["default"], inputs["time_control"]["default"])
    ids=json.loads(plan["matrix"])
    jobs=document["jobs"]
    if (ids!=list(range(1,21)) or int(jobs["play"]["strategy"]["max-parallel"])!=20
            or jobs["play"]["strategy"]["matrix"]["shard"]!="${{ fromJSON(needs.build.outputs.matrix) }}"):
        raise ValueError("invalid matrix")
    if any(job["runs-on"]!="ubuntu-24.04" for job in jobs.values()):
        raise ValueError("every job must use the same Linux runner image")
    parts=[load_book(DATA/"v3/cloud_openings.json",500,i,20) for i in ids]
    rows=[r for part in parts for r in part]
    if (any(len(part)!=25 for part in parts) or len({r["id"] for r in rows})!=500
        or len({(r["rule"],canonical(r["moves"])) for r in rows})!=500
        or Counter(r["rule"] for r in rows)!={1:250,2:250}):
        raise ValueError("overlapping or unbalanced cloud opening schedule")
    controls=[sprt(seq,0,5)["decision"] for seq in ([4]*500,[0]*500,[2]*3)]
    if controls!=["H1","H0","inconclusive"]:raise ValueError(controls)
    result=dict(actionlint=version,workflow_sha256=sha(path),shards=len(ids),games=2*len(rows),
                unique_pairs=500,hypothesis_controls=controls)
    output=ROOT/"validation/v3/workflow.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    dump(output,result)
    print(result)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--actionlint",type=Path,default=Path("actionlint"))
    check(p.parse_args().actionlint)

