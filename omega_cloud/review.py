"""Attribute clock and placement penalties from the archived PGN-style game ledgers."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory
import numpy as np
from matches import FULL, schedule
from datasets import SEED, dump
from replay import verdict

def download(run, name, folder):
    subprocess.run(["gh","run","download",str(run),"-n",name,"-D",str(folder)],check=True)

def end_review(record):
    turn=record["turns"][-1]
    result={"opening":record["opening"],"leg":record["leg"],"rule":record["rule"],"reason":record["reason"],
            "offender":turn["engine"],"ply":record["opening_plies"]+len(record["turns"]),
            "engine_reason":turn.get("omega",{}).get("reason")}
    if record["reason"]=="time":
        assert turn["wall_ns"]>=turn["before_ns"] and turn["after_ns"]==0
        return dict(result,wall_ms=turn["wall_ns"]/1e6,bank_ms=turn["before_ns"]/1e6)
    if record["reason"]=="illegal_coordinate":
        x,y=turn["x"],turn["y"]
        assert not (0<=x<15 and 0<=y<15) or x+y*15 in record["moves"]
        return result
    history=record["moves"][:-1]
    assert record["rule"]==2 and len(history)%2==0 and turn["color"]==0
    grid={(p%15,p//15):i%2 for i,p in enumerate(history)}
    placed=(turn["x"],turn["y"])
    grid[placed]=0
    assert verdict(grid,placed,0,2)==record["reason"]
    del grid[placed]
    counts=Counter()
    for p in range(225):
        xy=p%15,p//15
        if xy in grid: continue
        grid[xy]=0
        counts[verdict(grid,xy,0,2) or "legal"]+=1
        del grid[xy]
    return dict(result,legal_alternatives=counts["legal"]+counts["five"],empty_points=sum(counts.values()),
                placement_outcomes=dict(counts))

def compare_screens(screens):
    def index(games):
        return {(g["rule"],g["opening"],g["leg"]):g for g in games}
    two,four=index(screens[2]),index(screens[4])
    assert two.keys()==four.keys() and len(two)==200
    observations=defaultdict(lambda:defaultdict(lambda:np.zeros(2)))
    def points(row):
        win=row["winner"]==FULL
        return np.array([win,win+0.5*(row["winner"] is None)])
    for key,a in two.items():
        b=four[key]
        assert (a["black"],a["white"])==(b["black"],b["white"])
        observations[key[0]][key[1]] += (points(b)-points(a))/2
    rng=np.random.default_rng(SEED)
    bootstrap=np.zeros((100_000,2))
    mean=np.zeros(2)
    for group in observations.values():
        x=np.array(list(group.values()))
        mean+=x.sum(axis=0)
        for start in range(0,len(bootstrap),1024):
            part=bootstrap[start:start+1024]
            part+=x[rng.integers(len(x),size=(len(part),len(x)))].sum(axis=1)
    n=sum(len(group) for group in observations.values())
    ci=np.quantile(100*bootstrap/n,[.025,.975],axis=0,method="inverted_cdf")
    return dict(pairs=n,win_pp=float(100*mean[0]/n),score_pp=float(100*mean[1]/n),
                win_ci95_pp=ci[:,0].tolist(),score_ci95_pp=ci[:,1].tolist())

def review(ids, out):
    output={"runs":{},"penalties":[],"orig_4t_by_colour":{}}
    main=[]
    screens={}
    for run in ids:
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            download(run,"omega-result",root/"result")
            summary=json.loads((root/"result/summary.json").read_bytes())
            games=json.loads((root/"result/verified-games.json").read_bytes())
            stage=summary["stage"]
            intra=summary["identity"]["intra_op"]
            scores=summary["scores"]["all"]
            assert len(games)==scores["games"]==summary["audit"]["games"]
            assert sum(g["winner"]==FULL for g in games)==scores["W"] and sum(g["winner"] is None for g in games)==scores["D"]
            plan={row["id"]:i%20 for i,row in enumerate(schedule(stage))}
            offenders=[g for g in games if g["reason"].startswith("forbidden") or g["reason"] in ("time","illegal_coordinate")]
            assert len(offenders)==summary["audit"]["illegal_moves"]+summary["audit"]["hard_flags"]
            by_shard=defaultdict(list)
            for g in offenders: by_shard[plan[g["opening"]]].append(g)
            for shard,rows in by_shard.items():
                download(run,f"games-{shard}",root)
                with tarfile.open(root/f"shard-{shard}.tgz") as tape:
                    for g in rows:
                        prefix=f"shard-{shard}/games/"
                        with tape.extractfile(prefix+f"{g['opening']}_{g['leg']}.json") as f:
                            record=json.load(f)
                        assert all(record[k]==v for k,v in g.items())
                        output["penalties"].append(dict(run=run,intra=intra,stage=stage,**end_review(record)))
            if intra==4 and stage in ("screen","extension"):
                main.extend(games)
            if stage=="screen":
                screens[intra]=games
            data=json.loads(subprocess.check_output(["gh","run","view",str(run),"--json","createdAt,jobs,status,conclusion"]))
            assert data["status"]=="completed" and data["conclusion"]=="success"
            finish=max(j["completedAt"] for j in data["jobs"])
            total_s=(datetime.fromisoformat(finish)-datetime.fromisoformat(data["createdAt"])).total_seconds()
            output["runs"][run]=dict(stage=stage,intra=intra,total_s=total_s,play_s=summary["play_span_s"],
                                     peak_games=summary["peak_games"],peak_vcpus=summary["peak_vcpus"])
    assert len(main)==len({(r["opening"],r["leg"]) for r in main})==1000
    output["screen_4t_minus_2t"]=compare_screens(screens)
    for rule in (1,2):
        for colour in ("black","white"):
            subset=[r for r in main if r["rule"]==rule and r[colour]==FULL]
            output["orig_4t_by_colour"][f"{rule}/{colour}"]=[sum(r["winner"]==FULL for r in subset),
                    sum(r["winner"] is None for r in subset),sum(r["winner"]=="orig" for r in subset)]
    out.mkdir()
    dump(out/"review.json",output)
    print(json.dumps(output,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("ids",help="JSON list of audited match run IDs")
    p.add_argument("out",type=Path)
    a=p.parse_args()
    review(json.loads(a.ids),a.out)
