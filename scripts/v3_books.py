"""Freeze distinct cloud openings and an outcome-stratified historical endgame sample."""
from collections import Counter
import json
import random

from datasets import DATA, ROOT, SEED, canonical, dump, sha
from rules import valid_history


def run():
    v2=DATA/"v2"
    dest=DATA/"v3"
    dest.mkdir(exist_ok=True)
    book=json.loads((v2/"openings.json").read_bytes())
    screen=json.loads((v2/"screen.json").read_bytes())
    cloud=list(book)
    for rule,prefix in ((1,"S"),(2,"R")):
        used={canonical(o["moves"]) for o in cloud if o["rule"]==rule}
        counts=Counter(o["mother"] for o in cloud if o["rule"]==rule)
        pools={m:sorted((r for r in screen if r["rule"]==rule and r["mother"]==m),
                        key=lambda r:abs(r["baseline_black_score"]-.5)) for m in counts}
        for index in range(151,251):
            mother=min(counts,key=lambda m:(counts[m],m))
            while True:
                row=pools[mother].pop(0)
                key=canonical(row["moves"])
                if key not in used: break
            if not valid_history(row["moves"],rule): raise ValueError("illegal opening")
            cloud.append(dict(id=f"{prefix}{index:03d}",**row))
            counts[mother]+=1
            used.add(key)
        assert len(used)==250
    random.Random(SEED+3).shuffle(cloud)
    dump(dest/"cloud_openings.json",cloud)
    mothers={r["id"]:r["mother"] for r in book}
    bins={(rule,draw):[] for rule in (1,2) for draw in (True,False)}
    for p in sorted((ROOT/"match_v2/games").glob("*.json")):
        g=json.loads(p.read_bytes())
        if len(g["moves"])>=130 and g["reason"] in ("five","five_or_more","full_board"):
            bins[g["rule"],g["winner"] is None].append((p,g))
    long=[]
    for rule,prefix in ((1,"S"),(2,"R")):
        used=set()
        for draw,quota in ((False,4),(True,16)):
            rng=random.Random(SEED+rule+int(draw))
            rng.shuffle(bins[rule,draw])
            for p,g in bins[rule,draw]:
                ply=rng.randint(120,min(170,len(g["moves"])-8))
                moves=g["moves"][:ply]
                key=canonical(moves)
                if key in used: continue
                if not valid_history(moves,rule): raise ValueError(p)
                used.add(key)
                long.append(dict(id=f"{prefix}{len(used):03d}",rule=rule,mother=mothers[g["opening"]],
                                 moves=moves,source="match_v2/games/"+p.name,source_sha256=sha(p),
                                 historical_draw=draw, ply=ply))
                if len(used)==quota: break
    assert all(sum(o["rule"]==r for o in long)==16 for r in (1,2))
    dump(dest/"long_openings.json",long)
    print("cloud 500 distinct pairs; long 32 pairs", Counter((o["rule"],o["historical_draw"]) for o in long))


if __name__=="__main__":
    run()

