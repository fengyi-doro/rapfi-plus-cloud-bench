"""Replay historical long games through the production classifier and independent W/L/D oracles."""
from collections import Counter
from functools import lru_cache
import json
import random
import subprocess
import sys

from datasets import ROOT, DATA, SEED, dump, sha
from rules import BITS, BOARD, ID, bits, exact, forbidden, has_win, valid_history
from v3_rank_controls import controls
sys.path.insert(0, str(ROOT))
from build import compile_unit, link


def literal(b, w, stm, rule):
    @lru_cache(None)
    def visit(black, white, turn):
        holes = BOARD ^ (black|white)
        if not holes: return 0
        own, opp = (white,black) if turn else (black,white)
        best = -1
        while holes:
            bit = holes & -holes
            holes ^= bit
            p = ID[bit]
            if forbidden(own,opp,p,rule,turn): continue
            if has_win(own|bit,p,exact(rule,turn)): return 1
            v = -visit(black,white|bit,0) if turn else -visit(black|bit,white,1)
            best = max(best,v)
            if best == 1: break
        return best
    v=visit(b,w,stm)
    states=visit.cache_info().misses
    visit.cache_clear()
    return v,states


def run():
    dest=ROOT/"validation/v3"
    dest.mkdir(parents=True,exist_ok=True)
    exe=ROOT/"build/v3/pairing_probe.exe"
    obj=exe.with_suffix(".o")
    tree=ROOT/"src_plus"
    subprocess.run(compile_unit(ROOT/"scripts/v3_pairing_probe.cpp",tree,obj,["-I"+str(tree)]),check=True)
    subprocess.run(link([obj],exe),check=True)
    groups={r:{k:[] for k in ("decisive","draw")} for r in (1,2)}
    for path in sorted((ROOT/"match_v2/games").glob("*.json")):
        g=json.loads(path.read_bytes())
        if len(g["moves"])>=121 and g["reason"] in ("five","five_or_more","full_board"):
            groups[g["rule"]]["draw" if g["winner"] is None else "decisive"].append((path,g))
    games=[]
    for rule in (1,2):
        for group,items in groups[rule].items():
            random.Random(SEED+rule+len(group)).shuffle(items)
            games.extend(items[:24])
    payload="".join(f"{i} {g['rule']} {len(g['moves'])} "+" ".join(map(str,g["moves"]))+"\n"
                    for i,(_,g) in enumerate(games))
    out=subprocess.check_output([exe],input=payload,text=True)
    lines=out.splitlines()
    if lines[0]!="graphs 4608": raise ValueError(lines[0])
    counts=Counter()
    truths=Counter()
    samples=[]
    for line in lines[1:]:
        idx,ply,draw,lower,pair_lo,pair_hi=map(int,line.split())
        path,g=games[idx]
        rule,stm=g["rule"],ply%2
        counts[f"nodes_{rule}"]+=1
        counts[f"exact_{rule}"]+=draw==1
        counts[f"lower_{rule}"]+=lower==2
        counts["coordinate_witnesses"]+=pair_lo+pair_hi
        counts["rank_only_lower"]+=lower==2 and not pair_lo
        counts["rank_used_exact"]+=draw==1 and not (pair_lo and pair_hi)
        b,w=bits(g["moves"][:ply])
        truth=None
        if 225-ply<=5 or (225-ply in (6,7) and idx%9==0):
            truth,states=literal(b,w,stm,rule)
            counts["minimax_positions"]+=1
            counts["minimax_states"]+=states
        elif ply>=len(g["moves"])-5 and g["winner"] is not None:
            stones=(b,w)
            wins=[sum(has_win(stones[c]|bit,p,exact(rule,c)) for p,bit in enumerate(BITS)
                      if not (b|w)&bit) for c in (0,1)]
            if wins[stm]: truth=1
            elif wins[1-stm]>=2: truth=-1
            if truth is not None: counts["decisive_tactics"]+=1
        if truth is not None:
            if (draw and truth!=0) or (lower and truth<0):
                raise ValueError((path.name,ply,truth,draw,lower))
            truths[str(truth)]+=1
            samples.append([path.stem,ply,truth,draw,lower])
    if not all(counts[k] for k in ("exact_1","lower_1","lower_2","coordinate_witnesses","minimax_positions")):
        raise ValueError(counts)
    fixtures=controls()
    inputs="".join(f"{i} {b['rule']} {len(b['moves'])} "+" ".join(map(str,b["moves"]))+"\n"
                   for i,b in enumerate(fixtures))
    extra=subprocess.check_output([exe],input=inputs,text=True).splitlines()[1:]
    checks=[]
    for record in extra:
        i,ply,draw,lower,pair_lo,_=map(int,record.split())
        b=fixtures[i]
        if ply != len(b["moves"]): continue
        value,states=literal(*bits(b["moves"]),b["stm"],b["rule"])
        eligible=b["rule"]==1 or b["stm"]==1
        if value != b["truth"] or draw or (lower==2)!=(eligible and value==1):
            raise ValueError(("decisive control",b["rule"],ply,value,draw,lower))
        checks.append(dict(rule=b["rule"],ply=ply,truth=value,states=states,lower=lower==2,
                           rank_only=lower==2 and not pair_lo))
    if len(checks)!=8 or sum(x["rank_only"] for x in checks)!=3:
        raise ValueError(("rank-only controls",checks))
    dump(dest/"rank_controls.json",fixtures)
    dump(dest/"pairing_samples.json",samples)
    dump(dest/"pairing.json",dict(games=len(games),rule_counts=dict(Counter(g["rule"] for _,g in games)),
        game_ids=[p.stem for p,_ in games], rank_graphs=4608,literal_truth=dict(truths),
        counts=dict(counts),decisive_controls=checks,false_claims=0, probe_sha256=sha(exe),
        playing_sha256=sha(ROOT/"pbrain-rapfi-plus-v3.exe")))
    print(len(games),dict(counts),dict(truths), flush=True)


if __name__=="__main__":
    run()

