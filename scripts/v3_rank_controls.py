"""Generate legal nearly full boards with known one-ply wins or unavoidable double threats."""
import random
from rules import BITS, ID, bits, exact, has_win, targets, valid_history


def controls():
    rng=random.Random(250318163)
    cases=[]
    for rule in (1,2):
        for stm in (0,1):
            for decisive in (1,-1):
                for attempt in range(10000):
                    phase=rng.randrange(4)
                    colors={p:int((p%15+2*(p//15)+phase)%4>=2) for p in range(225)}
                    threat_colors=[1-stm,1-stm]+([stm] if decisive==1 else [])
                    rows=rng.sample(range(1,14), len(threat_colors))
                    protected=set()
                    for y,side in zip(rows,threat_colors):
                        x=rng.randrange(1,10)
                        gap=rng.randrange(5)
                        for k in range(-1,6):
                            p=15*y+x+k
                            protected.add(p)
                            if k==gap: del colors[p]
                            else: colors[p]=1-side if k in (-1,5) else side
                    count=218+stm
                    quotas=[(count+1)//2,count//2]
                    for side in (0,1):
                        surplus=sum(c==side for c in colors.values())-quotas[side]
                        if surplus<0: break
                        cells=[p for p,c in colors.items() if p not in protected and c==side]
                        for p in rng.sample(cells,surplus): del colors[p]
                    if len(colors)!=count: continue
                    lists=[[p for p,c in colors.items() if c==side] for side in (0,1)]
                    if list(map(len,lists))!=quotas: continue
                    stones=[sum(BITS[p] for p in pts) for pts in lists]
                    if any(has_win(stones[c],p,False) for p,c in colors.items()): continue
                    wins=[targets(stones[c],stones[1-c],exact(rule,c),4) for c in (0,1)]
                    if (bool(wins[stm])!=(decisive==1) or wins[1-stm].bit_count()<2): continue
                    for v in lists: rng.shuffle(v)
                    moves=[lists[i%2][i//2] for i in range(count)]
                    if valid_history(moves,rule):
                        cases.append(dict(rule=rule,moves=moves,truth=decisive,stm=stm))
                        break
                else:
                    raise RuntimeError(("could not construct a valid control",rule,stm,decisive))
    return cases

