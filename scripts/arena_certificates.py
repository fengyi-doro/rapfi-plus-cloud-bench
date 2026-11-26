"""Construct replayable forced-four lines and disjoint defensive pairs for the referee."""
from collections import Counter
from itertools import combinations
from rules import BITS, FIVES, ID, exact, forbidden, targets

WINDOWS=sorted({(line,flanks) for rows in FIVES for line,flanks,_ in rows})


def standard_pairs(stones, stm):
    certificates=[]
    occupied=stones[0]|stones[1]
    for attacker in (0,1):
        live=[line & ~occupied for line,flanks in WINDOWS
              if not (line & stones[1-attacker] or flanks & stones[attacker])]
        if len(live)>64 or 0 in live: return None
        singles={s for s in live if s.bit_count()==1}
        if len(singles)>1 or (singles and attacker==stm): return None
        pre=next(iter(singles),0)
        edges=[]
        for bits in live:
            if pre & bits: continue
            edge=[]
            while bits:
                bit=bits & -bits
                edge.append(ID[bit])
                bits^=bit
            edges.append(edge)
        edges.sort(key=len)
        weights=Counter(pair for edge in edges for pair in combinations(edge,2))
        used=set()
        replies=[]
        for edge in edges:
            if any(a in edge and b in edge for a,b in replies): continue
            pairs=[pair for pair in combinations(edge,2) if not used.intersection(pair)]
            if not pairs: return None
            pair=max(pairs,key=weights.get)
            used.update(pair)
            replies.append(list(pair))
        certificates.append(dict(attacker=attacker,preplay=ID[pre] if pre else None,pairs=replies))
    return certificates


def forced_line(stones, last, rule, turn, budget=80):
    score=turn.get("eval") or ""
    if not score.startswith("+M") or not score[2:].isdigit():
        return None
    own,opp=stones[last],stones[1-last]
    attack=turn["y"]*15+turn["x"]
    remaining=[budget]

    def after_attack(mine, other, p, depth):
        if targets(other,mine,exact(rule,1-last),4):
            return None
        wins=targets(mine,other,exact(rule,last),4)
        if not wins: return None
        bit=wins & -wins
        q=ID[bit]
        rest=wins ^ bit
        if rest:
            return [[p,q,ID[rest & -rest],2]]
        if forbidden(other,mine,q,rule,1-last):
            return [[p,q,-1,3]]
        if depth==0 or not remaining[0]: return None
        other |= bit
        danger=targets(other,mine,exact(rule,1-last),4)
        if danger.bit_count()>1: return None
        candidates=targets(mine,other,exact(rule,last),3)
        if danger: candidates &= danger
        while candidates and remaining[0]:
            bit=candidates & -candidates
            candidates^=bit
            remaining[0]-=1
            child=ID[bit]
            if forbidden(mine,other,child,rule,last): continue
            child_stones=mine | bit
            reply=after_attack(child_stones,other,child,depth-1)
            if reply: return [[p,q,-1,1],*reply]
        return None
    return after_attack(own,opp,attack,4)

