"""Independent rule oracle. Python bitboards use 16-bit rows with a wall column."""
from functools import lru_cache
from itertools import combinations

SIDE = 15
BITS = [1 << (16*(p//15)+p%15) for p in range(225)]
ID = {b:p for p,b in enumerate(BITS)}
BOARD = sum(BITS)
FIVES = [[] for _ in BITS]
SIXES = [[] for _ in BITS]
STRAIGHTS = [[] for _ in BITS]
STARTS = {}

def bit(x,y):
    return BITS[15*y+x] if 0<=x<15 and 0<=y<15 else 0

for dx,dy in ((1,0),(0,1),(1,1),(-1,1)):
    shift = dx+dy*16
    starts = 0
    for x in range(15):
        for y in range(15):
            for n,table in ((4,STRAIGHTS),(5,FIVES),(6,SIXES)):
                cells = [bit(x+i*dx,y+i*dy) for i in range(n)]
                if not all(cells):
                    continue
                before,after = bit(x-dx,y-dy),bit(x+n*dx,y+n*dy)
                mask = sum(cells)
                if n==5:
                    starts |= cells[0]
                for b in cells:
                    table[ID[b]].append((mask,before|after,
                        bit(x-2*dx,y-2*dy)|bit(x+(n+1)*dx,y+(n+1)*dy)))
    STARTS[shift] = starts

def exact(rule,side):
    return rule==1 or (rule==2 and side==0)

def has_win(own,p,requires_five):
    return any(own & m == m and (not requires_five or not own & ends)
               for m,ends,_ in FIVES[p])

@lru_cache(65536)
def black_status(black,white,p):
    """Return 1 five, -1 foul, or 0 ordinary legal. Black already occupies p."""
    if has_win(black,p,True):
        return 1
    if any(black & m == m for m,_,_ in SIXES[p]):
        return -1
    fours = {black&m for m,ends,_ in FIVES[p] if not white&m and not black&ends
             and (black&m).bit_count()==4}
    if len(fours)>1:
        return -1
    threes = {}
    occupied = black|white
    for m,ends,beyond in STRAIGHTS[p]:
        if ends.bit_count()!=2 or ends & occupied or beyond & black or m & white:
            continue
        stones = black & m
        if stones.bit_count()==3:
            threes.setdefault(stones,set()).add(m^stones)
    if len(threes)>=2:
        real = 0
        for extensions in threes.values():
            if any(black_status(black|ext,white,ID[ext])==0 for ext in extensions):
                real += 1
                if real==2:
                    return -1
    return 0

def forbidden(own,opp,p,rule,side):
    return rule==2 and side==0 and black_status(own|BITS[p],opp,p)==-1

def valid_history(moves,rule):
    stones=[0,0]
    for i,p in enumerate(moves):
        a=i%2
        if not 0<=p<225 or (stones[0]|stones[1]) & BITS[p] or forbidden(stones[a],stones[1-a],p,rule,a):
            return False
        stones[a] |= BITS[p]
        if has_win(stones[a],p,exact(rule,a)):
            return False
    return True

TRIPLES = list(combinations(range(5),3))
@lru_cache(32768)
def targets(own,opp,needs_five,n):
    out = 0
    for s,valid in STARTS.items():
        if needs_five:
            valid &= ~((own<<s)|(own>>(5*s)))
        if not valid:
            continue
        a,b,c,d,e = (own>>(i*s) for i in range(5))
        if n==4:
            abc, de = a&b&c,d&e
            out |= ((valid&b&c&de) | ((valid&a&c&de)<<s) | ((valid&a&b&de)<<(2*s))
                    | ((valid&abc&e)<<(3*s)) | ((valid&abc&d)<<(4*s)))
        else:
            slots=(a,b,c,d,e)
            free = BOARD & ~(own|opp)
            for group in TRIPLES:
                i,j,k=group
                start=valid&slots[i]&slots[j]&slots[k]
                if not start:
                    continue
                holes = [h for h in range(5) if h not in group]
                u,v = holes
                start &= (free>>(s*u)) & (free>>(s*v))
                out |= (start<<(s*u)) | (start<<(s*v))
    return out & BOARD & ~(own|opp)

def bits(moves):
    return sum(BITS[p] for p in moves[::2]),sum(BITS[p] for p in moves[1::2])

def replay(own,opp,side,rule,steps):
    for i,(attack,block,other,kind) in enumerate(steps):
        m=BITS[attack]
        if (own|opp)&m or forbidden(own,opp,attack,rule,side):
            return False
        own |= m
        if kind==0:
            return i==len(steps)-1 and has_win(own,attack,exact(rule,side))
        if has_win(own,attack,exact(rule,side)) or targets(opp,own,exact(rule,1-side),4):
            return False
        wins=targets(own,opp,exact(rule,side),4)
        if not wins & BITS[block]:
            return False
        if kind==2:
            return i==len(steps)-1 and block!=other and bool(wins & BITS[other])
        if forbidden(opp,own,block,rule,1-side):
            return kind==3 and i==len(steps)-1
        if kind!=1:
            return False
        opp |= BITS[block]
    return False

def replies_in_tree(own,opp,side,rule,steps):
    """Enumerate every legal reply in the AND tree, including an inert reply.

    The designated block recurses. Each other reply must allow the named
    immediate completion on the resulting board. Count the checked branches.
    """
    m,block,other,kind=steps[0]
    assert not (own|opp)&BITS[m] and not forbidden(own,opp,m,rule,side)
    own |= BITS[m]
    if kind==0:
        assert len(steps)==1 and has_win(own,m,exact(rule,side))
        return 0
    count=0
    for reply in range(-1,225):
        b=0 if reply==-1 else BITS[reply]
        if b&(own|opp) or (reply>=0 and forbidden(opp,own,reply,rule,1-side)):
            continue
        parried=opp|b
        assert reply<0 or not has_win(parried,reply,exact(rule,1-side))
        count+=1
        if reply==block and kind==1:
            count+=replies_in_tree(own,parried,side,rule,steps[1:])
        else:
            target=other if reply==block else block
            assert target>=0 and not (own|parried)&BITS[target]
            assert has_win(own|BITS[target],target,exact(rule,side))
    return count

@lru_cache(65536)
def has_vcf(own,opp,side,rule,horizon):
    own_exact,opp_exact=exact(rule,side),exact(rule,1-side)
    if targets(own,opp,own_exact,4):
        return True
    if horizon<3:
        return False
    candidates=targets(own,opp,own_exact,3)
    danger=targets(opp,own,opp_exact,4)
    if danger:
        if danger.bit_count()>1:
            return False
        candidates &= danger
    while candidates:
        m=candidates & -candidates
        candidates ^= m
        if forbidden(own,opp,ID[m],rule,side):
            continue
        win=targets(own|m,opp,own_exact,4)
        if win and (win.bit_count()>1 or forbidden(opp,own|m,ID[win],rule,1-side)
                    or has_vcf(own|m,opp|win,side,rule,horizon-2)):
            return True
    return False
