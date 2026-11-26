"""Count exact VQ-index collisions in both perspectives on the frozen real game histories."""
from array import array
import json
import struct
import lz4.frame
from datasets import DATA, ROOT, dump

DIRECTIONS = ((1,0),(0,1),(1,-1),(1,1))
POW = [3**i for i in range(12)]
SHAPES = 442503


def border(left, right):
    l, r = 5-left, 5-right
    if r == 0:
        return 0 if l == 0 else 2*POW[11]+sum(POW[11-i] for i in range(1,l))
    low, high = min(l,r),min(l,r-1)
    return (POW[11]+2*sum(POW[:low])+2*sum(POW[10-i] for i in range(high))
            +sum(POW[i] for i in range(low,r-1))+sum(POW[10-i] for i in range(high,l-1)))


def profile():
    tables = {}
    for p in ROOT.glob("mix9svq*.bin.lz4"):
        raw = lz4.frame.decompress(p.read_bytes())
        magic,_,_,_,description = struct.unpack_from("<5I",raw)
        assert magic == 0xacd8cc6a
        offset = 20+description+2*65536*64*10//8
        tables[p.name] = array("H",raw[offset:offset+4*SHAPES])
    empty = []
    for p in range(225):
        x,y=p%15,p//15
        left,right,top,bottom = min(x,5),min(14-x,5),min(y,5),min(14-y,5)
        empty.append([border(left,right),border(top,bottom),border(min(left,bottom),min(right,top)),
                      border(min(left,top),min(right,bottom))])
    count = {name:dict(changes=0,zero=0,center_zero=0) for name in tables}
    for row in json.loads((DATA/"positions.json").read_bytes()):
        for side in (0,1):
            suffix = ("freestyle_bsmix","standard_bs15","renju_bs15_"+("black","white")[side])[row["rule"]]
            name = "mix9svq"+suffix+".bin.lz4"
            table, tally = tables[name], count[name]
            state = [s.copy() for s in empty]
            for ply,p in enumerate(row["moves"]):
                for d,(dx,dy) in enumerate(DIRECTIONS):
                    base = (d//2)*SHAPES
                    for dist in range(-5,6):
                        x,y=p%15+dist*dx,p//15+dist*dy
                        if 0<=x<15 and 0<=y<15:
                            cell=state[15*y+x]
                            old=cell[d]
                            cell[d] += (1+((ply%2)^side))*POW[dist+5]
                            assert cell[d] < SHAPES
                            hit = table[base+old] == table[base+cell[d]]
                            tally["changes"] += 1
                            tally["zero"] += hit
                            tally["center_zero"] += hit and dist == 0
    dump(ROOT/"validation/codebook.json",count)
    print("Codebook index collisions",count)


if __name__ == "__main__":
    profile()
