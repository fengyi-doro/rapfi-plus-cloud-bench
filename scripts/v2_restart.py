"""Compare fresh and resident processes after an intervening search, RESTART, TT clear and core swap."""
import json
from datasets import DATA,ROOT,dump,sha
from engine import Engine,filename,topology
import psutil

def observe(e,rule,board):
    e.select(rule)
    e.sync("INFO timeout_match 0\nINFO timeout_turn 0\nINFO max_depth 12\nINFO show_detail 0")
    e.clean_position(board["moves"])
    messages=e.sync("YXSHOWHASHUSAGE")
    assert "MESSAGE Transposition table full: 0.0%" in messages,messages
    result=e.think()
    assert result["depth"]==12 and result["nodes"]==board["pilot_nodes"]
    e.sync("STOP")
    return [result[k] for k in ("x","y","eval","nodes","depth")]


def run():
    hw=topology()
    psutil.Process().cpu_affinity(hw["helper"])
    first,second=hw["performance"][:2]
    boards=json.loads((DATA/"bench_positions.json").read_bytes())
    rows=[]
    for v in ("orig","plus"):
        expected={}
        for rule in (1,2):
            group=[b for b in boards if b["rule"]==rule]
            with Engine(v,second) as e:
                expected[rule]=[observe(e,rule,b) for b in group[:2]]
        with Engine(v,first) as e:
            for rule in (1,2):
                group=[b for b in boards if b["rule"]==rule]
                for j,b in enumerate(group[:2]):
                    e.ps.cpu_affinity([first])
                    e.core=first
                    observe(e,rule,group[-1])
                    e.ps.cpu_affinity([second])
                    e.core=second
                    obtained=observe(e,rule,b)
                    assert obtained==expected[rule][j] and all(x["effective_mask"]==1<<second for x in e.affinity_masks().values())
                    rows.append(dict(variant=v,rule=rule,board=b["id"],root=obtained))
    dump(ROOT/"validation/v2_restart.json",dict(cases=rows,hash_after_clear="0.0%",mismatches=0,
         exe={v:sha(filename(v)) for v in ("orig","plus")}))
    print("Fresh/resident parity",len(rows),"root moves, values and exact nodes",flush=True)

if __name__=="__main__":run()

