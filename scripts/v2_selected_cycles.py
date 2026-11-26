"""Direct per-function active-thread cycle probes restricted to the four retail-sampled hotspots."""
import json
import os
from pathlib import Path
import subprocess
import psutil
from datasets import DATA,ROOT,sha,dump
from engine import Engine,filename,topology
from v2_build import BASE,TRIALS,update_layout,compile_unit,link

OUT=TRIALS/"profile"
DEST=ROOT/"build/narrowcycles"
NAMES=["other root work","Evaluation::mix9svq::Accumulator::move","starBlock4","Board::move",
       "Evaluation::mix9svq::Accumulator::evaluatePolicy"]
HEADER='''extern "C" void enterSelected(unsigned);
extern "C" void exitSelected(unsigned);
namespace {
struct CycleInterval {
    unsigned id;
    explicit CycleInterval(unsigned i):id(i){ enterSelected(id); }
    ~CycleInterval(){ exitSelected(id); }
};
}
'''

def build():
    DEST.mkdir(parents=True,exist_ok=True)
    tree=ROOT/"src_plus"
    manifest=json.loads((BASE/"manifest.json").read_bytes())
    objects=manifest["objects"].copy()
    for name,sites in {"eval/mix9svqnnue.cpp":[("void Accumulator::move(",1),("inline void starBlock4(",2),
                              ("void Accumulator::evaluatePolicy(",4)],
         "game/board.cpp":[("void Board::move(Pos pos)",3)],
         "search/ab/search.cpp":[("const RootMove *ABSearcher::searchMain(",0)]}.items():
        frozen=TRIALS/"baseline"/name
        s=(frozen if frozen.exists() else tree/name).read_text(encoding="utf8")
        for prefix,idx in sites:
            assert s.count(prefix)==1,prefix
            brace=s.index("{",s.index(prefix))+1
            s=s[:brace]+f"\n    CycleInterval measure({idx});"+s[brace:]
        path=DEST/Path(name).name
        path.write_text(HEADER+s,encoding="utf8")
        obj=DEST/(name.replace("/","_")+".o")
        subprocess.run(compile_unit(path,tree,obj,["-iquote",str((tree/name).parent)]),check=True)
        objects[manifest["sources"].index(name)]=str(obj)
    hook=DEST/"hook.o"
    subprocess.run(compile_unit(ROOT/"scripts/selected_cycles.cpp",tree,hook),check=True)
    subprocess.run(link([*objects,str(hook)],filename("narrowcycles")),check=True)
    update_layout("narrowcycles")


def run():
    boards=json.loads((DATA/"bench_positions.json").read_bytes())
    hw=topology()
    psutil.Process().cpu_affinity(hw["helper"])
    raw=OUT/"selected_cycles.log"
    raw.write_text("")
    os.environ["RAPFI_CYCLES_FILE"]=str(raw)
    records=[]
    with Engine("narrowcycles",hw["performance"][0]) as e:
        e.sync("INFO timeout_turn 0\nINFO timeout_match 0\nINFO max_depth 12")
        for rule in range(3):
            e.select(rule)
            for rep in range(4):
                for b in (r for r in boards if r["rule"]==rule):
                    e.clean_position(b["moves"])
                    result=e.think(measure_cycles=True)
                    assert result["depth"]==12 and result["nodes"]==b["pilot_nodes"]
                    records.append(dict(rep=rep,id=b["id"],**result))
                    e.sync("STOP")
    values=[list(map(int,l.split())) for l in raw.read_text().splitlines()]
    assert len(values)==len(records)==96 and all(sum(r[2::2])==r[0] for r in values)
    total=sum(r[0] for r in values)
    ranking=[]
    for j,name in enumerate(NAMES):
        cycles=sum(r[2+2*j] for r in values)
        calls=sum(r[3+2*j] for r in values)
        corrected=sum(max(0,r[2+2*j]-r[1]*r[3+2*j]) for r in values)
        ranking.append(dict(function=name,cycles=cycles,calls=calls,corrected=corrected,pct=cycles/total*100))
    dump(OUT/"selected_searches.json",records)
    dump(OUT/"selected_ranking.json",dict(positions=24,reps=4,total_scope_cycles=total,
         baseline_exe=sha(filename("p13")),probe_exe=sha(filename("narrowcycles")),ranking=ranking,
         timer_floor_cycles=values[0][1],method="exclusive active-thread cycle intervals for the four hot functions, remaining code measured in the root span; corrected estimate subtracts one adjacent counter read per call"))
    print("Selected direct cycles",ranking,flush=True)


if __name__=="__main__":
    import sys
    if "--measure-only" not in sys.argv:build()
    if "--build-only" not in sys.argv:run()

