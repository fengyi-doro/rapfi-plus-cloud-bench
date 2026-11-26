"""Cross-check the uninstrumented RIP ranking by timing all surviving project C++ call boundaries."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil
from datasets import DATA,ROOT,dump,sha
from engine import Engine,topology
from v2_build import update_layout
from v2_profile import symbolize
sys.path.insert(0,str(ROOT))
from build import compile_unit,link

DEST=ROOT/"build/cycletrace"
OUT=ROOT/"bench/v2/profile"


def build():
    tree=ROOT/"src_plus"
    DEST.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/"build/p13/manifest.json").read_bytes())
    original=(tree/"search/ab/search.cpp").read_text(encoding="utf8")
    marker="const RootMove *ABSearcher::searchMain(SearchThread &th)\n{"
    declaration=('extern "C" void rapfiProfileBegin();\nextern "C" void rapfiProfileEnd();\n'
       'struct SearchCycleProbe {\n'
       '    __attribute__((no_instrument_function,always_inline)) SearchCycleProbe() { rapfiProfileBegin(); }\n'
       '    __attribute__((no_instrument_function,always_inline)) ~SearchCycleProbe() { rapfiProfileEnd(); }\n};\n')
    assert original.count(marker)==1
    path=DEST/"instrumented_search.cpp"
    path.write_text(original.replace(marker,declaration+marker+"\n    SearchCycleProbe measure;"),encoding="utf8")
    def unit(pair):
        name,existing=pair
        if not name.endswith(".cpp"):
            return str(ROOT/existing)
        frozen=OUT.parent/"baseline"/name
        source=path if name=="search/ab/search.cpp" else frozen if frozen.is_file() else tree/name
        obj=DEST/(name.replace("/","_")+".o")
        args=["-finstrument-functions-after-inlining","-iquote",str((tree/name).parent)]
        result=subprocess.run(compile_unit(source,tree,obj,args),capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace")[-1800:])
        return str(obj)
    with ThreadPoolExecutor(6) as p:
        objects=list(p.map(unit,zip(manifest["sources"],manifest["objects"],strict=True)))
    hooks=DEST/"cycle_hooks.o"
    subprocess.run(compile_unit(ROOT/"scripts/cycle_trace.cpp", tree,hooks),check=True)
    subprocess.run(link([*objects,str(hooks)],DEST/"pbrain-rapfi-cycletrace.exe"),check=True)
    update_layout("cycletrace")


def run():
    boards=json.loads((DATA/"bench_positions.json").read_bytes())
    hw=topology()
    psutil.Process().cpu_affinity(hw["helper"])
    log=OUT/"call_cycles.log"
    log.write_text("")
    os.environ["RAPFI_CYCLES_FILE"]=str(log)
    records=[]
    with Engine("cycletrace",hw["performance"][0]) as e:
        e.sync("INFO timeout_turn 0\nINFO timeout_match 0\nINFO max_depth 12")
        for row in boards:
            e.select(row["rule"])
            e.clean_position(row["moves"])
            a=e.think(measure_cycles=True)
            assert a["nodes"]==row["pilot_nodes"] and a["depth"]==12
            records.append(dict(id=row["id"],**a))
            e.sync("STOP")
    times,calls=Counter(),Counter()
    scope=outer=roots=0
    for line in log.read_text().splitlines():
        t,i,a,b=line.split()
        if t=="SEARCH":
            roots+=1
            scope+=int(a)
            outer+=int(b)
        else:
            k=int(i,16)
            times[k]+=int(a)
            calls[k]+=int(b)
    assert roots==len(boards)
    table=symbolize(DEST/"pbrain-rapfi-cycletrace.exe",0x140000000,
            [dict(ip=hex(0x140000000+k),samples=calls[k],cycles=v) for k,v in times.items()])
    indexed=Counter()
    entries=Counter()
    for row in table:
        indexed[row["family"]]+=row["cycles"]
        entries[row["family"]]+=row["samples"]
    indexed["Search::ABSearcher::searchMain"]+=outer
    entries["Search::ABSearcher::searchMain"]+=roots
    assigned=sum(indexed.values())
    ranks=[dict(function=k,exclusive_cycles=v,calls=entries[k],pct_of_attributed=100*v/assigned)
           for k,v in indexed.most_common()]
    dump(OUT/"call_sites.json",table)
    dump(OUT/"call_searches.json",records)
    dump(OUT/"call_ranking.json",dict(method="Clang after-inlining entry/exit instrumentation on project C++ translation units; each nested interval taken with QueryThreadCycleTime, excluding measured hook bodies",
      scope_cycles=scope,attributed_cycles=assigned,instrumentation_cycles=scope-assigned,positions=roots,
      ranking=ranks,exe=sha(DEST/"pbrain-rapfi-cycletrace.exe")))
    print("Exclusive worker cycles",ranks[:7],flush=True)


if __name__=="__main__":
    if "--measure-only" not in sys.argv:
        build()
    if "--build-only" not in sys.argv:
        run()

