"""Cycle-weighted instruction-pointer samples and exact per-search thread cycles on P1+P3."""
import argparse
import collections
import csv
import ctypes as C
import json
import mmap
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import psutil
from benchmark import DEPTH, setup
from datasets import DATA,ROOT,dump,sha
from engine import Engine,K,filename,topology
from bit_gate import CDB

OUT=ROOT/"bench/v2/profile"


class Symbol(C.Structure):
    _fields_=[("struct_size",C.c_ulong),("type_index",C.c_ulong),("reserved",C.c_uint64*2),
              ("index",C.c_ulong),("size",C.c_ulong),("module",C.c_uint64),("flags",C.c_ulong),
              ("value",C.c_uint64),("address",C.c_uint64),("register",C.c_ulong),("scope",C.c_ulong),
              ("tag",C.c_ulong),("name_len",C.c_ulong),("max_len",C.c_ulong),("name",C.c_char*1)]


class Line(C.Structure):
    _fields_=[("size",C.c_ulong),("key",C.c_void_p),("line",C.c_ulong),("filename",C.c_char_p),("address",C.c_uint64)]


def family(name):
    # One logical function may have separate instantiations for rule, move type or node type.
    name=re.sub(r"<.*","",name)
    return name.split("(")[0]


def symbolize(exe,base,rows):
    dbg=C.WinDLL(str(Path(CDB).with_name("dbghelp.dll")),use_last_error=True)
    h=C.c_void_p(-1)
    dbg.SymInitializeW.argtypes=[C.c_void_p,C.c_wchar_p,C.c_int]
    dbg.SymLoadModuleExW.argtypes=[C.c_void_p,C.c_void_p,C.c_wchar_p,C.c_wchar_p,C.c_uint64,C.c_ulong,C.c_void_p,C.c_ulong]
    dbg.SymLoadModuleExW.restype=C.c_uint64
    dbg.SymFromAddr.argtypes=[C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64),C.c_void_p]
    dbg.SymGetLineFromAddr64.argtypes=[C.c_void_p,C.c_uint64,C.POINTER(C.c_ulong),C.POINTER(Line)]
    dbg.SymCleanup.argtypes=[C.c_void_p]
    dbg.SymSetOptions(2|0x10|0x80000)
    assert dbg.SymInitializeW(h,str(exe.parent),False)
    assert dbg.SymLoadModuleExW(h,None,str(exe),None,base,0,None,0)==base,C.get_last_error()
    data=exe.read_bytes()
    pe=struct.unpack_from("<I",data,0x3c)[0]
    image_size=struct.unpack_from("<I",data,pe+24+56)[0]
    output=[]
    for raw in rows:
        addr=int(raw["ip"],16)
        name="outside engine image"
        loc=""
        if base<=addr<base+image_size:
            buf=C.create_string_buffer(C.sizeof(Symbol)+3072)
            symbol=Symbol.from_buffer(buf)
            symbol.struct_size=C.sizeof(Symbol)
            symbol.max_len=3000
            delta=C.c_uint64()
            name="unresolved engine instruction"
            if dbg.SymFromAddr(h,addr,C.byref(delta),buf):
                name=C.string_at(C.addressof(buf)+Symbol.name.offset,symbol.name_len).decode(errors="replace")
            lin=Line()
            lin.size=C.sizeof(lin)
            offset=C.c_ulong()
            if dbg.SymGetLineFromAddr64(h,addr,C.byref(offset),C.byref(lin)):
                loc=Path(os.fsdecode(lin.filename)).name+":"+str(lin.line)
        output.append(dict(ip=raw["ip"],rva=addr-base if base<=addr<base+image_size else None,
                           symbol=name,family=family(name),line=loc,samples=int(raw["samples"]),cycles=int(raw["cycles"])))
    dbg.SymCleanup(h)
    return output


def run(reps, variant="p13", out=OUT, core=None):
    out.mkdir(parents=True,exist_ok=True)
    observer=out/"worker_sample.exe"
    subprocess.run([sys.executable,"-m","ziglang","c++","-O2","-std=c++20",
                    str(ROOT/"scripts/worker_sample.cpp"),"-o",str(observer)],check=True)
    hw=topology()
    helpers = [c for c in hw["helper"] if c not in (14, 15)]
    core = hw["performance"][0] if core is None else core
    if core not in hw["performance"] or core in (14, 15) or not helpers:
        raise ValueError("unavailable profiling cores")
    psutil.Process().cpu_affinity(helpers)
    boards=json.loads((DATA/"bench_positions.json").read_bytes())
    trials=[]
    with Engine(variant,core) as e:
        modules=(C.c_void_p*64)()
        used=C.c_ulong()
        assert K.K32EnumProcessModules(int(e.proc._handle),modules,C.sizeof(modules),C.byref(used))
        base=modules[0]
        setup(e,0)
        e.clean_position(boards[0]["moves"])
        tid=e.think(measure_cycles=True)["worker_tid"]
        tag=f"RapfiCycles{os.getpid()}"
        with mmap.mmap(-1,8,tagname=tag) as gate:
            args=[str(observer),str(tid),tag,str(helpers[-1]),str(out/"samples.csv")]
            p=subprocess.Popen(args,stdout=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                assert p.stdout.readline().strip()==b"READY"
                window=0
                for rule in range(3):
                    setup(e,rule)
                    group=[b for b in boards if b["rule"]==rule]
                    for rep in range(reps):
                        for b in group:
                            e.clean_position(b["moves"])
                            struct.pack_into("<I",gate,0,window+1)
                            try:
                                obs=e.think(measure_cycles=True)
                            finally:
                                window+=2
                                struct.pack_into("<I",gate,0,window)
                            assert obs["worker_tid"]==tid and obs["depth"]==DEPTH
                            trials.append(dict(id=b["id"],rule=rule,rep=rep,**obs))
                            e.sync("STOP")
            finally:
                struct.pack_into("<I",gate,4,1)
                p.wait(timeout=8)
                p.stdout.close()
            assert p.returncode==0,p.returncode
    rows=symbolize(filename(variant),base,list(csv.DictReader((out/"samples.csv").open())))
    dump(out/"instructions.json",rows)
    dump(out/"searches.json",trials)
    by_name=collections.defaultdict(lambda:[0,0])
    for r in rows:
        v=by_name[r["family"]]
        v[0]+=r["cycles"]
        v[1]+=r["samples"]
    total=sum(x[0] for x in by_name.values())
    ranks=[dict(function=k,samples=v[1],cycles=v[0],pct=v[0]*100/total)
           for k,v in sorted(by_name.items(),key=lambda it:-it[1][0])]
    dump(out/"ranking.json",dict(method="random 95..215 us worker RIP samples weighted by adjacent QueryThreadCycleTime deltas within marked depth-12 search intervals",
         exe=sha(filename(variant)),core=core,observer_core=helpers[-1],worker_tid=tid,reps=reps,positions=len(boards),
         exact_worker_cycles=sum(x["worker_cycles"] for x in trials),weighted_cycles=total,samples=sum(v[1] for v in by_name.values()),ranking=ranks))
    print(ranks[:12],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--reps",type=int,default=12)
    p.add_argument("--variant", default="p13")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--core", type=int)
    args=p.parse_args()
    run(args.reps, args.variant, args.out, args.core)

