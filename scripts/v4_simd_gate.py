"""Check real PE instructions, affine integer boundaries and the convolution address map."""
import json
from pathlib import Path
import re
import subprocess
import sys
import statistics
import psutil

from bit_gate import CDB
from datasets import ROOT, dump, sha
sys.path.insert(0, str(ROOT))
from build import compile_unit, link


def run():
    dest=ROOT/"validation/v4"
    dest.mkdir(parents=True,exist_ok=True)
    directory=ROOT/"build/v4"
    tree=ROOT/"src_plus"
    manifest=json.loads((directory/"manifest.json").read_bytes())
    playing=Path(manifest["exe"])
    assert sha(playing)==manifest["sha256"]
    flags=["-I"+str(tree), *manifest["flags"]]

    result={}
    for isa in ("avx2","vnni"):
        extra=[x for x in flags if isa=="vnni" or x not in ("-mavxvnni","-DUSE_VNNI")]
        obj=directory/("affine_"+isa+".o")
        exe=obj.with_suffix(".exe")
        subprocess.run(compile_unit(ROOT/"scripts/v4_simd_check.cpp",tree,obj,extra),check=True)
        subprocess.run(link([obj],exe),check=True)
        result[isa]=json.loads(subprocess.check_output([exe]))
    samples={key:[] for key in ("avx2","vnni")}
    for repeat in range(8):
        for isa in (("avx2","vnni") if repeat%2==0 else ("vnni","avx2")):
            values=subprocess.check_output([directory/("affine_"+isa+".exe"),"--bench"],text=True)
            samples[isa].extend(map(float,values.split()))
    result["linear64_cpu"]=psutil.Process().cpu_affinity()
    result["linear64_median_ns"]={key:statistics.median(ns) for key,ns in samples.items()}
    result["linear64_samples_ns"]=samples
    obj=directory/"geometry_check.o"
    subprocess.run(compile_unit(ROOT/"scripts/v2_geometry_check.cpp",tree,obj,flags),check=True)
    objects=[o for o in manifest["objects"] if Path(o).name not in ("main.cpp.o","eval_mix9svqnnue.cpp.o")]
    exe=obj.with_suffix(".exe")
    subprocess.run(link([*objects,obj],exe),check=True)
    result["geometry"]=json.loads(subprocess.check_output([exe]))

    opcodes=("vpdpbusd","vpdpbssd","vpdpwssd","vpmaddubsw","vpmaddwd","vpmulhw")
    functions={"value":"Evaluation::mix9svq::Accumulator::evaluateValue",
               "policy":"Evaluation::mix9svq::Accumulator::evaluatePolicy",
               "move":"Evaluation::mix9svq::Accumulator::move"}
    for label,pe in [("v3",ROOT/"pbrain-rapfi-plus-v3.exe"),("v4",playing)]:
        module=pe.stem.replace("-","_")
        cmd="; ".join(f".echo BEGIN_{k}; uf {module}!{symbol}; .echo END_{k}" for k,symbol in functions.items())+"; q"
        assembly=subprocess.check_output([CDB,"-sins","-y",str(pe.parent),"-c",cmd,str(pe)],
                                         encoding="utf8",errors="replace",timeout=45)
        counts={}
        for k in functions:
            lines=re.search(rf"(?ms)^BEGIN_{k}\s*$\n(.*?)^END_{k}\s*$",assembly)[1]
            counts[k]={op:len(re.findall(rf"\b{op}\b",lines,re.I)) for op in opcodes}
        result[label+"_instructions"]=counts
        (dest/(label+"_heads_move.asm")).write_text(assembly,encoding="utf8")
    instructions=result["v4_instructions"]
    assert instructions["value"]["vpdpbusd"] and instructions["policy"]["vpdpbusd"]
    assert instructions["policy"]["vpdpwssd"] and instructions["move"]["vpmulhw"]
    result["exe"]=manifest["sha256"]
    dump(dest/"simd.json",result)
    print(result,flush=True)


if __name__=="__main__":
    run()
