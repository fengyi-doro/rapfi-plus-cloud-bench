"""Exhaustively compare the actual compiled accessors for both tactical labels and defence masks."""
import hashlib
from pathlib import Path
import subprocess
import sys
from datasets import ROOT,dump
sys.path.insert(0, str(ROOT))
from build import compile_unit, link


def run():
    streams = []
    tree = ROOT/"src_plus"
    for name, flag in (("dense", "0"), ("quotient", "1")):
        dest = ROOT/"build/v3-line-gate"/name
        dest.mkdir(parents=True,exist_ok=True)
        obj = dest/"v3_line_gate.o"
        defines=["-I"+str(tree), "-DRAPFI_LINE_QUOTIENT="+flag]
        table=dest/"pattern.o"
        subprocess.run(compile_unit(tree/"game/pattern.cpp", tree, table, defines),check=True)
        subprocess.run(compile_unit(ROOT/"scripts/v3_line_gate.cpp",tree,obj,defines),check=True)
        exe = dest/"v3_line_gate.exe"
        subprocess.run(link([obj, table], exe), check=True)
        streams.append(subprocess.check_output([exe]))
    dense, quotient = streams
    if len(dense) != 279633*3 or dense != quotient:
        raise ValueError("compiled tactical line triples disagree")
    dest=ROOT/"validation/v3"
    dest.mkdir(parents=True, exist_ok=True)
    dump(dest/"line_gate.json", dict(keys=len(dense)//3, rules=[14641,132496,132496],
         mismatches=0, raw_tuple_sha256=hashlib.sha256(dense).hexdigest(), quotient_bytes=41658))
    print(len(dense)//3,"line triples match")


if __name__ == "__main__":
    run()

