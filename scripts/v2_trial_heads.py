"""Read each proposed playing binary's full policy grids and float words against archived original."""
import argparse
import json
import subprocess
import sys
from bit_gate import capture
from datasets import DATA,ROOT,dump,sha
from engine import filename
from v2_build import TRIALS
sys.path.insert(0,str(ROOT))
from build import compile_unit,link

def run(names):
    rows=json.loads((DATA/"positions.json").read_bytes())
    original=json.loads((ROOT/"archive/v1/validation/heads_orig.json").read_bytes())
    assert sha(ROOT/"pbrain-rapfi-orig.exe")==json.loads((ROOT/"archive/v1/validation/heads.json").read_bytes())["exe"]["orig"]
    checks={}
    for v in names:
        out=TRIALS/"candidates"/v
        out.mkdir(parents=True,exist_ok=True)
        heads=capture(v,rows,"heads",exe=filename(v),out=out)
        mismatches=[p["id"] for p,a,b in zip(rows,original,heads,strict=True) if a!=b]
        checks[v]=dict(exe=sha(filename(v)),positions=len(rows),wdl_words=540,
                       policy_words=sum(2*x["empty"] for x in heads),mismatching_ids=mismatches)
        dump(out/"heads.json",checks[v])
        assert not mismatches,(v,mismatches)
        print(v,"all heads bit-exact",flush=True)
    if "orbit" in names:
        d=TRIALS/"candidates/orbit"
        tree=ROOT/"src_plus"
        obj=d/"kernel.o"
        subprocess.run(compile_unit(ROOT/"scripts/kernel_check.cpp",tree,obj,
            ["-I"+str(tree),"-I"+str(d/"eval"),"-I"+str(tree/"eval")]),check=True)
        exe=d/"kernel.exe"
        subprocess.run(link([obj],exe),check=True)
        kernel=json.loads(subprocess.check_output([exe]))
        dump(d/"kernel.json",kernel)
        print(kernel,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("names",nargs="+")
    run(p.parse_args().names)

