import json
import subprocess
from datasets import DATA,ROOT,dump
from v2_build import compile_unit,link

def run():
    if "geometry" not in json.loads((DATA/"v2/release.json").read_bytes())["selected"]:
        return
    tree=ROOT/"src_plus"
    dest=ROOT/"build/plus"
    obj=dest/"geometry_check.o"
    exe=dest/"geometry_check.exe"
    subprocess.run(compile_unit(ROOT/"scripts/v2_geometry_check.cpp",tree,obj,["-I"+str(tree)]),check=True)
    m=json.loads((dest/"manifest.json").read_bytes())
    objects=[o for o,n in zip(m["objects"],m["sources"],strict=True) if n not in ("eval/mix9svqnnue.cpp","main.cpp")]
    subprocess.run(link([*objects,str(obj)],exe),cwd=ROOT,check=True)
    data=json.loads(subprocess.check_output([exe],cwd=ROOT))
    dump(ROOT/"validation/v2_geometry.json",data)
    print("Move geometry",data,flush=True)

if __name__=="__main__":run()

