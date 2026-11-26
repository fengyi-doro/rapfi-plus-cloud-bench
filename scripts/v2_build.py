"""Archive the rollback baseline and link an isolated trial from same-flag production objects."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datasets import DATA, ROOT, dump, sha
from node_symbols import resolve

sys.path.insert(0, str(ROOT))
from build import compile_unit, link

TRIALS = ROOT / "bench/v2"
BASE = ROOT / "build/p13"


def update_layout(name):
    path = DATA / "node_symbols.json"
    layout=resolve(name)
    if name in ("orig","plus"):
        meta = json.loads(path.read_bytes())
        meta[name] = layout
        dump(path, meta)
    else:
        dump(DATA/"v2"/("nodes_"+name+".json"), layout)


def freeze():
    TRIALS.mkdir(exist_ok=True, parents=True)
    (DATA/"v2").mkdir(exist_ok=True)
    BASE.mkdir(exist_ok=True)
    if (DATA/"v2/release.json").exists() and (BASE/"manifest.json").is_file():
        existing=json.loads((BASE/"manifest.json").read_bytes())
        assert sha(ROOT/existing["exe"])==existing["sha256"]
        print("Verified archived P1+P3 baseline",existing["sha256"],flush=True)
        return
    manifest = json.loads((ROOT/"build/plus/manifest.json").read_bytes())
    exe = ROOT / "pbrain-rapfi-plus.exe"
    assert sha(exe) == manifest["sha256"]
    for obj in manifest["objects"]:
        shutil.copy2(ROOT/obj, BASE/Path(obj).name)
    for ext in ("exe","pdb"):
        shutil.copy2(exe.with_suffix("."+ext),BASE/f"pbrain-rapfi-p13.{ext}")
    shutil.copy2(exe.with_suffix(".pdb"),BASE/exe.with_suffix(".pdb").name)
    for file in ("eval/mix9svqnnue.cpp","game/board.cpp"):
        dest=TRIALS/"baseline"/file
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"src_plus"/file,dest)
    manifest["objects"]=[str((BASE/Path(o).name).relative_to(ROOT)) for o in manifest["objects"]]
    manifest["exe"]=str((BASE/"pbrain-rapfi-p13.exe").relative_to(ROOT))
    dump(BASE/"manifest.json",manifest)
    update_layout("plus")
    layouts=json.loads((DATA/"node_symbols.json").read_bytes())
    layouts["p13"]=dict(layouts["plus"])
    dump(DATA/"node_symbols.json",layouts)
    dump(DATA/"v2/nodes_p13.json",layouts["p13"])
    print("P1+P3 baseline",manifest["sha256"],flush=True)


def trial(name, files):
    dest = ROOT/"build"/name
    dest.mkdir(parents=True,exist_ok=True)
    tree = ROOT/"src_plus"
    base = json.loads((BASE/"manifest.json").read_bytes())
    objects=list(base["objects"])
    commands=[]
    for logical,path in files.items():
        obj = dest/(logical.replace("/","_")+".o")
        extra=["-iquote",str((tree/logical).parent)]
        command=compile_unit(Path(path),tree,obj,extra)
        subprocess.run(command,check=True,cwd=ROOT)
        objects[base["sources"].index(logical)] = str(obj.relative_to(ROOT))
        commands.append(command)
    exe=dest/f"pbrain-rapfi-{name}.exe"
    cmd=link(objects,exe)
    subprocess.run(cmd,cwd=ROOT,check=True)
    dump(dest/"manifest.json",dict(variant=name,objects=objects,sha256=sha(exe),
         exe=str(exe.relative_to(ROOT)),sources=base["sources"],commands=commands+[cmd]))
    update_layout(name)
    print(name,sha(exe),flush=True)


if __name__=="__main__":
    cmd=argparse.ArgumentParser()
    cmd.add_argument("name")
    cmd.add_argument("sources",nargs="*",help="logical/source.cpp=trial/source.cpp")
    args=cmd.parse_args()
    if args.name=="freeze":
        freeze()
    else:
        trial(args.name,dict(item.split("=",1) for item in args.sources))

