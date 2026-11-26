"""Resolve the playing PE's public Search::Engine and private counter layout from its own PDB."""
import json
import argparse
import re
import subprocess
from bit_gate import ADDR, CDB
from datasets import DATA, ROOT, dump, sha
from engine import filename


def resolve(variant):
    exe = filename(variant)
    module = exe.stem.replace("-","_")
    cmds = (f"lm m {module}; x {module}!Search::Engine; dt {module}!Search::SearchEngine threads_; "
            f"dt {module}!Search::SearchThread numNodes; q")
    result = subprocess.check_output([CDB, "-sins", "-y", str(exe.parent), "-c", cmds, str(exe)],
                                     cwd=ROOT, encoding="utf-8",errors="replace",timeout=25)
    def address(pattern):
        return int(re.search(pattern,result,re.M)[1].replace("`",""),16)
    base = address(rf"^({ADDR}) {ADDR}\s+{module}\s+")
    global_address = address(rf"^({ADDR}) {module}!Search::Engine = class")
    vector = int(re.search(r"\+0x([0-9a-f]+) threads_\s+:",result)[1],16)
    nodes = int(re.search(r"\+0x([0-9a-f]+) numNodes\s+:",result)[1],16)
    return dict(sha256=sha(exe), vector_rva=global_address-base+vector, nodes_offset=nodes)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("variants",nargs="*")
    variants=p.parse_args().variants or (("orig","plus") if (DATA/"v2/release.json").exists() else
                                         ("orig","plus","p1","p2","p3","p4","p123"))
    path=DATA/"node_symbols.json"
    existing=json.loads(path.read_bytes()) if path.exists() else {}
    legacy_changed=False
    for v in variants:
        layout=resolve(v)
        if v == "v4":
            (DATA/"v4").mkdir(exist_ok=True)
            dump(DATA/"v4/nodes_v4.json",layout)
        elif v in ("f9","v2","v3"):
            (DATA/"v3").mkdir(exist_ok=True)
            dump(DATA/"v3"/f"nodes_{v}.json",layout)
        else:
            existing[v]=layout
            legacy_changed=True
    if legacy_changed:
        dump(path,existing)
