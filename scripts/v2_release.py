"""Freeze the best additive combination, rebuild from the independent source tree, validate, and play."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datasets import DATA,ROOT,dump,sha
from v2_build import TRIALS,trial,update_layout
from v2_candidates import BASE,CASES,MIX,BOARD,orbit,prepared_geometry,packed_board,stable_pattern_counts
import v2_gate

TRANSFORMS={"geometry":(MIX,prepared_geometry),"orbit":(MIX,orbit),"pattern":(BOARD,packed_board),
            "counts":(BOARD,stable_pattern_counts)}

def command(script,*args):
    subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)


def materialize(tag, selected):
    dest=CASES/tag
    files={}
    for logical in (MIX,BOARD):
        current=(BASE/logical).read_text(encoding="utf8")
        relevant=[TRANSFORMS[s][1] for s in selected if TRANSFORMS[s][0]==logical]
        if not relevant:continue
        for transform in relevant:
            current=transform(current)
        path=dest/logical
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(current,encoding="utf8",newline="\n")
        files[logical]=path
    if "orbit" in selected and tag!="orbit":
        shutil.copy2(CASES/"orbit/eval/mix9_orbit.h",dest/"eval/mix9_orbit.h")
    return files


def select():
    individually={}
    for directory in ("gates","counts_gate"):
        scores=json.loads((TRIALS/directory/"summary.json").read_bytes())["comparisons"]
        individually.update(scores)
    ordered=sorted([s for s,r in individually.items() if r["admitted"]],
                   key=lambda s:individually[s]["wall_nps_pct"],reverse=True)
    for name in ordered:
        assert not json.loads((CASES/name/"heads.json").read_bytes())["mismatching_ids"]
    selected=ordered[:1]
    best=selected[0] if selected else "p13"
    steps=[]
    for n,name in enumerate(ordered[1:],1):
        options=[*selected,name]
        tag=f"stack{n}"
        trial(tag,materialize(tag,options))
        stage="gate_"+tag
        v2_gate.run([tag],6,stage,best)
        score=json.loads((TRIALS/stage/"summary.json").read_bytes())["comparisons"][tag]
        steps.append(dict(control=best,candidate=tag,added=name,gate=stage,admitted=score["admitted"]))
        if score["admitted"]:
            selected=options
            best=tag
    frozen=materialize("selected",selected)
    for logical in (MIX,BOARD):
        shutil.copy2(frozen.get(logical,BASE/logical),ROOT/"src_plus"/logical)
    if "orbit" in selected:
        shutil.copy2(CASES/"orbit/eval/mix9_orbit.h",ROOT/"src_plus/eval/mix9_orbit.h")
    release=dict(rollback="P2 restoration of upstream evaluatePolicy and removal of four-point helper",
                 individual=ordered,selected=selected,incremental=steps,best_trial=best,
                 source={logical:sha(ROOT/"src_plus"/logical) for logical in (MIX,BOARD)})
    dump(DATA/"v2/release.json",release)
    print("Frozen v2 release",release,flush=True)


def build_verify():
    original=sha(ROOT/"pbrain-rapfi-orig.exe")
    official=json.loads((ROOT/"archive/v1/match/design.json").read_bytes())["exe"]
    assert original==official["orig"] and sha(ROOT/"pbrain-rapfi-plus-v1.exe")==official["plus"]
    command("build.py","plus")
    update_layout("plus")
    release=json.loads((DATA/"v2/release.json").read_bytes())
    release["exe"]=dict(orig=original,plus=sha(ROOT/"pbrain-rapfi-plus.exe"),v1=official["plus"])
    dump(DATA/"v2/release.json",release)
    command("scripts/source_diff.py")
    command("scripts/probes.py")
    command("scripts/v2_geometry_check.py")
    command("scripts/bit_gate.py")
    command("scripts/tactics.py")
    command("scripts/mix10_gate.py")
    command("scripts/inspect_isa.py")
    command("scripts/v2_dispatch.py")
    command("scripts/v2_restart.py")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("select","validate","final","match","report","all"))
    args=p.parse_args()
    if args.stage in ("select","all"):select()
    if args.stage in ("validate","all"):build_verify()
    if args.stage in ("final","all"):command("scripts/benchmark.py","final","--reps",8)
    if args.stage in ("match","all"):command("scripts/arena.py","--v2","--out",ROOT/"match_v2")
    if args.stage in ("report","all"):command("scripts/v2_report.py")

