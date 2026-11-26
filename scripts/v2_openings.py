"""Expand each documented Gomocup 2025 mother opening only with moves in its tournament PSQs."""
import argparse
from collections import defaultdict
import hashlib
from itertools import islice
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import zipfile
from bit_gate import board_command
from datasets import DATA,ROOT,SEED,canonical,dump,sha
from rules import valid_history

OUT=DATA/"v2"
SOURCE=OUT/"gomocup2025results.zip"
URL="https://gomocup.org/static/tournaments/2025/results/gomocup2025results.zip"


def extract(archive=None):
    OUT.mkdir(exist_ok=True)
    if archive and archive.resolve()!=SOURCE.resolve():
        shutil.copy2(archive,SOURCE)
    assert sha(SOURCE)==json.loads((DATA/"corpus_metadata.json").read_bytes())["games_archive_sha256"]
    candidates=[]
    with zipfile.ZipFile(SOURCE) as z:
        for rule,label in ((1,"standard"),(2,"renju")):
            fname="openings/openings_"+label+".txt"
            blob=z.read(fname)
            assert blob==(DATA/"opening_sources"/f"2025_{label}_source.txt").read_bytes() or (
                    blob.splitlines()==(DATA/"opening_sources"/f"2025_{label}_source.txt").read_bytes().splitlines())
            families={}
            buckets=defaultdict(dict)
            for n,line in enumerate(blob.splitlines(),1):
                vals=list(map(int,re.findall(rb"-?\d+",line)))
                if not vals: continue
                moves=[x+7+(y+7)*15 for x,y in zip(vals[::2],vals[1::2],strict=True)]
                key=canonical(moves)
                families[len(moves),key]=n
                buckets[n][key]=dict(rule=rule,mother=n,base_plies=len(moves),moves=moves,source=fname,continuation_plies=0)
            assert len(families)==12
            lens=sorted({n for n,_ in families})
            names=sorted(n for n in z.namelist() if n.lower().startswith(label) and n.endswith(".psq"))
            random.Random(SEED+rule).shuffle(names)
            for name in names:
                coords=[]
                for line in z.read(name).splitlines()[1:]:
                    m=re.fullmatch(rb"(\d+),(\d+),(\d+)",line)
                    if not m: break
                    coords.append(int(m[1])-1+(int(m[2])-1)*15)
                possible=[(n,families[n,canonical(coords[:n])]) for n in lens
                          if len(coords)>=n and (n,canonical(coords[:n])) in families]
                assert len(possible)==1,name
                base,n=possible[0]
                for extra in range(2,9):
                    if len(coords)<=base+extra: continue
                    history=coords[:base+extra]
                    key=canonical(history)
                    if key not in buckets[n]:
                        buckets[n][key]=dict(rule=rule,mother=n,base_plies=base,moves=history,source=name,
                                            continuation_plies=extra)
            for n,table in sorted(buckets.items()):
                native=next(r for r in table.values() if not r["continuation_plies"])
                more=[r for r in table.values() if r["continuation_plies"]]
                random.Random(SEED+rule*100+n).shuffle(more)
                # Equal shortlist budgets prevent a more popular mother dominating the selection.
                chosen=[native]+list(islice((r for r in more if valid_history(r["moves"],rule)),96))
                assert len(chosen)==97 and valid_history(native["moves"],rule)
                candidates.extend(chosen)
    dump(OUT/"candidates.json",candidates)


def screen():
    rows=json.loads((OUT/"candidates.json").read_bytes())
    exe=ROOT/"pbrain-rapfi-orig.exe"
    for rule in (1,2):
        group=[r for r in rows if r["rule"]==rule]
        cmds=["START 15","INFO thread_num 1","INFO pondering 0","INFO hash_size 8192",f"INFO rule {1 if rule==1 else 4}"]
        for row in group:
            cmds += [board_command(row["moves"]),"TRACESEARCH"]
        cmds.append("END")
        run=subprocess.run([str(exe),"--config",str(ROOT/"config.toml")],cwd=ROOT,
            input=("\n".join(cmds)+"\n").encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            timeout=150,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        assert b"ERROR " not in run.stdout
        outputs=re.findall(rb"MESSAGE Evaluator Eval\[Black\]: (-?\d+) WR: ([.\d]+) LR: ([.\d]+) DR: ([.\d]+)",run.stdout)
        for row,vals in zip(group,outputs,strict=True):
            value,w,l,d=map(float,vals)
            row["baseline_black_score"]=w+.5*d
            row["baseline_wld"]=[w,l,d]
    chosen=[]
    for rule,prefix in ((1,"S"),(2,"R")):
        families=list(range(1,13))
        random.Random(SEED+rule).shuffle(families)
        selected=[]
        keys=set()
        for j,mother in enumerate(families):
            group=[r for r in rows if r["rule"]==rule and r["mother"]==mother]
            native=next(r for r in group if not r["continuation_plies"])
            pool=sorted((r for r in group if r["continuation_plies"]),key=lambda r:abs(r["baseline_black_score"]-.5))
            picked=0
            for row in [native]+pool:
                key=canonical(row["moves"])
                if key in keys:continue
                keys.add(key)
                selected.append(row)
                picked+=1
                if picked==12+(j<6):break
        assert len(keys)==len(selected)==150
        random.Random(SEED+rule*17).shuffle(selected)
        chosen.extend(dict(id=f"{prefix}{i:03d}",**r) for i,r in enumerate(selected,1))
    dump(OUT/"screen.json",rows)
    dump(OUT/"openings.json",chosen)
    dump(OUT/"opening_design.json",dict(official_url=URL,archive_sha256=sha(SOURCE),original_exe=sha(exe),
      rows_per_rule=12,shortlisted=len(rows),selected=len(chosen),seed=SEED,mother_quotas=[12,13],
      match_book_sha256=sha(OUT/"openings.json"),score="original TRACESEARCH black W+D/2 (4 decimal output)",
      matching="complete mother position under D4, within-color order ignored for identity; tournament coordinate order retained"))
    print("Frozen",len(chosen),"unique 2025 positions",sha(OUT/"openings.json"),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["extract","screen","all"])
    p.add_argument("--archive",type=Path)
    a=p.parse_args()
    if a.stage in ("all","extract"):extract(a.archive)
    if a.stage in ("all","screen"):screen()

