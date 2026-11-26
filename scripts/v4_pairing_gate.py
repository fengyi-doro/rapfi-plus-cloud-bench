"""Audit the production search windows and root witnesses with F10's independent coordinate oracle."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from datasets import ROOT, dump, sha


def audit_release(research, dest):
    sys.path.insert(0, str(research))
    import audit  # F10's Python oracle pins itself to efficiency core 12.
    from rank_reference import graph, outcome
    from rules import bits, exact, forbidden
    from engine import Engine
    sys.path.insert(0, str(ROOT))
    from build import compile_unit, link

    dest.mkdir(parents=True, exist_ok=True)
    games = {}
    for line in (research/"games.txt").read_text().splitlines():
        name, rule, count, *moves = line.split()
        assert int(count) == len(moves)
        games[name] = (int(rule), list(map(int, moves)), False, "F10")
    v3 = json.loads((ROOT/"validation/v3/pairing.json").read_bytes())
    for name in v3["game_ids"]:
        record = json.loads((ROOT/"match_v2/games"/(name+".json")).read_bytes())
        games["v3_"+name] = record["rule"], record["moves"], False, "v3"
    for i, case in enumerate(json.loads((ROOT/"validation/v3/rank_controls.json").read_bytes())):
        games["control_"+str(i)] = case["rule"], case["moves"], True, "control"

    manifest = json.loads((ROOT/"build/v4/manifest.json").read_bytes())
    exe = ROOT/"build/v4/pairing_probe.exe"
    obj = exe.with_suffix(".o")
    tree = ROOT/"src_plus"
    subprocess.run(compile_unit(ROOT/"scripts/v4_pairing_probe.cpp", tree, obj,
                                [*manifest["flags"], "-I"+str(tree)]), check=True)
    subprocess.run(link([obj],exe), check=True)
    payload = "".join(f"{name} {rule} {len(moves)} {int(last)} " + " ".join(map(str,moves))+"\n"
                      for name,(rule,moves,last,_) in games.items())
    evidence = dest/"pairing_witnesses.jsonl"
    with evidence.open("wb") as stream:
        subprocess.run([exe], input=payload.encode("ascii"), stdout=stream, check=True)

    counts, oracle, first = defaultdict(Counter), Counter(), {}
    time_samples, color_counts = [], Counter()
    late_v3 = {(name, ply): truth for name,ply,truth,*_ in
               json.loads((ROOT/"validation/v3/pairing_samples.json").read_bytes())}
    controls = json.loads((ROOT/"validation/v3/rank_controls.json").read_bytes())
    with evidence.open() as stream:
        for name,(rule,moves,last,group) in games.items():
            board = [0,0]
            c = counts[group]
            c["games"] += 1
            for ply in range(len(moves)+int(last)):
                tag, p, lower, upper, interior_draw, lo, hi, ns, mirror, root_lo, root_hi = json.loads(next(stream))
                assert (tag,p) == (name,ply)
                stm = ply % 2
                c["positions"] += 1
                if lower or upper or root_lo is not None:
                    free = audit.BOARD ^ (board[0]|board[1])
                    families = [audit.literal_family(board[s],board[1-s],exact(rule,s)) for s in (0,1)]
                    blanket = rule == 2 and audit.universal_black(*board)

                    def check(witness, defender, is_first):
                        attacker = 1-defender
                        proof = [tag, p, int(not is_first), *witness]
                        audit.verify_new(proof, [m for m,_ in families[attacker]], *board,
                                         rule, is_first, defender, blanket, oracle)
                        if witness[0] >= 0:
                            assert not forbidden(board[defender],board[attacker],witness[0],rule,defender)
                        c["witnesses_checked"] += 1

                    for bound, proof, side in [(lower,lo,stm), (upper,hi,1-stm)]:
                        if not bound: continue
                        if proof is not None:
                            check(proof, side, side==stm)
                        elif side==0 and mirror is not None:
                            check(mirror, side, side==stm)
                        else:
                            assert rule != 2 or side or blanket
                            mine, theirs = graph(families[side],True), graph(families[1-side],False)
                            rank = outcome(free,mine,theirs) if side==stm else -outcome(free,theirs,mine)
                            assert rank >= 0, (tag,ply,"rank",side)
                            c["rank_bounds_checked"] += 1
                    assert not interior_draw or (lower and upper)
                    if root_lo is not None:
                        assert root_hi is not None
                        check(root_lo, stm, True)
                        check(root_hi, 1-stm, False)
                        if group == "F10" and (rule not in first or ply < first[rule]["ply"]):
                            first[rule] = dict(game=name,ply=ply,pre=root_lo[0])
                        c["root_exact"] += 1
                    if mirror is not None and group == "F10": time_samples.append(ns)
                    if group == "F10" and rule == 2:
                        color_counts["black_lower" if stm==0 else "white_lower"] += lower
                        color_counts["white_upper" if stm==0 else "black_upper"] += upper
                    c["interior_exact"] += interior_draw
                    c["lower"] += lower
                    c["upper"] += upper
                truth = late_v3.get((name[3:],ply)) if group=="v3" else None
                if 225-ply <= 6 or (group=="control" and ply==len(moves)):
                    truth, states = audit.exact_value(*board, stm, rule)
                    c["literal_positions"] += 1
                    c["literal_states"] += states
                if truth is not None:
                    assert not lower or truth>=0, (tag,ply,truth,"lower")
                    assert not upper or truth<=0, (tag,ply,truth,"upper")
                    assert not (interior_draw or root_lo is not None) or truth==0, (tag,ply,truth,"draw")
                if group=="control" and ply==len(moves):
                    assert truth==controls[int(name.split("_")[-1])]["truth"]
                    c["decisive_controls"] += 1
                if ply < len(moves):
                    board[stm] |= audit.B[moves[ply]]
            assert tuple(board) == bits(moves)
        assert next(stream, None) is None
    assert counts["F10"]["positions"]==24691 and counts["F10"]["literal_positions"]==402
    assert counts["control"]["decisive_controls"]==8 and all(color_counts.values())
    assert first[1]["ply"]<=95 and first[2]["ply"]<=109, first
    fixtures={}
    with Engine("v4",12) as engine:
        for rule, case in first.items():
            name, ply, pre=case["game"],case["ply"],case["pre"]
            engine.select(rule)
            engine.sync("INFO timeout_turn 1000\nINFO timeout_match 10000\nINFO time_left 10000\n"
                        "INFO max_depth 12\nINFO show_detail 2")
            engine.clean_position(games[name][1][:ply])
            move=engine.think()
            engine.sync("STOP")
            assert (move["depth"],move["nodes"],move["eval"])==(0,0,"0"), (name,move)
            assert pre<0 or move["x"]+15*move["y"]==pre, (name,pre,move)
            fixtures[name]=dict(ply=ply,pre=pre,**move)
    dump(dest/"root_dispatch.json",dict(exe=manifest["sha256"],fixtures=fixtures))
    times = sorted(time_samples)
    result = dict(exe=manifest["sha256"], probe_sha256=sha(exe), corpus_sha256=sha(research/"games.txt"),
                  evidence_sha256=sha(evidence), counts=dict(counts), renju=color_counts,
                  independently_checked=oracle, earliest_root_draw=first, false_claims=0,
                  cheap_black_success_ns=dict(core=12,n=len(times),p50=statistics.median(times),
                                             p95=times[math.ceil(.95*len(times))-1]))
    dump(dest/"pairing.json",result)
    print(json.dumps(result, ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("research",type=Path)
    parser.add_argument("--out",type=Path,default=ROOT/"validation/v4")
    args=parser.parse_args()
    audit_release(args.research.resolve(),args.out.resolve())
